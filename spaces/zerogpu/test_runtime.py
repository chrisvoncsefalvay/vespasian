"""Session and child-process isolation without running an editor or model."""

import json
import contextvars
import io
import os
from pathlib import Path
import shutil
import stat
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import runtime
from bridge import Request


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bridge = SimpleNamespace(path=self.root / "worker.sock", secret="session-secret")
        self.config = {
            "goal": "Answer briefly", "editor": "openai", "api_key": "visitor-api-key",
            "model": "test/model", "editor_model": "gpt-6.1-sol", "budget": 0.5,
            "max_turns": 2, "max_new_tokens": 16, "example_limit": 5,
        }

    def test_optional_turn_cap_and_reasoning_reach_shared_controller(self):
        binary = self.root / "bin/vespasian"
        binary.parent.mkdir()
        binary.write_text("fixture")
        (self.root / "python_bridge.py").write_text("fixture")
        for turns in (None, 1, 75, 4294967295):
            for effort in ("low", "medium", "high"):
                config = {**self.config, "budget": 10, "max_turns": turns, "reasoning_effort": effort}
                runtime.validate(config)
                with patch.object(runtime, "ROOT", self.root), patch.dict(os.environ, {"VESPASIAN_SPACE_BINARY": str(binary)}):
                    args = runtime.command(config, "data.jsonl", "output", "revision")
                self.assertEqual(args[args.index("--reasoning-effort") + 1], effort)
                self.assertEqual(args[args.index("--budget") + 1], "10")
                self.assertEqual(args[args.index("--budget-policy") + 1], "strict")
                for flag, value in (("--seed", "42"), ("--max-seq-length", "1024"), ("--max-output-tokens", "2048"), ("--preserve-tolerance", "0")):
                    self.assertEqual(args[args.index(flag) + 1], value)
                if turns is None:
                    self.assertNotIn("--max-turns", args)
                else:
                    self.assertEqual(args[args.index("--max-turns") + 1], str(turns))
        for invalid in (0, -1, 1.5, True, float("nan"), float("inf"), 4294967296):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runtime.validate({**self.config, "max_turns": invalid})
        with self.assertRaisesRegex(ValueError, "reasoning"):
            runtime.validate({**self.config, "reasoning_effort": "none"})


    def test_all_quick_starters_use_the_paid_controller_with_strict_budget(self):
        import setup_helpers as setup
        binary = self.root / "bin/vespasian"
        binary.parent.mkdir()
        binary.write_text("fixture")
        (self.root / "python_bridge.py").write_text("fixture")
        for card in setup.quick_starter_cards():
            with self.subTest(starter=card["id"]):
                config = {**self.config, **setup.preset_config(card["id"])}
                runtime.validate(config)
                with patch.object(runtime, "ROOT", self.root), patch.dict(os.environ, {}, clear=True):
                    args = runtime.command(config, "prepared-gsm8k.jsonl", self.root / "output", "a" * 40)
                self.assertEqual(args[0], str(binary))
                self.assertEqual(args[1], "prepared-gsm8k.jsonl")
                self.assertEqual(args[args.index("--editor") + 1], "openai")
                self.assertEqual(args[args.index("--budget-policy") + 1], "strict")
                self.assertEqual(float(args[args.index("--budget") + 1]), 10)
                self.assertEqual(args[args.index("--reasoning-effort") + 1], "low")
                self.assertEqual(args[args.index("--thinking-mode") + 1], "off")
                self.assertEqual(args[args.index("--max-seq-length") + 1], "2048")
                self.assertEqual(args[args.index("--max-relative-norm") + 1], "0.1")
                self.assertEqual(args[args.index("--metric") + 1], "numeric_match")
                self.assertIn("--save-trajectory", args)
                self.assertNotIn("--max-turns", args)
                self.assertNotIn("evaluate", args)
                self.assertNotIn("--patch-backup", args)
                self.assertNotIn(config["api_key"], args)

    def test_inspection_settings_reach_controller_and_public_metadata(self):
        binary = self.root / "bin/vespasian"
        binary.parent.mkdir()
        binary.write_text("fixture")
        (self.root / "python_bridge.py").write_text("fixture")
        config = {**self.config, "thinking_mode": "on", "max_new_tokens": 32768,
                  "max_seq_length": 131072, "max_relative_norm": 0.75}
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "512"):
                runtime.validate(config)
        with patch.dict(os.environ, {"VESPASIAN_SPACE_MAX_NEW_TOKENS": "32768", "VESPASIAN_SPACE_MAX_SEQ_LENGTH": "131072"}):
            runtime.validate(config)
            with patch.object(runtime, "ROOT", self.root), patch.dict(os.environ, {"VESPASIAN_SPACE_BINARY": str(binary)}):
                args = runtime.command(config, "data.jsonl", "output", "revision")
        for flag, value in (("--thinking-mode", "on"), ("--max-new-tokens", "32768"),
                            ("--max-seq-length", "131072"), ("--max-relative-norm", "0.75")):
            self.assertEqual(args[args.index(flag) + 1], value)
        run = runtime.Run("visitor", config)
        self.addCleanup(shutil.rmtree, run.root)
        for field in ("thinking_mode", "max_new_tokens", "max_seq_length", "max_relative_norm"):
            self.assertEqual(run.state()["config"][field], config[field])
        self.assertNotIn("api_key", run.state()["config"])
        legacy = runtime.Run("visitor", self.config)
        self.addCleanup(shutil.rmtree, legacy.root)
        self.assertEqual(legacy.config["thinking_mode"], "off")
        self.assertEqual(legacy.config["max_seq_length"], 1024)
        self.assertEqual(legacy.config["max_relative_norm"], 0.1)

    def test_inspection_settings_reject_invalid_and_incompatible_values(self):
        for field, values in (("thinking_mode", ("none", True, None)),
                              ("max_seq_length", (31, 2049, 32.5, True, float("nan"))),
                              ("max_relative_norm", (0, 0.0000001, -0.1, 1.01, True, float("nan"), float("inf")))):
            for value in values:
                with self.subTest(field=field, value=value), patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
                    runtime.validate({**self.config, field: value})
        with self.assertRaisesRegex(ValueError, "compatible chat template"):
            runtime.validate({**self.config, "thinking_mode": "on", "prompt_format": "plain"})
        for thinking in ("off", "auto"):
            runtime.validate({**self.config, "thinking_mode": thinking, "prompt_format": "plain", "max_relative_norm": 1})

    def test_task_output_prompt_and_private_backup_flags_are_explicit(self):
        binary = self.root / "bin/vespasian"
        binary.parent.mkdir()
        binary.write_text("fixture")
        (self.root / "python_bridge.py").write_text("fixture")
        config = {**self.config, "hf_task": "text-generation", "save_format": "patch", "prompt_format": "plain", "patch_backup": True, "patch_repo": "visitor/patches", "patch_create": True}
        with patch.object(runtime, "ROOT", self.root), patch.dict(os.environ, {"VESPASIAN_SPACE_BINARY": str(binary)}):
            args = runtime.command(config, "data.jsonl", "output", "revision")
            default = runtime.command(self.config, "data.jsonl", "output", "revision")
        for flag, value in (("--task", "text-generation"), ("--save-format", "patch"), ("--prompt-format", "plain"), ("--patch-repo", "visitor/patches")):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--patch-backup", args)
        self.assertIn("--patch-create", args)
        self.assertNotIn("--patch-backup", default)
        with self.assertRaisesRegex(ValueError, "session token"):
            runtime.begin("visitor", config, False)
        with self.assertRaises(ValueError):
            runtime.validate({**config, "patch_backup": "true"})

    def test_child_drops_owner_credentials_and_has_its_own_home(self):
        ambient = {
            "PATH": "/usr/bin", "LANG": "C.UTF-8", "HF_TOKEN": "owner-hf-token",
            "OPENAI_API_KEY": "owner-openai-key", "ANTHROPIC_API_KEY": "owner-anthropic-key",
            "AWS_SECRET_ACCESS_KEY": "owner-aws-secret", "GITHUB_TOKEN": "owner-github-token",
            "HOME": "/owner", "HF_HOME": "/owner/hf", "PYTHONPATH": "/owner/modules",
            "HF_TOKEN_PATH": "/owner/secret-token", "HF_DATASETS_CACHE": "/owner/datasets",
        }
        with patch.dict(os.environ, ambient, clear=True):
            result = runtime.child_environment(self.root, self.bridge)
        for name in ("HF_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "PYTHONPATH", "HF_TOKEN_PATH", "HF_DATASETS_CACHE"):
            self.assertNotIn(name, result)
        self.assertEqual(result["PATH"], "/usr/bin")
        self.assertEqual(result["HOME"], str(self.root / "home"))
        self.assertEqual(result["HF_HOME"], str(self.root / "home/.cache/huggingface"))
        self.assertEqual(result["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertEqual(stat.S_IMODE((self.root / "home").stat().st_mode), 0o700)

    def test_child_only_receives_requested_visitor_provider_key(self):
        for editor, expected, absent in (("openai", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
                                         ("anthropic", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
            with self.subTest(editor=editor):
                result = runtime.child_environment(self.root, self.bridge, token="visitor-hf-token", api_key="visitor-provider-key", editor=editor)
                self.assertEqual(result["HF_TOKEN"], "visitor-hf-token")
                self.assertEqual(result["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "0")
                self.assertEqual(result[expected], "visitor-provider-key")
                self.assertNotIn(absent, result)
                self.assertEqual(result["VESPASIAN_WORKER_SECRET"], "session-secret")

    def test_run_state_redacts_secrets_and_session_directory(self):
        run = runtime.Run("session-a", self.config, "visitor-hf-token")
        self.addCleanup(shutil.rmtree, run.root)
        run.event({"type": "error", "message": f"visitor-hf-token visitor-api-key {run.root}",
                   "nested": [{"visitor-api-key": "visitor-hf-token"}]})
        encoded = json.dumps(run.state())
        self.assertNotIn("visitor-hf-token", encoded)
        self.assertNotIn("visitor-api-key", encoded)
        self.assertNotIn(str(run.root), encoded)
        self.assertIn("[redacted]", encoded)
        self.assertEqual(stat.S_IMODE(run.root.stat().st_mode), 0o700)

    def test_run_id_does_not_grant_access_to_other_session(self):
        run = runtime.Run("session-a", self.config)
        self.addCleanup(shutil.rmtree, run.root)
        with patch.dict(runtime.RUNS, {run.id: run}, clear=True):
            self.assertIs(runtime.get_run("session-a", {"run_id": run.id}), run)
            with self.assertRaisesRegex(ValueError, "this session"):
                runtime.get_run("session-b", {"run_id": run.id})

    def test_completed_elapsed_time_stays_fixed_during_result_inspection(self):
        run = runtime.Run("session-a", self.config)
        self.addCleanup(shutil.rmtree, run.root)
        run.started = 10.0
        run.finished_at = 32.5
        run.finished = True
        with patch.object(runtime.time, "monotonic", return_value=999.0):
            self.assertEqual(run.state()["elapsed"], 22.5)

    def test_command_restores_executable_modes_without_exposing_tokens(self):
        binary = self.root / "bin/vespasian"
        binary.parent.mkdir()
        binary.write_text("binary fixture")
        shim = self.root / "python_bridge.py"
        shim.write_text("launcher fixture")
        binary.chmod(0o644)
        shim.chmod(0o644)
        with patch.object(runtime, "ROOT", self.root), patch.dict(os.environ, {}, clear=True):
            command = runtime.command(self.config, self.root / "input.jsonl", self.root / "output", "a" * 40)
        self.assertTrue(binary.stat().st_mode & stat.S_IXUSR)
        self.assertTrue(shim.stat().st_mode & stat.S_IXUSR)
        self.assertEqual(command[command.index("--python") + 1], str(shim))
        self.assertEqual(command[command.index("--revision") + 1], "a" * 40)
        self.assertNotIn("visitor-api-key", " ".join(command))

    def test_untouched_optional_controls_produce_valid_process_arguments(self):
        binary = self.root / "bin/vespasian"
        binary.parent.mkdir()
        binary.write_text("binary fixture")
        (self.root / "python_bridge.py").write_text("launcher fixture")
        with patch.object(runtime, "ROOT", self.root), patch.dict(os.environ, {}, clear=True):
            args = runtime.command({**self.config, "controls": None}, self.root / "input.jsonl", self.root / "output", "a" * 40)
        self.assertEqual(args[args.index("--controls") + 1], "")
        self.assertTrue(all(isinstance(value, str) for value in args))

    def test_strict_prices_are_checked_before_command_and_zero_drop_survives(self):
        binary = self.root / "bin/vespasian"
        binary.parent.mkdir()
        binary.write_text("binary fixture")
        (self.root / "python_bridge.py").write_text("launcher fixture")
        config = {**self.config, "health_max_drop": 0}
        with patch.object(runtime, "ROOT", self.root), patch.dict(os.environ, {}, clear=True):
            args = runtime.command(config, self.root / "input.jsonl", self.root / "output", "a" * 40)
            tracked = runtime.command({**config, "input_price": " ", "output_price": None, "health_max_drop": None}, self.root / "input.jsonl", self.root / "output", "a" * 40)
            for invalid in ({"input_price": 2}, {"output_price": 10},
                            {"editor_model": "custom-model", "input_price": 2, "output_price": 10},
                            {"editor_model": "custom-model"}, {"editor": "anthropic", "editor_model": "gpt-6.1-sol"}):
                with self.subTest(invalid=invalid):
                    for operation in (lambda c: runtime.validate(c), lambda c: runtime.command(c, "data", "output", "revision")):
                        with self.assertRaisesRegex(ValueError, "--budget-policy auto"):
                            operation({**config, **invalid})
        self.assertEqual(args[args.index("--health-max-drop") + 1], "0")
        self.assertEqual(args[args.index("--budget-policy") + 1], "strict")
        for flag in ("--input-price", "--output-price", "--health-max-drop"):
            self.assertNotIn(flag, tracked)
        for editor, model in (("openai", "gpt-6.1-sol"), ("anthropic", "claude-sonnet-5-5")):
            runtime.validate({**config, "editor": editor, "editor_model": model})

    def test_completed_report_exposes_only_public_results_and_redacts_secrets(self):
        run = runtime.Run("session-a",self.config,"visitor-hf-token")
        self.addCleanup(shutil.rmtree,run.root,ignore_errors=True)
        run.output.mkdir()
        report = {"metric":"exact_match","best_turn":1,"reload_verified":True,
                  "final_metrics":{"test":{"count":2,"exact_match":.5}},
                  "baseline_metrics":{"validation":{"count":3,"exact_match":0}},
                  "checkpoint":str(run.output/"checkpoint"),
                  "final_scoring_error":f"visitor-api-key visitor-hf-token {run.root}",
                  "history":[{"prompt":"private dataset","api_key":"visitor-api-key"}],
                  "api_key":"visitor-api-key","hf_token":"visitor-hf-token", "unknown":"hidden",
                  "budget":{"spent_usd":.25,"note":"visitor-api-key"}}
        (run.output/"report.json").write_text(json.dumps(report))
        run.load_report()
        public = run.state()["report"]
        self.assertEqual(public["final_metrics"]["test"],{"count":2,"exact_match":.5})
        self.assertEqual(public["checkpoint"],"checkpoint")
        self.assertTrue(public["reload_verified"])
        self.assertFalse(set(public)&{"history","api_key","hf_token","unknown"})
        encoded = json.dumps(public)
        self.assertNotIn("visitor-api-key",encoded)
        self.assertNotIn("visitor-hf-token",encoded)
        self.assertNotIn(str(run.root),encoded)
        self.assertIn("[redacted]",encoded)
        run.secrets = ()
        self.assertNotIn("visitor-api-key",json.dumps(run.state()))

    def test_report_preserves_measured_behavior_and_redacts_secrets(self):
        run = runtime.Run("visitor", self.config)
        self.addCleanup(shutil.rmtree, run.root)
        run.output.mkdir()
        behavior = {"baseline_behavior": {"looping": 2, "note": "visitor-api-key"},
                    "best_behavior": {"looping": 0}, "final_behavior": {"truncated": 1},
                    "unknown": "must stay private"}
        (run.output / "report.json").write_text(json.dumps(behavior))
        run.load_report()
        public = run.state()["report"]
        self.assertEqual(set(public), {"baseline_behavior", "best_behavior", "final_behavior"})
        self.assertEqual(public["best_behavior"], {"looping": 0})
        self.assertEqual(public["final_behavior"], {"truncated": 1})
        self.assertNotIn("visitor-api-key", json.dumps(public))

    def test_missing_or_incomplete_report_keeps_public_state_usable(self):
        run = runtime.Run("session-a",self.config)
        self.addCleanup(shutil.rmtree,run.root,ignore_errors=True)
        run.load_report()
        self.assertEqual(run.state()["report"],{})
        run.output.mkdir()
        (run.output/"report.json").write_text('{"final_metrics":')
        run.load_report()
        self.assertEqual(run.state()["report"],{})

    def test_preservation_limit_is_public_but_editor_credentials_are_not(self):
        run = runtime.Run("session-a",{**self.config,"health_max_drop":.1,"hf_token":"visitor-hf-token"})
        self.addCleanup(shutil.rmtree,run.root,ignore_errors=True)
        config = run.state()["config"]
        self.assertEqual(config["health_max_drop"],.1)
        self.assertNotIn("api_key",config)
        self.assertNotIn("hf_token",config)

    def test_untouched_required_textboxes_are_rejected_before_start(self):
        for field, message in (("goal", "editing objective"), ("api_key", "editor API key")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                runtime.validate({**self.config, field: None})

    def test_cleanup_removes_only_expired_completed_runs(self):
        now = 10000.0
        expired = runtime.Run("expired-session", self.config)
        retained = runtime.Run("recent-session", self.config)
        active = runtime.Run("active-session", self.config)
        for run in (expired, retained, active):
            self.addCleanup(shutil.rmtree, run.root, ignore_errors=True)
            (run.root / "private-data.jsonl").write_text("private dataset")
        expired.finished = retained.finished = True
        expired.finished_at = now - runtime.RETENTION_SECONDS
        retained.finished_at = now - runtime.RETENTION_SECONDS + 1
        active.started = now - runtime.RETENTION_SECONDS * 2
        with patch.dict(runtime.RUNS, {run.id: run for run in (expired, retained, active)}, clear=True):
            runtime.cleanup(now=now)
            self.assertNotIn(expired.id, runtime.RUNS)
            self.assertFalse(expired.root.exists())
            self.assertEqual(set(runtime.RUNS), {retained.id, active.id})
            self.assertTrue((retained.root / "private-data.jsonl").is_file())
            self.assertTrue((active.root / "private-data.jsonl").is_file())
            runtime.cleanup(now=now + 1)
            self.assertEqual(set(runtime.RUNS), {active.id})
            self.assertFalse(retained.root.exists())

    def test_failed_run_clears_session_home_and_engine_state_even_if_archive_fails(self):
        run = runtime.Run("session-a", self.config, "visitor-hf-token")
        self.addCleanup(shutil.rmtree, run.root, ignore_errors=True)
        for name in ("home", "engines"):
            (run.root / name).mkdir()
            (run.root / name / "private-state").write_text("sensitive state")
        with patch.object(run, "bundle", side_effect=OSError("disk full")):
            states = list(run.execute(self.config, "visitor-hf-token", [], Mock()))
        self.assertTrue(states[-1]["finished"])
        self.assertIsNotNone(run.finished_at)
        self.assertEqual(run.secrets, ())
        self.assertFalse((run.root / "home").exists())
        self.assertFalse((run.root / "engines").exists())

    def make_pending_run(self):
        run = runtime.Run("session-a", self.config)
        self.addCleanup(shutil.rmtree, run.root, ignore_errors=True)
        self.addCleanup(run._clear_dispatch)
        request = Request(run.root / "engines/test", {"id": 1, "command": "evaluate"})
        run.bridge = SimpleNamespace(next=Mock(side_effect=[request, None]))
        return run, request

    def test_each_allocation_uses_a_new_request_without_restarting_controller(self):
        run, request = self.make_pending_run()
        credential = contextvars.ContextVar("test_credential")
        used, calls = [], []

        def dispatch(directory, payload, *, boundaries):
            self.assertTrue(boundaries)
            calls.append((directory, payload))
            for batch in range(3):
                yield {"type": "gpu_allocation"}
                current = credential.get()
                if current in used:
                    raise RuntimeError("Expired ZeroGPU proxy token")
                used.append(current)
                yield {"type": "sample", "batch": batch}
            yield {"type": "result", "id": 1, "ok": True}

        for index in range(3):
            token = credential.set(f"request-{index}")
            try:
                list(run.advance(dispatch))
            finally:
                credential.reset(token)
            self.assertEqual(len(used), index + 1)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(run.pending_dispatch)
        responses = list(request.responses.queue)
        self.assertEqual([row["type"] for row in responses], ["sample"] * 3 + ["result"])
        self.assertTrue(responses[-1]["ok"])

    def test_overlapping_browser_events_do_not_advance_a_run_twice(self):
        run, request = self.make_pending_run()
        entered, release = threading.Event(), threading.Event()
        calls = []

        def dispatch(*args, **kwargs):
            yield {"type": "gpu_allocation"}
            calls.append(threading.get_ident())
            entered.set()
            self.assertTrue(release.wait(2))
            yield {"type": "result", "id": 1, "ok": True}

        thread = threading.Thread(target=lambda: list(run.advance(dispatch)))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            states = list(run.advance(dispatch))
            self.assertEqual(len(states), 1)
            self.assertEqual(len(calls), 1)
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(request.responses.qsize(), 1)

    def test_preparation_is_background_and_does_not_capture_the_browser_request(self):
        run = runtime.Run("session-a", self.config)
        self.addCleanup(shutil.rmtree, run.root, ignore_errors=True)
        entered, release = threading.Event(), threading.Event()
        browser_context = contextvars.ContextVar("browser_request", default=None)
        observed = []

        def prepare(*args):
            observed.append(browser_context.get())
            entered.set()
            release.wait(2)
            raise ValueError("fixture preparation finished")

        token = browser_context.set("old-request")
        try:
            with patch.object(runtime.hub, "prepare_data", side_effect=prepare):
                run.start(self.config, False, [{"id": "test/model", "available": True, "revision": "test"}])
                try:
                    self.assertTrue(entered.wait(2))
                    dispatch = Mock()
                    self.assertFalse(list(run.advance(dispatch))[-1]["finished"])
                    dispatch.assert_not_called()
                    with self.assertRaisesRegex(ValueError, "already started"):
                        run.start(self.config, False, [])
                finally:
                    release.set()
                    run.controller.join(2)
        finally:
            browser_context.reset(token)
        self.assertEqual(observed, [None])
        self.assertTrue(run.finished)
        self.assertEqual(run.secrets, ())

    def test_stop_between_batches_prevents_another_gpu_allocation(self):
        run, request = self.make_pending_run()
        calls, closed = [], []

        def dispatch(*args, **kwargs):
            try:
                for index in range(2):
                    yield {"type": "gpu_allocation"}
                    calls.append(index)
                    yield {"type": "sample", "batch": index}
            finally:
                closed.append(True)

        list(run.advance(dispatch))
        run.stop()
        list(run.advance(dispatch))
        self.assertEqual(calls, [0])
        run._clear_dispatch()
        self.assertEqual(closed, [True])

    def test_cpu_only_operations_do_not_need_an_allocation_marker(self):
        run, request = self.make_pending_run()
        def dispatch(*args, **kwargs):
            yield {"type": "result", "id": 1, "ok": True}
        list(run.advance(dispatch))
        self.assertIsNone(run.pending_dispatch)
        self.assertEqual(request.responses.get_nowait(), {"type": "result", "id": 1, "ok": True})

    def test_cancelled_browser_generator_releases_pending_operation(self):
        run, request = self.make_pending_run()
        closed = []
        def dispatch(*args, **kwargs):
            try:
                yield {"type": "gpu_allocation"}
                yield {"type": "sample"}
                yield {"type": "gpu_allocation"}
            finally:
                closed.append(True)
        list(run.advance(dispatch))
        response = run.advance(dispatch)
        next(response)
        response.close()
        self.assertTrue(run.cancelled.is_set())
        self.assertEqual(closed, [True])
        self.assertFalse(run.advance_lock.locked())

    def test_failed_pending_close_does_not_skip_final_cleanup(self):
        run = runtime.Run("session-a", self.config)
        self.addCleanup(shutil.rmtree, run.root, ignore_errors=True)
        run.pending_dispatch = SimpleNamespace(close=Mock(side_effect=RuntimeError("visitor-api-key release failed")))
        for name in ("engines", "home"):
            (run.root / name).mkdir()
        run._control(self.config, False, [])
        self.assertTrue(run.finished)
        self.assertEqual(run.error, "[redacted] release failed")
        self.assertEqual(run.secrets, ())
        self.assertFalse((run.root / "engines").exists())
        self.assertFalse((run.root / "home").exists())

    def test_concurrent_drains_preserve_controller_event_order(self):
        run, _ = self.make_pending_run()
        run.messages.put({"type": "phase", "message": "A"})
        run.messages.put({"type": "phase", "message": "B"})
        first_entered, second_entered, release = threading.Event(), threading.Event(), threading.Event()
        record = run.event
        def event(value):
            if value["message"] == "A":
                first_entered.set()
                release.wait(2)
            else:
                second_entered.set()
            record(value)
        with patch.object(run, "event", side_effect=event):
            first = threading.Thread(target=run.drain)
            second = threading.Thread(target=run.drain)
            first.start()
            try:
                self.assertTrue(first_entered.wait(2))
                second.start()
                self.assertFalse(second_entered.wait(0.05))
            finally:
                release.set()
                first.join(2)
                if second.ident is not None:
                    second.join(2)
        self.assertEqual([event["message"] for event in run.events], ["A", "B"])

    def test_abandoned_browser_stops_controller_and_releases_session(self):
        run = runtime.Run("session-a", self.config)
        self.addCleanup(shutil.rmtree, run.root, ignore_errors=True)
        process = SimpleNamespace(returncode=None, stdout=io.StringIO(""), stderr=io.StringIO(""))
        process.poll = lambda: process.returncode
        process.send_signal = lambda value: setattr(process, "returncode", -value)
        bridge = SimpleNamespace(path=run.root / "worker.sock", secret="fixture", close=Mock())
        run.last_seen = time.monotonic() - runtime.CLIENT_TIMEOUT_SECONDS - 1
        with patch.object(runtime.hub, "prepare_data", return_value=run.root / "data.jsonl"), \
             patch.object(runtime, "Bridge", return_value=bridge), \
             patch.object(runtime, "command", return_value=["fixture"]), \
             patch.object(runtime.subprocess, "Popen", return_value=process):
            run._control(self.config, False, [{"id": "test/model", "available": True, "revision": "test"}])
        self.assertTrue(run.finished)
        self.assertTrue(run.cancelled.is_set())
        self.assertIn("browser connection", run.error)
        self.assertEqual(run.secrets, ())
        bridge.close.assert_called_once()
        self.assertFalse((run.root / "home").exists())


if __name__ == "__main__":
    unittest.main()
