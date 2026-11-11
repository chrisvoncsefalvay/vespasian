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
            "model": "test/model", "editor_model": "test/editor", "budget": 0.5,
            "max_turns": 2, "max_new_tokens": 16, "example_limit": 5,
        }

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
