"""Session and child-process isolation without running an editor or model."""

import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import runtime


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


if __name__ == "__main__":
    unittest.main()
