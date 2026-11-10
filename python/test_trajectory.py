"""Trajectory publication contracts with a fake Hub. No remote writes occur."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import trajectory


HEAD = "a" * 40
COMMIT = "b" * 40


class HubError(Exception):
    def __init__(self, status):
        self.response = SimpleNamespace(status_code=status)
        super().__init__("Authorization: Bearer hf_secret_for_test")


def row(sequence, kind, payload):
    return {"schema": "vespasian.trajectory", "version": 1, "sequence": sequence,
            "timestamp_unix_ms": 1700000000000 + sequence, "type": kind, "payload": payload}


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.run.mkdir()
        self.rows = [
            row(0, "run_start", {"run_id": "test-run", "version": "0.1.2", "model": "Qwen/Qwen3.5-0.8B",
                                 "goal": "Copy accurately", "health_checks": [{"check": "copying", "examples": 20}]}),
            row(1, "editor_request", {"system": "Return a bounded edit", "context": {"prompt": "Supplied prompt", "target": "Reference"},
                                        "schema": {"type": "object"}, "tag": "turn-001"}),
            row(2, "editor_response", {"response": {"operation": "scale", "strength": 0.01, "rationale": "Measured correction"},
                                        "usage": {"input_tokens": 100, "output_tokens": 30}, "spent_usd": 0.02}),
            row(3, "turn", {"turn": 1, "accepted": True, "health": {"copying": {"score": 1.0, "count": 20, "metric": "exact_copy"}}}),
            row(4, "run_end", {"status": "complete", "result": {"accepted_turns": 1}}),
        ]
        self.write()
        self.api = Mock()

        def info(**kwargs):
            return SimpleNamespace(private=True, sha=kwargs.get("revision", HEAD))

        self.api.dataset_info.side_effect = info
        self.api.list_repo_files.return_value = ["README.md", "unrelated.jsonl"]
        self.uploaded = {}

        def commit(**kwargs):
            self.uploaded = {operation.path_in_repo: Path(operation.path_or_fileobj).read_bytes()
                             for operation in kwargs["operations"]}
            return SimpleNamespace(oid=COMMIT)

        def download(**kwargs):
            destination = Path(kwargs["local_dir"]) / kwargs["filename"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.uploaded[kwargs["filename"]])
            return str(destination)

        self.api.create_commit.side_effect = commit
        self.api.hf_hub_download.side_effect = download

    def write(self):
        (self.run / "trajectory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def call(self, **kwargs):
        args = {"run": self.run, "repo": "owner/trajectories", "visibility": "private", "api": self.api,
                "operation": lambda **args: SimpleNamespace(**args)}
        args.update(kwargs)
        return trajectory.publish(**args)

    def test_completed_file_is_snapshotted_exactly_and_retains_measurements(self):
        source = (self.run / "trajectory.jsonl").read_bytes()
        expected = hashlib.sha256(source).hexdigest()
        with trajectory.prepare(self.run) as prepared:
            self.assertEqual(prepared.summary["status"], "complete")
            self.assertEqual(prepared.summary["records"], 5)
            self.assertEqual(prepared.summary["trajectory"]["sha256"], expected)
            self.assertEqual((prepared.directory / f"trajectories/{expected}.jsonl").read_bytes(), source)
            self.assertEqual({item["path"] for item in prepared.files},
                             {f"trajectories/{expected}.jsonl", f"manifests/{expected}.json"})
        self.assertFalse(prepared.directory.exists())

    def test_corpus_mode_is_retained_in_trajectory_metadata(self):
        self.rows[0]["payload"]["dataset_mode"] = "corpus"
        self.write()
        with trajectory.prepare(self.run) as prepared:
            content = next((prepared.directory / item["path"]).read_text()
                           for item in prepared.files if item["path"].endswith(".jsonl"))
            self.assertEqual(json.loads(content.splitlines()[0])["payload"]["dataset_mode"], "corpus")

    def test_incomplete_failed_and_cancelled_runs_are_not_labelled_complete(self):
        for status in [None, "failed", "cancelled"]:
            with self.subTest(status=status):
                if status is None:
                    self.rows = self.rows[:4]
                else:
                    self.rows = self.rows[:4] + [row(4, "run_end", {"status": status, "result": {"error": "Stopped"}})]
                self.write()
                with trajectory.prepare(self.run) as prepared:
                    self.assertEqual(prepared.summary["status"], status or "incomplete")

    def test_optional_token_is_not_required_for_local_preparation_or_cached_login_api(self):
        with patch.dict(os.environ, {}, clear=True):
            result = self.call()
        self.assertTrue(result["verified"])
        self.assertNotIn("token", self.api.create_commit.call_args.kwargs)

    def test_existing_repo_gets_only_content_addressed_files_and_pinned_commit(self):
        result = self.call()
        self.api.create_repo.assert_not_called()
        self.api.create_commit.assert_called_once()
        args = self.api.create_commit.call_args.kwargs
        self.assertEqual(args["repo_type"], "dataset")
        self.assertEqual(args["parent_commit"], HEAD)
        self.api.list_repo_files.assert_called_once_with(repo_id="owner/trajectories", repo_type="dataset", revision=HEAD)
        self.assertEqual(len(self.uploaded), 2)
        self.assertNotIn("README.md", self.uploaded)
        self.assertNotIn("unrelated.jsonl", self.uploaded)
        self.assertEqual(result["repo_url"], "https://huggingface.co/datasets/owner/trajectories")
        self.assertTrue(result["verified"])
        self.assertEqual(result["commit"], COMMIT)
        self.assertEqual(result["status"], "complete")
        for call in self.api.hf_hub_download.call_args_list:
            self.assertEqual(call.kwargs["revision"], COMMIT)
            self.assertEqual(call.kwargs["repo_type"], "dataset")
            self.assertTrue(call.kwargs["force_download"])

    def test_new_private_and_public_repositories_use_explicit_visibility(self):
        for private in [True, False]:
            with self.subTest(private=private):
                info = SimpleNamespace(private=private, sha=HEAD)
                final = SimpleNamespace(private=private, sha=COMMIT)
                self.api.dataset_info.side_effect = [HubError(404), info, info, final]
                self.api.list_repo_files.return_value = []
                self.api.create_repo.reset_mock()
                result = self.call(visibility="private" if private else "public", create=True)
                self.api.create_repo.assert_called_once_with(repo_id="owner/trajectories", repo_type="dataset", private=private, exist_ok=False)
                self.assertTrue(result["created"])
                self.assertIn("README.md", self.uploaded)
                self.assertEqual(result["private"], private)

    def test_existing_repo_with_create_is_reused(self):
        self.call(create=True)
        self.api.create_repo.assert_not_called()

    def test_creation_race_rechecks_repository_and_visibility(self):
        info = SimpleNamespace(private=True, sha=HEAD)
        final = SimpleNamespace(private=True, sha=COMMIT)
        self.api.dataset_info.side_effect = [HubError(404), info, info, final]
        self.api.create_repo.side_effect = HubError(409)
        result = self.call(create=True)
        self.assertFalse(result["created"])
        self.api.create_commit.assert_called_once()

    def test_missing_repository_requires_explicit_creation(self):
        self.api.dataset_info.side_effect = HubError(404)
        with self.assertRaisesRegex(trajectory.TrajectoryError, "--create"):
            self.call()
        self.api.create_repo.assert_not_called()
        self.api.create_commit.assert_not_called()

    def test_auth_errors_never_trigger_repository_creation(self):
        for status in [401, 403, 500]:
            self.api.dataset_info.side_effect = HubError(status)
            with self.subTest(status=status), self.assertRaises(trajectory.TrajectoryError) as caught:
                self.call(create=True)
            self.assertNotIn("hf_secret_for_test", str(caught.exception))
        self.api.create_repo.assert_not_called()

    def test_visibility_mismatch_is_never_mutated(self):
        with self.assertRaisesRegex(trajectory.TrajectoryError, "visibility"):
            self.call(visibility="public")
        self.api.create_commit.assert_not_called()
        self.api.update_repo_visibility.assert_not_called()

    def test_visibility_is_rechecked_before_commit(self):
        self.api.dataset_info.side_effect = [SimpleNamespace(private=True, sha=HEAD), SimpleNamespace(private=False, sha=HEAD)]
        with self.assertRaisesRegex(trajectory.TrajectoryError, "visibility"):
            self.call()
        self.api.create_commit.assert_not_called()

    def test_unknown_destination_revision_prevents_unpinned_update(self):
        self.api.dataset_info.side_effect = None
        self.api.dataset_info.return_value = SimpleNamespace(private=True, sha=None)
        with self.assertRaisesRegex(trajectory.TrajectoryError, "revision"):
            self.call()
        self.api.create_commit.assert_not_called()

    def test_snapshot_does_not_upload_run_logs_models_or_other_files(self):
        (self.run / "dataset.jsonl").write_text("Unselected source data")
        (self.run / "worker.log").write_text("hf_secret_for_test")
        (self.run / "checkpoint").mkdir()
        (self.run / "checkpoint/model.safetensors").write_bytes(b"not a trajectory")
        self.call()
        for name in self.uploaded:
            self.assertTrue(name.startswith(("trajectories/", "manifests/")))
        self.assertIn(b"Supplied prompt", next(data for name, data in self.uploaded.items() if name.endswith(".jsonl")))

    def test_changed_contents_get_distinct_paths(self):
        self.call()
        first = set(self.uploaded)
        self.rows[3]["payload"]["accepted"] = False
        self.write()
        self.call()
        self.assertTrue(first.isdisjoint(self.uploaded))

    def test_malformed_or_truncated_json_is_rejected_before_hub_access(self):
        path = self.run / "trajectory.jsonl"
        for data in [b"", b"not json\n", b'{"type":', b"\xff\n"]:
            path.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(trajectory.TrajectoryError):
                self.call(create=True)
        self.api.dataset_info.assert_not_called()

    def test_record_order_version_type_and_completion_are_validated(self):
        cases = [
            (1, "sequence", 3), (0, "version", 2), (0, "version", True),
            (1, "type", "raw_debug_log"), (2, "timestamp_unix_ms", 1),
            (1, "payload", "text"), (1, "type", "run_start"),
        ]
        for index, key, value in cases:
            original = self.rows[index][key]
            self.rows[index][key] = value
            self.write()
            with self.subTest(key=key, value=value), self.assertRaises(trajectory.TrajectoryError):
                self.call()
            self.rows[index][key] = original
        self.rows.append(row(5, "event", {}))
        self.write()
        with self.assertRaisesRegex(trajectory.TrajectoryError, "after its completion"):
            self.call()
        self.api.dataset_info.assert_not_called()

    def test_credential_fields_and_known_tokens_are_rejected_without_echoing_them(self):
        for payload in [{"api_key": "hf_secret_for_test"}, {"nested": [{"Authorization": "Bearer secret"}]},
                        {"error": "failed with hf_secret_for_test"}, {"prompt": "hf_" + "x" * 30}]:
            self.rows[1]["payload"] = payload
            self.write()
            with patch.dict(os.environ, {"HF_TOKEN": "hf_secret_for_test"}):
                with self.subTest(payload=payload), self.assertRaises(trajectory.TrajectoryError) as caught:
                    self.call()
            self.assertNotIn("hf_secret_for_test", str(caught.exception))
        self.api.dataset_info.assert_not_called()

    def test_unsupported_metadata_and_local_path_fields_are_rejected(self):
        self.rows[0]["payload"]["dataset_source"] = "/home/private/data.jsonl"
        self.write()
        with self.assertRaisesRegex(trajectory.TrajectoryError, "metadata"):
            self.call()
        del self.rows[0]["payload"]["dataset_source"]
        self.rows[1]["payload"]["checkpoint"] = "/home/private/checkpoint"
        self.write()
        with self.assertRaisesRegex(trajectory.TrajectoryError, "local path"):
            self.call()
        self.api.dataset_info.assert_not_called()

    def test_symlink_and_oversized_file_are_rejected(self):
        path = self.run / "trajectory.jsonl"
        path.rename(self.run / "other.jsonl")
        path.symlink_to(self.run / "other.jsonl")
        with self.assertRaisesRegex(trajectory.TrajectoryError, "regular"):
            self.call()
        path.unlink()
        self.write()
        with patch.object(trajectory, "MAX_FILE_BYTES", 10), self.assertRaisesRegex(trajectory.TrajectoryError, "file-size"):
            self.call()
        self.api.dataset_info.assert_not_called()

    def test_duplicate_object_keys_and_nonfinite_measurements_are_rejected(self):
        path = self.run / "trajectory.jsonl"
        data = path.read_text()
        path.write_text(data.replace('"sequence": 1', '"sequence": 1, "sequence": 1'))
        with self.assertRaisesRegex(trajectory.TrajectoryError, "duplicate"):
            self.call()
        self.rows[3]["payload"]["health"]["copying"]["score"] = float("inf")
        self.write()
        with self.assertRaisesRegex(trajectory.TrajectoryError, "nonfinite"):
            self.call()

    def test_readback_hash_mismatch_reports_that_commit_already_happened(self):
        def corrupt(**kwargs):
            path = Path(kwargs["local_dir"]) / "corrupt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("changed bytes")
            return str(path)
        self.api.hf_hub_download.side_effect = corrupt
        with self.assertRaisesRegex(trajectory.TrajectoryError, f"committed as {COMMIT}"):
            self.call()
        self.api.create_commit.assert_called_once()

    def test_upload_failure_is_sanitised(self):
        self.api.create_commit.side_effect = HubError(403)
        with self.assertRaises(trajectory.TrajectoryError) as caught:
            self.call()
        self.assertNotIn("hf_secret_for_test", str(caught.exception))
        self.api.hf_hub_download.assert_not_called()

    def test_cli_visibility_is_explicit_and_no_token_argument_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            trajectory.main(["--run", str(self.run), "--repo", "owner/trajectories"])
        with patch.object(trajectory, "publish", return_value={"version": 1}) as call, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(trajectory.main(["--run", str(self.run), "--repo", "owner/trajectories", "--visibility", "private"]), 0)
        self.assertNotIn("token", call.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
