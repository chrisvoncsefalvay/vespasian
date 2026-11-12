"""Offline publication checks. No test writes to a remote repository."""
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

import publish


SHA = "a" * 40
COMMIT = "b" * 40


class HubError(Exception):
    def __init__(self, status):
        self.response = SimpleNamespace(status_code=status)
        super().__init__("Authorization: Bearer hf_super_secret")


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.checkpoint = self.run / "checkpoint"
        self.checkpoint.mkdir(parents=True)
        self.write("config.json", {"model_type": "gemma3_text", "_name_or_path": "/home/private/original"})
        self.write("tokenizer.json", {"model": {"vocab": {"hello": 1}}})
        (self.checkpoint / "model.safetensors").write_bytes(b"test-only-weight-bytes")
        self.metrics = {"validation": {"count": 2, "exact_match": 1.0, "token_f1": 1.0, "loss": 0.1}}
        self.report = {"status": "complete", "reload_verified": True, "metric": "exact_match", "turns": 3,
                       "accepted_turns": 1, "baseline_metrics": self.metrics, "best_metrics": self.metrics,
                       "final_metrics": self.metrics, "goal": "PRIVATE GOAL", "history": [{"prompt": "PRIVATE PROMPT"}],
                       "checkpoint": "/home/private/checkpoint", "api_key": "hf_super_secret"}
        (self.run / "report.json").write_text(json.dumps(self.report))
        (self.run / "model.json").write_text(json.dumps({"model": "google/gemma-3-1b-it", "revision": SHA,
                                                        "source": {"path": "/home/private/model"}}))
        (self.run / "run.json").write_text(json.dumps({"version": "0.1.1", "dataset_sha256": "c" * 64,
                                                      "dataset_source": "/home/private/dataset.jsonl", "goal": "PRIVATE GOAL"}))
        self.manifest()
        self.card_download = patch("publish._download_card", return_value="# Existing card").start()
        self.addCleanup(patch.stopall)
        self.api = Mock()
        self.api.model_info.return_value = SimpleNamespace(private=True, sha=SHA)
        self.api.list_repo_files.return_value = []
        self.api.create_commit.return_value = SimpleNamespace(oid=COMMIT)
        self.uploaded = {}

        def upload(**kwargs):
            self.uploaded = {op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in kwargs["operations"]
                             if hasattr(op, "path_or_fileobj")}
            return SimpleNamespace(oid=COMMIT)

        self.api.create_commit.side_effect = upload

    def write(self, name, value):
        path = self.checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def manifest(self, extra=None):
        files = [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                 for p in self.checkpoint.iterdir() if p.is_file()]
        if extra:
            files.append(extra)
        self.receipt = {"source_unchanged": True, "path": str(self.checkpoint), "files": files}
        (self.run / "checkpoint.json").write_text(json.dumps(self.receipt))

    def call(self, **kwargs):
        arguments = {"run": self.run, "repo": "owner/model", "visibility": "private", "api": self.api,
                     "operation": lambda **kw: SimpleNamespace(**kw),
                     "delete_operation": lambda **kw: SimpleNamespace(**kw)}
        arguments.update(kwargs)
        return publish.publish(**arguments)

    def test_card_pipeline_uses_verified_saved_architecture_and_safe_task(self):
        self.write("config.json", {"model_type": "qwen3_5", "vision_config": {"model_type": "qwen3_5_vision"}})
        self.manifest()
        settings = json.loads((self.run / "run.json").read_text())
        settings["hf_task"] = "text-classification"
        (self.run / "run.json").write_text(json.dumps(settings))
        with publish.prepare(self.run, "owner/model") as prepared:
            card = (prepared.directory / "README.md").read_text()
            self.assertIn("pipeline_tag: image-text-to-text", card)
            self.assertIn("text-classification", card)
            self.assertIn("evaluations use text only", card)
        settings["hf_task"] = "PRIVATE UNKNOWN TASK"
        (self.run / "run.json").write_text(json.dumps(settings))
        with publish.prepare(self.run, "owner/model") as prepared:
            self.assertNotIn("PRIVATE UNKNOWN TASK", (prepared.directory / "README.md").read_text())

    def test_health_publication_contains_scores_without_monitoring_examples(self):
        health = {"copying": {"label": "PRIVATE LABEL", "score": 0.75, "count": 4, "metric": "exact_text", "samples": ["PRIVATE PROMPT"]}}
        self.report.update({name: health for name in ("baseline_health", "best_health", "final_health")})
        (self.run / "report.json").write_text(json.dumps(self.report))
        self.call()
        values = json.loads(self.uploaded["vespasian-results.json"])
        self.assertEqual(values["final_health"], {"copying": {"score": 0.75, "count": 4, "metric": "exact_text"}})
        for content in self.uploaded.values():
            self.assertNotIn(b"PRIVATE PROMPT", content)
            self.assertNotIn(b"PRIVATE LABEL", content)
        self.assertIn(b"Health checks", self.uploaded["README.md"])
        for invalid in [float("nan"), 1.5, True, "SECRET"]:
            with self.assertRaises(publish.PublishError):
                publish._health({"copying": {"score": invalid, "count": 4, "metric": "exact_text"}})

    def test_existing_private_repo_gets_one_atomic_curated_commit(self):
        result = self.call()
        self.assertEqual(result["commit"], COMMIT)
        self.assertTrue(result["private"])
        self.assertFalse(result["created"])
        self.api.create_repo.assert_not_called()
        self.api.create_commit.assert_called_once()
        self.assertEqual(self.api.create_commit.call_args.kwargs["parent_commit"], SHA)
        self.api.list_repo_files.assert_called_once_with(repo_id="owner/model", repo_type="model", revision=SHA)
        self.assertEqual(set(self.uploaded), {"config.json", "tokenizer.json", "model.safetensors", "README.md",
                                             "vespasian-results.json", "vespasian-provenance.json", "vespasian-badge.svg"})
        self.assertEqual(result["repo_url"], "https://huggingface.co/owner/model")
        for item in result["files"]:
            self.assertEqual(item["sha256"], hashlib.sha256(self.uploaded[item["path"]]).hexdigest())

    def test_prompt_only_results_keep_mode_and_only_measured_metrics(self):
        metadata = json.loads((self.run / "run.json").read_text())
        metadata["dataset_mode"] = "prompts"
        (self.run / "run.json").write_text(json.dumps(metadata))
        self.report.update(dataset_mode="prompts", metric="semantic")
        metrics = {"validation": {"count": 2, "semantic": 0.75}}
        for name in ("baseline_metrics", "best_metrics", "final_metrics"):
            self.report[name] = metrics
        (self.run / "report.json").write_text(json.dumps(self.report))
        self.call()
        self.assertEqual(json.loads(self.uploaded["vespasian-provenance.json"])["dataset_mode"], "prompts")
        results = json.loads(self.uploaded["vespasian-results.json"])
        self.assertEqual(results["final_metrics"], metrics)
        self.assertNotIn("loss", results["final_metrics"]["validation"])

    def test_numeric_accuracy_survives_result_publication(self):
        self.report["metric"] = "numeric_match"
        for name in ("baseline_metrics", "best_metrics", "final_metrics"):
            self.report[name] = {"validation": {"count": 4, "numeric_match": .75, "loss": .5}}
        (self.run / "report.json").write_text(json.dumps(self.report))
        self.call()
        results = json.loads(self.uploaded["vespasian-results.json"])
        self.assertEqual(results["metric"], "numeric_match")
        self.assertEqual(results["final_metrics"]["validation"]["numeric_match"], .75)

    def test_new_public_repo_uses_explicit_visibility(self):
        info = SimpleNamespace(private=False, sha=SHA)
        self.api.model_info.side_effect = [HubError(404), info, info]
        result = self.call(visibility="public", create=True)
        self.assertTrue(result["created"])
        self.assertFalse(result["private"])
        self.api.create_repo.assert_called_once_with(repo_id="owner/model", repo_type="model", private=False, exist_ok=False)

    def test_new_private_repo_uses_explicit_visibility(self):
        info = SimpleNamespace(private=True, sha=SHA)
        self.api.model_info.side_effect = [HubError(404), info, info]
        result = self.call(create=True)
        self.assertTrue(result["created"])
        self.api.create_repo.assert_called_once_with(repo_id="owner/model", repo_type="model", private=True, exist_ok=False)

    def test_existing_repo_with_create_is_reused(self):
        self.call(create=True)
        self.api.create_repo.assert_not_called()

    def test_missing_repo_needs_create(self):
        self.api.model_info.side_effect = HubError(404)
        with self.assertRaisesRegex(publish.PublishError, "--create"):
            self.call()
        self.api.create_repo.assert_not_called()
        self.api.create_commit.assert_not_called()

    def test_visibility_mismatch_never_changes_or_uploads_repo(self):
        with self.assertRaisesRegex(publish.PublishError, "visibility"):
            self.call(visibility="public")
        self.api.create_repo.assert_not_called()
        self.api.create_commit.assert_not_called()
        self.api.update_repo_visibility.assert_not_called()

    def test_visibility_is_rechecked_before_commit(self):
        self.api.model_info.side_effect = [SimpleNamespace(private=True, sha=SHA), SimpleNamespace(private=False, sha=SHA)]
        with self.assertRaisesRegex(publish.PublishError, "visibility"):
            self.call()
        self.api.create_commit.assert_not_called()

    def test_creation_race_rechecks_visibility(self):
        info = SimpleNamespace(private=True, sha=SHA)
        self.api.model_info.side_effect = [HubError(404), info, info]
        self.api.create_repo.side_effect = HubError(409)
        result = self.call(create=True)
        self.assertFalse(result["created"])
        self.api.create_commit.assert_called_once()

    def test_auth_error_cannot_be_mistaken_for_missing_repo(self):
        self.api.model_info.side_effect = HubError(401)
        with self.assertRaises(publish.PublishError) as caught:
            self.call(create=True)
        self.assertNotIn("hf_super_secret", str(caught.exception))
        self.api.create_repo.assert_not_called()

    def test_raw_data_logs_tokens_and_private_paths_are_not_uploaded(self):
        (self.run / "dataset.jsonl").write_text("PRIVATE DATASET ROW")
        (self.run / "editor.log").write_text("hf_super_secret")
        (self.checkpoint / "unlisted-prompts.jsonl").write_text("PRIVATE PROMPT")
        with patch.dict(os.environ, {"HF_TOKEN": "hf_super_secret"}):
            result = self.call()
        text = json.dumps(result) + "".join(value.decode() for value in self.uploaded.values())
        for private in ["PRIVATE GOAL", "PRIVATE PROMPT", "PRIVATE DATASET ROW", "hf_super_secret", "/home/private"]:
            self.assertNotIn(private, text)
        self.assertNotIn("_name_or_path", json.loads(self.uploaded["config.json"]))
        self.assertIn("_name_or_path", json.loads((self.checkpoint / "config.json").read_text()))

    def test_control_summary_contains_only_safe_aggregate_evidence(self):
        safe = {"passed": True, "status": "retained", "scope": "supplied_examples", "checked": 3,
                "applicable": 2, "unchanged": 4, "retention": 1.0}
        self.report["controls"] = dict(safe, controls="PRIVATE CONTROL TEXT",
            reason="PRIVATE REASON", checks=[{"prompt": "PRIVATE PROMPT", "reason": "PRIVATE REASON"}])
        (self.run / "report.json").write_text(json.dumps(self.report))
        self.call()
        summary = json.loads(self.uploaded["vespasian-results.json"])
        self.assertEqual(summary["controls"], safe)
        payload = "".join(value.decode() for value in self.uploaded.values())
        for private in ["PRIVATE CONTROL TEXT", "PRIVATE REASON", "PRIVATE PROMPT"]:
            self.assertNotIn(private, payload)

    def test_unmeasured_controls_keep_null_retention(self):
        self.report["controls"] = {"passed": True, "status": "not_exercised", "scope": "supplied_examples",
                                   "checked": 3, "applicable": 0, "unchanged": 2, "retention": None}
        (self.run / "report.json").write_text(json.dumps(self.report))
        self.call()
        summary = json.loads(self.uploaded["vespasian-results.json"])
        self.assertIsNone(summary["controls"]["retention"])
        self.assertIn("Retention: not measured.", self.uploaded["README.md"].decode())

    def test_legacy_runs_without_controls_remain_publishable(self):
        self.call()
        self.assertNotIn("controls", json.loads(self.uploaded["vespasian-results.json"]))

    def test_forbidden_manifest_file_is_rejected_before_network(self):
        self.write("dataset.json", {"prompt": "private"})
        self.manifest()
        with self.assertRaisesRegex(publish.PublishError, "allowlist"):
            self.call()
        self.api.model_info.assert_not_called()

    def test_path_traversal_and_symlink_escape_are_rejected(self):
        self.manifest({"path": "../private.json", "sha256": "a" * 64})
        with self.assertRaisesRegex(publish.PublishError, "unsafe"):
            self.call()
        self.manifest()
        weights = self.checkpoint / "model.safetensors"
        weights.unlink()
        weights.symlink_to(self.run / "report.json")
        with self.assertRaisesRegex(publish.PublishError, "symbolic"):
            self.call()
        self.api.model_info.assert_not_called()

    def test_weight_hash_mismatch_prevents_all_remote_mutation(self):
        (self.checkpoint / "model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(publish.PublishError, "hash"):
            self.call(create=True)
        self.api.model_info.assert_not_called()
        self.api.create_repo.assert_not_called()

    def test_incomplete_unverified_or_changed_source_run_is_rejected(self):
        for key, value in [("status", "running"), ("reload_verified", False)]:
            report = dict(self.report, **{key: value})
            (self.run / "report.json").write_text(json.dumps(report))
            with self.subTest(key=key), self.assertRaises(publish.PublishError):
                self.call()
        (self.run / "report.json").write_text(json.dumps(self.report))
        self.receipt["source_unchanged"] = False
        (self.run / "checkpoint.json").write_text(json.dumps(self.receipt))
        with self.assertRaisesRegex(publish.PublishError, "unchanged"):
            self.call()
        self.api.model_info.assert_not_called()

    def test_invalid_shard_reference_is_rejected(self):
        self.write("model.safetensors.index.json", {"weight_map": {"weight": "unverified.safetensors"}})
        self.manifest()
        with self.assertRaisesRegex(publish.PublishError, "unverified"):
            self.call()


    def test_card_has_local_branded_badge_and_verified_evidence(self):
        self.call()
        card = self.uploaded["README.md"].decode()
        self.assertIn("[![Created with Vespasian](vespasian-badge.svg)](https://chrisvoncsefalvay.com/vespasian/)", card)
        self.assertIn("pipeline_tag: text-generation", card)
        self.assertIn('base_model: "google/gemma-3-1b-it"', card)
        self.assertIn(SHA, card)
        self.assertIn("| validation | 2 | 1.000000 | 2 | 1.000000 |", card)
        self.assertIn("| test | not reported | not reported | not reported | not reported |", card)
        self.assertIn("private user objective", card)
        self.assertIn("No new licence", card)
        badge = self.uploaded["vespasian-badge.svg"].decode()
        self.assertIn("Created with Vespasian", badge)
        self.assertNotIn("href=", badge)


    def test_behaviour_category_and_tied_catalogue_are_safe_aggregates(self):
        settings = json.loads((self.run / "run.json").read_text())
        settings["target_type"] = "classification"
        (self.run / "run.json").write_text(json.dumps(settings))
        model = json.loads((self.run / "model.json").read_text())
        model["token_tables"] = [{"tied": True, "name": "PRIVATE TOKEN", "aliases": ["PRIVATE PROMPT"]}]
        (self.run / "model.json").write_text(json.dumps(model))
        self.call()
        card = self.uploaded["README.md"].decode()
        self.assertIn("Behaviour category: Classification", card)
        self.assertIn("1 tied input/output tables", card)
        self.assertIn("not evidence that a token-row edit was applied", card)
        self.assertNotIn("PRIVATE TOKEN", repr(self.uploaded))

    def test_existing_card_metadata_and_prose_are_preserved(self):
        old = "---\nlicense: mit\nlanguage: en\n---\n# Author card\nAuthor limitations.\n"
        self.card_download.return_value = old
        self.api.list_repo_files.return_value = ["README.md"]
        result = self.call()
        card = self.uploaded["README.md"].decode()
        self.assertTrue(card.startswith(old))
        self.card_download.assert_called_once_with(self.api, "owner/model", SHA)
        self.assertEqual(card.count("license: mit"), 1)
        item = next(item for item in result["files"] if item["path"] == "README.md")
        self.assertEqual(item["sha256"], hashlib.sha256(self.uploaded["README.md"]).hexdigest())
        self.card_download.return_value = card
        self.call()
        self.assertEqual(self.uploaded["README.md"].decode(), card)


    def test_existing_card_line_endings_are_preserved_exactly(self):
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=repr(newline)):
                old = newline.join(["---", "license: mit", "---", "# Author card", "Author limitations.", ""])
                self.card_download.return_value = old
                self.api.list_repo_files.return_value = ["README.md"]
                self.call()
                card = self.uploaded["README.md"]
                self.assertTrue(card.startswith(old.encode("utf-8")))
                self.card_download.return_value = card.decode("utf-8")
                self.call()
                self.assertEqual(self.uploaded["README.md"], card)

    def test_conflicting_existing_base_model_stops_upload(self):
        for metadata in ('base_model: other/model', 'base_model: [other/model]',
                         "base_model:\n- other/model", "base_model: [google/gemma-3-1b-it, other/model]"):
            with self.subTest(metadata=metadata):
                self.api.list_repo_files.return_value = ["README.md"]
                old = "---\n" + metadata + "\n---\nAuthor text."
                self.card_download.return_value = old
                with self.assertRaisesRegex(publish.PublishError, "base_model metadata does not match"):
                    self.call()
                self.api.create_commit.assert_not_called()
                self.assertEqual(self.card_download.return_value, old)

    def test_matching_yaml_base_model_forms_are_preserved(self):
        for metadata in ('base_model: "google/gemma-3-1b-it"',
                         "base_model: [google/gemma-3-1b-it]",
                         "base_model:\n- google/gemma-3-1b-it"):
            with self.subTest(metadata=metadata):
                self.api.list_repo_files.return_value = ["README.md"]
                old = "---\n" + metadata + "\n---\nAuthor text."
                self.card_download.return_value = old
                self.call()
                self.assertTrue(self.uploaded["README.md"].decode().startswith(old))

    def test_republishing_generated_card_with_changed_base_stops_upload(self):
        self.call()
        self.card_download.return_value = self.uploaded["README.md"].decode()
        self.api.create_commit.reset_mock()
        self.api.list_repo_files.return_value = ["README.md"]
        model = json.loads((self.run / "model.json").read_text())
        model["model"] = "Qwen/Qwen3.5-0.8B"
        (self.run / "model.json").write_text(json.dumps(model))
        with self.assertRaisesRegex(publish.PublishError, "base_model metadata does not match"):
            self.call()
        self.api.create_commit.assert_not_called()

    def test_unverified_base_cannot_preserve_claim_of_known_base(self):
        self.api.list_repo_files.return_value = ["README.md"]
        self.card_download.return_value = "---\nbase_model: google/gemma-3-1b-it\n---\nAuthor text."
        (self.run / "model.json").write_text(json.dumps({"model": "/private/local/model"}))
        with self.assertRaisesRegex(publish.PublishError, "base_model metadata does not match"):
            self.call()
        self.api.create_commit.assert_not_called()

    def test_unreadable_existing_card_stops_publication(self):
        self.api.list_repo_files.return_value = ["README.md"]
        self.card_download.side_effect = publish.PublishError("Could not read existing card")
        with self.assertRaises(publish.PublishError):
            self.call()
        self.api.create_commit.assert_not_called()

    def test_ambiguous_managed_card_stops_publication(self):
        self.api.list_repo_files.return_value = ["README.md"]
        self.card_download.return_value = publish._CARD_START * 2
        with self.assertRaises(publish.PublishError):
            self.call()
        self.api.create_commit.assert_not_called()

    def test_incomplete_scoring_is_explicit_without_publishing_error(self):
        self.report["final_scoring_error"] = "PRIVATE PROMPT hf_super_secret /home/private/data"
        self.report["final_metrics"] = {}
        (self.run / "report.json").write_text(json.dumps(self.report))
        self.call()
        card = self.uploaded["README.md"].decode()
        self.assertIn("WARNING: final scoring was incomplete", card)
        self.assertNotIn("hf_super_secret", card)
        self.assertNotIn("PRIVATE PROMPT", card)

    def test_untrusted_base_model_cannot_inject_yaml(self):
        (self.run / "model.json").write_text(json.dumps({"model": "unknown/model\nlicense: fake", "revision": "PRIVATE"}))
        self.call()
        card = self.uploaded["README.md"].decode()
        self.assertNotIn("license: fake", card)
        self.assertNotIn("base_model:", card)
        self.assertIn("not publicly identified", card)

    def test_existing_shards_and_legacy_weights_are_removed_with_new_single_file(self):
        self.api.list_repo_files.return_value = ["model.safetensors.index.json", "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors", "pytorch_model.bin", "notes.txt", "other/model.safetensors",
            "quantised.gguf", "dataset.jsonl"]
        result = self.call()
        expected = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                    "model.safetensors.index.json", "pytorch_model.bin"]
        self.assertEqual(result["deleted_files"], expected)
        operations = self.api.create_commit.call_args.kwargs["operations"]
        deleted = [op for op in operations if not hasattr(op, "path_or_fileobj")]
        self.assertEqual([op.path_in_repo for op in deleted], expected)
        self.assertTrue(all(op.is_folder is False for op in deleted))
        self.api.create_commit.assert_called_once()

    def test_new_sharded_model_removes_stale_single_file(self):
        (self.checkpoint / "model.safetensors").unlink()
        (self.checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"sharded-test-only-weights")
        self.write("model.safetensors.index.json", {"weight_map": {"weight": "model-00001-of-00001.safetensors"}})
        self.manifest()
        self.api.list_repo_files.return_value = ["model.safetensors", "tokenizer.json", "README.md"]
        result = self.call()
        self.assertEqual(result["deleted_files"], ["model.safetensors"])
        self.assertIn("model-00001-of-00001.safetensors", self.uploaded)

    def test_unreadable_destination_file_listing_prevents_commit(self):
        self.api.list_repo_files.side_effect = HubError(403)
        with self.assertRaises(publish.PublishError) as caught:
            self.call()
        self.assertNotIn("hf_super_secret", str(caught.exception))
        self.api.create_commit.assert_not_called()

    def test_upload_error_is_sanitised(self):
        self.api.create_commit.side_effect = HubError(403)
        with self.assertRaises(publish.PublishError) as caught:
            self.call()
        self.assertNotIn("hf_super_secret", str(caught.exception))

    def test_cli_requires_explicit_visibility(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            publish.main(["--run", str(self.run), "--repo", "owner/model"])


if __name__ == "__main__":
    unittest.main()
