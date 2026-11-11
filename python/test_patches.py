"""Real tensor mutation tests for durable, private journals. No network."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import torch
from test_worker import engine, proposal
import patches


class PatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / ("a" * 40)
        self.e = engine()
        self.e.model.config.tie_word_embeddings = True
        self.e.model.tie_weights()
        self.e.model.save_pretrained(self.source)
        self.e.tokenizer.save_pretrained(self.source)
        self.e.source_dir = self.source
        self.e.source_files = self.e.manifest(self.source)
        self.e.register_editable()
        self.base = copy.deepcopy(self.e)
        self.directory = self.root / "patches"
        patches.initialize(self.e, self.directory, "verified/base")

    def row_edit(self, turn=1):
        info = self.e.inspect({"token_texts": ["a"]})
        name, token = info["token_tables"][0]["name"], info["token_candidates"][0]["id"]
        self.e.edit(proposal(self.e, operation="token_row_scale", targets=[name], token_ids=[token], source_token_ids=[], strength=0.03))
        receipt = patches.record_candidate(self.e, self.directory, turn)
        return name, token, receipt

    def equal(self, a, b):
        for name, value in a.model.state_dict().items():
            self.assertTrue(torch.equal(value, b.model.state_dict()[name]), name)

    def accept(self, turn=1):
        self.e.accept({})
        return patches.finalize(self.directory, turn, True)

    def test_baseline_and_privacy(self):
        receipt = patches.restore(copy.deepcopy(self.base), self.directory)
        self.assertEqual(receipt["turn"], 0)
        self.assertEqual(receipt["tensor_count"], 0)
        data = "".join(path.read_text() for path in self.directory.rglob("*.json"))
        for secret in (str(self.source), "alpha", "gamma", "HF_TOKEN"):
            self.assertNotIn(secret, data)
        self.assertIn("verified/base", data)
        self.assertEqual(patches.initialize(self.e, self.directory)["best"]["turn"], 0)

    def test_sparse_tied_replay_and_standalone_export(self):
        name, token, _ = self.row_edit()
        self.accept()
        fresh = copy.deepcopy(self.base)
        receipt = patches.restore(fresh, self.directory)
        self.assertTrue(receipt["base_verified"])
        self.equal(fresh, self.e)
        self.assertIs(fresh.model.get_input_embeddings().weight, fresh.model.get_output_embeddings().weight)
        self.assertEqual(fresh.row_changed[name], {token})
        bundle = self.root / "patches.zip"
        result = patches.export_bundle(self.directory, bundle)
        self.assertLess(result["best"]["tensor_bytes"], self.e.token_tables[name].numel() * 4)
        imported = self.root / "imported"
        patches.import_bundle(bundle, imported)
        restored = copy.deepcopy(self.base)
        patches.restore(restored, imported)
        self.equal(restored, self.e)

    def test_rejected_latest_and_pending_leave_best_unchanged(self):
        _, _, receipt = self.row_edit()
        self.assertEqual(receipt["best"]["turn"], 0)
        fresh = copy.deepcopy(self.base)
        patches.restore(fresh, self.directory, "latest")
        self.equal(fresh, self.e)
        self.e.rollback({})
        patches.finalize(self.directory, 1, False)
        receipt = patches.describe(self.directory)
        self.assertEqual(receipt["best"]["turn"], 0)
        self.assertEqual(receipt["latest"]["status"], "rejected")
        fresh = copy.deepcopy(self.base)
        patches.restore(fresh, self.directory)
        self.equal(fresh, self.base)

    def test_cumulative_sparse_and_dense(self):
        self.row_edit()
        self.accept()
        self.e.edit(proposal(self.e, operation="low_rank", strength=0.02))
        patches.record_candidate(self.e, self.directory, 2)
        self.accept(2)
        fresh = copy.deepcopy(self.base)
        patches.restore(fresh, self.directory)
        self.equal(fresh, self.e)
        self.assertTrue(fresh.changed)
        self.assertTrue(fresh.row_changed)
        self.assertEqual(patches.describe(self.directory)["best"]["turn"], 2)

    def test_corruption_or_base_mismatch_reject_before_changes(self):
        self.row_edit()
        self.accept()
        blob = next((self.directory / "blobs").iterdir())
        blob.write_bytes(blob.read_bytes()[:-1] + b"x")
        fresh = copy.deepcopy(self.base)
        with self.assertRaisesRegex(patches.PatchError, "checksum"):
            patches.restore(fresh, self.directory)
        self.equal(fresh, self.base)
        fresh.source_files[0]["sha256"] = "b" * 64
        with self.assertRaisesRegex(patches.PatchError, "base checkpoint"):
            patches.restore(fresh, self.directory)

    def test_source_file_mapping_and_ties_enter_base_identity(self):
        original = patches._base(self.base)
        other = copy.deepcopy(self.base)
        other.source_files[0]["path"] = "different.json"
        self.assertNotEqual(original["source_sha256"], patches._base(other)["source_sha256"])
        other = copy.deepcopy(self.base)
        other.token_table_info[0]["tied"] = False
        self.assertNotEqual(original["layout_sha256"], patches._base(other)["layout_sha256"])

    def test_fresh_base_required_and_decisions_idempotent(self):
        self.row_edit()
        with self.assertRaisesRegex(patches.PatchError, "freshly loaded"):
            patches.restore(self.e, self.directory)
        first = self.accept()
        self.assertEqual(first, patches.finalize(self.directory, 1, True))
        for turn, accepted in ((1, False), (2, True)):
            with self.assertRaises(patches.PatchError):
                patches.finalize(self.directory, turn, accepted)

    def test_interrupted_index_write_preserves_pointers(self):
        initial = (self.directory / "index.json").read_bytes()
        self.e.edit(proposal(self.e))
        atomic = patches._atomic
        def interrupt(path, data):
            if Path(path).name == "index.json":
                raise OSError("simulated interruption")
            return atomic(path, data)
        with patch.object(patches, "_atomic", side_effect=interrupt):
            with self.assertRaises(OSError):
                patches.record_candidate(self.e, self.directory, 1)
        self.assertEqual((self.directory / "index.json").read_bytes(), initial)
        self.assertEqual(patches.describe(self.directory)["latest"]["turn"], 0)
        result = patches.export_bundle(self.directory, self.root / "baseline.zip")
        self.assertEqual(result["files"], 2)

    def test_archive_path_traversal_rejected(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../secret", "not allowed")
        with self.assertRaises(patches.PatchError):
            patches.import_bundle(archive, self.root / "bad")
        self.assertFalse((self.root / "secret").exists())

    def test_materialize_without_dataset_uses_recorded_dtype(self):
        self.row_edit()
        self.accept()
        with patch("worker.load_model_components", return_value=(copy.deepcopy(self.base.model), self.base.tokenizer, None)) as loader:
            result = patches.materialize(self.directory, str(self.source), output=self.root / "model")
        self.assertTrue(result["base_verified"])
        self.assertTrue((self.root / "model" / "model.safetensors").is_file())
        self.assertEqual(loader.call_args.kwargs["dtype"], torch.float32)

    def api(self, private=True):
        root, calls = self.root / "remote", []
        class Fake:
            sha = "a" * 40
            def repo_info(self, **kwargs):
                calls.append(("info", kwargs))
                return SimpleNamespace(private=private, sha=self.sha)
            def create_repo(self, **kwargs):
                calls.append(("create", kwargs))
            def create_commit(self, **kwargs):
                calls.append(("commit", kwargs))
                for item in kwargs["operations"]:
                    target = root / item.path_in_repo
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(Path(item.path_or_fileobj).read_bytes())
                self.sha = "b" * 40
                return SimpleNamespace(oid=self.sha)
        def download(**kwargs):
            self.assertEqual(kwargs["token"], "session-token")
            self.assertEqual(kwargs["revision"], "b"*40)
            return str(root / kwargs["filename"])
        return Fake(), calls, download

    def test_private_backup_requires_consent_token_and_private_repo(self):
        api, calls, download = self.api()
        for consent, token in ((False, "session-token"), (True, ""), (True, False)):
            with self.assertRaises(patches.PatchError):
                patches.publish_private(self.directory, "owner/repo", token, consent=consent, api=api, operation=SimpleNamespace, download=download)
        self.assertFalse(calls)
        api, calls, download = self.api(False)
        with self.assertRaisesRegex(patches.PatchError, "private"):
            patches.publish_private(self.directory, "owner/repo", "session-token", consent=True, api=api, operation=SimpleNamespace, download=download)
        self.assertFalse(any(kind == "commit" for kind, _ in calls))

    def test_private_backup_pinned_commit_and_readback(self):
        self.row_edit()
        self.accept()
        api, calls, download = self.api()
        result = patches.publish_private(self.directory, "owner/repo", "session-token", consent=True, api=api, operation=SimpleNamespace, download=download)
        self.assertTrue(result["private"])
        self.assertTrue(result["verified"])
        commit = next(value for name, value in calls if name == "commit")
        self.assertEqual(commit["parent_commit"], "a"*40)
        self.assertTrue(all(item.path_in_repo.startswith(result["prefix"] + "/") for item in commit["operations"]))
        self.assertNotIn("session-token", json.dumps(result))
        self.assertFalse(any(name == "create" for name, _ in calls))


    def test_verified_unchanged_backup_does_not_reupload(self):
        self.row_edit()
        self.accept()
        api, calls, download = self.api()
        first = patches.publish_private(self.directory, "owner/repo", "session-token", consent=True, api=api, operation=SimpleNamespace, download=download)
        second = patches.publish_private(self.directory, "owner/repo", "session-token", consent=True, api=api, operation=SimpleNamespace, download=download)
        self.assertEqual(first["uploaded_files"], first["files"])
        self.assertEqual(second["uploaded_files"], 0)
        self.assertTrue(second["unchanged"])
        self.assertEqual(sum(kind == "commit" for kind, _ in calls), 1)
        self.e.edit(proposal(self.e, operation="scale", strength=0.02))
        patches.record_candidate(self.e, self.directory, 2)
        self.accept(2)
        third = patches.publish_private(self.directory, "owner/repo", "session-token", consent=True, api=api, operation=SimpleNamespace, download=download)
        self.assertLess(third["uploaded_files"], third["files"])

    def test_bfloat16_replay_and_materialization_keep_original_dtype(self):
        self.e.model.to(torch.bfloat16)
        self.e.load_dtype = "bfloat16"
        self.e.register_editable()
        self.base = copy.deepcopy(self.e)
        self.directory = self.root / "bfloat16-journal"
        patches.initialize(self.e, self.directory)
        self.row_edit()
        self.accept()
        fresh = copy.deepcopy(self.base)
        patches.restore(fresh, self.directory)
        self.equal(fresh, self.e)
        with patch("worker.load_model_components", return_value=(copy.deepcopy(self.base.model), self.base.tokenizer, None)) as loader:
            patches.materialize(self.directory, str(self.source), output=self.root / "bf16-model", device="cpu")
        self.assertEqual(loader.call_args.kwargs["dtype"], torch.bfloat16)

    def test_metadata_privacy_allowlist_and_symlink_blob_rejection(self):
        self.row_edit()
        self.accept()
        index = patches._index(self.directory)
        manifest = patches._manifest(self.directory, index["latest"])
        manifest["private_goal"] = "must not be uploaded"
        index["latest"] = patches._write_manifest(self.directory, manifest)
        patches._atomic(self.directory / "index.json", patches._json(index))
        with self.assertRaisesRegex(patches.PatchError, "Unexpected"):
            patches.export_bundle(self.directory, self.root / "bad.zip")

    def test_snapshot_rejects_concurrent_pointer_change(self):
        original = patches.shutil.copyfile
        changed = False
        def copy_and_change(source, destination):
            nonlocal changed
            result = original(source, destination)
            if not changed:
                changed = True
                path = self.directory / "index.json"
                path.write_bytes(path.read_bytes() + b" ")
            return result
        with patch.object(patches.shutil, "copyfile", side_effect=copy_and_change):
            with self.assertRaisesRegex(patches.PatchError, "changed while"):
                patches.export_bundle(self.directory, self.root / "race.zip")
        self.assertFalse((self.root / "race.zip").exists())


    def test_windows_uses_file_fsync_and_atomic_replace_without_directory_open(self):
        with patch.object(patches.sys, "platform", "win32"), patch.object(patches.os, "open") as open_directory:
            patches._sync_directory(self.root)
        open_directory.assert_not_called()
        with patch.object(patches.sys, "platform", "win32"), patch.object(patches.os, "fsync") as fsync:
            patches._atomic(self.root / "windows-atomic.json", b"{}")
        self.assertEqual(fsync.call_count, 1)
        self.assertEqual((self.root / "windows-atomic.json").read_bytes(), b"{}")

    def test_new_private_repository_gets_truthful_branded_storage_card(self):
        api, calls, download = self.api()
        original_info, original_create = api.repo_info, api.create_repo
        api.exists = False
        class Missing(Exception):
            response = SimpleNamespace(status_code=404)
        def info(**kwargs):
            if not api.exists:
                raise Missing()
            return original_info(**kwargs)
        def create(**kwargs):
            self.assertIs(kwargs["private"], True)
            self.assertFalse(kwargs["exist_ok"])
            original_create(**kwargs)
            api.exists = True
        api.repo_info, api.create_repo = info, create
        with self.assertRaises(patches.PatchError):
            patches.publish_private(self.directory, "owner/repo", "session-token", consent=True, api=api, operation=SimpleNamespace, download=download)
        self.assertFalse(api.exists)
        result = patches.publish_private(self.directory, "owner/repo", "session-token", consent=True, create=True, api=api, operation=SimpleNamespace, download=download)
        card = (self.root / "remote" / "README.md").read_text()
        self.assertIn("not a standalone model", card)
        self.assertIn("Created with Vespasian", card)
        self.assertIn("https://chrisvoncsefalvay.github.io/vespasian/", card)
        self.assertNotIn("pipeline_tag", card)
        self.assertNotIn("alpha", card)
        self.assertTrue((self.root / "remote" / "vespasian-badge.svg").is_file())
        self.assertTrue(result["verified"])

    def test_existing_repository_card_is_never_overwritten(self):
        api, calls, download = self.api()
        card = self.root / "remote" / "README.md"
        card.parent.mkdir()
        card.write_text("Owner's existing model card")
        patches.publish_private(self.directory, "owner/repo", "session-token", consent=True, api=api, operation=SimpleNamespace, download=download)
        self.assertEqual(card.read_text(), "Owner's existing model card")
        commit = next(value for kind, value in calls if kind == "commit")
        self.assertFalse(any(item.path_in_repo == "README.md" for item in commit["operations"]))

    @unittest.skipIf(os.name == "nt", "Symlink creation requires Windows privileges")
    def test_symlinked_tensor_blob_cannot_be_exported(self):
        self.row_edit()
        self.accept()
        blob = next((self.directory / "blobs").iterdir())
        outside = self.root / "outside.safetensors"
        blob.rename(outside)
        blob.symlink_to(outside)
        with self.assertRaisesRegex(patches.PatchError, "Symlink"):
            patches.export_bundle(self.directory, self.root / "unsafe.zip")

    def test_shared_dispatch_contract(self):
        receipt = patches.dispatch(self.e, {"command":"patch_describe", "path":str(self.directory)})
        self.assertEqual(receipt["best"]["status"], "baseline")
        receipt = patches.dispatch(self.e, {"command":"patch_export", "path":str(self.directory), "destination":str(self.root / "dispatch.zip")})
        self.assertTrue(Path(receipt["path"]).is_file())
        for request in ({"command":"patch_export", "path":str(self.directory)}, {"command":"unknown", "path":str(self.directory)}, {"command":"patch_describe"}):
            with self.assertRaises(patches.PatchError):
                patches.dispatch(self.e, request)


    def test_sparse_journal_never_transfers_the_whole_vocabulary_table(self):
        info = self.e.inspect({"token_texts": ["a"]})
        name, token = info["token_tables"][0]["name"], info["token_candidates"][0]["id"]
        self.e.edit(proposal(self.e, operation="token_row_scale", targets=[name], token_ids=[token], source_token_ids=[], strength=0.03))
        shape = tuple(self.e.token_tables[name].shape)
        original_cpu = torch.Tensor.cpu
        def small_cpu(tensor, *args, **kwargs):
            self.assertNotEqual(tuple(tensor.shape), shape, "Sparse patch copied the whole vocabulary table")
            return original_cpu(tensor, *args, **kwargs)
        fresh = copy.deepcopy(self.base)
        with patch.object(torch.Tensor, "cpu", small_cpu):
            patches.record_candidate(self.e, self.directory, 1)
            self.e.accept({})
            patches.finalize(self.directory, 1, True)
            patches.restore(fresh, self.directory)
        self.equal(fresh, self.e)


    def test_injected_owner_client_cannot_replace_session_credentials(self):
        api, calls, download = self.api()
        api.token = "owner-credential"
        with self.assertRaisesRegex(patches.PatchError, "session credentials"):
            patches.publish_private(self.directory, "owner/repo", "session-token", consent=True, api=api, operation=SimpleNamespace, download=download)
        self.assertFalse(calls)


    def test_mixed_float_profile_uses_requested_loader_dtype_not_uniform_cast(self):
        self.e.model.to(torch.bfloat16)
        self.e.model.model.norm.weight.data = self.e.model.model.norm.weight.data.float()
        self.e.load_dtype = "bfloat16"
        self.e.register_editable()
        self.base = copy.deepcopy(self.e)
        self.directory = self.root / "mixed-journal"
        receipt = patches.initialize(self.e, self.directory)
        self.assertEqual(receipt["base"]["floating_dtypes"], ["bfloat16", "float32"])
        self.assertEqual(receipt["base"]["load_dtype"], "bfloat16")
        self.row_edit()
        self.accept()
        model = copy.deepcopy(self.base.model)
        with patch("worker.load_model_components", return_value=(model, self.base.tokenizer, None)) as loader:
            result = patches.materialize(self.directory, str(self.source), output=self.root / "mixed-model", device="cpu")
        self.assertTrue(result["base_verified"])
        self.assertEqual(loader.call_args.kwargs["dtype"], torch.bfloat16)
        self.assertEqual(model.model.norm.weight.dtype, torch.float32)
        self.assertEqual(model.get_input_embeddings().weight.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
