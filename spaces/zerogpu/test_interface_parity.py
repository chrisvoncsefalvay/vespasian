"""Same-input parity across the persistent and allocation-restored model engines.

Uses local tiny transformers, without downloads or paid frontier calls. This is
an exact transport/state test on the same hardware, not a claim that independent
frontier samples or different accelerator arithmetic are deterministic.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["VESPASIAN_SPACE_SKIP_MODEL_LOAD"] = "1"
try:
    import spaces
    if not hasattr(spaces, "GPU"):
        raise ImportError("The source namespace is not the scheduler")
except ImportError:
    sys.modules["spaces"] = SimpleNamespace(GPU=lambda **kwargs: lambda function: function)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from test_worker import engine, proposal
from worker import Engine
import gpu


class InterfaceParityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = engine()
        self.original = copy.deepcopy(self.fixture.model.state_dict())
        self.revision = "b" * 40
        source = self.root / self.revision
        self.fixture.model.save_pretrained(source)
        self.fixture.tokenizer.save_pretrained(source)
        self.dataset = self.root / "dataset.jsonl"
        self.dataset.write_text("\n".join(json.dumps(row) for row in self.fixture.dataset))
        template = gpu.ModelTemplate(gpu.SUPPORTED_MODELS[0], self.revision,
                                     self.fixture.model, self.fixture.tokenizer)
        registry = patch.object(gpu, "MODEL_TEMPLATES", MappingProxyType({template.identifier: template}))
        registry.start()
        self.addCleanup(registry.stop)
        hub = patch("huggingface_hub.hf_hub_download", return_value=str(source / "config.json"))
        hub.start()
        self.addCleanup(hub.stop)
        self.state = self.root / "restored"
        self.request_id = 0
        self.direct = Engine(model_loader=gpu._model_loader)

    def space(self, command, **kwargs):
        self.request_id += 1
        events = list(gpu.dispatch(str(self.state), {"id": self.request_id, "command": command, **kwargs}))
        result = events[-1]
        self.assertEqual(result["type"], "result")
        self.assertTrue(result["ok"], result)
        return result["data"]

    def initialise(self, mode="paired"):
        if mode != "paired":
            rows = [{**row, "mode": mode, **({"prompt": "", "target": row["prompt"]}
                     if mode == "corpus" else {"target": ""})} for row in self.fixture.dataset]
            self.dataset.write_text("\n".join(json.dumps(row) for row in rows))
        request = {"model": gpu.SUPPORTED_MODELS[0], "revision": self.revision,
                   "dataset": str(self.dataset), "device": "cpu", "max_seq_length": 128, "seed": 17}
        direct = self.direct.initialise(request)
        restored = self.space("init", **request)
        self.assertEqual(direct, restored)
        return direct

    def compare_weights_and_rollouts(self):
        metadata, tensors = gpu._read_snapshot(self.state)
        restored = gpu._restore_engine(metadata, tensors, lambda event: None)
        for name, value in self.direct.model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.model.state_dict()[name]), name)
        self.assertEqual(self.direct.changed, restored.changed)
        self.assertEqual(self.direct.row_changed, restored.row_changed)
        request = {"splits": ["edit", "validation", "preserve", "test"], "max_new_tokens": 4}
        expected = self.direct.evaluate(request)
        with patch.object(gpu, "MAX_EXAMPLES", 1):
            actual = self.space("evaluate", **request)
        self.assertEqual(expected["samples"], actual["samples"])
        self.assertEqual(expected["metrics"], actual["metrics"])
        self.assertEqual(expected["behavior"], actual["behavior"])

    def test_all_supported_operations_match_through_accept_and_rollback(self):
        info = self.initialise()
        self.assertEqual(set(info["allowed_operations"]), {"scale", "low_rank", "noise", "direction_projection", "activation_rank_one", "activation_preserve", "token_row_scale", "token_row_interpolate", "token_row_residual"})
        self.compare_weights_and_rollouts()
        for index, operation in enumerate(op for op in info["allowed_operations"] if not op.startswith("token_row_") and op != "direction_projection"):
            with self.subTest(operation=operation):
                request = proposal(self.direct, operation=operation, seed=31 + index, strength=0.01)
                expected = self.direct.edit(request)
                actual = self.space("edit", **request)
                self.assertEqual({k: v for k, v in expected.items() if k != "elapsed_seconds"},
                                 {k: v for k, v in actual.items() if k != "elapsed_seconds"})
                self.compare_weights_and_rollouts()
                self.assertEqual(self.direct.accept({}), self.space("accept"))
                self.compare_weights_and_rollouts()
                rejected = proposal(self.direct, operation=operation, seed=87 + index, strength=0.02 if operation == "activation_preserve" else -0.01)
                self.direct.edit(rejected)
                self.space("edit", **rejected)
                self.assertEqual(self.direct.rollback({}), self.space("rollback"))
                self.compare_weights_and_rollouts()
        for name, value in self.fixture.model.state_dict().items():
            self.assertTrue(torch.equal(value, self.original[name]), name)

    def test_measured_direction_probes_and_projection_match_accept_and_rollback(self):
        self.initialise()
        self.compare_weights_and_rollouts()

        def without_local_ids(value):
            if isinstance(value, dict):
                return {k:without_local_ids(v) for k,v in value.items() if k not in {"direction_id","state_id","elapsed_seconds"}}
            if isinstance(value, list):
                return [without_local_ids(item) for item in value]
            return value

        for accept in (True,False):
            request = {"targets":[next(iter(self.direct.editable))],
                       "contexts":[{"row_id":"edit:0","source":"generation","positions":[-2,-1]}],
                       "contrast_contexts":[{"row_id":"preserve:0","source":"prompt","positions":[-1]}],"top_k":8}
            first = self.direct.probe_activations(request)
            second = self.space("probe_activations",**request)
            self.assertEqual(without_local_ids(first),without_local_ids(second))
            def edit_request(report):
                entry = report["probes"][0]
                return proposal(self.direct, targets=[entry["target"]], operation="direction_projection",rank=1,strength=.2,
                                direction_id=entry["direction_id"],columns=[row["column"] for row in entry["top_columns"][:3]])
            expected = self.direct.edit(edit_request(first))
            actual = self.space("edit",**edit_request(second))
            self.assertEqual(without_local_ids(expected),without_local_ids(actual))
            self.compare_weights_and_rollouts()
            if accept:
                self.assertEqual(self.direct.accept({}),self.space("accept"))
            else:
                self.assertEqual(self.direct.rollback({}),self.space("rollback"))
            self.compare_weights_and_rollouts()
        for name,value in self.fixture.model.state_dict().items():
            self.assertTrue(torch.equal(value,self.original[name]),name)

    def test_prompt_only_mode_has_the_same_restricted_capabilities(self):
        info = self.initialise("prompts")
        self.assertEqual(info["allowed_operations"], ["scale", "low_rank", "noise", "direction_projection", "token_row_scale", "token_row_interpolate"])
        self.compare_weights_and_rollouts()

    def inspect_both(self):
        request = {"token_texts": ["alpha beta"]}
        expected = self.direct.inspect(request)
        actual = self.space("inspect", **request)
        self.assertEqual(expected, actual)
        return expected

    def exercise_token_rows(self, tied=False):
        if tied:
            self.fixture.model.config.tie_word_embeddings = True
            self.fixture.model.tie_weights()
            self.original = copy.deepcopy(self.fixture.model.state_dict())
            self.fixture.model.save_pretrained(self.root / self.revision)
        self.initialise()
        inspection = self.inspect_both()
        self.assertEqual(len(inspection["token_tables"]), 1 if tied else 2)
        ids = [row["id"] for row in inspection["token_candidates"]][:2]
        self.assertEqual(len(ids), 2)
        for table in inspection["token_tables"]:
            name = table["name"]
            self.assertEqual(table["tied"], tied)
            if tied:
                self.assertEqual(set(table["roles"]), {"input_embedding", "output_embedding"})
            for operation in (op for op in table["operations"] if op.startswith("token_row_")):
                with self.subTest(tied=tied, table=name, operation=operation):
                    self.inspect_both()
                    selected_ids = ([row["id"] for row in inspection["token_candidates"] if row.get("edit_reference")][:2]
                                    if operation == "token_row_residual" else ids)
                    self.assertTrue(selected_ids)
                    request = proposal(self.direct, targets=[name], operation=operation,
                                       token_ids=selected_ids, source_token_ids=list(reversed(selected_ids)) if operation.endswith("interpolate") else [],
                                       strength=0.03, rank=16)
                    original = self.direct.token_tables[name].detach().clone()
                    expected, actual = self.direct.edit(request), self.space("edit", **request)
                    self.assertEqual({k: v for k, v in expected.items() if k != "elapsed_seconds"},
                                     {k: v for k, v in actual.items() if k != "elapsed_seconds"})
                    untouched = [i for i in range(original.shape[0]) if i not in selected_ids]
                    self.assertTrue(torch.equal(original[untouched], self.direct.token_tables[name][untouched]))
                    self.compare_weights_and_rollouts()
                    metadata, tensors = gpu._read_snapshot(self.state)
                    self.assertEqual(metadata["engine"]["row_pending_ids"][name], selected_ids)
                    self.assertEqual(tensors["row.pending." + name].shape[0], len(selected_ids))
                    self.assertNotIn("current." + name, tensors)
                    self.assertEqual(self.direct.rollback({}), self.space("rollback"))
                    self.compare_weights_and_rollouts()
                    self.assertTrue(torch.equal(original, self.direct.token_tables[name]))
                    self.inspect_both()
                    self.direct.edit(request)
                    self.space("edit", **request)
                    self.assertEqual(self.direct.accept({}), self.space("accept"))
                    self.compare_weights_and_rollouts()
        self.assertTrue(self.direct.row_changed)
        destination = self.root / "row-checkpoint"
        direct_destination = self.root / "direct-row-checkpoint"
        self.space("save", path=str(destination))
        self.direct.save({"path": str(direct_destination)})
        from transformers import LlamaForCausalLM
        saved = LlamaForCausalLM.from_pretrained(destination)
        direct_saved = LlamaForCausalLM.from_pretrained(direct_destination)
        for name, value in self.direct.model.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), saved.state_dict()[name]), name)
            self.assertTrue(torch.equal(value.cpu(), direct_saved.state_dict()[name]), name)
        if tied:
            self.assertIs(saved.get_input_embeddings().weight, saved.get_output_embeddings().weight)
        for name, value in self.fixture.model.state_dict().items():
            self.assertTrue(torch.equal(value, self.original[name]), name)

    def test_untied_token_rows_match_after_restore_rollback_and_reload(self):
        self.exercise_token_rows()

    def test_tied_token_rows_are_edited_once_and_match_after_reload(self):
        self.exercise_token_rows(tied=True)

    def test_missing_preservation_examples_disable_projected_edit_in_both_surfaces(self):
        self.dataset.write_text("\n".join(json.dumps(row) for row in self.fixture.dataset
                                         if row["split"] != "preserve"))
        info = self.initialise()
        self.assertNotIn("activation_preserve", info["allowed_operations"])
        self.assertIn("activation_rank_one", info["allowed_operations"])
        self.assertIn("token_row_residual", info["allowed_operations"])
        self.inspect_both()
        self.compare_weights_and_rollouts()

    def test_durable_latest_best_patches_match_and_restore_without_full_checkpoint(self):
        self.initialise()
        direct_journal, space_journal = self.root / "direct-journal", self.root / "space-journal"
        self.direct.dispatch({"command": "patch_init", "directory": str(direct_journal)})
        self.space("patch_init", directory=str(space_journal))
        inspection = self.inspect_both()
        table = next(row for row in inspection["token_tables"] if "token_row_residual" in row["operations"])
        ids = [row["id"] for row in inspection["token_candidates"] if row.get("edit_reference")][:1]
        request = proposal(self.direct, targets=[table["name"]], operation="token_row_residual",
                           token_ids=ids, source_token_ids=[], strength=0.02, rank=16)
        self.direct.edit(request)
        self.space("edit", **request)
        first = self.direct.dispatch({"command": "patch_candidate", "directory": str(direct_journal), "turn": 1})
        second = self.space("patch_candidate", directory=str(space_journal), turn=1)
        self.assertEqual(first["latest"], second["latest"])
        self.assertEqual(first["best"]["turn"], 0)
        self.direct.accept({}); self.space("accept")
        self.direct.dispatch({"command": "patch_finalize", "directory": str(direct_journal), "turn": 1, "accepted": True})
        saved = self.space("patch_finalize", directory=str(space_journal), turn=1, accepted=True)
        self.assertEqual(saved["best"]["turn"], 1)
        expected = copy.deepcopy(self.direct.model.state_dict())
        candidate = proposal(self.direct, operation="low_rank", strength=0.01)
        self.direct.edit(candidate); self.space("edit", **candidate)
        self.direct.dispatch({"command": "patch_candidate", "directory": str(direct_journal), "turn": 2})
        self.space("patch_candidate", directory=str(space_journal), turn=2)
        self.direct.rollback({}); self.space("rollback")
        self.direct.dispatch({"command": "patch_finalize", "directory": str(direct_journal), "turn": 2, "accepted": False})
        rejected = self.space("patch_finalize", directory=str(space_journal), turn=2, accepted=False)
        self.assertEqual(rejected["latest"]["status"], "rejected")
        self.assertEqual(rejected["best"], saved["best"])
        bundle = self.space("patch_export", directory=str(space_journal), path=str(self.root / "patches.zip"))
        self.assertTrue(Path(bundle["path"]).is_file())
        self.assertGreater(bundle["bytes"], 0)
        self.state = self.root / "fresh-restored"
        self.direct = Engine(model_loader=gpu._model_loader)
        self.initialise()
        left = self.direct.dispatch({"command": "patch_restore", "directory": str(direct_journal), "which": "best"})
        right = self.space("patch_restore", directory=str(space_journal), which="best")
        self.assertEqual(left, right)
        for name, value in expected.items():
            self.assertTrue(torch.equal(value, self.direct.model.state_dict()[name]), name)
        self.compare_weights_and_rollouts()
        self.assertFalse((self.root / "checkpoint").exists())

    def test_corpus_generation_and_activation_capability_match(self):
        info = self.initialise("corpus")
        self.assertIn("activation_rank_one", info["allowed_operations"])
        self.assertTrue(info["generation"]["do_sample"])
        self.compare_weights_and_rollouts()


if __name__ == "__main__":
    unittest.main()
