"""Bounded vocabulary-row edits against real tiny tied/untied transformers."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import LlamaForCausalLM
from test_worker import engine, CharacterTokenizer
from worker import Engine


def row_engine(tied=False, dtype=torch.float32):
    instance = engine()
    if tied:
        instance.model.config.tie_word_embeddings = True
        instance.model.tie_weights()
    instance.model.to(dtype=dtype)
    instance.register_editable()
    instance.inspect({"token_texts": ["abcdefgh"]})
    return instance


def row_proposal(instance, operation="token_row_scale", ids=None, sources=None, strength=0.05, **changes):
    value = {"targets": [next(iter(instance.token_tables))], "operation": operation,
             "token_ids": ids or instance.inspected_token_ids[:2], "source_token_ids": sources or [],
             "strength": strength, "rank": 1, "seed": 42, "rationale": "Test selected tokenizer rows",
             "strategy": "token_row_scaling", "plan": "Alter only named rows", "reflection": "No earlier candidates"}
    value.update(changes)
    return {"proposal": value}


class TokenRowTests(unittest.TestCase):
    def test_inspection_excludes_test_text_special_ids_and_discloses_topology(self):
        instance = row_engine()
        instance.dataset[-1] = {"prompt": "~", "target": "~", "split": "test"}
        result = instance.inspect({})
        ids = [row["id"] for row in result["token_candidates"]]
        self.assertNotIn(instance.tokenizer.encode("~")[0], ids)
        self.assertFalse(set(ids) & instance.special_token_ids())
        self.assertEqual(len(result["token_tables"]), 2)
        self.assertTrue(all(not table["tied"] for table in result["token_tables"]))
        self.assertEqual({role for table in result["token_tables"] for role in table["roles"]}, {"input_embedding", "output_embedding"})
        self.assertTrue(all("text" in row and "piece" in row for row in result["token_candidates"]))
        self.assertEqual(result["allowed_operations"][-3:], ["token_row_scale", "token_row_interpolate", "token_row_residual"])

    def test_tied_aliases_are_one_physical_table_and_edit_only_once(self):
        instance = row_engine(tied=True)
        self.assertEqual(len(instance.token_tables), 1)
        table_info = instance.token_table_info[0]
        self.assertTrue(table_info["tied"])
        self.assertEqual(table_info["roles"], ["input_embedding", "output_embedding"])
        self.assertEqual(len(table_info["aliases"]), 2)
        table = instance.token_tables[table_info["name"]]
        before = table.detach().clone()
        request = row_proposal(instance)
        result = instance.edit(request)
        ids = request["proposal"]["token_ids"]
        for token_id in ids:
            # The result is once, not squared, allowing small stored-norm rounding.
            self.assertTrue(torch.allclose(table[token_id], before[token_id] * 1.05, atol=1e-8))
        self.assertTrue(result["changed_tensors"][0]["tied"])
        self.assertIs(instance.model.get_input_embeddings().weight, instance.model.get_output_embeddings().weight)
        instance.rollback({})
        self.assertTrue(torch.equal(table, before))
        alias = next(name for name in table_info["aliases"] if name != table_info["name"])
        with self.assertRaisesRegex(ValueError, "outside the editable"):
            instance.edit(row_proposal(instance, targets=[alias]))
        with self.assertRaises(ValueError):
            # Duplicate is rejected earlier too; aliases must never cause two writes.
            instance.edit(row_proposal(instance, targets=[table_info["name"], alias]))

    def test_selected_rows_leave_every_other_weight_bit_exact_and_rollback(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                instance = row_engine(tied)
                before = copy.deepcopy(instance.model.state_dict())
                request = row_proposal(instance)
                name, ids = request["proposal"]["targets"][0], request["proposal"]["token_ids"]
                instance.edit(request)
                table_info = next(row for row in instance.token_table_info if row["name"] == name)
                for key, value in instance.model.state_dict().items():
                    if key in table_info["aliases"]:
                        mask = torch.ones(value.shape[0], dtype=torch.bool)
                        mask[ids] = False
                        self.assertTrue(torch.equal(value[mask], before[key][mask]), key)
                    else:
                        self.assertTrue(torch.equal(value, before[key]), key)
                self.assertFalse(instance.pending)
                self.assertEqual(instance.row_pending[name]["values"].numel(), len(ids) * instance.token_tables[name].shape[1])
                instance.rollback({})
                for key, value in instance.model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]), key)
                self.assertFalse(instance.row_pending)

    def test_overlapping_interpolation_reads_all_sources_before_any_write(self):
        instance = row_engine()
        name = next(iter(instance.token_tables)); table = instance.token_tables[name]
        ids = instance.inspected_token_ids[:2]
        before = table.detach().clone()
        result = instance.edit(row_proposal(instance, "token_row_interpolate", ids=ids, sources=list(reversed(ids))))
        for detail in result["changed_tensors"][0]["rows"]:
            token_id, source_id = detail["token_id"], detail["source_token_id"]
            expected = before[token_id] + detail["effective_fraction"] * (before[source_id] - before[token_id])
            self.assertTrue(torch.equal(table[token_id], expected))
            self.assertLessEqual(detail["relative_norm"], 0.05)
        instance.accept({})
        self.assertEqual(instance.row_changed[name], set(ids))

    def test_interpolation_caps_large_source_direction_and_bfloat_stored_delta(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                instance = row_engine(dtype=dtype)
                name = next(iter(instance.token_tables)); table = instance.token_tables[name]
                ids = instance.inspected_token_ids[:2]
                with torch.no_grad():
                    table[ids[1]].mul_(1000)
                before = table.detach().clone()
                result = instance.edit(row_proposal(instance, "token_row_interpolate", ids=ids[:1], sources=ids[1:], strength=0.1))
                detail = result["changed_tensors"][0]["rows"][0]
                self.assertLess(detail["effective_fraction"], 0.1)
                measured = torch.linalg.vector_norm(table[ids[0]].float() - before[ids[0]].float())
                self.assertLessEqual(float(measured), 0.1 * float(torch.linalg.vector_norm(before[ids[0]].float())))
                instance.rollback({})
                self.assertTrue(torch.equal(table, before))

    def test_row_selectors_are_grounded_bounded_nonspecial_and_operation_specific(self):
        instance = row_engine()
        ids = instance.inspected_token_ids
        cases = [dict(token_ids=[]), dict(token_ids=[ids[0], ids[0]]), dict(token_ids=[True]),
                 dict(token_ids=[-1]), dict(token_ids=[100]), dict(token_ids=[2]), dict(token_ids=[98]),
                 dict(token_ids=ids * 4), dict(source_token_ids=[ids[1]]), dict(token_ids="1")]
        for changes in cases:
            with self.subTest(changes=changes):
                request = row_proposal(instance);request["proposal"].update(changes)
                with self.assertRaises(ValueError):instance.edit(request)
                self.assertFalse(instance.pending or instance.row_pending)
        for changes in (dict(source_token_ids=[]), dict(source_token_ids=[ids[0],ids[1]]),
                        dict(source_token_ids=[2, ids[0]]), dict(strength=-0.1)):
            request=row_proposal(instance,"token_row_interpolate",ids=ids[:2],sources=ids[2:4]);request["proposal"].update(changes)
            with self.assertRaises(ValueError):instance.edit(request)
        with self.assertRaisesRegex(ValueError,"outside the editable"):
            instance.edit(row_proposal(instance, operation="scale"))
        with self.assertRaises(ValueError):
            instance.edit(row_proposal(instance, targets=[next(iter(instance.editable))]))

    def test_accept_then_rollback_preserves_retained_rows_and_export_reload(self):
        instance = row_engine(tied=True)
        instance.edit(row_proposal(instance));instance.accept({})
        retained = copy.deepcopy(instance.model.state_dict())
        instance.edit(row_proposal(instance,strength=-0.01));instance.rollback({})
        for name,value in instance.model.state_dict().items():self.assertTrue(torch.equal(value,retained[name]))
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"checkpoint"
            receipt=instance.save({"path":str(path)})
            loaded=LlamaForCausalLM.from_pretrained(path)
            self.assertIs(loaded.get_input_embeddings().weight,loaded.get_output_embeddings().weight)
            for name,value in loaded.state_dict().items():self.assertTrue(torch.equal(value,retained[name]))
            self.assertTrue(receipt["changed_token_rows"])

    def test_zero_nonfinite_and_pending_rows_fail_without_partial_edits(self):
        for bad in (0.,float('nan'),float('inf')):
            instance=row_engine();table=next(iter(instance.token_tables.values()));ids=instance.inspected_token_ids[:2]
            with torch.no_grad():table[ids[1]].fill_(bad)
            before=table.detach().clone()
            with self.assertRaises(ValueError):instance.edit(row_proposal(instance,ids=ids))
            self.assertTrue(torch.equal(torch.isnan(table),torch.isnan(before)))
            self.assertTrue(torch.equal(torch.nan_to_num(table),torch.nan_to_num(before)))
            self.assertFalse(instance.row_pending)
        instance=row_engine();instance.edit(row_proposal(instance))
        with self.assertRaises(ValueError):instance.inspect({})
        with self.assertRaises(ValueError):instance.edit(row_proposal(instance))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):instance.save({"path":str(Path(root)/"checkpoint")})

    def test_repeated_sources_noop_and_emit_failure_are_safe(self):
        instance = row_engine(dtype=torch.bfloat16)
        instance.edit(row_proposal(instance, strength=0.000001))
        instance.accept({})
        self.assertEqual(instance.row_changed, {})
        instance = row_engine()
        ids = instance.inspected_token_ids[:3]
        result = instance.edit(row_proposal(instance, "token_row_interpolate", ids=ids[:2], sources=[ids[2], ids[2]]))
        self.assertEqual(result["changed_tensors"][0]["source_token_ids"], [ids[2], ids[2]])
        instance.rollback({})
        before = copy.deepcopy(instance.model.state_dict())
        def fail_emit(event):
            raise RuntimeError("Transport interrupted")
        instance.emit = fail_emit
        with self.assertRaisesRegex(RuntimeError, "Transport interrupted"):
            instance.edit(row_proposal(instance))
        self.assertFalse(instance.row_pending)
        for name, value in instance.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_nonfinite_interpolation_source_never_mutates_destination(self):
        instance = row_engine()
        ids = instance.inspected_token_ids[:2]
        table = next(iter(instance.token_tables.values()))
        with torch.no_grad():
            table[ids[1]].fill_(float("inf"))
        before = table.detach().clone()
        with self.assertRaises(ValueError):
            instance.edit(row_proposal(instance, "token_row_interpolate", ids=ids[:1], sources=ids[1:]))
        self.assertTrue(torch.equal(table, before))
        self.assertFalse(instance.row_pending)

    def test_model_vision_control_tokens_are_excluded_even_with_plain_tokenizer(self):
        instance = row_engine()
        control_ids = instance.inspected_token_ids[:2]
        instance.model.config.vision_start_token_id = control_ids[0]
        instance.model.config.vision_end_token_id = control_ids[1]
        result = instance.inspect({"token_texts": ["abcdefgh"]})
        self.assertFalse(set(control_ids) & {row["id"] for row in result["token_candidates"]})
        with self.assertRaises(ValueError):
            instance.edit(row_proposal(instance, ids=control_ids))

    def test_inspection_limits_are_enforced_and_no_gradients_enabled(self):
        instance=row_engine()
        for request in ({"token_texts":["x"]*17},{"token_texts":["x"*257]},{"token_texts":"x"}):
            with self.assertRaises(ValueError):instance.inspect(request)
        instance.edit(row_proposal(instance));instance.accept({})
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in instance.model.parameters()))


if __name__ == '__main__':unittest.main()
