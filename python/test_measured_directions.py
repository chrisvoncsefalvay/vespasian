"""Measured forward-pass edit directions, not ROME/MEMIT or efficacy claims."""
import copy
import unittest
from unittest.mock import patch
import torch
from test_token_rows import row_engine, row_proposal
from test_worker import engine, proposal


class MeasuredDirectionTests(unittest.TestCase):
    def residual_request(self, instance, **kwargs):
        table = next(item for item in instance.token_table_info if 'output_embedding' in item['roles'])
        ids = [i for i in instance.encode(instance.dataset[0])[1] if i not in instance.special_token_ids()]
        return row_proposal(instance, operation='token_row_residual', targets=[table['name']], ids=ids, **kwargs)

    def test_residual_uses_real_head_input_and_teacher_forced_targets(self):
        instance = row_engine()
        request = self.residual_request(instance)
        ids = request['proposal']['token_ids']; head = instance.model.get_output_embeddings()
        captured = []
        handle = head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().float().clone()))
        prompt, target = instance.encode(instance.dataset[0])
        inputs = torch.tensor([prompt + target])
        with torch.inference_mode():
            logits = instance.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=len(target)+1).logits[0, -(len(target)+1):-1].float()
        handle.remove()
        keys = captured[-1][0, -(len(target)+1):-1]
        contrast = (torch.tensor(target)[:, None] == torch.tensor(ids)[None, :]).float() - logits.softmax(-1)[:, ids]
        expected = contrast.T @ (keys / keys.square().sum(-1, keepdim=True).clamp_min(1e-12))
        expected /= expected.norm(dim=-1, keepdim=True)
        actual, evidence = instance.output_row_directions(request['proposal']['targets'][0], ids, 1, 42)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertEqual(evidence['positions'], len(target))
        self.assertFalse(evidence['autograd'])
        self.assertTrue(all(p.grad is None for p in instance.model.parameters()))

    def test_residual_tied_untied_bfloat_bounds_untouched_rows_and_rollback(self):
        for tied in (False, True):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(tied=tied, dtype=dtype):
                    instance = row_engine(tied=tied, dtype=dtype)
                    request = self.residual_request(instance)
                    name = request['proposal']['targets'][0]; ids = request['proposal']['token_ids']
                    table = instance.token_tables[name]; before = table.detach().clone()
                    result = instance.edit(request)
                    mask = torch.ones(table.shape[0], dtype=torch.bool); mask[ids] = False
                    self.assertTrue(torch.equal(table[mask], before[mask]))
                    self.assertTrue(all(row['relative_norm'] <= .05 for row in result['changed_tensors'][0]['rows']))
                    self.assertGreater(result['changed_tensors'][0]['changed_elements'], 0)
                    self.assertIn('direction_evidence', result['changed_tensors'][0])
                    instance.rollback({})
                    self.assertTrue(torch.equal(table, before))

    def test_residual_rejects_input_only_table_unobserved_ids_and_negative_strength(self):
        instance = row_engine()
        input_name = next(item['name'] for item in instance.token_table_info if item['roles'] == ['input_embedding'])
        request = self.residual_request(instance); request['proposal']['targets'] = [input_name]
        with self.assertRaisesRegex(ValueError, 'does not support'):
            instance.edit(request)
        request = self.residual_request(instance); request['proposal']['token_ids'] = [instance.tokenizer.encode('h')[0]]
        with self.assertRaisesRegex(ValueError, 'edit reference'):
            instance.edit(request)
        with self.assertRaisesRegex(ValueError, 'positive strength'):
            instance.edit(self.residual_request(instance, strength=-.05))
        self.assertFalse(instance.row_pending)

    def test_residual_stored_noop_is_rejected_and_restored(self):
        instance = row_engine(dtype=torch.bfloat16)
        before = copy.deepcopy(instance.model.state_dict())
        with self.assertRaisesRegex(ValueError, 'empty stored update'):
            instance.edit(self.residual_request(instance, strength=0.000001))
        self.assertFalse(instance.row_pending)
        for name, value in instance.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_residual_reads_no_test_or_validation_rows(self):
        first, second = row_engine(), row_engine()
        for row in second.dataset:
            if row['split'] in {'validation', 'test'}:
                row['target'] = 'different'; row['prompt'] = 'different'
        request = self.residual_request(first)
        a, _ = first.output_row_directions(request['proposal']['targets'][0], request['proposal']['token_ids'], 1, 42)
        b, _ = second.output_row_directions(request['proposal']['targets'][0], request['proposal']['token_ids'], 1, 42)
        self.assertTrue(torch.equal(a, b))

    def test_residual_fails_atomically_on_nonfinite_head_capture(self):
        instance = row_engine(); before = copy.deepcopy(instance.model.state_dict())
        with patch.object(instance, 'output_row_directions', side_effect=ValueError('bad direction')):
            with self.assertRaisesRegex(ValueError, 'bad direction'):
                instance.edit(self.residual_request(instance))
        self.assertFalse(instance.row_pending)
        for name, value in instance.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_preservation_projection_annuls_observed_keys(self):
        instance = engine(); name = next(iter(instance.editable)); captured = []
        handle = instance.model.get_submodule(name.removesuffix('.weight')).register_forward_pre_hook(lambda module,args: captured.append(args[0].detach().float().reshape(-1,args[0].shape[-1]).clone()))
        row = next(row for row in instance.dataset if row['split'] == 'preserve')
        prompt, target = instance.encode(row); inputs = torch.tensor([prompt+target])
        with torch.inference_mode():
            instance.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=1)
        handle.remove(); keys = captured[-1][len(prompt)-1:len(prompt)+len(target)-1]
        direction = instance.activation_directions([name], 1, 42, preserve=True)[name]
        relative = (direction @ keys.T).norm() / keys.norm()
        self.assertLess(float(relative), 2e-5)
        result = instance.edit(proposal(instance, operation='activation_preserve'))
        self.assertIn('preserve_key_ranks', result['changed_tensors'][0]['direction_evidence'])
        instance.rollback({})
        self.assertFalse(instance.pending)

    def test_preserve_requires_data_and_rejects_zero_remaining_key_space(self):
        instance = engine(); instance.dataset = [r for r in instance.dataset if r['split'] != 'preserve']
        self.assertNotIn('activation_preserve', instance.allowed_operations())
        with self.assertRaises(ValueError):
            instance.edit(proposal(instance, operation='activation_preserve'))
        instance = engine(); edit = instance.dataset[0]
        instance.dataset = [edit, {**edit, 'split':'preserve'}]
        before = copy.deepcopy(instance.model.state_dict())
        with self.assertRaises(ValueError):
            instance.edit(proposal(instance, operation='activation_preserve'))
        for name, value in instance.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_unsupported_head_and_prompt_mode_do_not_offer_residual(self):
        instance = row_engine()
        class SpecialHead(torch.nn.Linear):
            pass
        old = instance.model.get_output_embeddings()
        instance.model.set_output_embeddings(SpecialHead(old.in_features,old.out_features,bias=False))
        instance.register_editable()
        self.assertNotIn('token_row_residual', instance.allowed_operations())
        instance = row_engine()
        for row in instance.dataset:
            row['mode'] = 'prompts'; row['target'] = ''
        instance.register_editable()
        self.assertNotIn('token_row_residual', instance.allowed_operations())
        self.assertNotIn('activation_preserve', instance.allowed_operations())

if __name__ == '__main__':
    unittest.main()
