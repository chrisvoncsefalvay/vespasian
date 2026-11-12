"""Forward-only probes, measured column projection, and exact replay contracts."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from test_worker import engine, proposal
from test_token_rows import row_engine
import patches
import probes


def probe(instance, **changes):
    request = {"command": "probe_activations", "targets": [next(iter(instance.editable))],
               "contexts": [{"row_id": "edit:0", "source": "prompt", "positions": [-1]}], "top_k": 8}
    request.update(changes)
    return instance.dispatch(request)


def measured(instance, result=None, **changes):
    result = result or probe(instance)
    entry = result["probes"][0]
    request = proposal(instance, operation="direction_projection", rank=1, strength=.2)
    request["proposal"].update(targets=[entry["target"]], direction_id=entry["direction_id"],
                               columns=[column["column"] for column in entry["top_columns"][:3]])
    request["proposal"].update(changes)
    return request


class ProbeTests(unittest.TestCase):
    def test_probe_is_real_forward_mean_and_contrast_without_mutation(self):
        instance = engine(); name = next(iter(instance.editable))
        before = copy.deepcopy(instance.model.state_dict()); captured = []
        module = instance.model.get_submodule(name.removesuffix('.weight'))
        handle = module.register_forward_hook(lambda m, args, output: captured.append(output.detach()[0,-1].float().cpu().clone()))
        with patch('torch.autograd.backward', side_effect=AssertionError('No backward')):
            report = probe(instance, contrast_contexts=[{"row_id":"preserve:0","source":"reference","positions":[-1]}])
        handle.remove()
        row = report['probes'][0]
        expected = captured[0] - captured[1]; expected /= expected.norm()
        vector = torch.tensor(instance.probe_cache[row['direction_id']]['vector'])
        self.assertTrue(torch.equal(vector, expected))
        self.assertNotIn('vector', json.dumps(report))
        for column in row['top_columns']:
            self.assertAlmostEqual(column['projection'], float(vector @ module.weight[:,column['column']]), places=6)
        self.assertFalse(row['evidence']['autograd'])
        self.assertTrue(all(p.grad is None for p in instance.model.parameters()))
        for name, value in instance.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertEqual(len(module._forward_hooks), 0)

    def test_only_eligible_module_layouts_and_visible_contexts_are_supported(self):
        instance = engine(); modules = probes.catalogue(instance)
        self.assertTrue(any('q_proj' in item['name'] for item in modules))
        self.assertFalse(any(item['name'] == 'lm_head.weight' for item in modules))
        before = copy.deepcopy(instance.model.state_dict())
        for bad in [
            {'targets':['lm_head.weight']},
            {'contexts':[{'row_id':'test:0','source':'prompt','positions':[-1]}]},
            {'contexts':[{'row_id':'edit:0','source':'unknown','positions':[-1]}]},
            {'contexts':[{'row_id':'edit:0','source':'prompt','positions':[-999]}]},
            {'contexts':[{'row_id':'edit:0','source':'prompt','positions':[-1,-1]}]},
            {'contexts':[{'row_id':'edit:0','source':'prompt','positions':[True]}]},
            {'top_k':129}, {'targets':[]}, {'unexpected':'code'},
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                probe(instance, **bad)
        self.assertFalse(instance.probe_cache)
        for name, value in instance.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_zero_or_nonfinite_capture_fails_and_removes_hooks(self):
        instance = engine(); name = next(iter(instance.editable)); module = instance.model.get_submodule(name.removesuffix('.weight'))
        identical = [{'row_id':'edit:0','source':'prompt','positions':[-1]}]
        with self.assertRaisesRegex(ValueError, 'zero or nonfinite'):
            probe(instance, contrast_contexts=identical)
        handle = module.register_forward_hook(lambda m,a,o: o * float('nan'))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            probe(instance)
        self.assertEqual(len(module._forward_hooks), 1)
        handle.remove(); self.assertFalse(instance.probe_cache)

    def test_generation_probe_requires_current_evaluation_and_never_caches_test(self):
        instance = engine(); selected = [{'row_id':'validation:0','source':'generation','positions':[-1]}]
        with self.assertRaisesRegex(ValueError, 'Evaluate'):
            probe(instance, contexts=selected)
        report = instance.evaluate({'splits':['validation','test'],'max_new_tokens':4})
        self.assertIn('behavior', report)
        self.assertEqual(set(instance.probe_evaluations), {'validation:0'})
        result = probe(instance, contexts=selected)
        evidence = result['probes'][0]['evidence']['contexts'][0]
        self.assertEqual(evidence['context_tokens'], evidence['prompt_tokens'] + report['samples'][0]['generated_tokens'])

    def test_masked_projection_matches_formula_exactly_and_rolls_back(self):
        instance = engine(); request = measured(instance); p = request['proposal']
        name = p['targets'][0]; weight = instance.editable[name]; before = weight.detach().clone()
        u = torch.tensor(instance.probe_cache[p['direction_id']]['vector']); columns = p['columns']
        expected = before.clone(); expected[:,columns] += -p['strength'] * torch.outer(u, u @ before[:,columns])
        result = instance.edit(request)
        self.assertTrue(torch.equal(weight, expected))
        self.assertLessEqual(result['changed_tensors'][0]['relative_norm'], .1)
        mask = torch.ones(weight.shape[1],dtype=torch.bool); mask[columns] = False
        self.assertTrue(torch.equal(weight[:,mask], before[:,mask]))
        instance.evaluate({'splits':['validation'],'max_new_tokens':4})
        self.assertIn(p['direction_id'], instance.probe_cache)
        instance.rollback({}); self.assertTrue(torch.equal(weight, before))
        instance.edit(request); instance.rollback({})
        self.assertTrue(torch.equal(weight, before))

    def test_reviewed_bound_is_separate_from_projection_strength(self):
        instance = engine(); result = probe(instance); entry = result['probes'][0]
        weight = instance.editable[entry['target']]; before = weight.detach().clone()
        u = torch.tensor(instance.probe_cache[entry['direction_id']]['vector'])
        ratio = float(torch.outer(u,u @ weight).norm()/weight.norm())
        request = measured(instance,result,columns=list(range(weight.shape[1])),strength=.11/ratio)
        with self.assertRaisesRegex(ValueError,'max_relative_norm'):
            instance.edit(request)
        self.assertFalse(instance.pending); self.assertTrue(torch.equal(weight,before))
        instance.max_relative_norm = .12
        edited = instance.edit(request)
        self.assertAlmostEqual(edited['changed_tensors'][0]['relative_norm'], .11, places=5)
        instance.rollback({})

    def test_bfloat_projection_tied_embeddings_and_atomic_rejection(self):
        for tied in (False, True):
            instance = row_engine(tied=tied,dtype=torch.bfloat16); request = measured(instance)
            before = copy.deepcopy(instance.model.state_dict())
            instance.edit(request); instance.rollback({})
            for name,value in instance.model.state_dict().items():
                self.assertTrue(torch.equal(value,before[name]))
            self.assertEqual(instance.model.get_input_embeddings().weight is instance.model.get_output_embeddings().weight,tied)
            bad = copy.deepcopy(request); bad['proposal']['columns'] = [0,0]
            with self.assertRaisesRegex(ValueError,'distinct'):
                instance.edit(bad)
            self.assertFalse(instance.pending)

    def test_handles_expire_on_accept_or_unexpected_model_change(self):
        instance = engine(); request = measured(instance)
        instance.edit(request); instance.accept({})
        with self.assertRaisesRegex(ValueError,'handle'):
            instance.edit(request)
        request = measured(instance)
        with torch.no_grad():
            instance.model.get_input_embeddings().weight[0].add_(1)
        with self.assertRaisesRegex(ValueError,'handle'):
            instance.edit(request)
        self.assertFalse(instance.pending)

    def test_accept_promotes_measured_candidate_outputs_without_reevaluation(self):
        instance = engine()
        instance.evaluate({'splits':['edit','validation'],'max_new_tokens':3})
        baseline = copy.deepcopy(instance.probe_evaluations)
        request = measured(instance)
        instance.edit(request)
        candidate = instance.evaluate({'splits':['validation'],'max_new_tokens':4})
        self.assertEqual(instance.probe_evaluations,baseline)
        self.assertEqual(instance.probe_candidate_evaluations['validation:0']['tokens'],candidate['samples'][0]['token_ids'])
        candidate_id = instance.probe_candidate_state_id
        instance.accept({})
        self.assertFalse(instance.probe_cache)
        self.assertEqual(instance.probe_state_id,candidate_id)
        self.assertEqual(set(instance.probe_evaluations),{'validation:0'})
        self.assertFalse(instance.probe_candidate_evaluations)
        probe(instance,contexts=[{'row_id':'validation:0','source':'generation','positions':[-1]}])
        # No stale baseline-only generation can masquerade as newly measured.
        with self.assertRaisesRegex(ValueError,'Evaluate'):
            probe(instance,contexts=[{'row_id':'edit:0','source':'generation','positions':[-1]}])

    def test_corrupt_dimensions_values_target_or_restore_state_rejects_atomically(self):
        for modification in ('dimension','nan','state','target','weight'):
            instance = engine(); request = measured(instance); p = request['proposal']; cached = instance.probe_cache[p['direction_id']]
            if modification == 'dimension': cached['vector'] = [1.]
            if modification == 'nan': cached['vector'][0] = float('nan')
            if modification == 'state': cached['state_id'] = 'old'
            if modification == 'target': cached['target'] = 'other.weight'
            if modification == 'weight':
                with torch.no_grad(): instance.editable[p['targets'][0]][0,0].add_(1)
                instance.refresh_probe_versions()  # Wrong cold snapshot cannot reuse a handle.
            before = copy.deepcopy(instance.model.state_dict())
            with self.subTest(modification=modification), self.assertRaises(ValueError): instance.edit(request)
            self.assertFalse(instance.pending)
            for name,value in instance.model.state_dict().items(): self.assertTrue(torch.equal(value,before[name]))

    def test_json_cold_state_roundtrip_produces_identical_edit(self):
        instance = engine(); instance.evaluate({'splits':['edit'],'max_new_tokens':4})
        request = measured(instance, probe(instance,contexts=[{'row_id':'edit:0','source':'generation','positions':[-1]}]))
        restored = engine(); restored.model.load_state_dict(instance.model.state_dict())
        for name in ('probe_state_id','probe_cache','probe_evaluations','thinking_mode','max_relative_norm','dataset'):
            setattr(restored,name,json.loads(json.dumps(getattr(instance,name),allow_nan=False)))
        restored.refresh_probe_versions()
        first, second = instance.edit(request), restored.edit(request)
        self.assertEqual(first['changed_tensors'],second['changed_tensors'])
        for name,value in instance.model.state_dict().items(): self.assertTrue(torch.equal(value,restored.model.state_dict()[name]))

    def test_private_patch_replay_restores_exact_values_without_direction_handles(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/'base'; journal = Path(directory)/'patch'
            instance = engine(); instance.model.save_pretrained(base)
            instance.source_dir = base; instance.source_files = instance.manifest(base)
            patches.initialize(instance,journal)
            request = measured(instance); instance.edit(request)
            patches.record_candidate(instance,journal,1); instance.accept({}); patches.finalize(journal,1,True)
            restored = engine(); restored.source_dir = base; restored.source_files = instance.source_files
            restored.dispatch({'command':'patch_restore','directory':str(journal),'which':'best'})
            self.assertFalse(restored.probe_cache)
            for name,value in instance.model.state_dict().items(): self.assertTrue(torch.equal(value,restored.model.state_dict()[name]))
            a=instance.evaluate({'splits':['validation'],'max_new_tokens':4}); b=restored.evaluate({'splits':['validation'],'max_new_tokens':4})
            self.assertEqual(a['samples'],b['samples'])

    def test_batched_rows_keep_global_ids_and_cache_merges(self):
        instance = engine(); instance.dataset.insert(1,{'prompt':'other','target':'o','split':'edit'})
        instance.ensure_row_ids(); rows = instance.dataset
        try:
            for row in rows[:2]:
                instance.dataset=[row]; result=instance.evaluate({'splits':['edit'],'max_new_tokens':2})
                self.assertEqual(result['samples'][0]['row_id'],row['row_id'])
        finally: instance.dataset=rows
        self.assertEqual(set(instance.probe_evaluations),{'edit:0','edit:1'})
        probe(instance,contexts=[{'row_id':'edit:1','source':'generation','positions':[-1]}])

    def test_thinking_modes_are_explicit_and_auto_defers_to_template(self):
        instance=engine(); instance.tokenizer.chat_template='{{ enable_thinking }}'
        for mode in ('off','on','auto'):
            instance.thinking_mode=mode
            with patch.object(instance.tokenizer,'apply_chat_template',wraps=instance.tokenizer.apply_chat_template) as apply:
                instance.prompt_tokens('question')
                if mode=='auto': self.assertNotIn('enable_thinking',apply.call_args.kwargs)
                else: self.assertEqual(apply.call_args.kwargs['enable_thinking'],mode=='on')
        instance.thinking_mode='on'; instance.tokenizer.chat_template='{{ messages }}'
        with self.assertRaisesRegex(ValueError,'unavailable'): instance.prompt_tokens('question')

    def test_long_output_request_is_bounded_without_running_long_inference(self):
        instance=engine(); instance.max_seq_length=40000
        with patch.object(instance.model,'generate',return_value=torch.tensor([[1]*7+[2]])) as generate:
            instance.evaluate({'splits':['edit'],'max_new_tokens':4096})
            self.assertEqual(generate.call_args.kwargs['max_new_tokens'],4096)
        for maximum in (True,32769,0):
            with self.assertRaises(ValueError): instance.evaluate({'max_new_tokens':maximum})
        instance.max_seq_length=128
        with self.assertRaisesRegex(ValueError,'exceeds'): instance.evaluate({'max_new_tokens':4096})


if __name__=='__main__': unittest.main()
