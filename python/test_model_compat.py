"""Offline synthetic compatibility checks, distinct from tested public checkpoints."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from worker import Engine, load_model_components, model_structure, validate_model_contract

class ModelCompatibilityTests(unittest.TestCase):
    def fixtures(self, root, family):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        model = (GPT2LMHeadModel(GPT2Config(vocab_size=16,n_embd=16,n_layer=1,n_head=2,n_positions=64,bos_token_id=0,eos_token_id=1,pad_token_id=3)) if family=='gpt2' else
                 LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,bos_token_id=0,eos_token_id=1,pad_token_id=3)))
        vocabulary={word:i for i,word in enumerate(['<bos>','<eos>','<unk>','<pad>','alpha','beta','gamma','a','b','c','hello','world','one','two','three','four'])}
        backend=Tokenizer(WordLevel(vocabulary,unk_token='<unk>'));backend.pre_tokenizer=Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,bos_token='<bos>',eos_token='<eos>',unk_token='<unk>',pad_token='<pad>')
        source=root/'source';model.save_pretrained(source);tokenizer.save_pretrained(source)
        dataset=root/'data.jsonl';dataset.write_text('\n'.join(json.dumps(r) for r in [{'prompt':'alpha','target':'a','split':'edit'},{'prompt':'beta','target':'b','split':'validation'},{'prompt':'gamma','target':'c','split':'preserve'}]))
        return source,dataset

    def test_local_llama_and_gpt2_load_edit_rollback_export_reload(self):
        for family in ('llama','gpt2'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);source,data=self.fixtures(root,family)
                e=Engine();info=e.initialise({'model':str(source),'dataset':str(data),'device':'cpu','max_seq_length':64,'prompt_format':'plain'})
                self.assertEqual(info['architecture'],family)
                self.assertEqual(info['generation']['conditioning'],'plain_prompt')
                if family=='gpt2':
                    self.assertFalse(e.editable)
                    self.assertEqual(set(info['allowed_operations']),{'token_row_scale','token_row_interpolate','token_row_residual'})
                before=copy.deepcopy(e.model.state_dict());baseline=e.evaluate({'max_new_tokens':2})
                inspected=e.inspect({'token_texts':['a']});token=next(x['id'] for x in inspected['token_candidates'] if x['edit_reference'])
                table=next(x for x in inspected['token_tables'] if 'token_row_residual' in x['operations'])
                proposal={'targets':[table['name']],'operation':'token_row_residual','token_ids':[token],'source_token_ids':[],'strength':0.03,'rank':1,'seed':3,'rationale':'Offline compatibility test'}
                e.edit({'proposal':proposal});e.rollback({})
                self.assertTrue(all(torch.equal(v,e.model.state_dict()[k]) for k,v in before.items()))
                self.assertEqual(baseline['samples'],e.evaluate({'max_new_tokens':2})['samples'])
                e.inspect({'token_texts':['a']});e.edit({'proposal':proposal});e.accept({})
                destination=root/'saved';e.save({'path':str(destination)})
                reloaded=Engine();reloaded.initialise({'model':str(destination),'dataset':str(data),'device':'cpu','max_seq_length':64,'prompt_format':'plain'})
                self.assertTrue(all(torch.equal(v,reloaded.model.state_dict()[k]) for k,v in e.model.state_dict().items()))
                self.assertEqual(e.evaluate({'max_new_tokens':2})['samples'],reloaded.evaluate({'max_new_tokens':2})['samples'])

    def test_missing_template_requires_explicit_plain_choice(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,data=self.fixtures(Path(temporary),'gpt2')
            with self.assertRaisesRegex(ValueError,'Explicitly choose'):
                Engine().initialise({'model':str(source),'dataset':str(data),'device':'cpu'})

    def test_optional_forward_argument_is_not_sent_to_unsupported_models(self):
        class Legacy(torch.nn.Module):
            def forward(self,input_ids): return input_ids
        e=Engine();e.model=Legacy()
        self.assertEqual(e.forward(input_ids='result',logits_to_keep=2),'result')
        class Older(torch.nn.Module):
            def forward(self,input_ids,num_logits_to_keep=0): return num_logits_to_keep
        e.model=Older();self.assertEqual(e.forward(input_ids='x',logits_to_keep=3),3)

    def test_encoder_decoder_and_quantized_configs_reject_before_weights(self):
        for config in (SimpleNamespace(is_encoder_decoder=True),SimpleNamespace(is_encoder_decoder=False,quantization_config={'bits':4}),SimpleNamespace(model_type="bert",is_decoder=False),SimpleNamespace(model_type="gpt2",architectures=["GPT2ForSequenceClassification"])):
            with patch('transformers.AutoConfig.from_pretrained',return_value=config),patch('transformers.AutoModelForCausalLM.from_pretrained') as load:
                with self.assertRaises(ValueError):load_model_components('example/model',revision=None,device='cpu',dtype=torch.float32)
                load.assert_not_called()

    def test_layer_metadata_uses_known_aliases(self):
        self.assertEqual(model_structure(SimpleNamespace(model_type='gpt2',n_layer=12))['num_hidden_layers'],12)
        with self.assertRaises(ValueError):model_structure(SimpleNamespace(model_type='custom',n_layer=True))

if __name__=='__main__':unittest.main()
