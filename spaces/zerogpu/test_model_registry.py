"""Operator-configured model choices, without downloads or GPU allocation."""
import os
from types import MappingProxyType
import unittest
from unittest.mock import patch
from test_gpu import gpu

class RegistryTests(unittest.TestCase):
    def test_compatible_operator_ids_are_not_limited_to_tested_shortlist(self):
        with patch.dict(os.environ, {'VESPASIAN_SPACE_MODELS':'example/custom-model','VESPASIAN_SPACE_MODEL_REVISIONS':'{}','VESPASIAN_SPACE_SKIP_MODEL_LOAD':'1'}):
            registry=gpu._preload_models()
        self.assertEqual(list(registry),['example/custom-model'])
        with patch.object(gpu,'MODEL_TEMPLATES',registry):
            self.assertEqual(gpu.model_options(),[('example/custom-model','example/custom-model')])
            self.assertIsNone(gpu.model_availability()[0]['tested_reference'])
            self.assertFalse(gpu.model_availability()[0]['available'])
            with self.assertRaisesRegex(ValueError,'unavailable'):
                gpu._model_loader('another/not-configured',revision=None,device='cpu',dtype=None)

    def test_invalid_ids_and_unconfigured_revisions_are_rejected(self):
        for ids,revisions in [('../local/path','{}'),('example/model','{"other/model":"abc"}'),('', '{}')]:
            with self.subTest(ids=ids),patch.dict(os.environ,{'VESPASIAN_SPACE_MODELS':ids,'VESPASIAN_SPACE_MODEL_REVISIONS':revisions,'VESPASIAN_SPACE_SKIP_MODEL_LOAD':'1'}):
                with self.assertRaises(ValueError):gpu._preload_models()

    def test_verified_reference_is_distinct_from_current_revision(self):
        name=gpu.SUPPORTED_MODELS[1]
        with patch.object(gpu,'MODEL_TEMPLATES',MappingProxyType({name:gpu.ModelTemplate(name,'different')})):
            value=gpu.model_availability()[0]
            self.assertEqual(value['revision'],'different')
            self.assertEqual(value['tested_reference']['revision'],'dcc83ea841ab6100d6b47a070329e1ba4cf78752')
            self.assertIn('newer primitives not',value['tested_reference']['checks'])

if __name__=='__main__':unittest.main()
