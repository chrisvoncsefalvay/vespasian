"""Deployment limits are explicit and identical for every Space entry point."""

import os
import unittest
from unittest.mock import patch

from limits import space_limits


class DeploymentLimitTests(unittest.TestCase):
    def test_defaults_and_operator_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(space_limits(), {"max_new_tokens": 512, "max_seq_length": 2048})
        with patch.dict(os.environ, {"VESPASIAN_SPACE_MAX_NEW_TOKENS": "32768", "VESPASIAN_SPACE_MAX_SEQ_LENGTH": "131072"}):
            self.assertEqual(space_limits(), {"max_new_tokens": 32768, "max_seq_length": 131072})

    def test_invalid_operator_limits_fail_instead_of_silent_clamping(self):
        for name, values in (("MAX_NEW_TOKENS", ("0", "32769", "1.5", "NaN", "", "true")),
                             ("MAX_SEQ_LENGTH", ("31", "131073", "1024.5", "inf", ""))):
            for value in values:
                with self.subTest(name=name, value=value), patch.dict(os.environ, {"VESPASIAN_SPACE_" + name: value}):
                    with self.assertRaisesRegex(ValueError, "VESPASIAN_SPACE_" + name):
                        space_limits()
