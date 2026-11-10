"""Fixed health datasets and scorers, without network or model dependencies."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from health import HealthError, MMLU_SUBJECTS, canonical, parse_selection, prepare, score, validate_manifest


class HealthTests(unittest.TestCase):
    def test_catalogue_contains_all_fifty_seven_subjects(self):
        self.assertEqual(len(MMLU_SUBJECTS), 57)
        self.assertEqual(len(set(MMLU_SUBJECTS)), 57)

    def test_selection_names_and_counts_are_explicit(self):
        self.assertEqual(parse_selection("mmlu:anatomy:7"), {"check": "mmlu:anatomy", "examples": 7})
        for value in ("copying", "copying:0", "copying:-1", "copying:1.5", "copying:1001", "arbitrary:1", "mmlu:unknown:1"):
            with self.subTest(value=value), self.assertRaises(HealthError):
                parse_selection(value)

    def test_local_checks_need_no_hub_runtime_or_credentials(self):
        with patch("health._runtime", side_effect=AssertionError("Unexpected Hub access")):
            result = prepare([parse_selection("copying:3"), parse_selection("arithmetic:2"), parse_selection("json:4")], 42)
        self.assertEqual(len(result["rows"]), 9)
        self.assertEqual([check["count"] for check in result["checks"]], [3, 2, 4])
        validate_manifest(result)

    def test_seeded_examples_are_stable_independent_of_selection_order(self):
        checks = [parse_selection("copying:5"), parse_selection("arithmetic:5"), parse_selection("json:5")]
        first = prepare(checks, 42)
        second = prepare(list(reversed(checks)), 42)
        self.assertEqual({r["id"]: r for r in first["rows"]}, {r["id"]: r for r in second["rows"]})
        self.assertNotEqual(first["rows_sha256"], prepare(checks, 43)["rows_sha256"])
        self.assertEqual(first, prepare(checks, 42))

    def test_no_duplicate_generated_examples_at_maximum_count(self):
        for check in ("copying", "arithmetic", "json"):
            result = prepare([parse_selection(check + ":1000")], 3)
            self.assertEqual(len(result["rows"]), 1000)
            self.assertEqual(len({row["prompt"] for row in result["rows"]}), 1000)

    def test_duplicate_selection_and_invalid_seed_are_rejected(self):
        selection = parse_selection("copying:1")
        for selections, seed in (([], 1), ([selection, selection], 1), ([selection], -1), ([selection], 2**32), ([selection], True)):
            with self.subTest(selections=selections, seed=seed), self.assertRaises(HealthError):
                prepare(selections, seed)

    def test_mmlu_uses_pinned_test_subset_and_records_indices(self):
        data = [{"question": f"Question {i}", "choices": ["one", "two", "three", "four"], "answer": i % 4} for i in range(10)]
        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha="a" * 40)
        loader = Mock(return_value=data)
        result = prepare([parse_selection("mmlu:anatomy:3")], 7, api=api, loader=loader)
        loader.assert_called_once_with("cais/mmlu", "anatomy", split="test", revision="a" * 40)
        source = result["checks"][0]["source"]
        self.assertEqual(source["revision"], "a" * 40)
        self.assertEqual(source["shots"], 0)
        self.assertEqual(source["sample_indices"], [row["source_index"] for row in result["rows"]])
        self.assertEqual(len(set(source["sample_indices"])), 3)
        self.assertTrue(all(row["prompt"].endswith("Answer:") for row in result["rows"]))
        validate_manifest(result)

    def test_mmlu_fails_instead_of_silently_reducing_count(self):
        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha="a" * 40)
        with self.assertRaisesRegex(HealthError, "has 2 test examples"):
            prepare([parse_selection("mmlu:anatomy:3")], 7, api=api, loader=Mock(return_value=[{}, {}]))

    def test_mmlu_invalid_rows_and_unpinned_revisions_fail(self):
        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha="main")
        with self.assertRaisesRegex(HealthError, "immutable"):
            prepare([parse_selection("mmlu:anatomy:1")], 7, api=api, loader=Mock())
        api.dataset_info.return_value = SimpleNamespace(sha="a" * 40)
        for row in ({}, {"question": "q", "choices": ["a", "b", "c", "d"], "answer": True}):
            with self.assertRaisesRegex(HealthError, "invalid"):
                prepare([parse_selection("mmlu:anatomy:1")], 7, api=api, loader=Mock(return_value=[row]))

    def test_loader_errors_do_not_echo_credentials(self):
        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha="a" * 40)
        with self.assertRaises(HealthError) as raised:
            prepare([parse_selection("mmlu:anatomy:1")], 7, api=api, loader=Mock(side_effect=ValueError("hf_private_secret")))
        self.assertNotIn("hf_private_secret", str(raised.exception))

    def test_manifest_hash_and_counts_detect_changes(self):
        original = prepare([parse_selection("copying:2")], 5)
        changed = copy.deepcopy(original)
        changed["rows"][0]["target"] = "tampered"
        with self.assertRaisesRegex(HealthError, "hash"):
            validate_manifest(changed)
        changed = copy.deepcopy(original)
        changed["checks"][0]["count"] = 3
        with self.assertRaisesRegex(HealthError, "definitions"):
            validate_manifest(changed)
        changed = copy.deepcopy(original)
        changed["rows"][1] = changed["rows"][0]
        changed["rows_sha256"] = hashlib.sha256(canonical(changed["rows"])).hexdigest()
        with self.assertRaisesRegex(HealthError, "duplicate"):
            validate_manifest(changed)

    def test_manifest_requires_seed_and_source_provenance(self):
        original = prepare([parse_selection("copying:1")], 5)
        for change in ({"seed": True}, {"seed": -1}):
            with self.assertRaisesRegex(HealthError, "seed"):
                validate_manifest({**original, **change})
        changed = copy.deepcopy(original)
        changed["checks"][0]["source"]["seed"] = 99
        with self.assertRaisesRegex(HealthError, "provenance"):
            validate_manifest(changed)
        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha="a" * 40)
        rows = [{"question":"q", "choices":["one","two","three","four"], "answer":1}]
        original = prepare([parse_selection("mmlu:anatomy:1")], 5, api=api, loader=Mock(return_value=rows))
        for field, value in (("revision", "main"), ("revision", "a" * 41), ("split", "dev"), ("sample_indices", [1])):
            changed = copy.deepcopy(original)
            changed["checks"][0]["source"][field] = value
            with self.subTest(field=field), self.assertRaises(HealthError):
                validate_manifest(changed)

    def test_copying_is_character_and_spacing_exact(self):
        self.assertTrue(score("copying", "A  b", "A  b"))
        for value in ("a  b", "A b", " A  b", "A  b ", '"A  b"'):
            self.assertFalse(score("copying", value, "A  b"))

    def test_copying_allows_exactly_one_final_line_break(self):
        target = "A  b"
        for value in (target, target + "\n", target + "\r\n"):
            self.assertTrue(score("copying", value, target))
        for value in (target + "\n\n", target + "\r\n\r\n", target + "\n\r\n", target + "\r", target + " \n", target + "\t\n", " " + target + "\n", "a  b\n", "A b\n"):
            self.assertFalse(score("copying", value, target))
        self.assertTrue(score("copying", " spaced \n", " spaced "))
        self.assertFalse(score("copying", " spaced\n", " spaced "))

    def test_arithmetic_accepts_only_a_numeric_answer(self):
        for value in ("42", "+42", " 42\n", "0042"):
            self.assertTrue(score("arithmetic", value, "42"))
        for value in ("42 is the answer", "6 * 7 = 42", "42.0", "4 2", "NaN"):
            self.assertFalse(score("arithmetic", value, "42"))

    def test_json_requires_exact_payload_and_types(self):
        target = '{"active":true,"count":1}'
        self.assertTrue(score("json", '{ "count": 1, "active": true }', target))
        for value in ('{"active":1,"count":1}', '{"active":true,"count":1.0}', '{"active":true,"count":1,"extra":0}', '{"active":true,"count":1,"count":1}', '```json\n' + target + '\n```', '{"active":true,"count":NaN}'):
            self.assertFalse(score("json", value, target))

    def test_cli_creates_an_explicit_manifest_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            command = [sys.executable, str(Path(__file__).with_name("health.py")), "--selection", "copying:2", "--seed", "1", "--output", str(path)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["count"], 2)
            self.assertEqual(summary["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            validate_manifest(json.loads(path.read_text()))
            before = path.read_bytes()
            retry = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
