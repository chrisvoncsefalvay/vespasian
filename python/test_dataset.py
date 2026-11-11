"""Offline checks for revision pinning, field mapping and safe publication."""
import hashlib
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import dataset


SHA = "a" * 40


class Stream:
    info = SimpleNamespace(config_name="default")

    def __init__(self, rows):
        self.rows = rows
        self.consumed = 0

    def __iter__(self):
        for row in self.rows:
            self.consumed += 1
            yield row


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "data.jsonl"
        self.receipt = Path(self.directory.name) / "receipt.json"
        self.api = Mock()
        self.api.dataset_info.return_value = SimpleNamespace(sha=SHA, card_data={"task_categories": ["text-generation"]})

    def fetch(self, rows, **kwargs):
        self.stream = Stream(rows)
        self.loader = Mock(return_value=self.stream)
        return dataset.fetch(repo="org/data", split="train", output=self.output,
            receipt=self.receipt, api=self.api, loader=self.loader,
            runtime_version="5.1.0", **kwargs)

    def test_pins_revision_limits_consumption_and_hashes_exact_bytes(self):
        receipt = self.fetch([{"prompt": f"Question {i}?", "target": " Café\n", "id": i} for i in range(6)], limit=3, revision="main")
        self.api.dataset_info.assert_called_once_with(repo_id="org/data", revision="main")
        self.loader.assert_called_once_with("org/data", name=None, split="train", revision=SHA, streaming=True)
        self.assertEqual(self.stream.consumed, 3)
        self.assertEqual(receipt["rows"], 3)
        self.assertEqual(receipt["revision"], SHA)
        self.assertEqual(receipt["config"], "default")
        self.assertEqual(receipt["ignored_columns"], ["id"])
        self.assertEqual(receipt["sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(json.loads(self.output.read_text().splitlines()[0])["target"], " Café\n")
        self.assertEqual(json.loads(self.receipt.read_text()), receipt)

    def test_nested_explicit_mapping(self):
        self.fetch([{"item": {"question": "Hello?", "answer": "Hello."}, "unused": 2}], prompt_field="item.question", target_field="item.answer")
        self.assertEqual(json.loads(self.output.read_text()), {"prompt": "Hello?", "target": "Hello."})

    def test_shared_column_array_expands_pairs_and_limit_counts_examples(self):
        rows = [{"examples": [{"input": f"Question {i}", "target": i} for i in range(4)], "extra": "ignored"},
                {"examples": [{"input": "Unused", "target": "Unused"}]}]
        receipt = self.fetch(rows, limit=3)
        self.assertEqual(self.stream.consumed, 1)
        self.assertEqual(receipt["source_rows"], 1)
        self.assertEqual(receipt["rows"], 3)
        self.assertEqual(receipt["selection"], "first_examples")
        self.assertEqual(receipt["mapping"], "paired_array")
        self.assertEqual(receipt["prompt_field"], "examples[].input")
        self.assertEqual(receipt["target_field"], "examples[].target")
        self.assertEqual(receipt["ignored_columns"], ["extra"])
        self.assertEqual(receipt["transformations"], {"number_to_text": 3})
        self.assertEqual([json.loads(line) for line in self.output.read_text().splitlines()],
                         [{"prompt": f"Question {i}", "target": str(i)} for i in range(3)])

    def test_limit_does_not_validate_unselected_array_entries(self):
        self.fetch([{"examples": [{"input": "First", "target": "One"}, {"input": "Second"}]}],
                   prompt_field="examples[].input", target_field="examples[].target", limit=1)
        self.assertEqual(json.loads(self.output.read_text())["target"], "One")

    def test_numeric_and_bracket_indexes_select_the_same_nested_item(self):
        self.fetch([{"examples": [{"input": "First", "target": "One"}, {"input": "Second", "target": "Two"}]}],
                   prompt_field="examples.1.input", target_field="examples[1].target")
        self.assertEqual(json.loads(self.output.read_text()), {"prompt": "Second", "target": "Two"})

    def test_identical_or_independent_array_selectors_are_rejected(self):
        row = {"examples": [{"input": "Q", "target": "A"}], "answers": [{"target": "A"}]}
        for prompt, target, message in [
            ("examples[].input", "answers[].target", "same array"),
            ("examples[].input", "examples[0].target", "same array"),
            ("examples.0.input", "examples[0].input", "different"),
            ("examples", "examples", "different"),
        ]:
            with self.subTest(prompt=prompt, target=target), self.assertRaisesRegex(dataset.DatasetError, message):
                self.fetch([row], prompt_field=prompt, target_field=target)
            self.assertFalse(self.output.exists())

    def test_invalid_array_entries_never_disappear_or_shift_pairing(self):
        for items in [[], [{"input": "Q", "target": "A"}, {"input": "Q2"}],
                      [{"input": "Q", "target": "A"}, {"input": "Q2", "target": ""}],
                      [{"input": "Q", "target": "A"}, {"input": 2, "target": "B"}]]:
            with self.subTest(items=items), self.assertRaises(dataset.DatasetError):
                self.fetch([{"examples": items}], prompt_field="examples[].input", target_field="examples[].target")
            self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_fallback_score_maps_preserve_all_correct_answers(self):
        rows = [{"examples": [
            {"input": "Direct", "target": "known", "target_scores": None},
            {"input": "Single", "target": None, "target_scores": '{"wrong":0,"right":1}'},
            {"input": "Multiple", "target": None, "target_scores": {"first": 1, "second": 0, "third": 1}},
        ]}]
        receipt = self.fetch(rows)
        values = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual(receipt["target_field"], "examples[].target ?? examples[].target_scores")
        self.assertEqual(values[0], {"prompt": "Direct", "target": "known"})
        self.assertEqual(values[1]["target"], "B")
        self.assertIn("A. wrong\nB. right", values[1]["prompt"])
        self.assertEqual(json.loads(values[2]["target"]), ["A", "C"])
        self.assertIn("JSON array", values[2]["prompt"])
        self.assertEqual(receipt["transformations"], {"field_only": 1, "binary_choice_scores": 2})
        self.assertEqual(receipt["target_sources"], {"examples[].target": 1, "examples[].target_scores": 2})

    def test_explicit_target_never_silently_uses_scores(self):
        with self.assertRaisesRegex(dataset.DatasetError, "missing or null"):
            self.fetch([{"examples": [{"input": "Q", "target": None, "target_scores": '{"A":1}'}]}],
                       prompt_field="examples[].input", target_field="examples[].target")
        self.assertFalse(self.output.exists())

    def test_ambiguous_or_invalid_score_maps_are_rejected(self):
        for scores in [{"a": 0, "b": 0}, {"a": 0.5, "b": 1}, {"a": float("nan")},
                       {"a": True}, {"": 1}, {}, '["a"]', '{"a":0,"a":1}', 'not json']:
            with self.subTest(scores=scores), self.assertRaises(dataset.DatasetError):
                self.fetch([{"prompt": "Q", "target_scores": scores}],
                           target_field="target_scores")
            self.assertFalse(self.output.exists())

    def test_numeric_targets_are_finite_and_never_booleans(self):
        for target in [float("inf"), float("nan"), True]:
            with self.subTest(target=target), self.assertRaises(dataset.DatasetError):
                self.fetch([{"prompt": "Q", "target": target}])
        receipt = self.fetch([{"prompt": "Q", "target": 2.5}])
        self.assertEqual(json.loads(self.output.read_text())["target"], "2.5")
        self.assertEqual(receipt["transformations"], {"number_to_text": 1})

    def test_corpus_expands_nested_text_without_prompt_or_target_mapping(self):
        receipt = self.fetch([{"documents": [{"content": "First passage"}, {"content": "Second passage"}], "id": "x"}],
                             dataset_mode="corpus", text_field="documents[].content", limit=1)
        self.assertEqual(json.loads(self.output.read_text()), {"mode": "corpus", "prompt": "", "target": "First passage"})
        self.assertEqual(receipt["dataset_mode"], "corpus")
        self.assertEqual(receipt["text_field"], "documents[].content")
        self.assertIsNone(receipt["prompt_field"])
        self.assertIsNone(receipt["target_field"])
        self.assertEqual(receipt["rows"], 1)
        self.assertEqual(receipt["source_rows"], 1)
        self.assertEqual(receipt["ignored_columns"], ["id"])

    def test_corpus_auto_detects_a_text_array(self):
        receipt = self.fetch([{"documents": ["First passage", "Second passage"]}], dataset_mode="corpus")
        self.assertEqual(receipt["text_field"], "documents[]")
        self.assertEqual([json.loads(line)["target"] for line in self.output.read_text().splitlines()], ["First passage", "Second passage"])

    def test_prompts_mode_does_not_require_references(self):
        receipt = self.fetch([{"examples": [{"input": "Question one"}, {"input": "Question two"}]}], dataset_mode="prompts")
        self.assertEqual(receipt["prompt_field"], "examples[].input")
        self.assertIsNone(receipt["target_field"])
        self.assertEqual([json.loads(line) for line in self.output.read_text().splitlines()],
                         [{"mode": "prompts", "prompt": value, "target": ""} for value in ["Question one", "Question two"]])

    def test_single_field_modes_reject_missing_empty_or_nontext_examples(self):
        for mode, row in [("corpus", {"text": ""}), ("corpus", {"text": 12}),
                          ("prompts", {"input": []}), ("prompts", {"unrelated": "Q"})]:
            with self.subTest(mode=mode, row=row), self.assertRaises(dataset.DatasetError):
                self.fetch([row], dataset_mode=mode)
            self.assertFalse(self.output.exists())

    def test_invalid_dataset_mode_fails_before_hub_access(self):
        with self.assertRaisesRegex(dataset.DatasetError, "dataset mode"):
            self.fetch([], dataset_mode="unknown")
        self.api.dataset_info.assert_not_called()

    def test_aliases_are_recorded(self):
        receipt = self.fetch([{"input": "Hello?", "output": "Hello."}])
        self.assertEqual((receipt["prompt_field"], receipt["target_field"]), ("input", "output"))

    def test_import_auto_selects_main_at_pinned_revision(self):
        configs = Mock(return_value=["main", "socratic"])
        receipt = self.fetch([{"question": "Why?", "answer": "Because."}], config_names=configs)
        configs.assert_called_once_with("org/data", revision=SHA)
        self.loader.assert_called_once_with("org/data", name="main", split="train", revision=SHA, streaming=True)
        self.assertIsNone(receipt["config_requested"])

    def test_instruction_alias_precedes_optional_input(self):
        receipt = self.fetch([{"instruction": "Hello?", "input": "", "output": "Hello."}])
        self.assertEqual(receipt["prompt_field"], "instruction")
        self.assertEqual(receipt["ignored_columns"], ["input"])
        self.assertEqual(receipt["mapping"], "field_only")

    def test_explicit_missing_field_never_falls_back(self):
        with self.assertRaisesRegex(dataset.DatasetError, "no prompt field"):
            self.fetch([{"input": "Hello?", "output": "Hello."}], prompt_field="prompt")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.receipt.exists())

    def test_non_string_is_not_silently_coerced(self):
        with self.assertRaisesRegex(dataset.DatasetError, "nonempty text"):
            self.fetch([{"prompt": "Hello?", "target": ["Hello."]}])
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_empty_split_is_rejected(self):
        with self.assertRaisesRegex(dataset.DatasetError, "no rows"):
            self.fetch([])
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.write_text("existing\n")
        with self.assertRaisesRegex(dataset.DatasetError, "already exists"):
            self.fetch([{"prompt": "Hello?", "target": "Hello."}])
        self.assertEqual(self.output.read_text(), "existing\n")
        self.api.dataset_info.assert_not_called()

    def test_second_publication_collision_rolls_back_only_owned_output(self):
        original_link = dataset.os.link

        def racing_link(source, destination):
            if destination == self.receipt:
                destination.write_text("concurrent file")
            return original_link(source, destination)

        with patch.object(dataset.os, "link", side_effect=racing_link):
            with self.assertRaisesRegex(dataset.DatasetError, "already exists"):
                self.fetch([{"prompt": "Hello?", "target": "Hello."}])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.receipt.read_text(), "concurrent file")
        self.assertEqual([p.name for p in Path(self.directory.name).iterdir()], ["receipt.json"])

    def test_unknown_revision_does_not_load(self):
        self.api.dataset_info.return_value = SimpleNamespace(sha=None)
        with self.assertRaisesRegex(dataset.DatasetError, "immutable revision"):
            self.fetch([{"prompt": "Hello?", "target": "Hello."}])
        self.loader.assert_not_called()

    def test_invalid_limits_and_local_paths_do_not_request_hub(self):
        for limit in (0, -1, 100001, True):
            with self.subTest(limit=limit), self.assertRaisesRegex(dataset.DatasetError, "limit"):
                self.fetch([], limit=limit)
        with self.assertRaises(dataset.DatasetError):
            dataset.fetch(repo="../private", split="train", output=self.output, receipt=self.receipt, api=self.api, loader=Mock())
        self.api.dataset_info.assert_not_called()

    def test_upstream_errors_do_not_expose_credentials(self):
        self.api.dataset_info.side_effect = RuntimeError("Authorization: Bearer secret-value")
        with self.assertRaises(dataset.DatasetError) as caught:
            self.fetch([])
        self.assertNotIn("secret-value", str(caught.exception))

    def test_iteration_failure_leaves_no_partial_files(self):
        def failing_rows():
            yield {"prompt": "Hello?", "target": "Hello."}
            raise RuntimeError("access_token=secret")

        with self.assertRaisesRegex(dataset.DatasetError, "streaming failed") as caught:
            self.fetch(failing_rows())
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock()
        self.api.dataset_info.return_value = SimpleNamespace(sha=SHA, card_data={"task_categories": ["text-generation"]})
        self.configs = Mock(return_value=["main"])
        self.splits = Mock(return_value=["train", "test"])

    def preview(self, rows=None, **kwargs):
        self.stream = Stream(rows if rows is not None else [
            {"question": f"Question {index}?", "answer": f"Answer {index}", "id": index}
            for index in range(8)
        ])
        self.loader = Mock(return_value=self.stream)
        return dataset.preview(repo="org/data", api=self.api, loader=self.loader,
            config_names=self.configs, split_names=self.splits, runtime_version="5.1.0", **kwargs)

    def test_preview_pins_every_call_and_reads_only_three_rows(self):
        result = self.preview(revision="main")
        self.api.dataset_info.assert_called_once_with(repo_id="org/data", revision="main")
        self.configs.assert_called_once_with("org/data", revision=SHA)
        self.splits.assert_called_once_with("org/data", config_name="main", revision=SHA)
        self.loader.assert_called_once_with("org/data", name="main", split="train", revision=SHA, streaming=True)
        self.assertEqual(self.stream.consumed, 3)
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual((result["prompt_field"], result["target_field"]), ("question", "answer"))
        self.assertIsNone(result["error"])
        self.assertIsNone(result["mapping_error"])
        self.assertFalse(result["truncated"])

    def test_multi_configuration_returns_choices_without_loading(self):
        self.configs.return_value = ["arithmetic", "socratic"]
        result = self.preview()
        self.assertEqual(result["configs"], ["arithmetic", "socratic"])
        self.assertIsNone(result["config"])
        self.assertIn("Choose", result["error"])
        self.splits.assert_not_called()
        self.loader.assert_not_called()

    def test_main_configuration_is_selected_when_available(self):
        self.configs.return_value = ["main", "socratic"]
        result = self.preview()
        self.assertEqual(result["config"], "main")
        self.assertEqual(len(result["rows"]), 3)
        self.assertIsNone(result["error"])

    def test_literal_default_configuration_is_selected(self):
        self.configs.return_value = ["other", "default"]
        result = self.preview()
        self.assertEqual(result["config"], "default")
        self.assertIsNone(result["error"])

    def test_bad_configuration_and_split_return_choices(self):
        result = self.preview(config="missing")
        self.assertEqual(result["configs"], ["main"])
        self.assertIn("configuration", result["error"])
        self.loader.assert_not_called()
        result = self.preview(config="main", split="missing")
        self.assertEqual(result["splits"], ["train", "test"])
        self.assertIn("split", result["error"])
        self.loader.assert_not_called()

    def test_bad_mapping_retains_raw_rows_and_columns(self):
        result = self.preview(prompt_field="wrong", target_field="answer")
        self.assertIn("wrong", result["mapping_error"])
        self.assertEqual(result["rows"][0]["question"], "Question 0?")
        self.assertIn({"name": "question", "dtype": "string"}, result["columns"])
        self.assertEqual(result["target_field"], "answer")
        self.assertIsNone(result["error"])

    def test_non_text_mapping_and_duplicate_mapping_are_rejected(self):
        result = self.preview(prompt_field="id", target_field="answer")
        self.assertIn("nonempty text", result["mapping_error"])
        result = self.preview(prompt_field="question", target_field="question")
        self.assertIn("different", result["mapping_error"])

    def test_nested_columns_and_mapping_are_visible(self):
        result = self.preview(rows=[{"item": {"question": "Why?", "answer": "Because."}}],
            prompt_field="item.question", target_field="item.answer")
        self.assertIn({"name": "item.question", "dtype": "string"}, result["columns"])
        self.assertEqual(result["rows"][0]["item"]["answer"], "Because.")
        self.assertIsNone(result["mapping_error"])

    def test_nested_array_preview_uses_identical_mapping_and_keeps_raw_rows(self):
        row = {"name": "record", "examples": [{"input": "Question one", "target": 1}, {"input": "Question two", "target": "Two"}]}
        result = self.preview(rows=[row])
        self.assertIsNone(result["mapping_error"])
        self.assertEqual(result["rows"], [row])
        self.assertIn({"name": "examples[].input", "dtype": "string"}, result["columns"])
        self.assertEqual(result["mapped_rows"], [{"row_index": 0, **pair} for pair in dataset.mapped_pairs(row, result["prompt_field"], result["target_field"])])
        self.assertEqual([pair["target"] for pair in result["mapped_rows"]], ["1", "Two"])

    def test_score_map_preview_discloses_choice_transformation(self):
        result = self.preview(rows=[{"examples": [{"input": "Which?", "target": None, "target_scores": '{"one":1,"two":1}'}]}])
        pair = result["mapped_rows"][0]
        self.assertEqual(pair["target_source"], "examples[].target_scores")
        self.assertEqual(pair["transformation"]["correct_count"], 2)
        self.assertEqual(pair["transformation"]["answer_format"], "json_label_array")
        self.assertEqual(json.loads(pair["target"]), ["A", "B"])
        self.assertIn("Choices:", pair["prompt"])

    def test_mapped_preview_is_bounded_and_reports_omitted_pairs(self):
        row = {"examples": [{"input": f"Q{i}", "target": f"A{i}"} for i in range(30)]}
        result = self.preview(rows=[row])
        self.assertEqual(len(result["mapped_rows"]), 20)
        self.assertTrue(result["mapped_rows_truncated"])
        self.assertEqual(result["mapped_rows"][-1]["example_index"], 19)
        self.assertEqual(len(row["examples"]), 30)

    def test_single_field_preview_matches_canonical_mode(self):
        result = self.preview(rows=[{"text": "A corpus passage"}], dataset_mode="corpus")
        self.assertEqual(result["dataset_mode"], "corpus")
        self.assertEqual(result["text_field"], "text")
        self.assertIsNone(result["mapping_error"])
        self.assertEqual(result["mapped_rows"][0]["prompt"], "")
        self.assertEqual(result["mapped_rows"][0]["target"], "A corpus passage")
        result = self.preview(rows=[{"input": "Question?"}], dataset_mode="prompts")
        self.assertIsNone(result["mapping_error"])
        self.assertEqual(result["mapped_rows"][0]["prompt"], "Question?")
        self.assertEqual(result["mapped_rows"][0]["target"], "")
        self.assertIsNone(result["target_field"])

    def test_long_cells_are_explicitly_truncated_without_changing_source(self):
        row = {"prompt": "x" * 5000, "target": "Good"}
        result = self.preview(rows=[row])
        self.assertEqual(len(result["rows"][0]["prompt"]), 4000)
        self.assertEqual(len(row["prompt"]), 5000)
        self.assertTrue(result["truncated"])
        self.assertIn("rows[0].prompt", result["truncated_fields"])

    def test_row_payload_has_a_total_character_budget(self):
        row = {str(index): "x" * 8000 for index in range(200)}
        row.update(prompt="Hello", target="Hi")
        result = self.preview(rows=[row])
        self.assertLess(len(json.dumps(result)), 40000)
        self.assertTrue(result["truncated"])

    def test_upstream_preview_errors_never_expose_credentials(self):
        self.configs.side_effect = RuntimeError("Authorization: Bearer secret-value")
        result = self.preview()
        self.assertIn("configurations", result["error"])
        self.assertNotIn("secret-value", json.dumps(result))
        self.loader.assert_not_called()

    def test_invalid_preview_size_does_not_call_hub(self):
        for size in (0, -1, 6, True):
            with self.subTest(size=size), self.assertRaises(dataset.DatasetError):
                self.preview(preview_rows=size)
        self.api.dataset_info.assert_not_called()

    def test_cli_preview_needs_no_output_and_prints_one_json_object(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(dataset, "preview", return_value={"rows": [], "error": None}) as preview:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = dataset.main(["--preview", "--repo", "org/data"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"rows": [], "error": None})
        preview.assert_called_once_with(repo="org/data", split="train", config=None,
            revision=None, prompt_field=None, target_field=None, preview_rows=3, dataset_mode="paired", text_field=None, hf_task=None, model_id=None)

    def test_import_still_requires_both_output_paths(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(dataset.main(["--repo", "org/data"]), 1)


if __name__ == "__main__":
    unittest.main()
