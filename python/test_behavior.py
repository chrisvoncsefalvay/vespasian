"""Offline behavioral diagnostics: observations must not become invented grades."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from behavior import analyze_sample, annotate_samples, summarize_evaluation


def row(prediction="hello", **kwargs):
    return {"split": "edit", "mode": "paired", "prediction": prediction,
            "target": "hello", "exact_match": False, "finish_reason": "stop",
            "generated_tokens": 1, **kwargs}


class BehaviorTests(unittest.TestCase):
    def signals(self, value):
        return analyze_sample(value)["signals"]

    def test_empty_and_short_traces(self):
        self.assertEqual(summarize_evaluation([])["sample_count"], 0)
        self.assertIn("empty_output", self.signals(row("")))
        self.assertNotIn("repetition", self.signals(row("yes yes yes")))
        self.assertEqual(analyze_sample({})["evidence"]["status"], "no_prediction")
        self.assertEqual(summarize_evaluation([{"split":"edit"}])["diagnosed_count"], 0)

    def test_substantial_consecutive_repetition_not_ordinary_reuse(self):
        repeated = row("I should now calculate the answer before I proceed. " * 12)
        details = analyze_sample(repeated)
        self.assertIn("repetition", details["signals"])
        evidence = details["evidence"]["repetition"]
        self.assertGreaterEqual(evidence["copies"], 3)
        self.assertGreaterEqual(evidence["span_tokens"], 24)
        self.assertIn("repetition", self.signals(row("wait " * 100)))
        long_block = " ".join(f"word{i}" for i in range(60)) + " "
        self.assertIn("repetition", self.signals(row(long_block * 4)))
        for text in ["One phrase is repeated. One phrase is repeated.",
                     "\n".join(f"The result for item {i} is {i * 2}." for i in range(100)),
                     "- " * 100, "A name appears in chapter one. A name returns later."]:
            self.assertNotIn("repetition", self.signals(row(text)), text)

    def test_length_is_evidence_not_inferred_from_size_or_empty_output(self):
        self.assertIn("length_truncation", self.signals(row("7", finish_reason="length")))
        self.assertNotIn("terminated", self.signals(row("7", finish_reason="length")))
        self.assertIn("terminated", self.signals(row("long " * 1000, finish_reason="eos")))
        self.assertNotIn("length_truncation", self.signals(row("", finish_reason=None)))
        self.assertIn("unknown_termination", self.signals(row("", finish_reason=None)))

    def test_numeric_and_json_equivalence_is_whole_output_only(self):
        for actual, expected in [("7.0", "7"), ("+7", "7"),
                                 ('{"b": 2, "a": 1}', '{"a":1,"b":2}'), ('[1.0]', '[1]')]:
            sample = row(actual, target=expected)
            self.assertIn("format_only_mismatch", self.signals(sample))
            self.assertNotIn("content_difference", self.signals(sample))
        for actual, expected in [("The answer is 7.", "7"), ("7 then 8", "7"),
                                 ('```json\n{"a":1}\n```', '{"a":1}'), ("salve!", "salve"),
                                 ('{"a":1,"a":2}', '{"a":2}'), ("NaN", "NaN")]:
            self.assertIn("unclassified_reference_mismatch", self.signals(row(actual, target=expected)))
        for actual, expected in [("8", "7"), ('{"a":2}', '{"a":1}'), ("true", "1"), ('[1,2]', '[2,1]')]:
            self.assertIn("content_difference", self.signals(row(actual, target=expected)))

    def test_unpaired_and_missing_reference_scores_have_no_content_grade(self):
        for sample in [row("7", mode="prompts"), row("7", mode="corpus"), row("7", exact_match=None), row("7", target=None)]:
            self.assertNotIn("reference_mismatch", self.signals(sample))
            self.assertNotIn("content_difference", self.signals(sample))
        self.assertNotIn("reference_mismatch", self.signals(row("7", target="7", exact_match=True)))

    def test_preservation_failure_is_not_called_a_regression(self):
        self.assertIn("preservation_mismatch", self.signals(row("no", split="preserve")))
        self.assertNotIn("preservation_mismatch", self.signals(row("no", split="validation")))
        self.assertNotIn("preservation_mismatch", self.signals(row("hello", split="preserve", exact_match=True)))
        self.assertIn("not necessarily regressions", summarize_evaluation([])["limitations"])

    def test_sealed_tests_do_not_affect_counts_ids_or_text(self):
        samples = [row("SEALED " * 100, split="test"), row("wait " * 100, split="validation", row_id="test:secret"), row("no", split="preserve")]
        before = copy.deepcopy(samples)
        annotated = annotate_samples(samples)
        self.assertEqual(samples, before)
        self.assertNotIn("behavior", annotated[0])
        self.assertEqual(annotated[1]["row_id"], "validation:0")
        summary = summarize_evaluation(annotated)
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["signals"]["repetition"]["row_ids"], ["validation:0"])
        self.assertEqual(summary["signals"]["preservation_mismatch"]["row_ids"], ["preserve:0"])
        self.assertNotIn("SEALED", json.dumps(summary))
        self.assertNotIn("test:", json.dumps(summary))

    def test_summary_is_bounded_and_ignores_untrusted_annotations(self):
        samples = [row("wait " * 32, behavior={"signals": ["invented"]}) for _ in range(1000)]
        summary = summarize_evaluation(samples)
        self.assertEqual(summary["signals"]["repetition"]["count"], 1000)
        self.assertEqual(len(summary["signals"]["repetition"]["row_ids"]), 8)
        self.assertTrue(summary["signals"]["repetition"]["row_ids_truncated"])
        self.assertLess(len(json.dumps(summary)), 6000)
        self.assertNotIn("invented", json.dumps(summary))

    def test_batched_ids_are_stable_and_supplied_ids_are_validated(self):
        annotated = annotate_samples([row("wait " * 30, split="validation")], row_ids=["validation:42"])
        self.assertEqual(annotated[0]["row_id"], "validation:42")
        self.assertEqual(summarize_evaluation(annotated)["signals"]["repetition"]["row_ids"], ["validation:42"])
        for ids in [[], ["test:42"], ["validation:SECRET"]]:
            with self.assertRaises(ValueError):
                annotate_samples([row(split="validation")], row_ids=ids)


class PublishedExcerptTests(unittest.TestCase):
    """Published fragments are isolated diagnostic inputs, never training rows."""

    @classmethod
    def setUpClass(cls):
        cls.fixture_path = Path(__file__).parent / "fixtures" / "behavior" / "p50c_fwd1_excerpts.json"
        cls.fixture = json.loads(cls.fixture_path.read_text())
        cls.episodes = {episode["problem_id"]: episode for episode in cls.fixture["episodes"]}

    @staticmethod
    def observe_excerpt(episode, part):
        # The upstream logger exposes a cutoff boolean, not an EOS reason. Do
        # not treat upstream correctness or extracted answers as our scores.
        metadata = episode["metadata"]
        return analyze_sample({"prediction": episode["excerpts"][part],
                               "mode": "prompts", "text_scope": "published_excerpt_only",
                               "generated_tokens": metadata["tokens"],
                               "finish_reason": "length" if metadata["cut"] else None})

    def test_exact_published_excerpts_keep_provenance_and_isolation(self):
        self.assertEqual(self.fixture["usage"], "offline_diagnostic_tests_only")
        source = self.fixture["source"]
        self.assertEqual(source["license"], "CC-BY-4.0")
        self.assertTrue(source["url"].endswith("/exp8_math/runs/p50c/tool_calls.jsonl#L4"))
        self.assertEqual(source["record_number"], 4)
        self.assertEqual(source["field"], "out")
        self.assertTrue(source["original_model"])
        self.assertFalse(source["held_out"])
        # Pin the verified published bytes, including logger whitespace.
        hashes = {'gsm8k_train_5410': {'start': 'a5fc31998af317fd24ad93c03a04f3e14c104d02118cdeacb33e5fddbf56af1e', 'mid': '7331edc6e864ed733b81d331f75c78ecf317c160a1d87bc31525cd881ec0bef8', 'end': '48c91c1e07f38bbc13e02bdc72e22b9049e38b0ba3774b6feb2667ac640c0a9b'}, 'gsm8k_train_2805': {'start': 'f3e06199ca1aeadb995a3bd5b87bf8514c667140f78965ecda3ab0c7d6f84ce3', 'mid': '2c05d122951e5e3e94177c4f963cbe362e31c16f15dd53acc1b4d5f9add3983e', 'end': '6ea5ab711f9db8623640b675c6a863e00407310805a76c79c9bee129494651d2'}}
        for episode in self.episodes.values():
            self.assertEqual(set(episode["excerpts"]), {"start", "mid", "end"})
            for part, excerpt in episode["excerpts"].items():
                self.assertEqual(len(excerpt), 250)
                self.assertEqual(hashlib.sha256(excerpt.encode()).hexdigest(), hashes[episode["problem_id"]][part])
        self.assertTrue(self.episodes["gsm8k_train_5410"]["excerpts"]["mid"].endswith("  "))
        self.assertTrue(self.episodes["gsm8k_train_2805"]["excerpts"]["end"].endswith(r"\boxed{23}"))
        for dataset in (Path(__file__).resolve().parent.parent / "examples").rglob("*.jsonl"):
            for problem_id in self.episodes:
                self.assertNotIn(problem_id, dataset.read_text(), str(dataset))

    def test_published_loop_is_detected_in_observed_middle_and_end(self):
        episode = self.episodes["gsm8k_train_5410"]
        self.assertEqual(episode["metadata"], {"sample_index":0,"tokens":32768,"cut":True,"correct":False,"pred":None})
        for part in ["mid", "end"]:
            observed = self.observe_excerpt(episode, part)
            self.assertIn("repetition", observed["signals"], part)
            self.assertIn("length_truncation", observed["signals"])
            self.assertNotIn("reference_mismatch", observed["signals"])
            self.assertGreaterEqual(observed["evidence"]["repetition"]["copies"], 3)
        self.assertNotIn("repetition", self.observe_excerpt(episode, "start")["signals"])

    def test_published_productive_excerpts_are_not_called_loops_or_graded(self):
        episode = self.episodes["gsm8k_train_2805"]
        self.assertEqual(episode["metadata"], {"sample_index":0,"tokens":1336,"cut":False,"correct":True,"pred":"23"})
        for part in ["start", "mid", "end"]:
            observed = self.observe_excerpt(episode, part)
            self.assertNotIn("repetition", observed["signals"], part)
            self.assertNotIn("length_truncation", observed["signals"])
            self.assertNotIn("format_only_mismatch", observed["signals"])
            self.assertNotIn("reference_mismatch", observed["signals"])
            self.assertIn("unknown_termination", observed["signals"])
            self.assertNotIn("terminated", observed["signals"])


if __name__ == "__main__":
    unittest.main()
