"""Setup choices agree with the engine and keep credentials out of summaries."""

from pathlib import Path
import re
import os
from unittest.mock import patch
import unittest

import hub
import setup_helpers as setup


class SetupTests(unittest.TestCase):
    def test_runtime_defaults_and_optional_turn_limit(self):
        for name in setup.PRESETS:
            config = setup.preset_config(name)
            self.assertEqual(config["budget"], 10)
            self.assertIsNone(config["max_turns"])
            self.assertEqual(config["reasoning_effort"], "low")
            self.assertIn("No turn cap", setup.review_html(config))
        config = {**setup.preset_config("greetings"), "editor": "openai", "editor_model": "gpt-6.1-sol", "api_key": "test"}
        self.assertEqual(setup.validate_step(config, "run"), {})
        config["max_turns"] = "   "
        self.assertEqual(setup.validate_step(config, "run"), {})
        self.assertIsNone(setup.normalize_config(config)["max_turns"])
        config["max_turns"] = 75
        self.assertEqual(setup.validate_step(config, "run"), {})
        config["reasoning_effort"] = "none"
        self.assertIn("reasoning_effort", setup.validate_step(config, "run"))
        self.assertIsNone(setup.normalize_config({"max_turns": ""})["max_turns"])

    def config(self):
        return {**setup.preset_config("greetings"), "editor": "openai", "editor_model": "gpt-6.1-sol",
                "api_key": "editor-secret", "hf_token": "hub-secret"}

    def test_inspection_defaults_review_and_shared_deployment_caps(self):
        for name in setup.PRESETS:
            config = setup.preset_config(name)
            self.assertEqual(config["thinking_mode"], "off")
            self.assertEqual(config["max_seq_length"], 1024)
            self.assertEqual(config["max_relative_norm"], 0.1)
        config = {**self.config(), "thinking_mode": "on", "max_new_tokens": 32768,
                  "max_seq_length": 131072, "max_relative_norm": 1}
        with patch.dict(os.environ, {}, clear=True):
            errors = setup.validate_step(config, "run")
            self.assertIn("max_new_tokens", errors)
            self.assertIn("max_seq_length", errors)
        with patch.dict(os.environ, {"VESPASIAN_SPACE_MAX_NEW_TOKENS": "32768", "VESPASIAN_SPACE_MAX_SEQ_LENGTH": "131072"}):
            self.assertEqual(setup.validate_step(config, "run"), {})
        review = setup.review_html(config)
        for value in ("Target thinking", ">on<", "32768", "131072", "Direction-edit relative norm cap", "truncated"):
            self.assertIn(value, review)
        for field, values in (("thinking_mode", ("none", True)),
                              ("max_relative_norm", (0, 0.0000001, 1.1, float("nan"), float("inf"), True))):
            for value in values:
                self.assertIn(field, setup.validate_step({**self.config(), field: value}, "run"))
        self.assertIn("thinking_mode", setup.validate_step({**self.config(), "thinking_mode": "on", "prompt_format": "plain"}, "run"))

    def test_tracking_discards_a_stale_numeric_limit(self):
        config = {"health_policy": "track", "health_limit_pp": 0, "health_max_drop": .2}
        normal = setup.normalise_config(config)
        self.assertIsNone(normal["health_max_drop"])
        self.assertEqual(config["health_max_drop"], .2)
        self.assertEqual(setup.normalise_config({"health_policy": "enforce", "health_limit_pp": 10})["health_max_drop"], .1)
        self.assertEqual(setup.normalise_config({"health_policy": "enforce", "health_limit_pp": 0})["health_max_drop"], 0)

    def test_legacy_explicit_limit_and_invalid_new_limit(self):
        self.assertEqual(setup.normalise_config({"health_max_drop": .15})["health_max_drop"], .15)
        for value in (None, "", -1, 101, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                setup.normalise_config({"health_policy": "enforce", "health_limit_pp": value})

    def test_unpaired_modes_only_offer_goal_scoring(self):
        for mode in ("corpus", "prompts"):
            self.assertEqual(setup.metric_options(mode), [("Goal score", "semantic")])
            self.assertEqual(setup.metric_for_mode(mode, "exact_match"), "semantic")
            config = {**self.config(), "dataset_mode": mode, "metric": "exact_match"}
            self.assertIn("metric", setup.validate_step(config, "behaviour"))
        self.assertEqual(setup.metric_for_mode("paired", "token_f1"), "token_f1")

    def test_data_requires_matching_successful_previews(self):
        config = self.config()
        preview = hub.preview_upload(config["dataset_file"], config)
        self.assertIn("dataset_file", setup.validate_step(config, "data", preview))
        preview["_current"] = True
        self.assertEqual(setup.validate_step(config, "data", preview), {})
        config.update(test_dataset_source="alice/test", test_dataset_split="test")
        self.assertIn("test_dataset_source", setup.validate_step(config, "data", preview))
        self.assertEqual(setup.validate_step(config, "data", preview, {"_current": True, "mapped_rows": [{"prompt": "x"}]}), {})
        preview["mapping_error"] = "Choose nested fields in the shared column"
        self.assertIn("prompt_field", setup.validate_step(config, "data", preview))

    def test_fingerprint_tracks_only_relevant_selection(self):
        config = self.config()
        original = setup.fingerprint(config)
        config.update(api_key="another-key", hf_token="another-token", budget=7, goal="Changed objective")
        self.assertEqual(setup.fingerprint(config), original)
        config["prompt_field"] = "messages[0].content"
        self.assertNotEqual(setup.fingerprint(config), original)
        test_fingerprint = setup.fingerprint(config, test=True)
        config["dataset_source"] = "alice/another-primary"
        self.assertEqual(setup.fingerprint(config, test=True), test_fingerprint)
        config["test_dataset_split"] = "validation"
        self.assertNotEqual(setup.fingerprint(config, test=True), test_fingerprint)

    def test_fingerprint_survives_gradio_blank_and_number_roundtrip(self):
        config = setup.preset_config("greetings")
        expected = setup.fingerprint(config)
        roundtrip = {key: None if value == "" else value for key, value in config.items()}
        roundtrip["example_limit"] = float(config["example_limit"])
        self.assertEqual(setup.fingerprint(roundtrip), expected)

    def test_all_local_presets_are_usable_and_have_exact_counts(self):
        for name in ("greetings", "style", "style-prompts"):
            with self.subTest(name=name):
                config = setup.preset_config(name)
                self.assertTrue(Path(config["dataset_file"]).is_file())
                preview = hub.preview_upload(config["dataset_file"], config)
                self.assertIsNone(preview["mapping_error"])
                self.assertEqual(preview["total_rows"], 8)
                self.assertTrue(preview["mapped_rows"])
                self.assertEqual(setup.validate_step(config, "behaviour"), {})
                counts = setup.split_summary(config, preview)
                self.assertTrue(counts["exact"])
                self.assertEqual(counts["total"], 8)
                self.assertGreater(counts["edit"], 0)
                self.assertGreater(counts["validation"], 0)
        config = setup.preset_config("greetings")
        config["health_checks"].clear()
        self.assertEqual(setup.preset_config("greetings")["health_checks"], ["copying"])

    def test_four_quick_starters_load_complete_bundled_datasets(self):
        import json
        import hashlib
        manifest = json.loads(setup.STARTER_MANIFEST.read_text())
        self.assertEqual(len(setup.quick_starter_cards()), 4)
        self.assertEqual({card["id"] for card in setup.quick_starter_cards()}, {item["id"] for item in manifest["starters"]})
        for item in manifest["starters"]:
            with self.subTest(starter=item["id"]):
                config = setup.preset_config(item["id"])
                path = Path(config["dataset_file"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
                preview = hub.preview_upload(str(path), config)
                self.assertIsNone(preview["mapping_error"])
                self.assertIsNone(preview["error"])
                self.assertEqual(preview["counts"], item["counts"])
                self.assertEqual(preview["total_rows"], sum(item["counts"].values()))
                self.assertEqual(config["example_limit"], preview["total_rows"])
                self.assertTrue(all(preview["counts"][split] > 0 for split in ("edit", "validation", "test", "preserve")))
                self.assertFalse(config["patch_backup"])
                self.assertFalse(config["patch_create"])
                self.assertEqual(config["budget"], 10)
                self.assertIsNone(config["max_turns"])
                self.assertEqual(config["reasoning_effort"], "low")
                self.assertEqual(setup.validate_step(config, "behaviour"), {})
                self.assertIn("dataset_file", setup.validate_step(config, "data", preview))
        cards = setup.quick_starter_cards()
        cards[0]["title"] = "Changed copy"
        self.assertNotEqual(cards[0]["title"], setup.quick_starter_cards()[0]["title"])

    def test_hub_split_summary_is_labelled_as_estimate(self):
        counts = setup.split_summary(setup.preset_config("gsm8k"))
        self.assertEqual((counts["edit"], counts["validation"], counts["test"]), (13, 3, 0))
        self.assertFalse(counts["exact"])
        self.assertEqual(counts["label"], "Planned maximum examples")

    def test_review_escapes_values_without_rendering_credentials(self):
        config = self.config()
        config.update(goal='<script>alert("goal")</script>', controls="keep things", health_policy="enforce", health_limit_pp=10)
        rendered = setup.review_html(config, hub.preview_upload(config["dataset_file"], config))
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script", rendered)
        self.assertNotIn("editor-secret", rendered)
        self.assertNotIn("hub-secret", rendered)
        self.assertIn("10 percentage points", rendered)
        self.assertIn("4 edit · 2 validation · 1 test · 1 preserve", rendered)
        config.update(health_limit_pp="", input_price="invalid")
        self.assertIn("Choose a valid preservation limit", setup.review_html(config))

    def test_strict_space_rejects_custom_models_and_price_overrides(self):
        config = self.config()
        config["editor_model"] = "custom-editor"
        errors = setup.validate_step(config, "run")
        self.assertEqual(set(errors), {"editor_model"})
        self.assertIn("--budget-policy auto", errors["editor_model"])
        config.update(input_price="1.25", output_price="6")
        self.assertEqual(set(setup.validate_step(config, "run")), {"editor_model", "input_price", "output_price"})
        # Normalisation may preserve imported prices, but validation must reject them.
        normal = setup.normalise_config(config)
        self.assertEqual((normal["input_price"], normal["output_price"]), (1.25, 6.))
        self.assertIn("input_price", setup.validate_step(normal, "editor"))
        for bad in ("no", "nan", "inf", 0, -1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                setup.normalise_config({"input_price": bad})
        config.update(editor_model="gpt-6.1-sol", input_price=" ", output_price="")
        self.assertEqual(setup.validate_step(config, "editor"), {})
        for value in (2, "2", 0, False, "no", float("nan")):
            for field in ("input_price", "output_price"):
                self.assertIn(field, setup.validate_step({**config, field: value}, "editor"))
        for provider, models in setup.EDITOR_RATES.items():
            for model in models:
                self.assertEqual(setup.validate_step({**config, "editor": provider, "editor_model": model}, "run"), {})
        self.assertIn("editor_model", setup.validate_step({**config, "editor": "anthropic", "editor_model": "gpt-6.1-sol"}, "run"))

    def test_budget_rates_match_controller_and_provider_choices(self):
        source = ROOT / "src/budget.rs"
        if not source.is_file():
            self.skipTest("Controller source is not in this bundle")
        rates = {model: (float(incoming), float(outgoing)) for model, incoming, outgoing in
                 re.findall(r'"([^"]+)" => \(([\d.]+), ([\d.]+)(?:, [\d.]+)?\)', source.read_text())}
        for provider, models in setup.EDITOR_RATES.items():
            self.assertEqual([value for _, value in setup.editor_choices(provider)], list(models))
            for model, expected in models.items():
                self.assertEqual(expected, rates[model])

    def test_provider_switch_remembers_custom_model(self):
        model, memory = setup.choose_editor("anthropic", "my-custom-openai", {"_provider": "openai"})
        self.assertEqual(model, "claude-sonnet-5-5")
        model, memory = setup.choose_editor("openai", "claude-opus-5-5", memory)
        self.assertEqual(model, "my-custom-openai")
        self.assertEqual(memory["anthropic"], "claude-opus-5-5")

    def test_suggestions_use_the_data_shape_and_selected_fields(self):
        self.assertIn("examples[].answer", setup.suggested_objective("factual", "paired", "examples[].answer"))
        self.assertIn("documents[].text", setup.suggested_objective("style", "corpus", text_field="documents[].text"))
        self.assertNotIn("reference", setup.suggested_objective("custom", "prompts").lower())

    def test_task_choice_and_patch_preferences_are_explicit(self):
        config = self.config()
        self.assertEqual(config["hf_task"], "text-generation")
        self.assertFalse(config["patch_backup"])
        self.assertEqual(config["save_format"], "model")
        self.assertIn("hf_task", setup.validate_step({**config, "hf_task": None}, "data"))
        self.assertNotEqual(setup.fingerprint(config), setup.fingerprint({**config, "hf_task": "translation"}))
        self.assertIn("patch_repo", setup.validate_step({**config, "patch_backup": True}, "editor"))
        self.assertEqual(setup.validate_step({**config, "patch_backup": True, "patch_repo": "visitor/patches"}, "editor"), {})
        self.assertIn("patch_repo", setup.validate_step({**config, "patch_backup": "true", "patch_repo": "visitor/patches"}, "editor"))
        self.assertIn("Patches only", setup.review_html({**config, "save_format": "patch"}))
        self.assertNotIn("<script>", setup.task_help({"warning": "<script>unsafe</script>"}))


ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    unittest.main()
