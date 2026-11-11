"""Behaviour checks using a small, real randomly initialised transformer."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from worker import Engine, load_model_components, model_structure, normalise, token_f1


class CharacterTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(character) % 96 for character in text]

    def apply_chat_template(self, messages, **kwargs):
        return [1] + self.encode(messages[0]["content"] + ":")

    def decode(self, tokens, skip_special_tokens=True, **kwargs):
        return "".join(chr(32 + (token - 3) % 96) for token in tokens if token >= 3)

    def save_pretrained(self, directory):
        (Path(directory) / "tokenizer_config.json").write_text('{"test_tokenizer": true}')


def engine():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=100, hidden_size=24, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0))
    model.requires_grad_(False)
    result = Engine()
    result.model = model.eval()
    result.tokenizer = CharacterTokenizer()
    result.editable = {name: parameter for name, parameter in model.named_parameters() if name.endswith(".mlp.down_proj.weight")}
    result.dataset = [
        {"prompt": "alpha", "target": "a", "split": "edit"},
        {"prompt": "beta", "target": "b", "split": "validation"},
        {"prompt": "gamma", "target": "c", "split": "preserve"},
        {"prompt": "delta", "target": "d", "split": "test"},
    ]
    return result


def proposal(instance, **changes):
    value = {"targets": [next(iter(instance.editable))], "operation": "low_rank", "strength": 0.05, "rank": 2, "seed": 31, "rationale": "Test a residual direction"}
    value.update(changes)
    return {"proposal": value}


class WorkerTests(unittest.TestCase):
    def test_component_loader_pins_revision_and_disables_remote_code(self):
        from unittest.mock import MagicMock
        tokenizer = CharacterTokenizer()
        for architecture in ("qwen3_5", "gemma3_text"):
            model = MagicMock()
            model.to.return_value = model
            processor = SimpleNamespace(tokenizer=tokenizer)
            with self.subTest(architecture=architecture), \
                 patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(model_type=architecture)) as config_loader, \
                 patch("transformers.AutoProcessor.from_pretrained", return_value=processor) as processor_loader, \
                 patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as tokenizer_loader, \
                 patch("transformers.Qwen3_5ForConditionalGeneration.from_pretrained", return_value=model) as qwen_loader, \
                 patch("transformers.Gemma3ForCausalLM.from_pretrained", return_value=model) as gemma_loader:
                loaded, actual_tokenizer, actual_processor = load_model_components("test/model", revision="pinned", device="cpu", dtype=torch.float32)
                self.assertIs(loaded, model)
                self.assertIs(actual_tokenizer, tokenizer)
                self.assertIs(actual_processor, processor if architecture == "qwen3_5" else None)
                calls = [config_loader, processor_loader if architecture == "qwen3_5" else tokenizer_loader, qwen_loader if architecture == "qwen3_5" else gemma_loader]
                for loader in calls:
                    self.assertFalse(loader.call_args.kwargs["trust_remote_code"])
                    self.assertEqual(loader.call_args.kwargs["revision"], "pinned")
                self.assertEqual(calls[-1].call_args.kwargs["attn_implementation"], "eager")
                model.requires_grad_.assert_called_with(False)

    def test_injected_loader_preserves_initialisation_and_provenance(self):
        from unittest.mock import Mock
        fixture = engine()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            fixture.model.save_pretrained(source)
            fixture.tokenizer.save_pretrained(source)
            dataset = Path(directory) / "dataset.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in fixture.dataset))
            loader = Mock(return_value=(fixture.model, fixture.tokenizer, None))
            events = []
            instance = Engine(emit=events.append, model_loader=loader)
            result = instance.initialise({"model": str(source), "dataset": str(dataset), "device": "cpu", "seed": 19})
            loader.assert_called_once_with(str(source), revision=None, device="cpu", dtype=torch.float32)
            self.assertEqual(result["counts"], {"edit": 1, "validation": 1, "preserve": 1, "test": 1})
            self.assertEqual(result["source"]["files"], Engine.manifest(source))
            self.assertEqual(result["source"]["path"], str(source.resolve()))
            self.assertEqual(result["architecture"], "llama")
            self.assertEqual(instance.seed, 19)
            self.assertEqual(len(instance.encoded), len(fixture.dataset))
            self.assertEqual(set(instance.editable), set(fixture.editable))
            self.assertFalse(instance.model.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in instance.model.parameters()))
            self.assertEqual(events[-1]["message"], "Model ready")

    def unpaired(self, mode):
        instance = engine()
        instance.dataset_mode = mode
        instance.dataset = [dict(row, mode=mode, **({"prompt": "", "target": row["prompt"]} if mode == "corpus" else {"target": ""})) for row in instance.dataset]
        return instance

    def test_corpus_uses_start_token_and_separate_reference_encodings(self):
        instance = self.unpaired("corpus")
        with patch.object(instance.tokenizer, "apply_chat_template", side_effect=AssertionError("Corpus has no chat prompt")):
            first, second = [instance.encode(row) for row in instance.dataset[:2]]
            self.assertEqual(first[0], [1])
            self.assertEqual(second[0], [1])
            self.assertNotEqual(first[1], second[1])
            self.assertEqual(first[1][-1], instance.tokenizer.eos_token_id)
            with patch.object(instance.model, "generate", wraps=instance.model.generate) as generate:
                result = instance.evaluate({"splits": ["validation"], "max_new_tokens": 8})
            self.assertEqual(generate.call_args.kwargs["input_ids"].tolist(), [[1]])
            self.assertTrue(generate.call_args.kwargs["do_sample"])
            self.assertEqual(set(result["metrics"]["validation"]), {"count", "loss"})
            self.assertTrue(math.isfinite(result["metrics"]["validation"]["loss"]))
            self.assertEqual(instance.generation_policy()["conditioning"], "unconditional")

    def test_corpus_sampling_is_repeatable_order_independent_and_restores_rng(self):
        instance = self.unpaired("corpus")
        request = {"splits": ["edit", "validation", "preserve"], "max_new_tokens": 12}
        rng = torch.random.get_rng_state().clone()
        first = instance.evaluate(request)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        instance.dataset.reverse()
        second = instance.evaluate(request)
        keyed = lambda result: {row["target"]: row for row in result["samples"]}
        self.assertEqual(keyed(first), keyed(second))
        self.assertEqual(len({row["generation_seed"] for row in first["samples"]}), 3)
        self.assertEqual(keyed(first), keyed(self.unpaired("corpus").evaluate(request)))
        old_seed = instance.generation_seed(instance.dataset[0])
        instance.seed += 1
        self.assertNotEqual(old_seed, instance.generation_seed(instance.dataset[0]))

    def test_unconditional_start_token_falls_back_to_eos(self):
        instance = self.unpaired("corpus")
        instance.model.config.bos_token_id = None
        self.assertEqual(instance.start_token(), 2)
        instance.tokenizer.bos_token_id = 99
        self.assertEqual(instance.start_token(), 99)
        instance.tokenizer.bos_token_id = 999
        self.assertEqual(instance.start_token(), 2)
        instance.tokenizer.eos_token_id = None
        with self.assertRaisesRegex(ValueError, "BOS token"):
            instance.start_token()

    def test_corpus_answer_ids_do_not_collide_for_empty_prompts(self):
        instance = self.unpaired("corpus")
        for row in instance.dataset:
            row["split"] = "validation"
        events = []
        instance.emit = events.append
        instance.evaluate({"splits": ["validation"], "max_new_tokens": 4})
        examples = [event["example"] for event in events if event["type"] == "answer"]
        self.assertEqual(len({row["id"] for row in examples}), 4)
        self.assertTrue(all(row["prompt"] == "" and row["mode"] == "corpus" for row in examples))

    def test_prompt_only_generation_never_measures_nonexistent_reference(self):
        instance = self.unpaired("prompts")
        with patch.object(instance, "loss", side_effect=AssertionError("No reference exists")):
            result = instance.evaluate({"splits": ["validation", "preserve"], "max_new_tokens": 4})
        self.assertEqual(result["metrics"], {"validation": {"count": 1}, "preserve": {"count": 1}})
        for row in result["samples"]:
            self.assertEqual(row["target"], "")
            for missing in ("loss", "exact_match", "token_f1", "generation_seed"):
                self.assertNotIn(missing, row)
        self.assertNotIn("activation_rank_one", instance.allowed_operations())
        original = {name: value.clone() for name, value in instance.editable.items()}
        with self.assertRaisesRegex(ValueError, "requires reference"):
            instance.edit(proposal(instance, operation="activation_rank_one"))
        self.assertFalse(instance.pending)
        self.assertTrue(all(torch.equal(original[name], value) for name, value in instance.editable.items()))
        instance.edit(proposal(instance, operation="scale"))
        instance.rollback({})
        self.assertTrue(all(torch.equal(original[name], value) for name, value in instance.editable.items()))

    def test_corpus_activation_edit_is_forward_only_and_rollback_restores_samples(self):
        instance = self.unpaired("corpus")
        request = {"splits": ["validation"], "max_new_tokens": 8}
        baseline = instance.evaluate(request)
        with patch("torch.autograd.backward", side_effect=AssertionError("Backpropagation is forbidden")):
            result = instance.edit(proposal(instance, operation="activation_rank_one", rank=1))
        self.assertGreater(result["changed_tensors"][0]["changed_elements"], 0)
        instance.rollback({})
        restored = instance.evaluate(request)
        self.assertEqual(baseline["samples"], restored["samples"])
        self.assertEqual(baseline["metrics"], restored["metrics"])

    def test_health_checks_keep_conditional_generation_in_unpaired_modes(self):
        from health import prepare, parse_selection
        reference = engine()
        reference.health_set = prepare([parse_selection("copying:1")], 7)
        expected = reference.health({"max_new_tokens": 4})
        for mode in ("corpus", "prompts"):
            instance = self.unpaired(mode)
            instance.health_set = reference.health_set
            with patch.object(instance.model, "generate", wraps=instance.model.generate) as generate:
                actual = instance.health({"max_new_tokens": 4})
            self.assertFalse(generate.call_args.kwargs["do_sample"])
            self.assertGreater(generate.call_args.kwargs["input_ids"].shape[-1], 1)
            self.assertEqual(actual["metrics"], expected["metrics"])
            self.assertEqual(actual["samples"], expected["samples"])

    def test_loader_requires_mode_consistency_and_unique_corpus_references(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            corpus, prompts = self.unpaired("corpus").dataset, self.unpaired("prompts").dataset
            for rows, mode in ((corpus, "corpus"), (prompts, "prompts")):
                path.write_text("\n".join(json.dumps(row) for row in rows))
                instance = engine()
                self.assertEqual(instance.load_dataset(str(path))["validation"], 1)
                self.assertEqual(instance.dataset_mode, mode)
            for rows in ([corpus[0], {**corpus[1], "target": " ALPHA "}], [corpus[0], prompts[1]],
                         [corpus[0], {**corpus[1], "prompt": "not unconditional"}], [prompts[0], {**prompts[1], "target": "not prompt-only"}]):
                path.write_text("\n".join(json.dumps(row) for row in rows))
                with self.assertRaises(ValueError):
                    engine().load_dataset(str(path))

    def test_structure_uses_the_actual_text_config_layer_count(self):
        instance = engine()
        self.assertEqual(model_structure(instance.model.config), {"architecture":"llama", "num_hidden_layers":2})
        nested = SimpleNamespace(model_type="qwen3_5", num_hidden_layers=999, text_config=SimpleNamespace(model_type="qwen3_5_text", num_hidden_layers=24))
        self.assertEqual(model_structure(nested), {"architecture":"qwen3_5_text", "num_hidden_layers":24})
        self.assertEqual(model_structure(SimpleNamespace(model_type="gemma3_text", num_hidden_layers=26)), {"architecture":"gemma3_text", "num_hidden_layers":26})
        for layers in (None, True, 0, -1, 2.5):
            with self.subTest(layers=layers), self.assertRaisesRegex(ValueError, "layer count"):
                model_structure(SimpleNamespace(model_type="gemma3_text", num_hidden_layers=layers))

    def test_tensor_events_start_before_changes_and_apply_after_measured_bounds(self):
        instance = engine()
        names = list(instance.editable)[:2]
        original = {name: instance.editable[name].detach().clone() for name in names}
        events = []

        def observe(event):
            events.append(event)
            if event["type"] != "tensor_edit":
                return
            tensor = instance.editable[event["tensor"]]
            if event["status"] == "started":
                self.assertTrue(torch.equal(tensor, original[event["tensor"]]))
                self.assertNotIn("relative_norm", event)
            else:
                self.assertTrue(torch.isfinite(tensor).all())
                self.assertGreater(event["changed_elements"], 0)
                self.assertLessEqual(event["relative_norm"], 0.03)

        instance.emit = observe
        result = instance.edit(proposal(instance, targets=names, operation="scale", strength=0.03))
        tensor_events = [event for event in events if event["type"] == "tensor_edit"]
        self.assertEqual([event["status"] for event in tensor_events], ["started", "applied", "started", "applied"])
        self.assertEqual([event["tensor"] for event in tensor_events], [names[0], names[0], names[1], names[1]])
        self.assertEqual([event["current"] for event in tensor_events], [0, 1, 1, 2])
        self.assertTrue(all(event["total"] == 2 and event["operation"] == "scale" for event in tensor_events))
        self.assertEqual([event["current"] for event in events if event["type"] == "progress"], [1, 2])
        for event, detail in zip(tensor_events[1::2], result["changed_tensors"]):
            for key in ("relative_norm", "changed_elements", "shape"):
                self.assertEqual(event[key], detail[key])
        self.assertFalse(instance.changed)
        instance.rollback({})
        for name in names:
            self.assertTrue(torch.equal(instance.editable[name], original[name]))

    def test_failed_surgery_never_emits_applied_for_failed_tensor_or_retained(self):
        instance = engine()
        names = list(instance.editable)[:2]
        with torch.no_grad():
            instance.editable[names[1]].zero_()
        original = {name: instance.editable[name].detach().clone() for name in names}
        events = []
        instance.emit = events.append
        with self.assertRaisesRegex(ValueError, "nonzero norms"):
            instance.edit(proposal(instance, targets=names, operation="scale", strength=0.03))
        tensor_events = [event for event in events if event["type"] == "tensor_edit"]
        self.assertEqual([event["status"] for event in tensor_events], ["started", "applied", "started"])
        self.assertEqual(tensor_events[-1]["tensor"], names[1])
        self.assertFalse(any(event.get("retained") or event.get("accepted") for event in events))
        self.assertFalse(instance.changed)
        self.assertFalse(instance.pending)
        for name in names:
            self.assertTrue(torch.equal(instance.editable[name], original[name]))

    def test_actual_edit_updates_only_named_tensor_and_rolls_back_exactly(self):
        instance = engine()
        before = {name: parameter.detach().clone() for name, parameter in instance.model.named_parameters()}
        with patch("torch.autograd.backward", side_effect=AssertionError("Backpropagation is forbidden")):
            report = instance.edit(proposal(instance))
        name = report["changed_tensors"][0]["name"]
        self.assertGreater(report["changed_tensors"][0]["changed_elements"], 0)
        self.assertLessEqual(report["changed_tensors"][0]["relative_norm"], 0.050001)
        for key, parameter in instance.model.named_parameters():
            self.assertFalse(parameter.requires_grad)
            if key != name:
                self.assertTrue(torch.equal(parameter, before[key]), key)
        instance.rollback({})
        for key, parameter in instance.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[key]), key)

    def test_non_edit_targets_do_not_change_activation_surgery(self):
        left, right = engine(), engine()
        for row in right.dataset:
            if row["split"] != "edit":
                row["target"] = "different answer"
        # Include every fixture row if the edit-only filter regresses.
        left.edit(proposal(left, operation="activation_rank_one", rank=4))
        right.edit(proposal(right, operation="activation_rank_one", rank=4))
        for lhs, rhs in zip(left.model.parameters(), right.model.parameters()):
            self.assertTrue(torch.equal(lhs, rhs))

    def test_rejected_proposal_and_nonfinite_delta_preserve_weights(self):
        instance = engine()
        before = copy.deepcopy(instance.model.state_dict())
        with self.assertRaises(ValueError):
            instance.edit(proposal(instance, targets=["lm_head.weight"]))
        with patch("torch.randn", side_effect=lambda shape, **kwargs: torch.full(shape, float("inf"))):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                instance.edit(proposal(instance))
        self.assertFalse(instance.pending)
        for key, tensor in instance.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[key]), key)

    def test_pending_edit_cannot_be_overwritten_or_saved(self):
        instance = engine()
        instance.edit(proposal(instance))
        with self.assertRaisesRegex(ValueError, "previous edit"):
            instance.edit(proposal(instance))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "current edit"):
                instance.save({"path": str(Path(directory) / "checkpoint")})

    def test_accepted_checkpoint_serialises_all_parameters_and_reloads(self):
        instance = engine()
        instance.edit(proposal(instance))
        instance.accept({})
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "checkpoint"
            saved = instance.save({"path": str(destination)})
            self.assertTrue(any(file["path"].endswith(".safetensors") for file in saved["files"]))
            self.assertEqual(len(saved["changed_tensors"]), 1)
            loaded = LlamaForCausalLM.from_pretrained(destination)
            for name, value in loaded.state_dict().items():
                self.assertTrue(torch.equal(value, instance.model.state_dict()[name]), name)
            with self.assertRaisesRegex(ValueError, "already exists"):
                instance.save({"path": str(destination)})

    def test_evaluation_returns_only_requested_splits(self):
        instance = engine()
        result = instance.evaluate({"splits": ["validation", "preserve"], "max_new_tokens": 4})
        self.assertEqual(set(result["metrics"]), {"validation", "preserve"})
        self.assertEqual(len(result["samples"]), 2)
        for metric in result["metrics"].values():
            self.assertGreater(metric["loss"], 0)
            self.assertTrue(0 <= metric["token_f1"] <= 1)
            self.assertEqual(metric["count"], 1)

    def test_objective_answers_stream_before_next_row_with_completed_progress(self):
        instance = engine()
        events = []
        instance.emit = events.append
        counts_before_generate = []
        generate = instance.model.generate

        def observed_generate(*args, **kwargs):
            counts_before_generate.append(sum(event["type"] == "answer" for event in events))
            return generate(*args, **kwargs)

        request = {"splits": ["validation", "preserve"], "max_new_tokens": 4, "label": "turn-003", "turn": 3}
        with patch.object(instance.model, "generate", side_effect=observed_generate):
            result = instance.evaluate(request)
        self.assertEqual(counts_before_generate, [0, 1])
        self.assertEqual([event["type"] for event in events], ["progress", "answer", "progress", "answer", "progress", "progress"])
        self.assertEqual([event["current"] for event in events], [0, 1, 1, 2, 2, 2])
        for event in events:
            self.assertEqual(event["phase"], "evaluating")
            self.assertEqual(event["label"], "turn-003")
            self.assertEqual(event["turn"], 3)
            self.assertEqual(event["total"], 2)
        self.assertEqual(events[-1]["message"], "Evaluation complete")
        answers = [event for event in events if event["type"] == "answer"]
        for answer, sample in zip(answers, result["samples"]):
            self.assertEqual({key: answer["example"][key] for key in sample}, sample)
            self.assertEqual(answer["example"]["output"], sample["prediction"])
            self.assertTrue(answer["example"]["id"].startswith(sample["split"] + ":"))
        answers[0]["example"]["token_ids"].append(999)
        self.assertNotIn(999, result["samples"][0]["token_ids"])
        without_observer = engine().evaluate(request)
        self.assertEqual(result["metrics"], without_observer["metrics"])
        self.assertEqual(result["samples"], without_observer["samples"])

    def test_objective_failure_never_reports_an_unscored_answer_or_completion(self):
        instance = engine()
        events = []
        instance.emit = events.append
        loss = instance.loss
        calls = 0

        def invalid_second_loss(row):
            nonlocal calls
            calls += 1
            return loss(row) if calls == 1 else torch.tensor(float("nan"))

        with patch.object(instance, "loss", side_effect=invalid_second_loss):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                instance.evaluate({"splits": ["validation", "preserve"], "max_new_tokens": 4, "label": "baseline"})
        self.assertEqual([event["type"] for event in events], ["progress", "answer", "progress"])
        self.assertEqual([event["current"] for event in events], [0, 1, 1])
        self.assertTrue(all("turn" not in event for event in events))
        self.assertTrue(all(event.get("message") != "Evaluation complete" for event in events))

    def test_display_events_bound_text_without_mutating_the_full_sample(self):
        instance = engine()
        events = []
        instance.emit = events.append
        text = "A" * 20000
        sample = {"prompt": text, "target": text, "prediction": text, "split": "validation", "token_ids": [3, 2], "finish_reason": "stop"}
        original = copy.deepcopy(sample)
        instance.answer("evaluating", "baseline", None, 1, 1, sample)
        self.assertEqual(sample, original)
        event = events[0]
        self.assertEqual(set(event["truncated_fields"]), {"prompt", "target", "prediction", "output"})
        self.assertEqual(len(event["example"]["output"]), 16384)
        self.assertEqual(event["example"]["token_ids"], [3, 2])

    def test_oversized_examples_and_duplicate_prompts_are_rejected(self):
        instance = engine()
        instance.max_seq_length = 4
        with self.assertRaisesRegex(ValueError, "exceeds"):
            instance.encode(instance.dataset[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            rows = [instance.dataset[0], {**instance.dataset[1], "prompt": " ALPHA "}]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "repeated prompt"):
                instance.load_dataset(str(path))

    def test_bfloat16_projection_respects_actual_storage_norm(self):
        instance = engine()
        instance.model.to(dtype=torch.bfloat16)
        report = instance.edit(proposal(instance, strength=0.001))
        self.assertLessEqual(report["changed_tensors"][0]["relative_norm"], 0.001001)

    def test_surgeries_are_seeded_and_distinct(self):
        for operation in ("low_rank", "noise"):
            first, second, third = engine(), engine(), engine()
            first.edit(proposal(first, operation=operation, seed=7))
            second.edit(proposal(second, operation=operation, seed=7))
            third.edit(proposal(third, operation=operation, seed=8))
            name = next(iter(first.editable))
            self.assertTrue(torch.equal(first.editable[name], second.editable[name]))
            self.assertFalse(torch.equal(first.editable[name], third.editable[name]))

    def test_scale_is_direct_multiplication_within_norm_bound(self):
        instance = engine()
        name = next(iter(instance.editable))
        original = instance.editable[name].detach().clone()
        report = instance.edit(proposal(instance, operation="scale", strength=-0.03))
        self.assertLessEqual(report["changed_tensors"][0]["relative_norm"], 0.030001)
        self.assertTrue(torch.allclose(instance.editable[name], original * 0.97, atol=1e-5))

    def test_activation_surgery_uses_edit_targets_without_backpropagation(self):
        first, second = engine(), engine()
        first.dataset[0]["target"] = "ax"
        second.dataset[0]["target"] = "az"
        with patch("torch.autograd.backward", side_effect=AssertionError("Backpropagation is forbidden")):
            report = first.edit(proposal(first, operation="activation_rank_one", rank=1))
            second.edit(proposal(second, operation="activation_rank_one", rank=1))
        name = next(iter(first.editable))
        self.assertFalse(torch.equal(first.editable[name], second.editable[name]))
        self.assertLessEqual(report["changed_tensors"][0]["relative_norm"], 0.050001)
        self.assertFalse(first.model.get_submodule(name.removesuffix(".weight"))._forward_hooks)

    def test_gemma_reference_uses_assistant_turn_terminator(self):
        instance = engine()
        instance.model.config.model_type = "gemma3_text"
        instance.tokenizer.convert_tokens_to_ids = lambda token: 106 if token == "<end_of_turn>" else None
        _, target = instance.encode(instance.dataset[0])
        self.assertEqual(target[-1], 106)
        self.assertNotEqual(target[-1], instance.tokenizer.eos_token_id)

    def test_protocol_recovers_after_invalid_request(self):
        process = subprocess.run([sys.executable, str(Path(__file__).with_name("worker.py"))], input='[]\n{"id":2,"command":"shutdown"}\n', text=True, capture_output=True, timeout=10, check=True)
        lines = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertFalse(lines[0]["ok"])
        self.assertTrue(lines[1]["ok"])
        self.assertEqual(lines[1]["id"], 2)

    def test_metrics_have_explicit_normalisation(self):
        self.assertEqual(normalise(" A  b\n"), "a b")
        self.assertEqual(token_f1("a a b", "a b b"), 2 / 3)
        self.assertEqual(token_f1("", "x"), 0)

    def test_health_generation_is_separate_from_edit_examples_and_keeps_weights(self):
        from health import prepare, parse_selection
        instance = engine()
        instance.health_set = prepare([parse_selection("copying:1"), parse_selection("arithmetic:1"), parse_selection("json:1")], 42)
        original_data = copy.deepcopy(instance.dataset)
        original_weights = copy.deepcopy(instance.model.state_dict())
        with patch("torch.autograd.backward", side_effect=AssertionError("Backpropagation is forbidden")):
            report = instance.dispatch({"command": "health", "max_new_tokens": 4})
        self.assertEqual(set(report["metrics"]), {"copying", "arithmetic", "json"})
        self.assertEqual(len(report["samples"]), 3)
        self.assertEqual(instance.dataset, original_data)
        self.assertFalse(instance.encoded)
        for name, value in instance.model.state_dict().items():
            self.assertTrue(torch.equal(value, original_weights[name]))
        self.assertTrue(all(not parameter.requires_grad for parameter in instance.model.parameters()))

    def test_health_answers_stream_scoring_and_choice_probabilities(self):
        from health import prepare, parse_selection
        instance = engine()
        instance.health_set = prepare([parse_selection("arithmetic:1")], 42)
        instance.health_set["checks"].append({"check":"mmlu:anatomy", "label":"MMLU · anatomy", "metric":"choice_logprob_accuracy"})
        instance.health_set["rows"].append({"id":"mmlu:anatomy:test:0", "check":"mmlu:anatomy", "prompt":"Q?\nA. x\nB. y\nC. z\nD. w\nAnswer:", "target":"B", "choices":list("ABCD")})
        events = []
        instance.emit = events.append
        result = instance.health({"max_new_tokens": 4, "label": "Health · turn 2", "turn": 2})
        self.assertEqual([event["type"] for event in events], ["progress", "answer", "progress", "answer", "progress", "progress"])
        self.assertEqual([event["current"] for event in events], [0, 1, 1, 2, 2, 2])
        self.assertTrue(all(event["phase"] == "health" and event["turn"] == 2 and event["label"] == "Health · turn 2" for event in events))
        answers = [event["example"] for event in events if event["type"] == "answer"]
        self.assertEqual(answers[0]["scoring"], "numeric_exact")
        for answer, sample in zip(answers, result["samples"]):
            self.assertEqual({key: answer[key] for key in sample}, sample)
            self.assertEqual(answer["output"], sample["prediction"])
        choice = answers[1]
        self.assertEqual(choice["scoring"], "choice_logprob_accuracy")
        self.assertEqual(choice["reference_choice"], "B")
        self.assertEqual(choice["predicted_choice"], result["samples"][1]["prediction"])
        self.assertEqual(set(choice["choice_probabilities"]), set("ABCD"))
        self.assertAlmostEqual(sum(choice["choice_probabilities"].values()), 1.0)
        ratio = choice["choice_probabilities"]["A"] / choice["choice_probabilities"]["B"]
        self.assertAlmostEqual(ratio, math.exp(choice["choice_logprobs"]["A"] - choice["choice_logprobs"]["B"]))
        self.assertNotIn("choice_probabilities", result["samples"][1])
        self.assertEqual(events[-1]["message"], "Health checks complete")

    def test_health_failure_keeps_completed_row_count_without_completion(self):
        from health import prepare, parse_selection
        instance = engine()
        instance.health_set = prepare([parse_selection("arithmetic:2")], 5)
        events = []
        instance.emit = events.append
        generate = instance.model.generate
        calls = 0

        def fail_second_generation(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("generation failed")
            return generate(*args, **kwargs)

        with patch.object(instance.model, "generate", side_effect=fail_second_generation):
            with self.assertRaisesRegex(ValueError, "generation failed"):
                instance.health({"max_new_tokens": 4})
        self.assertEqual([event["type"] for event in events], ["progress", "answer", "progress"])
        self.assertEqual([event["current"] for event in events], [0, 1, 1])
        self.assertTrue(all(event["label"] == "Health" and "turn" not in event for event in events))
        self.assertTrue(all(event.get("message") != "Health checks complete" for event in events))

    def test_health_mmlu_likelihoods_sum_all_continuation_tokens(self):
        instance = engine()
        row = {"prompt": "Q?\nA. x\nB. y\nC. z\nD. w\nAnswer:", "choices": list("ABCD")}
        with torch.inference_mode():
            predicted, scores, tokens = instance.health_choices(row)
            prompt = instance.tokenizer.encode(row["prompt"], add_special_tokens=True)
            sequence = torch.tensor([prompt + tokens["A"]])
            logits = instance.model(sequence, use_cache=False).logits[0, len(prompt)-1:-1].float()
            expected = -torch.nn.functional.cross_entropy(logits, torch.tensor(tokens["A"]), reduction="sum").item()
        self.assertAlmostEqual(scores["A"], expected, places=5)
        self.assertEqual(len(tokens["A"]), 2)
        self.assertEqual(predicted, max("ABCD", key=lambda choice: scores[choice]))

    def test_health_context_limit_and_missing_selection_are_explicit(self):
        from health import prepare, parse_selection
        instance = engine()
        with self.assertRaisesRegex(ValueError, "no health checks"):
            instance.health({})
        instance.health_set = prepare([parse_selection("copying:1")], 1)
        instance.max_seq_length = 20
        with self.assertRaisesRegex(ValueError, "exceeds"):
            instance.health({"max_new_tokens": 4})

    def test_health_rollouts_restore_exactly_after_rejected_weight_change(self):
        from health import prepare, parse_selection
        instance = engine()
        instance.health_set = prepare([parse_selection("arithmetic:1")], 5)
        baseline = instance.health({"max_new_tokens": 4})
        instance.edit(proposal(instance))
        instance.health({"max_new_tokens": 4})
        instance.rollback({})
        restored = instance.health({"max_new_tokens": 4})
        self.assertEqual(baseline["samples"], restored["samples"])
        self.assertEqual(baseline["metrics"], restored["metrics"])


if __name__ == "__main__":
    unittest.main()
