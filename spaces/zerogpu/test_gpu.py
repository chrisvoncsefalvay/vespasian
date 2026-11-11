"""Real tiny-transformer checks; no Hub downloads, scheduler or frontier calls."""
import copy
import contextvars
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["VESPASIAN_SPACE_SKIP_MODEL_LOAD"] = "1"
try:
    import spaces
    if not hasattr(spaces, "GPU"):
        raise ImportError("Only the source tree's spaces namespace is installed")
except ImportError:
    # The production module always requires spaces. Only this test harness may
    # replace its scheduler when using the existing core-only test environment.
    sys.modules["spaces"] = SimpleNamespace(GPU=lambda **kwargs: lambda function: function)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from transformers import LlamaForCausalLM
from test_worker import CharacterTokenizer, engine, proposal
from worker import Engine
import gpu


class GPUStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = engine()
        self.original = copy.deepcopy(self.fixture.model.state_dict())
        self.revision = "a" * 40
        self.source = self.root / self.revision
        self.fixture.model.save_pretrained(self.source)
        self.fixture.tokenizer.save_pretrained(self.source)
        self.dataset = self.root / "dataset.jsonl"
        self.dataset.write_text("\n".join(json.dumps(row) for row in self.fixture.dataset))
        template = gpu.ModelTemplate(gpu.SUPPORTED_MODELS[0], self.revision, self.fixture.model, self.fixture.tokenizer)
        self.registry = patch.object(gpu, "MODEL_TEMPLATES", MappingProxyType({template.identifier: template}))
        self.registry.start()
        self.addCleanup(self.registry.stop)
        self.hub = patch("huggingface_hub.hf_hub_download", return_value=str(self.source / "config.json"))
        self.hub.start()
        self.addCleanup(self.hub.stop)
        self.state = self.root / "worker-a"
        self.request_id = 0

    def call(self, command, state=None, **kwargs):
        self.request_id += 1
        events = list(gpu.dispatch(str(state or self.state), {"id": self.request_id, "command": command, **kwargs}))
        self.assertTrue(events, command)
        self.assertEqual(events[-1]["type"], "result")
        self.assertTrue(events[-1]["ok"], events[-1])
        json.dumps(events, allow_nan=False)
        return events[-1]["data"], events

    def initialise(self, state=None, **kwargs):
        return self.call("init", state=state, model=gpu.SUPPORTED_MODELS[0], revision=self.revision, dataset=str(self.dataset), device="cpu", max_seq_length=128, **kwargs)

    def assert_template_unchanged(self):
        for name, value in self.fixture.model.state_dict().items():
            self.assertTrue(torch.equal(value, self.original[name]), name)

    def test_pending_accept_and_rollback_survive_separate_allocations(self):
        self.initialise()
        self.call("edit", **proposal(self.fixture, operation="scale"))
        metadata, tensors = gpu._read_snapshot(self.state)
        name = next(iter(self.fixture.editable))
        pending_weight = tensors["current." + name].clone()
        self.assertTrue(torch.equal(tensors["pending." + name], self.original[name]))
        with patch.object(gpu, "_gpu_dispatch", side_effect=AssertionError("CPU selection must not allocate a GPU")):
            self.call("accept")
        self.call("edit", **proposal(self.fixture, operation="noise", seed=123))
        with patch.object(gpu, "_gpu_dispatch", side_effect=AssertionError("CPU rollback must not allocate a GPU")):
            self.call("rollback")
        metadata, tensors = gpu._read_snapshot(self.state)
        self.assertEqual(metadata["engine"]["changed"], [name])
        self.assertEqual(metadata["engine"]["pending_names"], [])
        self.assertEqual(set(tensors), {"current." + name})
        self.assertTrue(torch.equal(tensors["current." + name], pending_weight))
        self.assert_template_unchanged()

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 is unavailable")
    def test_cuda_bfloat16_replacements_and_rollback_are_bit_exact(self):
        template = gpu.ModelTemplate(gpu.SUPPORTED_MODELS[0], self.revision, copy.deepcopy(self.fixture.model).to(dtype=torch.bfloat16), self.fixture.tokenizer)
        registry = patch.object(gpu, "MODEL_TEMPLATES", MappingProxyType({template.identifier: template}))
        registry.start()
        self.addCleanup(registry.stop)
        self.call("init", model=gpu.SUPPORTED_MODELS[0], revision=self.revision, dataset=str(self.dataset), device="cuda", max_seq_length=128)
        self.call("edit", **proposal(self.fixture, operation="scale", strength=0.01))
        name = next(iter(self.fixture.editable))
        metadata, tensors = gpu._read_snapshot(self.state)
        expected_original = self.original[name].to(torch.bfloat16)
        self.assertEqual(metadata["loader"]["dtype"], "bfloat16")
        self.assertEqual(tensors["current." + name].device.type, "cpu")
        self.assertEqual(tensors["current." + name].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(tensors["pending." + name], expected_original))
        self.call("accept")
        retained = tensors["current." + name].clone()
        self.call("edit", **proposal(self.fixture, operation="noise", strength=0.02))
        self.call("rollback")
        self.assertTrue(torch.equal(gpu._read_snapshot(self.state)[1]["current." + name], retained))
        self.assert_template_unchanged()

    def test_qwen_template_copy_preserves_canonical_mixed_precision_buffers(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5VisionConfig
        from worker import load_model_components
        text = Qwen3_5TextConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=16, layer_types=["full_attention"] * 2,
            max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]})
        vision = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
            out_hidden_size=32, num_position_embeddings=16, patch_size=2, spatial_merge_size=2, temporal_patch_size=1)
        config = Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=96,
            video_token_id=97, vision_start_token_id=98, vision_end_token_id=99)
        source = self.root / "tiny-qwen"
        Qwen3_5ForConditionalGeneration(config).save_pretrained(source)
        processor = SimpleNamespace(tokenizer=CharacterTokenizer())
        with patch("transformers.AutoProcessor.from_pretrained", return_value=processor):
            canonical, tokenizer, loaded_processor = load_model_components(str(source), revision=None, device="cpu", dtype=torch.bfloat16)
            template = gpu.ModelTemplate(gpu.SUPPORTED_MODELS[0], self.revision, canonical, tokenizer, loaded_processor)
            with patch.object(gpu, "MODEL_TEMPLATES", MappingProxyType({template.identifier: template})):
                copied, _, _ = gpu._model_loader(template.identifier, revision=self.revision, device="cpu", dtype=torch.bfloat16)
                canonical_buffers, copied_buffers = dict(canonical.named_buffers()), dict(copied.named_buffers())
                self.assertEqual(canonical.dtype, torch.bfloat16)
                self.assertTrue(any(value.dtype == torch.float32 and "inv_freq" in name for name, value in canonical_buffers.items()))
                self.assertEqual(set(copied_buffers), set(canonical_buffers))
                for name, value in canonical_buffers.items():
                    self.assertEqual(copied_buffers[name].dtype, value.dtype, name)
                    self.assertTrue(torch.equal(copied_buffers[name], value), name)
                inputs = torch.tensor([[1, 3, 5, 7, 9]])
                with torch.inference_mode():
                    expected = canonical(input_ids=inputs, use_cache=False).logits
                    actual = copied(input_ids=inputs, use_cache=False).logits
                self.assertTrue(torch.equal(expected, actual))

                def canonical_float32(identifier, *, revision, device, dtype):
                    return load_model_components(str(source), revision=None, device=device, dtype=dtype)

                with patch.object(gpu, "load_model_components", side_effect=canonical_float32) as loader:
                    float32, _, _ = gpu._model_loader(template.identifier, revision=self.revision, device="cpu", dtype=torch.float32)
                loader.assert_called_once_with(template.identifier, revision=self.revision, device="cpu", dtype=torch.float32)
                self.assertEqual(float32.dtype, torch.float32)
                self.assertEqual(canonical.dtype, torch.bfloat16)

    def test_startup_pins_resolved_revision_and_rejects_all_model_failure(self):
        with patch.dict(os.environ, {"VESPASIAN_SPACE_SKIP_MODEL_LOAD": "0", "VESPASIAN_SPACE_MODELS": gpu.SUPPORTED_MODELS[0]}), \
             patch.object(gpu, "load_model_components", return_value=(self.fixture.model, self.fixture.tokenizer, None)) as loader:
            templates = gpu._preload_models()
        self.assertEqual(templates[gpu.SUPPORTED_MODELS[0]].revision, self.revision)
        self.assertEqual(loader.call_args.kwargs["revision"], self.revision)
        self.assertEqual(loader.call_args.kwargs["device"], "cuda" if torch.cuda.is_available() else "cpu")
        with patch.dict(os.environ, {"VESPASIAN_SPACE_SKIP_MODEL_LOAD": "0", "VESPASIAN_SPACE_MODELS": gpu.SUPPORTED_MODELS[0]}), \
             patch.object(gpu, "load_model_components", side_effect=OSError("test unavailable model")):
            with self.assertRaisesRegex(RuntimeError, "No configured model"):
                gpu._preload_models()

    def test_private_sessions_keep_different_edits_and_datasets(self):
        second = self.root / "worker-b"
        self.initialise()
        self.dataset.write_text("\n".join(json.dumps({**row, "prompt": "other " + row["prompt"]}) for row in self.fixture.dataset))
        self.initialise(state=second)
        from concurrent.futures import ThreadPoolExecutor
        barrier = threading.Barrier(2)
        original_edit = Engine.edit

        def overlap(instance, request):
            barrier.wait(timeout=5)
            return original_edit(instance, request)

        def edit(state, strength):
            return list(gpu.dispatch(str(state), {"id": 99, "command": "edit", **proposal(self.fixture, operation="scale", strength=strength)}))

        with patch.object(Engine, "edit", overlap), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(edit, self.state, 0.02), pool.submit(edit, second, -0.03)]
            for future in futures:
                self.assertTrue(future.result()[-1]["ok"])
        first_metadata, first_tensors = gpu._read_snapshot(self.state)
        second_metadata, second_tensors = gpu._read_snapshot(second)
        self.assertNotEqual(first_metadata["engine"]["dataset"], second_metadata["engine"]["dataset"])
        name = next(iter(self.fixture.editable))
        self.assertFalse(torch.equal(first_tensors["current." + name], second_tensors["current." + name]))
        self.assertNotEqual(first_metadata["model_version"], second_metadata["model_version"])
        self.call("rollback", state=second)
        self.assertTrue(torch.equal(gpu._read_snapshot(self.state)[1]["current." + name], first_tensors["current." + name]))
        self.assert_template_unchanged()

    def test_failed_commit_preserves_pointer_and_retries_do_not_repeat_edit(self):
        self.initialise()
        before = (self.state / "CURRENT").read_bytes()
        request = {"id": 200, "command": "edit", **proposal(self.fixture)}
        with patch.object(gpu.os, "replace", side_effect=OSError("interrupted commit")):
            events = list(gpu.dispatch(str(self.state), request))
        self.assertFalse(events[-1]["ok"])
        self.assertEqual((self.state / "CURRENT").read_bytes(), before)
        first = list(gpu.dispatch(str(self.state), request))
        self.assertTrue(first[-1]["ok"])
        pointer = (self.state / "CURRENT").read_bytes()
        with patch.object(gpu, "_model_loader", side_effect=AssertionError("A committed request is replayed")):
            replay = list(gpu.dispatch(str(self.state), request))
        self.assertEqual(first, replay)
        self.assertEqual((self.state / "CURRENT").read_bytes(), pointer)
        self.assert_template_unchanged()

    def test_restore_keeps_encoded_rows_and_does_not_hash_source_again(self):
        self.initialise()
        self.call("edit", **proposal(self.fixture))
        self.call("accept")
        previous_location = gpu._read_metadata(self.state)[1]
        with patch.object(Engine, "manifest", side_effect=AssertionError("Provenance is already recorded")):
            result, _ = self.call("evaluate", splits=["validation"], max_new_tokens=3)
        self.assertEqual(len(result["samples"]), 1)
        metadata, _ = gpu._read_snapshot(self.state)
        self.assertEqual(len(metadata["engine"]["encoded"]), 4)
        self.assertEqual(len(metadata["engine"]["dataset"]), 4)
        location = gpu._read_metadata(self.state)[1]
        self.assertEqual((location / "tensors.safetensors").stat().st_ino, (previous_location / "tensors.safetensors").stat().st_ino)

    def test_export_reload_uses_real_checkpoint_loader_and_exact_saved_weights(self):
        self.initialise()
        self.call("edit", **proposal(self.fixture))
        self.call("accept")
        output = self.root / "export"
        saved, _ = self.call("save", path=str(output))
        self.assertTrue(saved["source_unchanged"])
        expected, _ = self.call("evaluate", splits=["validation", "test"], max_new_tokens=3)
        loaded_paths = []

        def fresh_loader(identifier, *, revision, device, dtype):
            loaded_paths.append(identifier)
            return LlamaForCausalLM.from_pretrained(identifier, dtype=dtype).to(device).eval(), CharacterTokenizer(), None

        with patch.object(gpu, "load_model_components", side_effect=fresh_loader):
            state = self.root / "fresh-checkpoint"
            self.call("init", state=state, model=str(output), dataset=str(self.dataset), device="cpu", max_seq_length=128)
            actual, _ = self.call("evaluate", state=state, splits=["validation", "test"], max_new_tokens=3)
        self.assertEqual(loaded_paths, [str(output), str(output)])
        self.assertEqual(actual["samples"], expected["samples"])
        self.assertEqual(actual["metrics"], expected["metrics"])
        self.assert_template_unchanged()

    def test_batched_evaluation_matches_core_and_streams_global_progress(self):
        self.fixture.dataset += [
            {"prompt": "epsilon", "target": "ef", "split": "validation"},
            {"prompt": "zeta", "target": "z", "split": "validation"},
        ]
        self.dataset.write_text("\n".join(json.dumps(row) for row in self.fixture.dataset))
        self.initialise()
        request = {"splits": ["validation", "preserve", "test"], "max_new_tokens": 3}
        expected = self.fixture.evaluate(request)
        with patch.object(gpu, "MAX_EXAMPLES", 2):
            actual, events = self.call("evaluate", **request)
        self.assertEqual(actual["samples"], expected["samples"])
        self.assertEqual(actual["metrics"], expected["metrics"])
        answers = [event for event in events if event["type"] == "answer"]
        self.assertEqual([event["current"] for event in answers], [1, 2, 3, 4, 5])
        self.assertEqual({event["total"] for event in answers}, {5})
        self.assertEqual(sum(event.get("message") == "Evaluation complete" for event in events), 1)
        self.assertEqual(len(gpu._read_snapshot(self.state)[0]["engine"]["dataset"]), 6)

    def test_batched_health_preserves_checks_hash_and_weighted_scores(self):
        from health import prepare, parse_selection
        manifest = prepare([parse_selection("copying:2"), parse_selection("arithmetic:2")], 7)
        path = self.root / "health.json"
        path.write_text(json.dumps(manifest))
        self.initialise(health_path=str(path))
        self.fixture.health_set = manifest
        expected = self.fixture.health({"max_new_tokens": 3})
        with patch.object(gpu, "MAX_EXAMPLES", 1):
            actual, _ = self.call("health", max_new_tokens=3)
        self.assertEqual(actual["samples"], expected["samples"])
        self.assertEqual(actual["metrics"], expected["metrics"])
        self.assertEqual(actual["rows_sha256"], manifest["rows_sha256"])
        self.assertEqual(gpu._read_snapshot(self.state)[0]["engine"]["health_set"], manifest)

    def test_unpaired_batched_modes_match_core_after_state_restoration(self):
        for mode in ("corpus", "prompts"):
            with self.subTest(mode=mode):
                fixture = engine()
                fixture.dataset_mode = mode
                fixture.dataset = [dict(row, mode=mode, **({"prompt": "", "target": row["prompt"]} if mode == "corpus" else {"target": ""})) for row in fixture.dataset]
                self.dataset.write_text("\n".join(json.dumps(row) for row in fixture.dataset))
                state = self.root / mode
                self.initialise(state=state)
                request = {"splits": ["validation", "preserve"], "max_new_tokens": 4}
                expected = fixture.evaluate(request)
                with patch.object(gpu, "MAX_EXAMPLES", 1):
                    actual, _ = self.call("evaluate", state=state, **request)
                self.assertEqual(actual["samples"], expected["samples"])
                self.assertEqual(actual["metrics"], expected["metrics"])

    def test_interrupted_batch_resumes_from_saved_group_without_repeating_it(self):
        self.initialise()
        request = {"id": 300, "command": "evaluate", "splits": ["validation", "preserve"], "max_new_tokens": 3}
        original_dispatch = gpu._gpu_dispatch
        attempted = []

        def interrupt_second(state_dir, batch_request):
            offset = batch_request["_space_batch"]["offset"]
            attempted.append(offset)
            if offset == 1:
                yield {"type": "result", "id": batch_request["id"], "ok": False, "error": "quota interrupted"}
            else:
                yield from original_dispatch(state_dir, batch_request)

        with patch.object(gpu, "MAX_EXAMPLES", 1), patch.object(gpu, "_gpu_dispatch", side_effect=interrupt_second):
            first = list(gpu.dispatch(str(self.state), request))
        self.assertFalse(first[-1]["ok"])
        self.assertEqual(attempted, [0, 1])
        attempted.clear()

        def resume(state_dir, batch_request):
            attempted.append(batch_request["_space_batch"]["offset"])
            yield from original_dispatch(state_dir, batch_request)

        with patch.object(gpu, "MAX_EXAMPLES", 1), patch.object(gpu, "_gpu_dispatch", side_effect=resume):
            resumed = list(gpu.dispatch(str(self.state), request))
        self.assertTrue(resumed[-1]["ok"])
        self.assertEqual(attempted, [1])
        self.assertEqual(len(resumed[-1]["data"]["samples"]), 2)

    def test_allocation_boundary_waits_for_the_resuming_request_context(self):
        credential = contextvars.ContextVar("request_credential", default="initial-request")
        observed = []

        def scheduled(state_dir, request):
            observed.append(credential.get())
            yield {"type": "result", "id": request["id"], "ok": True, "data": {}}

        with patch.object(gpu, "_gpu_dispatch", side_effect=scheduled):
            iterator = gpu.dispatch(str(self.state), {"id": 301, "command": "init"}, boundaries=True)
            self.assertEqual(next(iterator), {"type": "gpu_allocation"})
            self.assertEqual(observed, [])
            fresh_request = contextvars.copy_context()
            fresh_request.run(credential.set, "fresh-request")
            events = fresh_request.run(list, iterator)
        self.assertEqual(observed, ["fresh-request"])
        self.assertTrue(events[-1]["ok"])

    def test_batches_resume_on_fresh_callback_threads_without_changing_results(self):
        self.initialise()
        request = {"id": 302, "command": "evaluate", "splits": ["validation", "preserve", "test"], "max_new_tokens": 3}
        expected = self.fixture.evaluate(request)
        credential = contextvars.ContextVar("batch_request_credential", default="initial-request")
        original_dispatch = gpu._gpu_dispatch
        allocations, events = [], []

        def scheduled(state_dir, batch_request):
            allocations.append((batch_request["_space_batch"]["offset"], credential.get()))
            yield from original_dispatch(state_dir, batch_request)

        def advance_request(number):
            credential.set(f"request-{number}")
            for event in iterator:
                if event["type"] == "gpu_allocation":
                    return False
                events.append(event)
            return True

        with patch.object(gpu, "MAX_EXAMPLES", 1), patch.object(gpu, "_gpu_dispatch", side_effect=scheduled):
            iterator = gpu.dispatch(str(self.state), request, boundaries=True)
            self.assertEqual(next(iterator), {"type": "gpu_allocation"})
            self.assertEqual(allocations, [])
            with self.assertRaisesRegex(ValueError, "active operation"):
                with gpu._state_lock(self.state / "evaluations"):
                    pass
            completed = []
            for number in range(1, 4):
                thread = threading.Thread(target=lambda n=number: completed.append(advance_request(n)))
                thread.start()
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
        self.assertEqual(completed, [False, False, True])
        self.assertEqual(allocations, [(0, "request-1"), (1, "request-2"), (2, "request-3")])
        self.assertEqual(events[-1]["data"]["samples"], expected["samples"])
        self.assertEqual(events[-1]["data"]["metrics"], expected["metrics"])
        for journal in (self.state / "evaluations").glob("*.json"):
            self.assertNotIn("gpu_allocation", journal.read_text())
        with gpu._state_lock(self.state / "evaluations"):
            pass

    def test_expired_batch_retries_with_fresh_context_and_keeps_completed_batches(self):
        import gradio as gr

        self.initialise()
        request = {"id": 303, "command": "evaluate", "splits": ["validation", "preserve"], "max_new_tokens": 3}
        credential = contextvars.ContextVar("retry_request_credential", default="expired-request")
        original_dispatch = gpu._gpu_dispatch
        attempts, events = [], []

        def scheduled(state_dir, batch_request):
            offset = batch_request["_space_batch"]["offset"]
            attempts.append((offset, credential.get()))
            if offset == 1 and credential.get() == "expired-request":
                raise gr.Error("Expired ZeroGPU proxy token")
            yield from original_dispatch(state_dir, batch_request)

        with patch.object(gpu, "MAX_EXAMPLES", 1), patch.object(gpu, "_gpu_dispatch", side_effect=scheduled):
            iterator = gpu.dispatch(str(self.state), request, boundaries=True)
            self.assertEqual(next(iterator), {"type": "gpu_allocation"})
            for event in iterator:
                if event["type"] == "gpu_allocation":
                    break
                events.append(event)
            self.assertEqual(attempts, [(0, "expired-request")])
            self.assertEqual(next(iterator), {"type": "gpu_allocation"})
            self.assertEqual(attempts, [(0, "expired-request"), (1, "expired-request")])
            fresh_request = contextvars.copy_context()
            fresh_request.run(credential.set, "fresh-request")
            events.extend(fresh_request.run(list, iterator))
        self.assertEqual(attempts, [(0, "expired-request"), (1, "expired-request"), (1, "fresh-request")])
        self.assertTrue(events[-1]["ok"])
        self.assertEqual(len(events[-1]["data"]["samples"]), 2)
        self.assertEqual(len([event for event in events if event["type"] == "answer"]), 2)

    def test_proxy_retry_is_bounded_and_other_failures_are_not_retried(self):
        cases = (
            ("Expired ZeroGPU proxy token", True, False, 2),
            ("Expired ZeroGPU proxy token", False, False, 1),
            ("Expired ZeroGPU proxy token", True, True, 1),
            ("GPU quota exceeded", True, False, 1),
            ("Expired ZeroGPU proxy token: unrelated diagnostic", True, False, 1),
        )
        for message, boundaries, started, expected in cases:
            with self.subTest(message=message, boundaries=boundaries, started=started):
                attempts = []

                def fail(state_dir, request):
                    attempts.append(request["id"])
                    if started:
                        yield {"type": "progress", "message": "Model operation started"}
                    raise RuntimeError(message)

                with patch.object(gpu, "_gpu_dispatch", side_effect=fail):
                    events = list(gpu.dispatch(str(self.state), {"id": 304, "command": "init"}, boundaries=boundaries))
                self.assertEqual(len(attempts), expected)
                self.assertEqual(sum(event["type"] == "gpu_allocation" for event in events), expected if boundaries else 0)
                self.assertFalse(events[-1]["ok"])
                self.assertEqual(events[-1]["error"], message)

    def test_cpu_commands_have_no_allocation_boundary(self):
        self.initialise()
        self.call("edit", **proposal(self.fixture))
        with patch.object(gpu, "_gpu_dispatch", side_effect=AssertionError("CPU command allocated a GPU")):
            for number, command in enumerate(("rollback", "shutdown"), start=310):
                events = list(gpu.dispatch(str(self.state), {"id": number, "command": command}, boundaries=True))
                self.assertTrue(events[-1]["ok"], events[-1])
                self.assertNotIn("gpu_allocation", {event["type"] for event in events})

    def test_closing_at_an_allocation_boundary_releases_the_batch_lock(self):
        self.initialise()
        request = {"id": 320, "command": "evaluate", "splits": ["validation"], "max_new_tokens": 3}
        with patch.object(gpu, "_gpu_dispatch", side_effect=AssertionError("Paused request allocated a GPU")):
            iterator = gpu.dispatch(str(self.state), request, boundaries=True)
            self.assertEqual(next(iterator), {"type": "gpu_allocation"})
            iterator.close()
        with gpu._state_lock(self.state / "evaluations"):
            pass

    def test_events_arrive_before_operation_finishes(self):
        self.initialise()
        released, complete = threading.Event(), threading.Event()
        original = Engine.evaluate

        def observed(instance, request):
            instance.progress("evaluating", "First event")
            released.wait(timeout=5)
            result = original(instance, request)
            complete.set()
            return result

        with patch.object(Engine, "evaluate", observed):
            iterator = gpu.dispatch(str(self.state), {"id": 400, "command": "evaluate", "splits": ["validation"], "max_new_tokens": 3})
            try:
                first = next(iterator)
                self.assertEqual(first["message"], "First event")
                self.assertFalse(complete.is_set())
            finally:
                released.set()
            rest = list(iterator)
        self.assertTrue(rest[-1]["ok"])
        self.assertTrue(complete.is_set())

    def test_availability_and_duration_need_no_model_tensor_access(self):
        self.initialise()
        with patch.object(gpu, "_model_loader", side_effect=AssertionError("CPU metadata only")):
            self.assertEqual(gpu.model_availability()[0]["revision"], self.revision)
            duration = gpu.estimate_duration(str(self.state), {"command": "evaluate", "max_new_tokens": 128})
        self.assertGreaterEqual(duration, 20)
        self.assertLessEqual(duration, 120)
        invalid = list(gpu.dispatch(str(self.state), {"id": 500, "command": "evaluate", "max_new_tokens": 129}))
        self.assertFalse(invalid[-1]["ok"])
        self.assertIn("128", invalid[-1]["error"])


if __name__ == "__main__":
    unittest.main()
