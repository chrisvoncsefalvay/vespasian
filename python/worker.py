#!/usr/bin/env python3
"""Persistent tensor engine for Vespasian's JSON-lines protocol."""
from __future__ import annotations

import collections
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def token_f1(prediction: str, target: str) -> float:
    predicted, expected = normalise(prediction).split(), normalise(target).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((collections.Counter(predicted) & collections.Counter(expected)).values())
    return 2.0 * overlap / (len(predicted) + len(expected))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return float(value)


def model_structure(config) -> dict[str, Any]:
    """Read text architecture metadata without inferring it from editable tensors."""
    text_config = getattr(config, "text_config", None) if getattr(config, "model_type", None) == "qwen3_5" else config
    architecture = getattr(text_config, "model_type", None)
    layers = getattr(text_config, "num_hidden_layers", None)
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError("The text model configuration has no architecture name")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("The text model configuration has no valid hidden-layer count")
    return {"architecture": architecture, "num_hidden_layers": layers}


class Engine:
    def __init__(self, emit=None):
        self.emit = emit or (lambda event: None)
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.dataset: list[dict[str, str]] = []
        self.encoded: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}
        self.pending: dict[str, Any] = {}
        self.pending_details: list[dict] = []
        self.changed: set[str] = set()
        self.original_hashes: dict[str, str] = {}
        self.source_dir: Path | None = None
        self.source_files: list[dict] = []
        self.max_seq_length = 512
        self.device = "cpu"
        self.health_set = None
        self.dataset_mode = "paired"
        self.seed = 42

    def progress(self, phase: str, message: str, current=0, total=0, *, label=None, turn=None):
        event = {"type": "progress", "phase": phase, "message": message, "current": current, "total": total}
        if label is not None:
            event["label"] = label
        if turn is not None:
            event["turn"] = turn
        self.emit(event)

    def answer(self, phase: str, label: str, turn, current: int, total: int, sample: dict, scoring=None):
        """Publish a bounded display copy while retaining the complete evaluation."""
        example = copy.deepcopy(sample)
        if "id" not in example:
            identity = sample.get("split", "") + "\0" + sample["prompt"]
            if sample.get("mode") == "corpus":
                identity += "\0" + sample["target"]
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            example["id"] = f"{sample.get('split', 'example')}:{digest}"
        example["output"] = sample["prediction"]
        if scoring is not None:
            example["scoring"] = scoring
        if "choice_logprobs" in sample:
            maximum = max(sample["choice_logprobs"].values())
            weights = {choice: math.exp(value - maximum) for choice, value in sample["choice_logprobs"].items()}
            total_weight = sum(weights.values())
            example["choice_probabilities"] = {choice: weight / total_weight for choice, weight in weights.items()}
            example["predicted_choice"] = sample["prediction"]
            example["reference_choice"] = sample["target"]
        truncated = []
        for field in ("prompt", "target", "prediction", "output"):
            if isinstance(example.get(field), str) and len(example[field]) > 16384:
                example[field] = example[field][:16384]
                truncated.append(field)
        event = {"type": "answer", "phase": phase, "label": label, "current": current, "total": total, "example": example}
        if turn is not None:
            event["turn"] = turn
        if truncated:
            event["truncated_fields"] = truncated
        self.emit(event)

    def ready(self):
        if self.model is None:
            raise ValueError("Load a model before starting this operation")

    def load_dataset(self, path: str):
        rows = []
        seen = set()
        dataset_mode = None
        with Path(path).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Dataset line {number}: invalid JSON") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Dataset line {number}: expected an object")
                if row.get("split") not in {"edit", "validation", "preserve", "test"}:
                    raise ValueError(f"Dataset line {number}: unknown split")
                mode = row.get("mode", "paired")
                if mode not in {"paired", "corpus", "prompts"}:
                    raise ValueError(f"Dataset line {number}: unknown dataset mode")
                if dataset_mode is not None and mode != dataset_mode:
                    raise ValueError("The dataset must use one mode throughout")
                dataset_mode = mode
                for key in ("prompt", "target"):
                    value = row.get(key)
                    absent = (mode == "corpus" and key == "prompt") or (mode == "prompts" and key == "target")
                    if not isinstance(value, str) or (not absent and not value.strip()) or (absent and value != ""):
                        raise ValueError(f"Dataset line {number}: invalid {key} for {mode} examples")
                normalised = normalise(row["target"] if mode == "corpus" else row["prompt"])
                if normalised in seen:
                    raise ValueError(f"Dataset line {number}: repeated {'text' if mode == 'corpus' else 'prompt'}")
                seen.add(normalised)
                selected = {key: row[key] for key in ("prompt", "target", "split")}
                if mode != "paired":
                    selected["mode"] = mode
                rows.append(selected)
        counts = collections.Counter(row["split"] for row in rows)
        if not counts["edit"] or not counts["validation"]:
            raise ValueError("The dataset needs edit and validation examples")
        self.dataset = rows
        self.dataset_mode = dataset_mode or "paired"
        self.encoded.clear()
        return dict(counts)

    def allowed_operations(self):
        operations = ["scale", "low_rank", "noise"]
        if all(row.get("mode", "paired") != "prompts" for row in self.dataset if row["split"] == "edit"):
            operations.append("activation_rank_one")
        return operations

    def generation_policy(self):
        if self.dataset_mode == "corpus":
            return {"conditioning": "unconditional", "do_sample": True, "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seed_policy": "run seed plus stable reference hash", "start_token_id": self.start_token()}
        return {"conditioning": "chat_prompt", "do_sample": False}

    def start_token(self):
        config = getattr(self.model.config, "text_config", self.model.config)
        for value in (getattr(self.tokenizer, "bos_token_id", None), getattr(config, "bos_token_id", None), getattr(self.tokenizer, "eos_token_id", None)):
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < self.model.get_input_embeddings().num_embeddings:
                return value
        raise ValueError("Unconditional generation needs a BOS token or an EOS start token")

    def generation_seed(self, row: dict):
        digest = hashlib.sha256(row["target"].encode("utf-8")).digest()
        return (self.seed + int.from_bytes(digest[:8], "big")) % (2**63)

    def initialise(self, request: dict):
        if self.model is not None:
            raise ValueError("This worker already has a model loaded")
        import torch
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Gemma3ForCausalLM, Qwen3_5ForConditionalGeneration

        self.max_seq_length = bounded_int(request.get("max_seq_length", 512), "max_seq_length", 16, 16384)
        seed = bounded_int(request.get("seed", 42), "seed", 0, 2**32 - 1)
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        requested_device = request.get("device", "auto")
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be auto, cpu or cuda")
        self.device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Choose --device cpu or install CUDA-enabled PyTorch")
        counts = self.load_dataset(request["dataset"])
        identifier = request["model"]
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Model identifier must contain text")
        revision = request.get("revision")
        kwargs = {"trust_remote_code": False}
        if revision:
            kwargs["revision"] = revision
        self.progress("loading", f"Loading {identifier}")
        if Path(identifier).is_dir():
            self.source_dir = Path(identifier).resolve()
            resolved_revision = None
        else:
            from huggingface_hub import hf_hub_download
            config_path = Path(hf_hub_download(identifier, "config.json", revision=revision))
            # Keep the snapshot path: resolving the file symlink would yield a blob hash.
            self.source_dir = config_path.parent
            resolved_revision = self.source_dir.name
            kwargs["revision"] = resolved_revision
        config = AutoConfig.from_pretrained(identifier, **kwargs)
        if config.model_type == "qwen3_5":
            model_class = Qwen3_5ForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(identifier, **kwargs)
            self.tokenizer = self.processor.tokenizer
        elif config.model_type == "gemma3_text":
            model_class = Gemma3ForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(identifier, **kwargs)
        else:
            raise ValueError(f"Unsupported architecture: {config.model_type}. Choose Qwen3.5-0.8B or Gemma-3-1B.")
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = model_class.from_pretrained(identifier, dtype=dtype, attn_implementation="eager", **kwargs).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        structure = model_structure(self.model.config)
        self.editable = {name: parameter for name, parameter in self.model.named_parameters() if name.endswith(".mlp.down_proj.weight")}
        if not self.editable:
            self.model = None
            raise ValueError("This checkpoint has no supported editable tensors")
        if self.source_dir:
            self.progress("loading", "Recording checkpoint provenance")
            self.source_files = self.manifest(self.source_dir)
        for row in self.dataset:
            self.encode(row)
        if request.get("health_path"):
            self.load_health(request["health_path"])
        self.progress("loading", "Model ready", 1, 1)
        return {"model": identifier, "dataset_mode": self.dataset_mode, "allowed_operations": self.allowed_operations(), "generation": self.generation_policy(), "revision": resolved_revision, "device": self.device, "dtype": str(dtype).replace("torch.", ""), **structure, "editable": [{"name": name, "shape": list(parameter.shape)} for name, parameter in self.editable.items()], "counts": counts, "source": {"path": str(self.source_dir) if self.source_dir else None, "files": self.source_files}, "seed": seed, "runtime": {"torch": torch.__version__, "transformers": __import__("transformers").__version__}}

    def encode(self, row: dict) -> tuple[list[int], list[int]]:
        mode = row.get("mode", "paired")
        key = (mode, row["prompt"], row["target"])
        if key in self.encoded:
            return self.encoded[key]
        if mode == "corpus":
            prompt = [self.start_token()]
        else:
            messages = [{"role": "user", "content": row["prompt"]}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if hasattr(prompt, "input_ids"):
                prompt = prompt.input_ids
            if isinstance(prompt, dict):
                prompt = prompt["input_ids"]
        target = [] if mode == "prompts" else self.tokenizer.encode(row["target"], add_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        # Chat answers use Gemma's assistant terminator. Corpus text uses EOS.
        if mode == "paired" and self.model is not None and getattr(self.model.config, "model_type", None) == "gemma3_text":
            eos = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if eos is None or eos == self.tokenizer.unk_token_id:
                raise ValueError("The Gemma tokenizer is missing its assistant turn terminator")
        if mode != "prompts" and eos is not None:
            target.append(eos)
        if not prompt or (mode != "prompts" and not target):
            raise ValueError("An example produced an empty prompt or target")
        if len(prompt) + len(target) > self.max_seq_length:
            raise ValueError(f"Example exceeds --max-seq-length ({len(prompt) + len(target)} tokens): {row['prompt'][:80]}")
        self.encoded[key] = (prompt, target)
        return prompt, target

    def loss(self, row: dict):
        import torch
        prompt, target = self.encode(row)
        if not target:
            raise ValueError("Reference loss is unavailable for prompt-only examples")
        tokens = torch.tensor([prompt + target], dtype=torch.long, device=self.device)
        output = self.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False, logits_to_keep=len(target) + 1)
        logits = output.logits[:, -(len(target) + 1):-1, :].float()
        expected = torch.tensor(target, dtype=torch.long, device=self.device)
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), expected)

    def evaluate(self, request: dict):
        self.ready()
        import torch
        splits = request.get("splits", ["validation", "preserve"])
        if not isinstance(splits, list) or not splits or len(set(splits)) != len(splits) or any(split not in {"edit", "validation", "preserve", "test"} for split in splits):
            raise ValueError("Evaluation splits must be a nonempty list of distinct split names")
        maximum = bounded_int(request.get("max_new_tokens", 64), "max_new_tokens", 1, 2048)
        rows = [row for row in self.dataset if row["split"] in splits]
        metrics: dict[str, dict] = {}
        samples = []
        self.model.eval()
        started = time.monotonic()
        label, turn = request.get("label", "Evaluation"), request.get("turn")
        self.progress("evaluating", f"{label} · 0/{len(rows)}", 0, len(rows), label=label, turn=turn)
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                mode = row.get("mode", "paired")
                prompt, _ = self.encode(row)
                if len(prompt) + maximum > self.max_seq_length:
                    raise ValueError(f"Prompt plus --max-new-tokens exceeds --max-seq-length: {row['prompt'][:80]}")
                input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
                arguments = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "max_new_tokens": maximum, "do_sample": False, "num_beams": 1, "use_cache": True, "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id}
                generation = {}
                if mode == "corpus":
                    seed = self.generation_seed(row)
                    device = torch.device(self.device)
                    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(seed)
                        output = self.model.generate(**{**arguments, "do_sample": True, "temperature": 0.8, "top_p": 0.95, "top_k": 0})
                    generation = {"generation_seed": seed, "start_token_id": prompt[0]}
                else:
                    output = self.model.generate(**arguments)
                generated = output[0, len(prompt):].tolist()
                prediction = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                eos_ids = self.model.generation_config.eos_token_id
                eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
                finish_reason = "stop" if generated and generated[-1] in eos_ids else "length"
                measured = {}
                if mode != "prompts":
                    value = float(self.loss(row).item())
                    if not math.isfinite(value):
                        raise ValueError("Evaluation produced a nonfinite target loss")
                    measured["loss"] = value
                if mode == "paired":
                    measured["exact_match"] = normalise(prediction) == normalise(row["target"])
                    measured["token_f1"] = token_f1(prediction, row["target"])
                totals = metrics.setdefault(row["split"], {key: 0.0 for key in measured} | {"count": 0})
                totals["count"] += 1
                for key, value in measured.items():
                    totals[key] += float(value)
                sample = {**row, "prediction": prediction, **measured, "generated_tokens": len(generated), "token_ids": generated, "finish_reason": finish_reason, **generation}
                samples.append(sample)
                self.answer("evaluating", label, turn, index, len(rows), sample)
                self.progress("evaluating", f"{label} · {row['split']} {index}/{len(rows)}", index, len(rows), label=label, turn=turn)
        for totals in metrics.values():
            for key in ("exact_match", "token_f1", "loss"):
                if key in totals:
                    totals[key] /= totals["count"]
        self.progress("evaluating", "Evaluation complete", len(rows), len(rows), label=label, turn=turn)
        return {"metrics": metrics, "samples": samples, "elapsed_seconds": time.monotonic() - started}

    def load_health(self, path: str):
        from health import validate_manifest
        with Path(path).open(encoding="utf-8") as source:
            self.health_set = validate_manifest(json.load(source))
        return {"checks": self.health_set["checks"], "rows_sha256": self.health_set["rows_sha256"]}

    def health_prompt(self, text: str) -> list[int]:
        tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        if hasattr(tokens, "input_ids"):
            tokens = tokens.input_ids
        if isinstance(tokens, dict):
            tokens = tokens["input_ids"]
        if not isinstance(tokens, list) or not tokens or any(not isinstance(token, int) for token in tokens):
            raise ValueError("A health prompt did not produce valid tokens")
        return tokens

    def health_choices(self, row: dict):
        """Zero-shot raw-prompt answer-letter likelihoods without length normalisation."""
        import torch
        prompt = self.tokenizer.encode(row["prompt"], add_special_tokens=True)
        if not prompt:
            raise ValueError("A MMLU prompt produced no tokens")
        values, choice_tokens = {}, {}
        for choice in row["choices"]:
            continuation = self.tokenizer.encode(" " + choice, add_special_tokens=False)
            if not continuation or len(prompt) + len(continuation) > self.max_seq_length:
                raise ValueError("A MMLU health example exceeds --max-seq-length. Increase the context limit")
            inputs = torch.tensor([prompt + continuation], dtype=torch.long, device=self.device)
            output = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=len(continuation) + 1)
            logits = output.logits[0, -(len(continuation) + 1):-1].float()
            expected = torch.tensor(continuation, dtype=torch.long, device=self.device)
            value = float(logits.log_softmax(dim=-1).gather(1, expected[:, None]).sum().item())
            if not math.isfinite(value):
                raise ValueError("A health answer-choice likelihood is not finite")
            values[choice], choice_tokens[choice] = value, continuation
        prediction = max(row["choices"], key=lambda choice: values[choice])
        return prediction, values, choice_tokens

    def health(self, request: dict):
        self.ready()
        if self.health_set is None:
            raise ValueError("This run has no health checks")
        import torch
        from health import score
        maximum = bounded_int(request.get("max_new_tokens", 64), "max_new_tokens", 1, 2048)
        metrics = {item["check"]: {"label": item["label"], "metric": item["metric"], "score": 0.0, "count": 0} for item in self.health_set["checks"]}
        samples = []
        rows = self.health_set["rows"]
        started = time.monotonic()
        self.model.eval()
        label, turn = request.get("label", "Health"), request.get("turn")
        self.progress("health", f"{label} · 0/{len(rows)}", 0, len(rows), label=label, turn=turn)
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                if row["check"].startswith("mmlu:"):
                    prediction, likelihoods, choice_tokens = self.health_choices(row)
                    detail = {"choice_logprobs": likelihoods, "choice_token_ids": choice_tokens, "finish_reason": "choice"}
                else:
                    prompt = self.health_prompt(row["prompt"])
                    if len(prompt) + maximum > self.max_seq_length:
                        raise ValueError("A health prompt plus --max-new-tokens exceeds --max-seq-length. Increase the context limit")
                    inputs = torch.tensor([prompt], dtype=torch.long, device=self.device)
                    output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=maximum, do_sample=False, num_beams=1, use_cache=True, pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
                    generated = output[0, len(prompt):].tolist()
                    prediction = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    eos = self.model.generation_config.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos]
                    detail = {"token_ids": generated, "generated_tokens": len(generated), "finish_reason": "stop" if generated and generated[-1] in eos else "length"}
                passed = score(row["check"], prediction, row["target"])
                metric = metrics[row["check"]]
                metric["count"] += 1
                metric["score"] += int(passed)
                sample = {"id": row["id"], "check": row["check"], "prompt": row["prompt"], "target": row["target"], "prediction": prediction, "passed": passed, **detail}
                samples.append(sample)
                self.answer("health", label, turn, index, len(rows), sample, scoring=metric["metric"])
                self.progress("health", f"{label} · {row['check']} {index}/{len(rows)}", index, len(rows), label=label, turn=turn)
        for metric in metrics.values():
            metric["score"] /= metric["count"]
        self.progress("health", "Health checks complete", len(rows), len(rows), label=label, turn=turn)
        return {"metrics": metrics, "samples": samples, "rows_sha256": self.health_set["rows_sha256"], "elapsed_seconds": time.monotonic() - started}

    def validate_proposal(self, proposal: dict):
        required = {"targets", "operation", "strength", "rank", "seed", "rationale"}
        if not isinstance(proposal, dict) or set(proposal) != required:
            raise ValueError("Edit proposal must contain targets, operation, strength, rank, seed and rationale")
        targets = proposal["targets"]
        if not isinstance(targets, list) or not 1 <= len(targets) <= 4 or any(not isinstance(name, str) for name in targets) or len(set(targets)) != len(targets):
            raise ValueError("Choose one to four distinct target tensors")
        if any(name not in self.editable for name in targets):
            raise ValueError("Edit proposal names a tensor outside the editable set")
        if not isinstance(proposal["rationale"], str) or not proposal["rationale"].strip():
            raise ValueError("Edit proposal requires a rationale")
        operation = proposal["operation"]
        if operation not in {"scale", "low_rank", "noise", "activation_rank_one"}:
            raise ValueError("Operation must be scale, low_rank, noise or activation_rank_one")
        if operation not in self.allowed_operations():
            raise ValueError("Activation surgery requires reference text. Choose scale, low_rank or noise for prompt-only data")
        strength = bounded_float(proposal["strength"], "strength", -0.1, 0.1)
        if abs(strength) < 1e-6:
            raise ValueError("Strength must have an absolute value of at least 0.000001")
        return targets, operation, strength, bounded_int(proposal["rank"], "rank", 1, 16), bounded_int(proposal["seed"], "seed", 0, 2147483647)

    def activation_directions(self, targets: list[str], rank: int, seed: int):
        """Approximate all reference tokens, including termination, with forward activations."""
        import torch
        rows = [row for row in self.dataset if row["split"] == "edit"]
        if not rows:
            raise ValueError("Activation surgery requires editing examples")
        selected = random.Random(seed).sample(rows, min(rank, len(rows)))
        captures = {}
        directions = {name: torch.zeros_like(self.editable[name], dtype=torch.float32) for name in targets}
        handles = []
        output_embeddings = self.model.get_output_embeddings().weight

        def capture(name):
            def hook(module, args, result):
                captures[name] = args[0].reshape(-1, args[0].shape[-1]).detach().float().clone()
            return hook

        try:
            for name in targets:
                module = self.model.get_submodule(name.removesuffix(".weight"))
                handles.append(module.register_forward_hook(capture(name)))
            with torch.inference_mode():
                for index, row in enumerate(selected, 1):
                    prompt, target = self.encode(row)
                    inputs = torch.tensor([prompt + target], dtype=torch.long, device=self.device)
                    captures.clear()
                    result = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=len(target) + 1)
                    predicted = result.logits[0, -(len(target) + 1):-1].argmax(dim=-1)
                    desired = torch.tensor(target, dtype=torch.long, device=self.device)
                    residual = output_embeddings[desired].float() - output_embeddings[predicted].float()
                    for name in targets:
                        # Position p-1 predicts the first answer token; subsequent
                        # teacher-forced positions predict its suffix and terminator.
                        keys = captures[name][len(prompt) - 1:len(prompt) + len(target) - 1]
                        if residual.shape[1] != directions[name].shape[0] or keys.shape != (len(target), directions[name].shape[1]):
                            raise ValueError("Activation and residual dimensions do not match the selected tensor")
                        directions[name].add_(residual.T @ (keys / keys.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)))
                    self.progress("editing", f"Reading edit activations {index}/{len(selected)}", index, len(selected))
            for name in targets:
                norm = torch.linalg.vector_norm(directions[name])
                if not torch.isfinite(norm) or norm <= 0:
                    raise ValueError("Selected edit examples already predict their reference tokens. Choose a different operation or seed")
                directions[name].div_(norm)
            return directions
        finally:
            for handle in handles:
                handle.remove()

    def edit(self, request: dict):
        self.ready()
        import torch
        if self.pending:
            raise ValueError("Accept or roll back the previous edit before starting another")
        targets, operation, strength, rank, seed = self.validate_proposal(request["proposal"])
        self.pending = {name: self.editable[name].detach().cpu().clone() for name in targets}
        started = time.monotonic()
        details = []
        try:
            with torch.inference_mode():
                activation_directions = self.activation_directions(targets, rank, seed) if operation == "activation_rank_one" else None
                for index, name in enumerate(targets):
                    self.emit({"type": "tensor_edit", "status": "started", "tensor": name, "operation": operation, "current": index, "total": len(targets)})
                    parameter = self.editable[name]
                    original = parameter.float().clone()
                    norm = float(torch.linalg.vector_norm(original).item())
                    if norm == 0 or not math.isfinite(norm):
                        raise ValueError("Editable tensors must have finite, nonzero norms")
                    generator = torch.Generator(device=parameter.device).manual_seed(seed + index)
                    if operation == "scale":
                        delta = original * strength
                    elif operation == "activation_rank_one":
                        delta = activation_directions[name] * (strength * norm)
                    elif operation == "low_rank":
                        if parameter.ndim != 2:
                            raise ValueError("Low-rank surgery requires a matrix")
                        left = torch.randn((parameter.shape[0], rank), device=parameter.device, dtype=torch.float32, generator=generator)
                        right = torch.randn((rank, parameter.shape[1]), device=parameter.device, dtype=torch.float32, generator=generator)
                        delta = left @ right
                        delta *= strength * norm / torch.linalg.vector_norm(delta)
                    else:
                        delta = torch.randn(parameter.shape, device=parameter.device, dtype=torch.float32, generator=generator)
                        delta *= strength * norm / torch.linalg.vector_norm(delta)
                    radius = norm * abs(strength)
                    parameter.copy_(original + delta)
                    measured = torch.linalg.vector_norm(parameter.float() - original)
                    # The bound applies to stored weights, including BF16 rounding.
                    for _ in range(24):
                        if measured <= radius:
                            break
                        margin = 0.99 if parameter.dtype in {torch.bfloat16, torch.float16} else 0.999999
                        delta *= min(margin, float(radius / measured) * margin)
                        parameter.copy_(original + delta)
                        measured = torch.linalg.vector_norm(parameter.float() - original)
                    if measured > radius:
                        raise ValueError("The tensor update could not satisfy its norm bound")
                    if not torch.isfinite(parameter).all():
                        raise ValueError("The edit produced nonfinite weights")
                    detail = {"name": name, "relative_norm": float(measured.item()) / norm, "shape": list(parameter.shape), "changed_elements": int(torch.count_nonzero(parameter.detach().cpu() != self.pending[name]).item())}
                    details.append(detail)
                    self.emit({"type": "tensor_edit", "status": "applied", "tensor": name, "operation": operation, "current": index + 1, "total": len(targets), "relative_norm": detail["relative_norm"], "changed_elements": detail["changed_elements"], "shape": list(detail["shape"])})
                    self.progress("editing", f"Applied {operation.replace('_', '-')} to tensor {index + 1}/{len(targets)}", index + 1, len(targets))
            self.pending_details = details
            return {"operation": operation, "strength": strength, "rank": rank, "seed": seed, "changed_tensors": details, "elapsed_seconds": time.monotonic() - started}
        except BaseException:
            self.rollback({})
            raise
        finally:
            self.model.requires_grad_(False)
            self.model.zero_grad(set_to_none=True)

    def accept(self, request: dict):
        self.ready()
        if not self.pending:
            raise ValueError("No edit is waiting to be accepted")
        self.changed.update(name for name, value in self.pending.items() if not __import__("torch").equal(value, self.editable[name].detach().cpu()))
        self.pending.clear()
        self.pending_details = []
        return {"accepted": True}

    def rollback(self, request: dict):
        self.ready()
        import torch
        with torch.no_grad():
            for name, original in self.pending.items():
                self.editable[name].copy_(original)
        self.pending.clear()
        self.pending_details = []
        self.model.requires_grad_(False)
        self.model.zero_grad(set_to_none=True)
        return {"restored": True}

    @staticmethod
    def manifest(directory: Path):
        return [{"path": str(path.relative_to(directory)), "sha256": file_hash(path), "bytes": path.stat().st_size} for path in sorted(directory.rglob("*")) if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".model", ".jinja", ".txt"} and ".cache" not in path.relative_to(directory).parts]

    def save(self, request: dict):
        self.ready()
        if self.pending:
            raise ValueError("Accept or roll back the current edit before saving")
        destination = Path(request["path"]).absolute()
        if destination.exists():
            raise ValueError(f"Checkpoint destination already exists: {destination}")
        if self.source_dir and destination.resolve().is_relative_to(self.source_dir):
            raise ValueError("Choose an output location outside the source checkpoint")
        self.progress("saving", "Saving selected checkpoint")
        destination.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(destination, safe_serialization=True)
        if self.processor is not None:
            self.processor.save_pretrained(destination)
        else:
            self.tokenizer.save_pretrained(destination)
        self.progress("saving", "Verifying saved files")
        files = self.manifest(destination)
        if self.source_dir:
            for original in self.source_files:
                source = self.source_dir / original["path"]
                if not source.is_file() or file_hash(source) != original["sha256"]:
                    raise ValueError(f"Source checkpoint changed during this run: {original['path']}")
        self.progress("saving", "Checkpoint saved", 1, 1)
        return {"path": str(destination), "files": files, "changed_tensors": sorted(self.changed), "source_unchanged": True if self.source_dir else None}

    def dispatch(self, request: dict):
        command = request.get("command")
        methods = {"init": self.initialise, "evaluate": self.evaluate, "health": self.health, "edit": self.edit, "accept": self.accept, "rollback": self.rollback, "save": self.save}
        if command == "shutdown":
            return {"stopped": True}
        if command not in methods:
            raise ValueError(f"Unknown worker command: {command}")
        return methods[command](request)


def main():
    # Keep third-party import and model diagnostics away from the protocol stream.
    protocol = sys.stdout

    def emit(value):
        protocol.write(json.dumps(value, allow_nan=False, ensure_ascii=False) + "\n")
        protocol.flush()

    engine = Engine(emit)
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Worker requests must be objects")
            with contextlib.redirect_stdout(sys.stderr):
                data = engine.dispatch(request)
            emit({"type": "result", "id": request.get("id"), "ok": True, "data": data})
            if request.get("command") == "shutdown":
                break
        except Exception as error:
            emit({"type": "result", "id": request.get("id") if isinstance(request, dict) else None, "ok": False, "error": str(error)})


if __name__ == "__main__":
    main()
