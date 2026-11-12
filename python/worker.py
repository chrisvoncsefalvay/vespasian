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
import uuid
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
    layers = next((getattr(text_config, name, None) for name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers") if isinstance(getattr(text_config, name, None), int) and not isinstance(getattr(text_config, name, None), bool)), None)
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError("The text model configuration has no architecture name")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("The text model configuration has no valid hidden-layer count")
    return {"architecture": architecture, "num_hidden_layers": layers}


def load_model_components(identifier: str, *, revision, device: str, dtype):
    """Load supported model components without changing run state or provenance."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Gemma3ForCausalLM, Qwen3_5ForConditionalGeneration

    kwargs = {"trust_remote_code": False}
    if revision:
        kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(identifier, **kwargs)
    if getattr(config, "is_encoder_decoder", False):
        raise ValueError("Encoder-decoder models are not supported; choose a compatible causal text-generation model")
    if getattr(config, "quantization_config", None):
        raise ValueError("Quantized checkpoints are not supported for direct weight editing")
    architecture_names = getattr(config, "architectures", None) or []
    if any(any(marker in name for marker in ("ForMaskedLM", "ForSequenceClassification", "ForTokenClassification", "ForQuestionAnswering")) for name in architecture_names if isinstance(name, str)):
        raise ValueError("The source checkpoint is not a causal text-generation model")
    encoder_families = {"bert", "roberta", "camembert", "xlm-roberta", "roformer", "electra", "rembert", "megatron-bert"}
    if config.model_type in encoder_families and not getattr(config, "is_decoder", False):
        raise ValueError("The source architecture is encoder-only, not causal text generation")
    processor = None
    if config.model_type == "qwen3_5":
        model_class = Qwen3_5ForConditionalGeneration
        processor = AutoProcessor.from_pretrained(identifier, **kwargs)
        tokenizer = processor.tokenizer
    elif config.model_type == "gemma3_text":
        model_class = Gemma3ForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(identifier, **kwargs)
    else:
        model_class = AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(identifier, **kwargs)
    if model_class is AutoModelForCausalLM:
        model, loading = model_class.from_pretrained(identifier, dtype=dtype, attn_implementation="eager", output_loading_info=True, **kwargs)
        parameter_names = set(dict(model.named_parameters()))
        if loading.get("mismatched_keys") or any(name in parameter_names for name in loading.get("missing_keys", [])):
            raise ValueError("The checkpoint did not fully initialise its causal model weights; no new random weights are allowed")
        model = model.to(device)
    else:
        model = model_class.from_pretrained(identifier, dtype=dtype, attn_implementation="eager", **kwargs).to(device)
    validate_model_contract(model)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer, processor


def validate_model_contract(model):
    """Check observed editing/generation interfaces, not architecture-name claims."""
    import torch
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("Only causal decoder text generation is supported")
    if getattr(model, "is_quantized", False) or getattr(model.config, "quantization_config", None):
        raise ValueError("Quantized checkpoints are not supported for direct editing")
    if not callable(getattr(model, "generate", None)) or not model.can_generate():
        raise ValueError("The checkpoint does not expose causal text generation")
    for role, module in (("input", model.get_input_embeddings()), ("output", model.get_output_embeddings())):
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2 or not weight.is_floating_point() or weight.is_meta:
            raise ValueError(f"The {role} embedding must expose an ordinary floating-point weight matrix")
    if model.get_input_embeddings().weight.shape[0] != model.get_output_embeddings().weight.shape[0]:
        raise ValueError("Input and output vocabulary sizes differ; this layout is unsupported")


class Engine:
    def __init__(self, emit=None, model_loader=None):
        self.emit = emit or (lambda event: None)
        self.model_loader = model_loader or load_model_components
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
        self.prompt_format = "auto"
        self.load_dtype = None
        self.seed = 42
        self.token_tables: dict[str, Any] = {}
        self.token_table_info: list[dict] = []
        self.inspected_token_ids: list[int] = []
        self.row_pending: dict[str, dict] = {}
        self.row_changed: dict[str, set[int]] = {}
        self.thinking_mode = "off"
        self.max_relative_norm = 0.1
        self.probe_state_id = uuid.uuid4().hex
        self.probe_cache: dict[str, dict] = {}
        self.probe_evaluations: dict[str, dict] = {}
        self.probe_candidate_evaluations: dict[str, dict] = {}
        self.probe_candidate_state_id = None
        self._probe_versions = None

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
        self.ensure_row_ids()
        self.dataset_mode = dataset_mode or "paired"
        self.encoded.clear()
        return dict(counts)

    def ensure_row_ids(self):
        counts = collections.Counter()
        for row in self.dataset:
            split = row["split"]
            if split != "test":
                row.setdefault("row_id", f"{split}:{counts[split]}")
                counts[split] += 1

    def reference_edits(self):
        return [row for row in self.dataset if row["split"] == "edit" and row.get("mode", "paired") != "prompts"]

    def preservation_rows(self):
        return [row for row in self.dataset if row["split"] == "preserve" and row.get("mode", "paired") != "prompts"]

    def residual_head_supported(self):
        import torch
        head = self.model.get_output_embeddings() if self.model is not None else None
        return type(head) is torch.nn.Linear and bool(self.reference_edits())

    def allowed_operations(self):
        operations = ["scale", "low_rank", "noise", "direction_projection"] if self.editable else []
        if self.editable and all(row.get("mode", "paired") != "prompts" for row in self.dataset if row["split"] == "edit"):
            operations.append("activation_rank_one")
            if self.reference_edits() and self.preservation_rows():
                operations.append("activation_preserve")
        if self.token_tables:
            operations.extend(["token_row_scale", "token_row_interpolate"])
            if self.residual_head_supported() and any("output_embedding" in item["roles"] for item in self.token_table_info):
                operations.append("token_row_residual")
        return operations

    def register_editable(self):
        """Expose bounded MLP matrices and canonical physical vocabulary tables."""
        import torch
        output_width = getattr(getattr(self.model.get_output_embeddings(), "weight", None), "shape", [0, 0])[1]
        self.editable = {name: value for name, value in self.model.named_parameters()
                         if name.endswith(".mlp.down_proj.weight") and value.ndim == 2
                         and value.is_floating_point() and not value.is_meta and min(value.shape) > 0
                         and value.shape[0] == output_width
                         and type(self.model.get_submodule(name.rsplit(".", 1)[0])) is torch.nn.Linear}
        self.token_tables, self.token_table_info = {}, []
        named = list(self.model.named_parameters(remove_duplicate=False))
        candidates = []
        for role, getter in (("input_embedding", "get_input_embeddings"), ("output_embedding", "get_output_embeddings")):
            module = getattr(self.model, getter, lambda: None)()
            value = getattr(module, "weight", None)
            if value is not None and value.ndim == 2 and value.is_floating_point() and min(value.shape) > 0:
                candidates.append((role, value))
        def identity(value):
            return (value.untyped_storage().data_ptr(), value.storage_offset(), tuple(value.shape), tuple(value.stride()))
        # Partial/reshaped aliases cannot safely share a token-row coordinate.
        unsafe = {value.untyped_storage().data_ptr() for _, value in candidates
                  for _, other in candidates if value.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()
                  and identity(value) != identity(other)}
        groups = {}
        for role, value in candidates:
            if value.untyped_storage().data_ptr() in unsafe:
                continue
            key = identity(value)
            if key not in groups:
                aliases = [name for name, parameter in named if identity(parameter) == key]
                if not aliases:
                    continue
                groups[key] = {"name": aliases[0], "aliases": aliases, "roles": [],
                               "shape": list(value.shape), "operations": ["token_row_scale", "token_row_interpolate"]}
                self.token_tables[aliases[0]] = value
            groups[key]["roles"].append(role)
        for metadata in groups.values():
            metadata["tied"] = len(metadata["roles"]) > 1
            if "output_embedding" in metadata["roles"] and self.residual_head_supported():
                metadata["operations"].append("token_row_residual")
            self.token_table_info.append(metadata)

    def editable_catalogue(self):
        mlp_operations = [op for op in self.allowed_operations() if not op.startswith("token_row_")]
        return ([{"name": name, "shape": list(value.shape), "operations": mlp_operations}
                 for name, value in self.editable.items()] + copy.deepcopy(self.token_table_info))

    def special_token_ids(self):
        ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        for owner in (self.tokenizer, self.model.config, getattr(self.model.config, "text_config", None), self.model.generation_config):
            for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
                value = getattr(owner, name, None)
                ids.update(value if isinstance(value, (list, tuple)) else [value])
        return {value for value in ids if isinstance(value, int) and not isinstance(value, bool)}

    def token_description(self, token_id):
        text = self.tokenizer.decode([token_id], skip_special_tokens=False)
        converter = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        piece = converter(token_id) if callable(converter) else text
        return {"id": token_id, "text": str(text), "piece": str(piece)}

    def inspect(self, request):
        """Read current model metadata/token rows; never inspect held-out examples."""
        self.ready()
        texts = request.get("token_texts", [])
        if not isinstance(texts, list) or len(texts) > 16 or any(not isinstance(text, str) or len(text) > 256 for text in texts):
            raise ValueError("Inspection accepts at most 16 token texts of 256 characters each")
        if self.pending or self.row_pending:
            raise ValueError("Accept or roll back the pending patch before inspection")
        sources = list(texts)
        for split in ("edit", "validation", "preserve"):
            for row in self.dataset:
                if row["split"] == split:
                    sources.extend([row.get("target", "")[:256], row.get("prompt", "")[:256]])
        special, selected = self.special_token_ids(), []
        rows = min((value.shape[0] for value in self.token_tables.values()), default=0)
        for text in sources:
            for token_id in self.tokenizer.encode(text, add_special_tokens=False):
                if isinstance(token_id, int) and not isinstance(token_id, bool) and 0 <= token_id < rows and token_id not in special and token_id not in selected:
                    selected.append(token_id)
                if len(selected) >= 128:
                    break
            if len(selected) >= 128:
                break
        self.inspected_token_ids = selected
        import torch
        candidates = []
        reference_ids = {i for row in self.reference_edits() for i in self.encode(row)[1]} - special
        for token_id in selected:
            norms = {name: float(torch.linalg.vector_norm(value[token_id].detach().float()).item())
                     for name, value in self.token_tables.items()}
            if all(math.isfinite(value) for value in norms.values()):
                candidates.append({**self.token_description(token_id), "row_norms": norms, "edit_reference": token_id in reference_ids})
        self.inspected_token_ids = [row["id"] for row in candidates]
        import probes
        return {"probe_modules": probes.catalogue(self), "max_relative_norm": self.max_relative_norm,
                "token_tables": copy.deepcopy(self.token_table_info), "token_candidates": candidates,
                "editable": self.editable_catalogue(), "allowed_operations": self.allowed_operations(),
                "dataset_mode": self.dataset_mode, "tokenizer_class": type(self.tokenizer).__name__,
                "policy": "Token pieces are tokenizer-specific. Test rows are sealed. Shared tables affect every listed role. Interpolation is bounded by each original row's norm.",
                "limits": {"max_selected_rows": 32, "max_candidates": 128, "max_relative_row_change": 0.1}}

    def forward(self, **kwargs):
        """Only send optional logit slicing when explicitly supported by this model."""
        import inspect
        parameters = inspect.signature(self.model.forward).parameters
        if "logits_to_keep" not in parameters:
            keep = kwargs.pop("logits_to_keep", None)
            if "num_logits_to_keep" in parameters and keep is not None:
                kwargs["num_logits_to_keep"] = keep
        return self.model(**kwargs)

    def prompt_tokens(self, text):
        if self.prompt_format == "plain":
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
        else:
            if not callable(getattr(self.tokenizer, "apply_chat_template", None)) or (hasattr(self.tokenizer, "chat_template") and not self.tokenizer.chat_template):
                raise ValueError("This tokenizer has no chat template. Explicitly choose prompt_format=plain for completion-style prompts, or choose a chat model")
            options = {}
            if self.thinking_mode != "auto":
                if self.thinking_mode == "on":
                    template = getattr(self.tokenizer, "chat_template", None)
                    getter = getattr(self.tokenizer, "get_chat_template", None)
                    if callable(getter):
                        template = getter()
                    if not isinstance(template, str) or "enable_thinking" not in template:
                        raise ValueError("This chat template does not expose enable_thinking; thinking_mode=on is unavailable")
                options["enable_thinking"] = self.thinking_mode == "on"
            tokens = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, **options)
            if hasattr(tokens, "input_ids"):
                tokens = tokens.input_ids
            if isinstance(tokens, dict):
                tokens = tokens["input_ids"]
        if not isinstance(tokens, list) or not tokens or any(isinstance(i, bool) or not isinstance(i, int) for i in tokens):
            raise ValueError("Prompt formatting did not produce valid token IDs")
        return tokens

    def generation_policy(self):
        if self.dataset_mode == "corpus":
            return {"conditioning": "unconditional", "do_sample": True, "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seed_policy": "run seed plus stable reference hash", "start_token_id": self.start_token()}
        return {"conditioning": "plain_prompt" if self.prompt_format == "plain" else "chat_prompt", "do_sample": False, "thinking_mode": self.thinking_mode}

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

        self.prompt_format = request.get("prompt_format", "auto")
        if self.prompt_format not in {"auto", "chat", "plain"}:
            raise ValueError("prompt_format must be auto, chat or plain")
        self.thinking_mode = request.get("thinking_mode", "off")
        if self.thinking_mode not in {"off", "on", "auto"}:
            raise ValueError("thinking_mode must be off, on or auto")
        if self.thinking_mode == "on" and self.prompt_format == "plain":
            raise ValueError("thinking_mode=on requires a compatible chat template")
        self.max_relative_norm = bounded_float(request.get("max_relative_norm", 0.1), "max_relative_norm", 0.000001, 1.0)
        self.max_seq_length = bounded_int(request.get("max_seq_length", 512), "max_seq_length", 16, 131072)
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
        if self.thinking_mode == "on" and self.dataset_mode == "corpus":
            raise ValueError("thinking_mode=on requires chat prompts; unconditional corpus generation has no chat template")
        identifier = request["model"]
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Model identifier must contain text")
        revision = request.get("revision")
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
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.load_dtype = str(dtype).removeprefix("torch.")
        self.model, self.tokenizer, self.processor = self.model_loader(
            identifier, revision=resolved_revision, device=self.device, dtype=dtype,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        structure = model_structure(self.model.config)
        self.register_editable()
        if not self.editable and not self.token_tables:
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
        return {"model": identifier, "dataset_mode": self.dataset_mode, "allowed_operations": self.allowed_operations(), "generation": self.generation_policy(), "revision": resolved_revision, "device": self.device, "dtype": str(dtype).replace("torch.", ""), **structure, "editable": self.editable_catalogue(), "token_tables": copy.deepcopy(self.token_table_info), "counts": counts, "source": {"path": str(self.source_dir) if self.source_dir else None, "files": self.source_files}, "seed": seed, "prompt_format": self.prompt_format, "thinking_mode": self.thinking_mode, "max_relative_norm": self.max_relative_norm, "runtime": {"torch": torch.__version__, "transformers": __import__("transformers").__version__}}

    def context_limit(self):
        # Model families differ in rotary scaling and effective context limits.
        # Respect the explicit run/platform bound; unsupported lengths fail in
        # the model rather than guessing support from one config attribute.
        return self.max_seq_length

    def refresh_probe_versions(self):
        import probes
        probes.refresh(self)

    def probe_activations(self, request):
        import probes
        return probes.measure(self, request)

    def encode(self, row: dict) -> tuple[list[int], list[int]]:
        mode = row.get("mode", "paired")
        key = (mode, row["prompt"], row["target"])
        if key in self.encoded:
            return self.encoded[key]
        if mode == "corpus":
            prompt = [self.start_token()]
        else:
            prompt = self.prompt_tokens(row["prompt"])
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
        if len(prompt) + len(target) > self.context_limit():
            raise ValueError(f"Example exceeds --max-seq-length ({len(prompt) + len(target)} tokens): {row['prompt'][:80]}")
        self.encoded[key] = (prompt, target)
        return prompt, target

    def loss(self, row: dict):
        import torch
        prompt, target = self.encode(row)
        if not target:
            raise ValueError("Reference loss is unavailable for prompt-only examples")
        tokens = torch.tensor([prompt + target], dtype=torch.long, device=self.device)
        output = self.forward(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False, logits_to_keep=len(target) + 1)
        logits = output.logits[:, -(len(target) + 1):-1, :].float()
        expected = torch.tensor(target, dtype=torch.long, device=self.device)
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), expected)

    def evaluate(self, request: dict):
        self.ready()
        import torch
        splits = request.get("splits", ["validation", "preserve"])
        if not isinstance(splits, list) or not splits or len(set(splits)) != len(splits) or any(split not in {"edit", "validation", "preserve", "test"} for split in splits):
            raise ValueError("Evaluation splits must be a nonempty list of distinct split names")
        maximum = bounded_int(request.get("max_new_tokens", 64), "max_new_tokens", 1, 32768)
        self.ensure_row_ids()
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
                if len(prompt) + maximum > self.context_limit():
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
                if row["split"] != "test":
                    from behavior import annotate_samples
                    sample = annotate_samples([sample], row_ids=[row["row_id"]])[0]
                samples.append(sample)
                self.answer("evaluating", label, turn, index, len(rows), sample)
                self.progress("evaluating", f"{label} · {row['split']} {index}/{len(rows)}", index, len(rows), label=label, turn=turn)
        for totals in metrics.values():
            for key in ("exact_match", "token_f1", "loss"):
                if key in totals:
                    totals[key] /= totals["count"]
        self.progress("evaluating", "Evaluation complete", len(rows), len(rows), label=label, turn=turn)
        from behavior import summarize_evaluation
        import probes
        probes.cache_evaluation(self, samples)
        return {"metrics": metrics, "samples": samples, "behavior": summarize_evaluation(samples), "elapsed_seconds": time.monotonic() - started}

    def load_health(self, path: str):
        from health import validate_manifest
        with Path(path).open(encoding="utf-8") as source:
            self.health_set = validate_manifest(json.load(source))
        return {"checks": self.health_set["checks"], "rows_sha256": self.health_set["rows_sha256"]}

    def health_prompt(self, text: str) -> list[int]:
        tokens = self.prompt_tokens(text)
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
            output = self.forward(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=len(continuation) + 1)
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
        maximum = bounded_int(request.get("max_new_tokens", 64), "max_new_tokens", 1, 32768)
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
                    if len(prompt) + maximum > self.context_limit():
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
        explanatory = {"strategy", "plan", "reflection"}
        selectors = {"token_ids", "source_token_ids", "direction_id", "columns"}
        if not isinstance(proposal, dict) or not required.issubset(proposal) or set(proposal) - required - explanatory - selectors:
            raise ValueError("Edit proposal must contain targets, operation, strength, rank, seed and rationale; only strategy, plan and reflection may be added")
        # Historical trajectories predate these public decision summaries. Keep
        # them replayable while validating any new metadata before tensor work.
        for field in explanatory & proposal.keys():
            if not isinstance(proposal[field], str) or not proposal[field].strip():
                raise ValueError(f"Edit proposal {field} must be a nonempty string")
        targets = proposal["targets"]
        if not isinstance(targets, list) or not 1 <= len(targets) <= 4 or any(not isinstance(name, str) for name in targets) or len(set(targets)) != len(targets):
            raise ValueError("Choose one to four distinct target tensors")
        operation = proposal["operation"]
        row_operation = operation in {"token_row_scale", "token_row_interpolate", "token_row_residual"}
        available = self.token_tables if row_operation else self.editable
        if any(name not in available for name in targets):
            raise ValueError("Edit proposal names a tensor outside the editable set for this operation")
        if row_operation:
            if len(targets) != 1:
                raise ValueError("Token-row operations select exactly one canonical table")
            self.validate_row_selectors(proposal, self.token_tables[targets[0]])
            metadata = next(item for item in self.token_table_info if item["name"] == targets[0])
            if operation not in metadata["operations"]:
                raise ValueError("Selected token table does not support this operation")
            if operation == "token_row_residual":
                reference_ids = {i for row in self.reference_edits() for i in self.encode(row)[1]}
                if any(i not in reference_ids for i in proposal["token_ids"]):
                    raise ValueError("Residual output rows must occur in edit reference tokens")
        elif proposal.get("token_ids", []) != [] or proposal.get("source_token_ids", []) != []:
            raise ValueError("MLP operations require empty token selectors")
        if not isinstance(proposal["rationale"], str) or not proposal["rationale"].strip():
            raise ValueError("Edit proposal requires a rationale")
        operation = proposal["operation"]
        if operation not in {"scale", "low_rank", "noise", "activation_rank_one", "activation_preserve", "token_row_scale", "token_row_interpolate", "token_row_residual", "direction_projection"}:
            raise ValueError("Unknown tensor operation")
        if operation not in self.allowed_operations():
            raise ValueError("Activation surgery requires reference text. Choose scale, low_rank or noise for prompt-only data")
        projection = operation == "direction_projection"
        if projection:
            if len(targets) != 1 or proposal["rank"] != 1:
                raise ValueError("Direction projection selects exactly one tensor and rank one")
            import probes
            probes.selected_direction(self, proposal)
        elif proposal.get("direction_id") not in (None, "") or proposal.get("columns", []) != []:
            raise ValueError("Only direction_projection accepts direction_id and columns")
        strength = bounded_float(proposal["strength"], "strength", 0.000001 if projection else -0.1, 16.0 if projection else 0.1)
        if abs(strength) < 1e-6:
            raise ValueError("Strength must have an absolute value of at least 0.000001")
        if operation in {"token_row_interpolate", "token_row_residual", "activation_preserve"} and strength <= 0:
            raise ValueError("This measured-direction operation requires positive strength")
        return targets, operation, strength, bounded_int(proposal["rank"], "rank", 1, 16), bounded_int(proposal["seed"], "seed", 0, 2147483647)

    def validate_row_selectors(self, proposal, table):
        ids, sources = proposal.get("token_ids"), proposal.get("source_token_ids", [])
        special = self.special_token_ids()
        def valid(values, distinct):
            return (isinstance(values, list) and 1 <= len(values) <= 32
                    and all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < table.shape[0]
                            and i not in special and i in self.inspected_token_ids for i in values)
                    and (not distinct or len(set(values)) == len(values)))
        if not valid(ids, True):
            raise ValueError("Select 1 to 32 distinct non-special token IDs from the current inspection")
        if proposal["operation"] == "token_row_interpolate":
            if not valid(sources, False) or len(sources) != len(ids) or any(a == b for a, b in zip(ids, sources)):
                raise ValueError("Interpolation requires one inspected source token per target, different from its target")
        elif sources != []:
            raise ValueError("This token-row operation requires empty source_token_ids")

    def edit_token_rows(self, proposal, strength, rank, seed):
        import torch
        started = time.monotonic()
        name, ids = proposal["targets"][0], proposal["token_ids"]
        parameter = self.token_tables[name]
        indices = torch.tensor(ids, device=parameter.device, dtype=torch.long)
        original = parameter.index_select(0, indices).detach().clone()
        self.row_pending = {name: {"ids": list(ids), "values": original.cpu().clone()}}
        metadata = next(item for item in self.token_table_info if item["name"] == name)
        operation = proposal["operation"]
        try:
            self.emit({"type": "tensor_edit", "status": "started", "tensor": name, "operation": operation,
                       "token_ids": ids, "aliases": metadata["aliases"], "roles": metadata["roles"], "tied": metadata["tied"], "current": 0, "total": len(ids)})
            with torch.inference_mode():
                # Capture all source rows before any destination is written: swaps
                # and source/destination overlaps are simultaneous, not cascading.
                sources = proposal.get("source_token_ids", [])
                source = parameter.index_select(0, torch.tensor(sources, device=parameter.device)).float().clone() if sources else None
                residual, evidence = self.output_row_directions(name, ids, rank, seed) if operation == "token_row_residual" else (None, None)
                stored = original.clone()
                details = []
                for index, token_id in enumerate(ids):
                    base = original[index].float()
                    norm = float(torch.linalg.vector_norm(base).item())
                    if not math.isfinite(norm) or norm <= 0:
                        raise ValueError("Token rows must have finite nonzero norms")
                    direction = residual[index] if residual is not None else (base if source is None else source[index] - base)
                    if not torch.isfinite(direction).all():
                        raise ValueError("Token-row direction must be finite")
                    radius = norm * abs(strength)
                    fraction = strength
                    direction_norm = float(torch.linalg.vector_norm(direction).item())
                    if residual is not None:
                        fraction = radius / direction_norm
                    elif source is not None and direction_norm > 0:
                        fraction = min(strength, radius / direction_norm)
                    delta = direction * fraction
                    candidate = (base + delta).to(parameter.dtype)
                    measured = float(torch.linalg.vector_norm(candidate.float() - base).item())
                    for _ in range(32):
                        if measured <= radius:
                            break
                        margin = 0.99 if parameter.dtype in {torch.bfloat16, torch.float16} else 0.999999
                        multiplier = min(margin, radius / measured * margin)
                        fraction *= multiplier
                        delta = direction * fraction
                        candidate = (base + delta).to(parameter.dtype)
                        measured = float(torch.linalg.vector_norm(candidate.float() - base).item())
                    if measured > radius or not torch.isfinite(candidate).all():
                        raise ValueError("Stored token-row update exceeds its norm bound")
                    stored[index] = candidate
                    description = self.token_description(token_id)
                    details.append({"token_id": token_id, "text": description["text"], "piece": description["piece"],
                                    "source_token_id": sources[index] if sources else None,
                                    "source_text": self.token_description(sources[index])["text"] if sources else None,
                                    "effective_fraction": fraction, "relative_norm": measured / norm,
                                    "changed_elements": int(torch.count_nonzero(candidate != original[index]).item())})
                if operation == "token_row_residual" and not torch.count_nonzero(stored != original).item():
                    raise ValueError("Measured residual rounds to an empty stored update")
                parameter.index_copy_(0, indices, stored)
                aggregate_norm = float(torch.linalg.vector_norm(original.float()).item())
                aggregate_delta = float(torch.linalg.vector_norm(stored.float() - original.float()).item())
                detail = {**copy.deepcopy(metadata), "relative_norm": aggregate_delta / aggregate_norm,
                          "changed_elements": sum(row["changed_elements"] for row in details),
                          "token_ids": list(ids), "source_token_ids": list(sources), "rows": details}
                if evidence is not None:
                    detail["direction_evidence"] = evidence
                self.pending_details = [detail]
                self.emit({"type": "tensor_edit", "status": "applied", "tensor": name, "operation": operation,
                           "current": len(ids), "total": len(ids), **detail})
                return {"operation": operation, "strength": strength, "rank": rank, "seed": seed,
                        "changed_tensors": [detail], "elapsed_seconds": time.monotonic() - started}
        except BaseException:
            self.rollback({})
            raise

    def output_row_directions(self, name, ids, rank, seed):
        """Direct output-row residuals from actual head input, without autograd."""
        import torch
        if not self.residual_head_supported():
            raise ValueError("Residual rows require an ordinary linear output head and references")
        head = self.model.get_output_embeddings()
        rows = self.reference_edits()
        selected = random.Random(seed).sample(rows, min(rank, len(rows)))
        directions = torch.zeros((len(ids), head.weight.shape[1]), device=head.weight.device, dtype=torch.float32)
        captured = []
        counts = {i: 0 for i in ids}
        positions = 0
        def capture(module, args):
            captured[:] = [args[0].detach().float().reshape(-1, args[0].shape[-1]).clone()]
        handle = head.register_forward_pre_hook(capture)
        try:
            with torch.inference_mode():
                for row in selected:
                    prompt, target = self.encode(row)
                    inputs = torch.tensor([prompt + target], device=self.device)
                    captured.clear()
                    result = self.forward(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=len(target) + 1)
                    if len(captured) != 1 or captured[0].shape[0] < len(target) + 1:
                        raise ValueError("Output-head activation layout is unsupported")
                    keys = captured[0][-(len(target) + 1):-1]
                    logits = result.logits[0, -(len(target) + 1):-1].float()
                    if keys.shape != (len(target), head.weight.shape[1]) or logits.shape != (len(target), head.weight.shape[0]):
                        raise ValueError("Output-head activation/logit dimensions do not match")
                    # Bound computation to 64 teacher-forced prediction positions per example.
                    keys, logits, target = keys[:64], logits[:64], target[:64]
                    probabilities = logits.softmax(dim=-1)[:, ids]
                    desired = torch.tensor(target, device=keys.device)
                    contrast = (desired[:, None] == torch.tensor(ids, device=keys.device)[None, :]).float() - probabilities
                    directions.add_(contrast.T @ (keys / keys.square().sum(-1, keepdim=True).clamp_min(1e-12)))
                    positions += len(target)
                    for i in ids:
                        counts[i] += target.count(i)
                if any(counts[i] == 0 for i in ids):
                    raise ValueError("Selected residual rows need observed reference occurrences within sampled edit positions; choose IDs, rank or seed accordingly")
                norms = directions.norm(dim=-1)
                if not torch.isfinite(directions).all() or bool((norms <= 1e-12).any()):
                    raise ValueError("Measured output-row residual direction is empty or nonfinite")
                directions /= norms[:, None]
                return directions, {"method": "target_minus_probability_head_input", "examples": len(selected), "positions": positions,
                                    "reference_occurrences": {str(i): counts[i] for i in ids}, "autograd": False,
                                    "limitation": "Teacher-forced local residual, not a causal or preservation guarantee; tied tables also change input embeddings."}
        finally:
            handle.remove()

    def activation_directions(self, targets: list[str], rank: int, seed: int, preserve=False):
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
                bases = {}
                if preserve:
                    protected = {name: [] for name in targets}
                    for row in random.Random(seed).sample(self.preservation_rows(), min(rank, len(self.preservation_rows()))):
                        prompt, target = self.encode(row)
                        inputs = torch.tensor([prompt + target], dtype=torch.long, device=self.device)
                        captures.clear()
                        self.forward(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=1)
                        for name in targets:
                            keys = captures[name][len(prompt) - 1:len(prompt) + len(target) - 1][:64]
                            protected[name].append(keys)
                    for name, parts in protected.items():
                        if not parts:
                            raise ValueError("Preservation-aware edits require preservation examples")
                        matrix = torch.cat(parts, dim=0)
                        if not torch.isfinite(matrix).all():
                            raise ValueError("Preservation keys must be finite")
                        _, singular, vectors = torch.linalg.svd(matrix, full_matrices=False)
                        tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular.max()
                        bases[name] = vectors[singular > tolerance]
                    self.direction_evidence = {"method": "contrastive_preserve_key_projection", "preserve_examples": min(rank, len(self.preservation_rows())),
                                               "preserve_key_ranks": {name: len(basis) for name, basis in bases.items()},
                                               "limitation": "Projection only protects the sampled finite key span before dtype rounding, not model-wide behaviour."}
                for index, row in enumerate(selected, 1):
                    prompt, target = self.encode(row)
                    inputs = torch.tensor([prompt + target], dtype=torch.long, device=self.device)
                    captures.clear()
                    result = self.forward(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=len(target) + 1)
                    predicted = result.logits[0, -(len(target) + 1):-1].argmax(dim=-1)
                    desired = torch.tensor(target, dtype=torch.long, device=self.device)
                    residual = output_embeddings[desired].float() - output_embeddings[predicted].float()
                    for name in targets:
                        # Position p-1 predicts the first answer token; subsequent
                        # teacher-forced positions predict its suffix and terminator.
                        keys = captures[name][len(prompt) - 1:len(prompt) + len(target) - 1]
                        if residual.shape[1] != directions[name].shape[0] or keys.shape != (len(target), directions[name].shape[1]):
                            raise ValueError("Activation and residual dimensions do not match the selected tensor")
                        if preserve:
                            basis = bases[name]
                            original_norm = keys.norm()
                            keys = keys - (keys @ basis.T) @ basis
                            if keys.norm() <= original_norm * 1e-6:
                                continue
                        directions[name].add_(residual.T @ (keys / keys.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)))
                    self.progress("editing", f"Reading edit activations {index}/{len(selected)}", index, len(selected))
            for name in targets:
                norm = torch.linalg.vector_norm(directions[name])
                if not torch.isfinite(norm) or norm <= 0:
                    raise ValueError("No usable measured activation direction remains after reference comparison or preservation projection. Choose a different operation or seed")
                directions[name].div_(norm)
            return directions
        finally:
            for handle in handles:
                handle.remove()

    def edit(self, request: dict):
        self.ready()
        import torch
        if self.pending or self.row_pending:
            raise ValueError("Accept or roll back the previous edit before starting another")
        targets, operation, strength, rank, seed = self.validate_proposal(request["proposal"])
        self.probe_candidate_state_id = uuid.uuid4().hex
        self.probe_candidate_evaluations.clear()
        if operation.startswith("token_row_"):
            return self.edit_token_rows(request["proposal"], strength, rank, seed)
        import probes
        selected = probes.selected_direction(self, request["proposal"]) if operation == "direction_projection" else None
        self.pending = {name: self.editable[name].detach().cpu().clone() for name in targets}
        started = time.monotonic()
        details = []
        try:
            with torch.inference_mode():
                activation_directions = self.activation_directions(targets, rank, seed, preserve=operation == "activation_preserve") if operation in {"activation_rank_one", "activation_preserve"} else None
                for index, name in enumerate(targets):
                    self.emit({"type": "tensor_edit", "status": "started", "tensor": name, "operation": operation, "current": index, "total": len(targets)})
                    parameter = self.editable[name]
                    original = parameter.float().clone()
                    norm = float(torch.linalg.vector_norm(original).item())
                    if norm == 0 or not math.isfinite(norm):
                        raise ValueError("Editable tensors must have finite, nonzero norms")
                    generator = torch.Generator(device=parameter.device).manual_seed(seed + index)
                    if operation == "direction_projection":
                        direction, columns, evidence = selected
                        column_ids = torch.tensor(columns, device=parameter.device)
                        delta = torch.zeros_like(original)
                        selected_values = original.index_select(1, column_ids)
                        delta[:, column_ids] = -strength * torch.outer(direction, direction @ selected_values)
                    elif operation == "scale":
                        delta = original * strength
                    elif operation in {"activation_rank_one", "activation_preserve"}:
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
                    radius = norm * (self.max_relative_norm if operation == "direction_projection" else abs(strength))
                    parameter.copy_(original + delta)
                    measured = torch.linalg.vector_norm(parameter.float() - original)
                    if operation == "direction_projection" and measured > radius:
                        raise ValueError("Direction projection exceeds the configured max_relative_norm; select fewer columns or a smaller strength")
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
                    if operation in {"activation_preserve", "direction_projection"} and detail["changed_elements"] == 0:
                        raise ValueError("Preservation-aware direction rounds to an empty stored update")
                    if operation == "direction_projection":
                        detail.update({"direction_id": request["proposal"]["direction_id"], "columns": list(columns),
                                       "max_relative_norm": self.max_relative_norm, "direction_evidence": evidence})
                    if operation == "activation_preserve":
                        detail["direction_evidence"] = copy.deepcopy(self.direction_evidence)
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
        if not self.pending and not self.row_pending:
            raise ValueError("No edit is waiting to be accepted")
        for name, pending in self.row_pending.items():
            changed = self.row_changed.setdefault(name, set())
            for index, token_id in enumerate(pending["ids"]):
                if not __import__("torch").equal(self.token_tables[name][token_id].detach().cpu(), pending["values"][index]):
                    changed.add(token_id)
            if not changed:
                self.row_changed.pop(name)
        self.row_pending.clear()
        self.changed.update(name for name, value in self.pending.items() if not __import__("torch").equal(value, self.editable[name].detach().cpu()))
        self.pending.clear()
        self.pending_details = []
        import probes
        probes.accept_candidate(self)
        return {"accepted": True}

    def rollback(self, request: dict):
        self.ready()
        import torch
        with torch.no_grad():
            for name, original in self.pending.items():
                self.editable[name].copy_(original)
            for name, pending in self.row_pending.items():
                parameter = self.token_tables[name]
                parameter.index_copy_(0, torch.tensor(pending["ids"], device=parameter.device), pending["values"].to(parameter.device))
        self.row_pending.clear()
        self.pending.clear()
        self.pending_details = []
        self.model.requires_grad_(False)
        self.model.zero_grad(set_to_none=True)
        self.probe_candidate_evaluations.clear()
        self.probe_candidate_state_id = None
        self.refresh_probe_versions()
        return {"restored": True}

    @staticmethod
    def manifest(directory: Path):
        return [{"path": str(path.relative_to(directory)), "sha256": file_hash(path), "bytes": path.stat().st_size} for path in sorted(directory.rglob("*")) if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".model", ".jinja", ".txt"} and ".cache" not in path.relative_to(directory).parts]

    def save(self, request: dict):
        self.ready()
        if self.pending or self.row_pending:
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
        return {"path": str(destination), "files": files, "changed_tensors": sorted(self.changed | set(self.row_changed)), "changed_token_rows": {name: sorted(ids) for name, ids in self.row_changed.items()}, "source_unchanged": True if self.source_dir else None}

    def patch(self, request: dict):
        """Durable local patch commands share the same engine across both surfaces."""
        import patches
        directory = request["directory"]
        command = request["command"]
        if command == "patch_init":
            return patches.initialize(self, directory, request.get("base_model"))
        if command == "patch_candidate":
            return patches.record_candidate(self, directory, request["turn"])
        if command == "patch_finalize":
            return patches.finalize(directory, request["turn"], request["accepted"])
        if command == "patch_restore":
            result = patches.restore(self, directory, request.get("which", "best"))
            import probes
            probes.invalidate(self)
            return result
        if command == "patch_export":
            return patches.export_bundle(directory, request["path"])
        raise ValueError("Unknown patch command")

    def dispatch(self, request: dict):
        command = request.get("command")
        if command in {"patch_init", "patch_candidate", "patch_finalize", "patch_restore", "patch_export"}:
            return self.patch(request)
        methods = {"init": self.initialise, "inspect": self.inspect, "probe_activations": self.probe_activations, "evaluate": self.evaluate, "health": self.health, "edit": self.edit, "accept": self.accept, "rollback": self.rollback, "save": self.save}
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
