"""Isolated, resumable Vespasian tensor operations for ZeroGPU.

Only strings and ordinary protocol dictionaries cross the GPU process boundary.
The startup models are read-only templates; every allocation owns its model copy.
"""
from __future__ import annotations

import spaces  # Must precede torch so ZeroGPU can register the startup weights.

import contextlib
import copy
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
from types import MappingProxyType
import uuid

import torch
from safetensors.torch import load_file, save_file

_ROOT = Path(__file__).resolve().parent.parent.parent
for _worker_dir in (Path(__file__).parent / "core", Path(__file__).parent / "python", _ROOT / "python"):
    if (_worker_dir / "worker.py").is_file():
        sys.path.insert(0, str(_worker_dir))
        break
from worker import Engine, load_model_components
from limits import space_limits
from behavior import summarize_evaluation


SUPPORTED_MODELS = ("Qwen/Qwen3.5-0.8B", "google/gemma-3-1b-it")  # Verified shortlist, not a compatibility whitelist.
TESTED_MODELS = {
    "Qwen/Qwen3.5-0.8B": {"revision": "2fc06364715b967f1860aea9cf38778875588b17", "checks": "CUDA token-row editing, rollback and reload"},
    "google/gemma-3-1b-it": {"revision": "dcc83ea841ab6100d6b47a070329e1ba4cf78752", "checks": "CUDA low-rank editing, rollback and reload; newer primitives not model-specifically verified"},
}

MAX_EXAMPLES = int(os.environ.get("VESPASIAN_SPACE_MAX_EXAMPLES", "16"))
MAX_TOTAL_EXAMPLES = int(os.environ.get("VESPASIAN_SPACE_MAX_TOTAL_EXAMPLES", "256"))
_LIMITS = space_limits()
MAX_NEW_TOKENS = _LIMITS["max_new_tokens"]
MAX_SEQ_LENGTH = _LIMITS["max_seq_length"]
GPU_SIZE = "large"
GPU_QUOTA_MULTIPLIER = 1  # spaces.zero.client: xlarge is 2x, large is 1x.
_FORMAT = 1
_GENERATION = re.compile(r"[0-9a-f]{32}\Z")
_COMMANDS = {"init", "inspect", "probe_activations", "evaluate", "health", "edit", "accept", "rollback", "save", "shutdown", "patch_init", "patch_candidate", "patch_finalize", "patch_restore", "patch_export"}
_ENGINE_FIELDS = (
    "dataset", "pending_details", "original_hashes", "source_files", "max_seq_length",
    "device", "health_set", "dataset_mode", "seed", "inspected_token_ids", "prompt_format",
    "thinking_mode", "max_relative_norm", "probe_state_id", "probe_cache", "probe_evaluations",
    "probe_candidate_state_id", "probe_candidate_evaluations",
)


@dataclass(frozen=True)
class ModelTemplate:
    identifier: str
    revision: str | None
    model: object = None
    tokenizer: object = None
    processor: object = None
    error: str | None = None


def _preload_models():
    identifiers = tuple(dict.fromkeys(part.strip() for part in os.environ.get(
        "VESPASIAN_SPACE_MODELS", SUPPORTED_MODELS[0],
    ).split(",") if part.strip()))
    if not identifiers or len(identifiers) > 8 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", identifier) or ".." in identifier for identifier in identifiers):
        raise ValueError("VESPASIAN_SPACE_MODELS must contain one to eight Hub repository IDs configured by the operator")
    revisions = json.loads(os.environ.get("VESPASIAN_SPACE_MODEL_REVISIONS", "{}"))
    if not isinstance(revisions, dict) or any(key not in identifiers or not isinstance(value, str) for key, value in revisions.items()):
        raise ValueError("VESPASIAN_SPACE_MODEL_REVISIONS must map configured models to revisions")
    parameter_limit = int(os.environ.get("VESPASIAN_SPACE_MAX_MODEL_PARAMETERS", "2000000000"))
    if parameter_limit < 1:
        raise ValueError("VESPASIAN_SPACE_MAX_MODEL_PARAMETERS must be positive")
    templates = {}
    for identifier in identifiers:
        if os.environ.get("VESPASIAN_SPACE_SKIP_MODEL_LOAD") == "1":
            templates[identifier] = ModelTemplate(identifier, None, error="Model loading is disabled")
            continue
        try:
            from huggingface_hub import hf_hub_download
            config_path = Path(hf_hub_download(identifier, "config.json", revision=revisions.get(identifier)))
            revision = config_path.parent.name
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            model, tokenizer, processor = load_model_components(identifier, revision=revision, device=device, dtype=dtype)
            if sum(parameter.numel() for parameter in model.parameters()) > parameter_limit:
                raise ValueError("Loaded model exceeds the operator-configured parameter limit")
            templates[identifier] = ModelTemplate(identifier, revision, model, tokenizer, processor)
        except Exception as error:
            print(f"Startup model {identifier} unavailable: {error}", file=sys.stderr)
            templates[identifier] = ModelTemplate(identifier, None, error=f"Model could not be loaded ({type(error).__name__}). Check repository access and the Space logs.")
    if os.environ.get("VESPASIAN_SPACE_SKIP_MODEL_LOAD") != "1" and not any(item.model is not None for item in templates.values()):
        raise RuntimeError("No configured model could be loaded. Check model access and the startup logs.")
    return MappingProxyType(templates)


# Startup CUDA registration is intentional; no inference runs outside @spaces.GPU.
MODEL_TEMPLATES = _preload_models()


def model_availability():
    """Return startup metadata without allocating a GPU or reading model tensors."""
    return [
        {"id": item.identifier, "revision": item.revision, "available": item.model is not None, "tested_reference": TESTED_MODELS.get(item.identifier),
         **({"error": item.error} if item.error else {})}
        for item in MODEL_TEMPLATES.values()
    ]


def model_options():
    """Configured startup options; availability is validated separately, never downloaded per session."""
    return [(item.identifier, item.identifier) for item in MODEL_TEMPLATES.values()]


def _model_loader(identifier, *, revision, device, dtype):
    # A saved checkpoint must be freshly loaded, including its own tokenizer and
    # configuration. Its evaluation is the controller's export verification.
    if Path(identifier).is_dir():
        return load_model_components(identifier, revision=revision, device=device, dtype=dtype)
    template = MODEL_TEMPLATES.get(identifier)
    if template is None or template.model is None:
        raise ValueError(f"Model is unavailable in this Space: {identifier}")
    if revision != template.revision:
        raise ValueError("Model revision differs from the Space's pinned startup revision")
    if template.model.dtype != dtype:
        # Architecture loaders retain selected FP32 buffers and parameters.
        # A blanket .to(dtype=...) would change their numerical behaviour.
        return load_model_components(identifier, revision=revision, device=device, dtype=dtype)
    model, tokenizer, processor = copy.deepcopy((template.model, template.tokenizer, template.processor))
    model.to(device=device)
    model.eval().requires_grad_(False)
    return model, tokenizer, processor


def _plain(value):
    """Validate the exact transport contract, including finite numeric values."""
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


@contextlib.contextmanager
def _state_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This model session already has an active operation") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read_metadata(directory):
    pointer = directory / "CURRENT"
    if not pointer.exists():
        return None, None
    generation = json.loads(pointer.read_text())["generation"]
    if not isinstance(generation, str) or not _GENERATION.fullmatch(generation):
        raise ValueError("Invalid model-state generation")
    location = directory / "generations" / generation
    metadata = json.loads((location / "metadata.json").read_text())
    if metadata.get("format") != _FORMAT:
        raise ValueError("Unsupported model-state format")
    return metadata, location


def _read_snapshot(directory):
    metadata, location = _read_metadata(directory)
    return metadata, load_file(str(location / "tensors.safetensors"), device="cpu") if location else {}


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())


def _commit_snapshot(directory, metadata, tensors, cancelled, reuse_tensors=None):
    """Write a complete generation before atomically advancing the state pointer."""
    generation = uuid.uuid4().hex
    generations = directory / "generations"
    generations.mkdir(exist_ok=True)
    location = generations / generation
    location.mkdir()
    tensor_path = location / "tensors.safetensors"
    if reuse_tensors is not None:
        os.link(reuse_tensors, tensor_path)
    else:
        save_file({key: value.detach().cpu().contiguous().clone() for key, value in tensors.items()}, str(tensor_path))
        with tensor_path.open("rb") as handle:
            os.fsync(handle.fileno())
    _write_json(location / "metadata.json", metadata)
    _fsync_directory(location)
    _fsync_directory(generations)
    pointer = directory / (".CURRENT-" + generation)
    _write_json(pointer, {"generation": generation})
    if cancelled():
        raise InterruptedError("Model operation was cancelled before state commit")
    os.replace(pointer, directory / "CURRENT")
    _fsync_directory(directory)


def _engine_snapshot(engine, loader, *, include_tensors=True):
    metadata = {"format": _FORMAT, "loader": loader}
    metadata["engine"] = {name: copy.deepcopy(getattr(engine, name)) for name in _ENGINE_FIELDS}
    metadata["engine"].update({
        "source_dir": str(engine.source_dir) if engine.source_dir else None,
        "changed": sorted(engine.changed),
        "pending_names": sorted(engine.pending),
        "encoded": [[*key, prompt, target] for key, (prompt, target) in engine.encoded.items()],
        "row_changed": {name: sorted(ids) for name, ids in engine.row_changed.items()},
        "row_pending_ids": {name: list(value["ids"]) for name, value in engine.row_pending.items()},
        "row_current_ids": {name: sorted(engine.row_changed.get(name, set()) | set(engine.row_pending.get(name, {}).get("ids", [])))
                            for name in set(engine.row_changed) | set(engine.row_pending)},
    })
    tensors = None
    if include_tensors:
        tensors = {"current." + name: engine.editable[name].detach().cpu().clone() for name in engine.changed | set(engine.pending)}
        tensors.update({"pending." + name: value.detach().cpu().clone() for name, value in engine.pending.items()})
        for name, ids in metadata["engine"]["row_current_ids"].items():
            table = engine.token_tables[name]
            tensors["row.current." + name] = table.index_select(0, torch.tensor(ids, device=table.device)).detach().cpu().clone()
        tensors.update({"row.pending." + name: value["values"].detach().cpu().clone() for name, value in engine.row_pending.items()})
    return metadata, tensors


def _restore_engine(metadata, tensors, emit):
    loader = metadata["loader"]
    engine = Engine(emit=emit, model_loader=_model_loader)
    engine.model, engine.tokenizer, engine.processor = _model_loader(
        loader["identifier"], revision=loader["revision"], device=loader["device"], dtype=getattr(torch, loader["dtype"]),
    )
    engine.load_dtype = loader["dtype"]
    stored = metadata["engine"]
    for name in _ENGINE_FIELDS:
        setattr(engine, name, copy.deepcopy(stored.get(name, getattr(engine, name))))
    engine.source_dir = Path(stored["source_dir"]) if stored["source_dir"] else None
    engine.changed = set(stored["changed"])
    engine.encoded = {tuple(item[:3]): (item[3], item[4]) for item in stored["encoded"]}
    engine.register_editable()
    engine.row_changed = {name: set(ids) for name, ids in stored.get("row_changed", {}).items()}
    row_current = stored.get("row_current_ids", {})
    row_pending = stored.get("row_pending_ids", {})
    expected = {"current." + name for name in engine.changed | set(stored["pending_names"])} | {"pending." + name for name in stored["pending_names"]}
    expected |= {"row.current." + name for name in row_current} | {"row.pending." + name for name in row_pending}
    if set(tensors) != expected:
        raise ValueError("Model snapshot tensor names do not match its metadata")
    with torch.no_grad():
        for name in engine.changed | set(stored["pending_names"]):
            target, value = engine.editable[name], tensors["current." + name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError("Model snapshot tensor shape or dtype does not match its source")
            target.copy_(value)
        for name, ids in row_current.items():
            if name not in engine.token_tables:
                raise ValueError("Snapshot token table is unavailable")
            table, value = engine.token_tables[name], tensors["row.current." + name]
            if (not ids or len(set(ids)) != len(ids) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < table.shape[0] for i in ids)
                    or value.shape != (len(ids), table.shape[1]) or value.dtype != table.dtype or not torch.isfinite(value).all()):
                raise ValueError("Invalid sparse token-row snapshot")
            table.index_copy_(0, torch.tensor(ids, device=table.device), value.to(table.device))
        for name, ids in row_pending.items():
            if name not in row_current or any(i not in row_current[name] for i in ids):
                raise ValueError("Pending token rows are missing from snapshot")
            value, table = tensors["row.pending." + name], engine.token_tables[name]
            if len(set(ids)) != len(ids) or value.shape != (len(ids), table.shape[1]) or value.dtype != table.dtype or not torch.isfinite(value).all():
                raise ValueError("Invalid pending token-row snapshot")
            engine.row_pending[name] = {"ids": list(ids), "values": value.clone()}
    engine.pending = {name: tensors["pending." + name].clone() for name in stored["pending_names"]}
    engine.ensure_row_ids()
    # Tensor version counters are process-local. The private snapshot restores
    # the associated state identity and exact values before binding this cache.
    engine.refresh_probe_versions()
    return engine


def _selected_rows(request, state):
    if request["command"] == "health":
        return (state.get("health_set") or {}).get("rows", [])
    return [row for row in state["dataset"] if row["split"] in request.get("splits", ["validation", "preserve"])]


def _validate_bounds(request, metadata=None):
    command = request.get("command")
    if command not in _COMMANDS:
        raise ValueError(f"Unknown worker command: {command}")
    if command == "init":
        maximum = request.get("max_seq_length", 512)
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum > MAX_SEQ_LENGTH:
            raise ValueError(f"This Space permits max_seq_length up to {MAX_SEQ_LENGTH}")
    if command == "probe_activations":
        targets = request.get("targets")
        if (not isinstance(targets, list) or not 1 <= len(targets) <= 4
                or any(not isinstance(name, str) or not name or len(name) > 2048 for name in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError("Choose one to four distinct probe modules")
        top_k = request.get("top_k", 64)
        if type(top_k) is not int or not 1 <= top_k <= 128:
            raise ValueError("Probe top_k must be between 1 and 128")
        state = (metadata or {}).get("engine", {})
        if state.get("pending_names") or state.get("row_pending_ids"):
            raise ValueError("Accept or roll back the pending patch before probing activations")
        visible = {row.get("row_id"): row for row in state.get("dataset", []) if row["split"] != "test"}
        for key in ("contexts", "contrast_contexts"):
            contexts = request.get(key, [] if key == "contrast_contexts" else None)
            if not isinstance(contexts, list) or len(contexts) > 8 or (key == "contexts" and not contexts):
                raise ValueError("Select one to eight probe contexts (contrast may be empty)")
            for context in contexts:
                if (not isinstance(context, dict) or set(context) != {"row_id", "source", "positions"}
                        or not isinstance(context["row_id"], str) or len(context["row_id"]) > 64
                        or context["source"] not in ("prompt", "reference", "generation")):
                    raise ValueError("Invalid probe context")
                if metadata and context["row_id"] not in visible:
                    raise ValueError("Probe row_id is unavailable; held-out test rows are sealed")
                positions = context["positions"]
                if (not isinstance(positions, list) or not 1 <= len(positions) <= 32
                        or any(type(i) is not int or not -MAX_SEQ_LENGTH <= i < MAX_SEQ_LENGTH for i in positions)
                        or len(set(positions)) != len(positions)):
                    raise ValueError("Choose one to 32 distinct bounded token positions")
                if metadata and context["source"] == "generation":
                    evaluated = state.get("probe_evaluations", {}).get(context["row_id"])
                    if not evaluated or evaluated.get("state_id") != state.get("probe_state_id"):
                        raise ValueError("Evaluate this visible row on the current model before probing its generation")
    if command in {"evaluate", "health"}:
        maximum = request.get("max_new_tokens", 64)
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= MAX_NEW_TOKENS:
            raise ValueError(f"This Space permits 1 to {MAX_NEW_TOKENS} generated tokens per example")
        if metadata:
            rows = _selected_rows(request, metadata["engine"])
            if len(rows) > MAX_TOTAL_EXAMPLES:
                raise ValueError(f"This Space permits up to {MAX_TOTAL_EXAMPLES} examples per evaluation")
            batch = request.get("_space_batch")
            if batch and not 0 < batch["count"] <= MAX_EXAMPLES:
                raise ValueError("Invalid evaluation batch size")


def _generation_rate():
    rate = float(os.environ.get("VESPASIAN_SPACE_SECONDS_PER_TOKEN", "0.05"))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("VESPASIAN_SPACE_SECONDS_PER_TOKEN must be finite and positive")
    return rate


def _evaluation_batch_size(request):
    per_example = 1.5 + min(request.get("max_new_tokens", 64), MAX_NEW_TOKENS) * _generation_rate()
    return max(1, min(MAX_EXAMPLES, math.floor((120 - 15) / per_example)))


def estimate_duration(state_dir: str, request: dict):
    """Conservative scheduling estimate; deployment receipts calibrate its rates.

    No GPU timing is claimed here. Generation is bounded by rows and new tokens;
    VESPASIAN_SPACE_SECONDS_PER_TOKEN can be tuned from measured acceptance runs.
    """
    command = request.get("command")
    if command in {"inspect", "patch_init", "patch_candidate", "patch_finalize", "patch_restore", "patch_export"}:
        return 20
    if command == "probe_activations":
        metadata, _ = _read_metadata(Path(state_dir))
        _validate_bounds(request, metadata)
        state = (metadata or {}).get("engine", {})
        rows = {row.get("row_id"): row for row in state.get("dataset", []) if row["split"] != "test"}
        encoded = {tuple(row[:3]): (len(row[3]), len(row[4])) for row in state.get("encoded", [])}
        tokens = 0
        contexts = request.get("contexts", []) + request.get("contrast_contexts", [])
        for context in contexts:
            row = rows.get(context["row_id"], {})
            prompt, target = encoded.get((row.get("mode", "paired"), row.get("prompt"), row.get("target")), (MAX_SEQ_LENGTH, 0))
            if context["source"] == "generation":
                target = len(state.get("probe_evaluations", {}).get(context["row_id"], {}).get("tokens", []))
            tokens += prompt + (target if context["source"] != "prompt" else 0)
        return min(120, max(20, math.ceil(15 + len(contexts) * 1.5 + tokens * 0.002)))
    if command == "save":
        return 90
    if command == "init":
        return 60
    if command == "edit":
        return 45 if request.get("proposal", {}).get("operation") == "activation_rank_one" else 25
    metadata, _ = _read_metadata(Path(state_dir))
    _validate_bounds(request, metadata)
    if not metadata:
        return 20
    count = request.get("_space_batch", {}).get("count", len(_selected_rows(request, metadata["engine"])))
    tokens = count * min(request.get("max_new_tokens", 64), MAX_NEW_TOKENS)
    rate = _generation_rate()
    return min(120, max(20, math.ceil(15 + count * 1.5 + tokens * rate)))


def _engine_command(engine, request):
    batch = request.get("_space_batch")
    if not batch:
        return engine.dispatch(request)
    engine.ensure_row_ids()
    original_dataset, original_health = engine.dataset, engine.health_set
    offset, count = batch["offset"], batch["count"]
    try:
        if request["command"] == "evaluate":
            engine.dataset = _selected_rows(request, {"dataset": engine.dataset})[offset:offset + count]
        else:
            rows = engine.health_set["rows"][offset:offset + count]
            checks = {row["check"] for row in rows}
            engine.health_set = {**engine.health_set, "rows": rows, "checks": [check for check in engine.health_set["checks"] if check["check"] in checks]}
        return engine.dispatch(request)
    finally:
        engine.dataset, engine.health_set = original_dataset, original_health


def _cpu_transition(command, metadata, tensors):
    if command == "shutdown":
        metadata["closed"] = True
        return {"stopped": True}
    state = metadata["engine"]
    pending = state["pending_names"]
    row_pending = state.get("row_pending_ids", {})
    if command == "accept" and not pending and not row_pending:
        raise ValueError("No edit is waiting to be accepted")
    changed = set(state["changed"])
    for name in pending:
        if command == "accept":
            if not torch.equal(tensors["current." + name], tensors["pending." + name]):
                changed.add(name)
        elif name in changed:
            tensors["current." + name] = tensors["pending." + name]
        else:
            tensors.pop("current." + name)
        tensors.pop("pending." + name)
    # An exact no-op proposal does not need a retained replacement tensor.
    for name in pending:
        if name not in changed:
            tensors.pop("current." + name, None)
    row_changed = {name: set(ids) for name, ids in state.get("row_changed", {}).items()}
    row_current = state.get("row_current_ids", {})
    for name, ids in row_pending.items():
        current = {token_id: value.clone() for token_id, value in zip(row_current[name], tensors["row.current." + name])}
        retained = row_changed.setdefault(name, set())
        for token_id, original in zip(ids, tensors["row.pending." + name]):
            if command == "accept":
                if not torch.equal(current[token_id], original):
                    retained.add(token_id)
            elif token_id in retained:
                current[token_id] = original.clone()
            else:
                current.pop(token_id)
        tensors.pop("row.pending." + name)
        kept_ids = sorted(retained)
        if kept_ids:
            row_current[name] = kept_ids
            tensors["row.current." + name] = torch.stack([current[token_id] for token_id in kept_ids])
        else:
            row_current.pop(name, None)
            row_changed.pop(name, None)
            tensors.pop("row.current." + name, None)
    state.update(changed=sorted(changed), pending_names=[], pending_details=[],
                 row_pending_ids={}, row_current_ids=row_current,
                 row_changed={name: sorted(ids) for name, ids in row_changed.items()})
    if command == "accept":
        retained_id = state.get("probe_candidate_state_id") or uuid.uuid4().hex
        retained = {key: value for key, value in state.get("probe_candidate_evaluations", {}).items()
                    if value.get("state_id") == retained_id}
        state.update(probe_state_id=retained_id, probe_cache={}, probe_evaluations=retained)
    state.update(probe_candidate_state_id=None, probe_candidate_evaluations={})
    return {"accepted": True} if command == "accept" else {"restored": True}


def _execute(state_dir, request, emit, cancelled):
    request = _plain(request)
    if not isinstance(request, dict):
        raise ValueError("Worker requests must be objects")
    directory = Path(state_dir)
    with _state_lock(directory):
        metadata, previous_location = _read_metadata(directory)
        tensors = load_file(str(previous_location / "tensors.safetensors"), device="cpu") if previous_location else {}
        previous = (metadata or {}).get("last_request")
        if previous and request.get("id") is not None and previous["request"].get("id") == request["id"]:
            if previous["request"] != request:
                raise ValueError("Request ID was reused with different arguments")
            for event in previous["events"]:
                emit(event)
            return previous["result"]
        _validate_bounds(request, metadata)
        command = request["command"]
        batch = request.get("_space_batch")
        if batch and (metadata or {}).get("model_version") != batch["model_version"]:
            raise ValueError("Model state changed during this evaluation")
        if metadata and metadata.get("closed"):
            raise ValueError("This model session has shut down")
        events = []

        def record(event):
            event = _plain(event)
            if batch and event.get("phase") in {"evaluating", "health"}:
                offset, total = batch["offset"], batch["total"]
                old_current, old_total = event.get("current", 0), event.get("total", 0)
                if event["type"] == "progress" and ((old_current == 0 and offset) or (event.get("message") in {"Evaluation complete", "Health checks complete"} and offset + batch["count"] < total)):
                    return
                event.update(current=offset + old_current, total=total)
                if "message" in event:
                    event["message"] = event["message"].replace(f"{old_current}/{old_total}", f"{offset + old_current}/{total}")
            events.append(event)
            emit(event)

        if command == "shutdown" and metadata is None:
            return {"type": "result", "id": request.get("id"), "ok": True, "data": {"stopped": True}}
        model_version = (metadata or {}).get("model_version")
        if command in {"accept", "rollback", "shutdown", "patch_finalize", "patch_export"}:
            if metadata is None:
                raise ValueError("Load a model before starting this operation")
            if command == "patch_finalize":
                import patches
                data = patches.finalize(request["directory"], request["turn"], request["accepted"])
            elif command == "patch_export":
                import patches
                data = patches.export_bundle(request["directory"], request["path"])
            else:
                data = _cpu_transition(command, metadata, tensors)
        else:
            if command == "init":
                if metadata is not None:
                    raise ValueError("This worker already has a model loaded")
                engine = Engine(emit=record, model_loader=_model_loader)
                data = engine.dispatch(request)
                loader = {"identifier": request["model"], "revision": data["revision"], "device": data["device"], "dtype": data["dtype"]}
            else:
                if metadata is None:
                    raise ValueError("Load a model before starting this operation")
                engine = _restore_engine(metadata, tensors, record)
                data = _engine_command(engine, request)
                loader = metadata["loader"]
            metadata, changed_tensors = _engine_snapshot(engine, loader, include_tensors=command in {"init", "edit", "patch_restore"})
            if changed_tensors is not None:
                tensors = changed_tensors
        metadata["model_version"] = uuid.uuid4().hex if command in {"init", "edit", "accept", "rollback", "patch_restore"} else model_version
        result = _plain({"type": "result", "id": request.get("id"), "ok": True, "data": data})
        metadata["last_request"] = {"request": request, "events": events, "result": result}
        reuse_tensors = previous_location / "tensors.safetensors" if previous_location and command in {"inspect", "probe_activations", "evaluate", "health", "save", "shutdown", "patch_init", "patch_candidate", "patch_finalize", "patch_export"} else None
        _commit_snapshot(directory, metadata, tensors, cancelled, reuse_tensors=reuse_tensors)
        return result


def _stream(state_dir, request):
    # The queue and callback stay entirely inside one GPU worker process. Only
    # plain event dictionaries are yielded across the scheduler's pickle boundary.
    messages = queue.Queue()
    stopped = threading.Event()

    def emit(event):
        if stopped.is_set():
            raise InterruptedError("Model operation was cancelled")
        messages.put(event)

    def run():
        try:
            result = _execute(state_dir, request, emit, stopped.is_set)
            messages.put(result)
        except Exception as error:
            messages.put({"type": "result", "id": request.get("id") if isinstance(request, dict) else None, "ok": False, "error": str(error)})
        finally:
            messages.put(None)

    thread = threading.Thread(target=run, name="vespasian-tensor-operation", daemon=True)
    thread.start()
    try:
        while True:
            event = messages.get()
            if event is None:
                break
            yield event
    finally:
        stopped.set()
        thread.join()


@spaces.GPU(duration=estimate_duration, size=GPU_SIZE)
def _gpu_dispatch(state_dir: str, request: dict):
    yield from _stream(state_dir, request)


def _scheduled_dispatch(state_dir, request, *, boundaries, observe=None):
    """Let the caller obtain a fresh browser request before each allocation."""
    request_id = uuid.uuid4().hex  # Opaque correlation, never the raw protocol payload.
    for attempt in range(2 if boundaries else 1):
        if boundaries:
            yield {"type": "gpu_allocation"}
        # Resuming the boundary reserves an attempt; polling or closing does not.
        receipt = {
            "allocation_id": uuid.uuid4().hex, "request_id": request_id,
            "operation": request.get("command") if request.get("command") in _COMMANDS else "unknown",
            "attempt": attempt + 1,
            "requested_max_duration_seconds": estimate_duration(state_dir, request),
            "gpu_size": GPU_SIZE, "quota_multiplier": GPU_QUOTA_MULTIPLIER,
            "status": "attempted", "wall_seconds": None,
        }
        # Record the bound BEFORE invoking the scheduler. This is not billing.
        if observe:
            observe(dict(receipt))
        started_at = time.monotonic()
        started = False
        outcome = "uncertain"
        operation_ok = None
        try:
            for event in _gpu_dispatch(state_dir, request):
                started = True
                if event.get("type") == "result":
                    operation_ok = bool(event.get("ok"))
                yield event
            outcome = "completed"
            return
        except Exception as error:
            message = getattr(error, "message", str(error))
            expired = not started and message == "Expired ZeroGPU proxy token"
            if expired:
                outcome = "rejected"
            if not boundaries or attempt or not expired:
                raise
            # Retry only at a new browser boundary with fresh credentials.
        finally:
            receipt.update(status=outcome, operation_ok=operation_ok, wall_seconds=max(0, time.monotonic() - started_at))
            if observe:
                observe(dict(receipt))


def _write_journal(path, value):
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    _write_json(temporary, value)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _aggregate(parts):
    metrics, samples = {}, []
    elapsed = 0.0
    for part in parts:
        samples.extend(part["samples"])
        elapsed += part["elapsed_seconds"]
        for name, metric in part["metrics"].items():
            metrics.setdefault(name, {key: value if isinstance(value, str) else 0 for key, value in metric.items()})
    # Re-sum complete samples in their original order, just as Engine does.
    # Averaging group means can otherwise introduce a different rounding path.
    for sample in samples:
        metric = metrics[sample["check"] if "check" in sample else sample["split"]]
        metric["count"] += 1
        if "check" in sample:
            metric["score"] += int(sample["passed"])
        else:
            for key in ("loss", "exact_match", "token_f1", "numeric_match"):
                if key in sample:
                    metric[key] += float(sample[key])
    for metric in metrics.values():
        for key in metric:
            if key != "count" and not isinstance(metric[key], str):
                metric[key] /= metric["count"]
    result = {"metrics": metrics, "samples": samples, "elapsed_seconds": elapsed}
    if parts and "rows_sha256" in parts[0]:
        result["rows_sha256"] = parts[0]["rows_sha256"]
    elif parts and "behavior" in parts[0]:
        result["behavior"] = summarize_evaluation(samples)
    return result


def _batched_dispatch(state_dir, request, *, boundaries=False, observe=None):
    directory = Path(state_dir)
    metadata, _ = _read_metadata(directory)
    _validate_bounds(request, metadata)
    if metadata is None:
        yield from _scheduled_dispatch(state_dir, request, boundaries=boundaries, observe=observe)
        return
    total = len(_selected_rows(request, metadata["engine"]))
    if not total:
        yield from _scheduled_dispatch(state_dir, request, boundaries=boundaries, observe=observe)
        return
    # Keep a retryable receipt outside the GPU boundary after each allocation.
    # Version binding prevents an old evaluation from observing newer edits.
    batch_size = _evaluation_batch_size(request)
    identity = {"request": request, "model_version": metadata["model_version"], "batch_size": batch_size}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    journals = directory / "evaluations"
    journals.mkdir(exist_ok=True)
    path = journals / (digest + ".json")
    with _state_lock(journals):
        journal = json.loads(path.read_text()) if path.exists() else {**identity, "batches": []}
        parts = []
        for index, offset in enumerate(range(0, total, batch_size)):
            if index < len(journal["batches"]):
                completed = journal["batches"][index]
                for event in completed["events"]:
                    yield event
                parts.append(completed["data"])
                continue
            batch_request = {**request, "id": f"{request.get('id')}:batch:{offset}", "_space_batch": {"offset": offset, "count": min(batch_size, total - offset), "total": total, "model_version": metadata["model_version"]}}
            events, result = [], None
            for event in _scheduled_dispatch(state_dir, batch_request, boundaries=boundaries, observe=observe):
                if event.get("type") == "gpu_allocation":
                    yield event
                elif event.get("type") == "result":
                    result = event
                else:
                    events.append(event)
                    yield event
            if result is None:
                raise ValueError("GPU evaluation ended without a result")
            if not result.get("ok"):
                yield {**result, "id": request.get("id")}
                return
            completed = {"events": events, "data": result["data"]}
            journal["batches"].append(completed)
            _write_journal(path, journal)
            parts.append(result["data"])
        yield {"type": "result", "id": request.get("id"), "ok": True, "data": _aggregate(parts)}


def dispatch(state_dir: str, request: dict, *, boundaries=False, observe=None):
    """Stream protocol events, optionally pausing before each GPU allocation.

    With boundaries enabled, the caller must consume ``gpu_allocation`` markers
    itself and resume at most one allocation in each fresh queued browser event.
    Markers are never persisted in evaluation journals or worker receipts.
    The optional observe callback runs on the CPU, before and after scheduler
    attempts, including generator closure. It must not raise or perform I/O.
    """
    try:
        if not isinstance(request, dict):
            raise ValueError("Worker requests must be objects")
        if request.get("command") in {"accept", "rollback", "shutdown", "patch_finalize", "patch_export"}:
            yield from _stream(state_dir, request)
        elif request.get("command") in {"evaluate", "health"}:
            yield from _batched_dispatch(state_dir, request, boundaries=boundaries, observe=observe)
        else:
            yield from _scheduled_dispatch(state_dir, request, boundaries=boundaries, observe=observe)
    except Exception as error:
        yield {"type": "result", "id": request.get("id") if isinstance(request, dict) else None, "ok": False, "error": str(error)}
