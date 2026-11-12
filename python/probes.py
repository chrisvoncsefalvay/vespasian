"""Bounded forward-only activation measurements and opaque direction handles.

Vectors remain private worker state, never frontier-tool results. A direction is
an observed module-output mean, optionally contrasted with another set of
positions. Projection edits use selected input columns of an eligible matrix.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid

MAX_HANDLES = 32
MAX_WIDTH = 32768
MAX_POSITIONS = 32
MAX_CONTEXTS = 8
MAX_COLUMNS = 256


def integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def provenance(engine):
    return hashlib.sha256(json.dumps({"source": engine.source_files, "thinking_mode": engine.thinking_mode,
                                      "prompt_format": engine.prompt_format}, sort_keys=True).encode()).hexdigest()


def weight_digest(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def versions(engine):
    return tuple((name, id(value), value._version) for name, value in engine.model.named_parameters())


def refresh(engine):
    engine._probe_versions = versions(engine)


def invalidate(engine):
    engine.probe_state_id = uuid.uuid4().hex
    engine.probe_cache.clear()
    engine.probe_evaluations.clear()
    engine.probe_candidate_evaluations.clear()
    engine.probe_candidate_state_id = None
    refresh(engine)


def accept_candidate(engine):
    state_id = engine.probe_candidate_state_id or uuid.uuid4().hex
    engine.probe_cache.clear()
    engine.probe_evaluations = {key: value for key, value in engine.probe_candidate_evaluations.items()
                                if value.get("state_id") == state_id}
    engine.probe_state_id = state_id
    engine.probe_candidate_evaluations = {}
    engine.probe_candidate_state_id = None
    refresh(engine)


def check_state(engine):
    current = versions(engine)
    if engine._probe_versions is not None and current != engine._probe_versions:
        invalidate(engine)
    else:
        engine._probe_versions = current


def catalogue(engine):
    import torch
    if engine.model is None:
        return []
    head = engine.model.get_output_embeddings()
    result = []
    for name, module in engine.model.named_modules():
        weight = getattr(module, "weight", None)
        if (type(module) is torch.nn.Linear and module is not head and name
                and isinstance(weight, torch.nn.Parameter) and weight.ndim == 2
                and weight.is_floating_point() and not weight.is_meta
                and 0 < min(weight.shape) and max(weight.shape) <= MAX_WIDTH):
            result.append({"name": name + ".weight", "input_width": weight.shape[1],
                           "output_width": weight.shape[0], "direction_projection": name + ".weight" in engine.editable})
    return result


def visible_rows(engine):
    counts = {}
    result = {}
    for row in engine.dataset:
        split = row["split"]
        if split == "test":
            continue
        index = counts.get(split, 0)
        counts[split] = index + 1
        result[row.get("row_id", f"{split}:{index}")] = row
    return result


def cache_evaluation(engine, samples):
    # Pending measurements remain separate until acceptance. Rollback must keep
    # the previous best observations, but accept needs the newly measured ones.
    pending = bool(engine.pending or engine.row_pending)
    if not pending:
        check_state(engine)
    cache = engine.probe_candidate_evaluations if pending else engine.probe_evaluations
    state_id = engine.probe_candidate_state_id if pending else engine.probe_state_id
    if not state_id:
        return
    rows = visible_rows(engine)
    # Replacing evaluated splits also drops stale outputs for changed datasets.
    for sample in samples:
        row_id = sample.get("row_id")
        row = rows.get(row_id)
        if row is None or sample.get("split") == "test":
            continue
        generated = sample.get("token_ids", [])
        prompt, _ = engine.encode(row)
        if (len(cache) < 256 or row_id in cache) and len(prompt) + len(generated) <= engine.max_seq_length:
            cache[row_id] = {"state_id": state_id,
                "prompt": row["prompt"], "target": row["target"], "tokens": list(generated)}
    # Bound all retained generation tokens, including long-output evaluation.
    total = 0
    for row_id in reversed(list(cache)):
        total += len(cache[row_id]["tokens"])
        if total > 262144:
            del cache[row_id]


def contexts(engine, requests, *, required=True):
    if not isinstance(requests, list) or len(requests) > MAX_CONTEXTS or (required and not requests):
        raise ValueError("Select one to eight probe contexts (contrast may be empty)")
    rows = visible_rows(engine)
    prepared = []
    for item in requests:
        if not isinstance(item, dict) or set(item) != {"row_id", "source", "positions"}:
            raise ValueError("A probe context requires row_id, source and positions")
        row_id, source = item["row_id"], item["source"]
        if not isinstance(row_id, str) or row_id not in rows:
            raise ValueError("Probe row_id is unavailable; held-out test rows are sealed")
        if source not in {"prompt", "reference", "generation"}:
            raise ValueError("Probe source must be prompt, reference or generation")
        row = rows[row_id]
        prompt, target = engine.encode(row)
        tokens = list(prompt)
        if source == "reference":
            if not target:
                raise ValueError("This row has no reference tokens")
            tokens += target
        elif source == "generation":
            evaluated = engine.probe_evaluations.get(row_id)
            if (not evaluated or evaluated["state_id"] != engine.probe_state_id
                    or evaluated["prompt"] != row["prompt"] or evaluated["target"] != row["target"]):
                raise ValueError("Evaluate this visible row on the current model before probing its generation")
            tokens += evaluated["tokens"]
        selected = item["positions"]
        if not isinstance(selected, list) or not 1 <= len(selected) <= MAX_POSITIONS:
            raise ValueError("Select one to 32 token positions per context")
        positions = [integer(p, "token position", -len(tokens), len(tokens)-1) % len(tokens) for p in selected]
        if len(set(positions)) != len(positions):
            raise ValueError("Probe token positions must be distinct after resolving negative indices")
        if len(tokens) > engine.context_limit():
            raise ValueError("Probe context exceeds the configured sequence length")
        prepared.append({"row_id": row_id, "source": source, "tokens": tokens, "positions": positions,
                         "prompt_tokens": len(prompt)})
    return prepared


def measure(engine, request):
    import torch
    engine.ready()
    if engine.pending or engine.row_pending:
        raise ValueError("Accept or roll back the pending patch before probing activations")
    check_state(engine)
    allowed_keys = {"id", "command", "targets", "contexts", "contrast_contexts", "top_k"}
    if set(request) - allowed_keys:
        raise ValueError("Unexpected activation probe fields")
    targets = request.get("targets")
    available = {item["name"]: item for item in catalogue(engine)}
    if (not isinstance(targets, list) or not 1 <= len(targets) <= 4
            or any(not isinstance(name, str) or name not in available for name in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError("Choose one to four distinct supported probe modules from probe_modules")
    top_k = integer(request.get("top_k", 64), "top_k", 1, 128)
    positive = contexts(engine, request.get("contexts"))
    negative = contexts(engine, request.get("contrast_contexts", []), required=False)
    aggregates = {name: [[], []] for name in targets}
    inputs = {name: [] for name in targets}
    position_summaries = {name: [] for name in targets}
    handles, captured = [], {}
    active = None

    def hook(name):
        def capture(module, args, output):
            if active is None or name in captured:
                raise ValueError("Module was invoked multiple times; this activation layout is unsupported")
            x = args[0] if args else None
            if (not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor)
                    or x.ndim != 3 or output.ndim != 3 or x.shape[:2] != output.shape[:2]
                    or x.shape[0] != 1 or x.shape[1] != len(active["tokens"])
                    or x.shape[-1] != available[name]["input_width"] or output.shape[-1] != available[name]["output_width"]):
                raise ValueError("Module activation layout is unsupported; expected batch, token, feature")
            indices = torch.tensor(active["positions"], device=x.device)
            selected_x = x.detach()[0].index_select(0, indices).float().cpu()
            selected_y = output.detach()[0].index_select(0, indices.to(output.device)).float().cpu()
            if not torch.isfinite(selected_x).all() or not torch.isfinite(selected_y).all():
                raise ValueError("Activation probe produced nonfinite measurements")
            captured[name] = (selected_x, selected_y)
        return capture

    try:
        for name in targets:
            handles.append(engine.model.get_submodule(name.removesuffix(".weight")).register_forward_hook(hook(name)))
        engine.model.eval()
        with torch.inference_mode():
            for group, items in enumerate((positive, negative)):
                for active in items:
                    captured.clear()
                    tokens = torch.tensor([active["tokens"]], dtype=torch.long, device=engine.device)
                    engine.forward(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False, logits_to_keep=1)
                    if set(captured) != set(targets):
                        raise ValueError("Selected module was not reached by text inference; probe unavailable")
                    for name, (x, y) in captured.items():
                        aggregates[name][group].append(y)
                        if group == 0:
                            inputs[name].append(x.abs())
                        position_summaries[name].append({"row_id": active["row_id"], "source": active["source"],
                            "group": "contrast" if group else "context", "positions": active["positions"],
                            "prompt_tokens": active["prompt_tokens"], "context_tokens": len(active["tokens"]),
                            "input_norms": x.norm(dim=1).tolist(), "output_norms": y.norm(dim=1).tolist()})
    finally:
        for handle in handles:
            handle.remove()

    results, staged = [], {}
    for name in targets:
        positive_values = torch.cat(aggregates[name][0])
        vector = positive_values.mean(dim=0)
        if negative:
            vector = vector - torch.cat(aggregates[name][1]).mean(dim=0)
        norm = float(vector.norm())
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("Measured direction is zero or nonfinite; select different contexts or positions")
        vector = vector / norm
        module = engine.model.get_submodule(name.removesuffix(".weight"))
        # Compute one projection vector, not a full delta, on the model device.
        with torch.inference_mode():
            projection = (vector.to(module.weight.device) @ module.weight.detach().float()).cpu()
        mean_input = torch.cat(inputs[name]).mean(dim=0)
        score = projection.abs() * mean_input
        if not torch.isfinite(projection).all() or not torch.isfinite(score).all():
            raise ValueError("Selected module has nonfinite projection measurements")
        indices = torch.argsort(score, descending=True, stable=True)[:min(top_k, len(score))].tolist()
        direction_id = "direction_" + uuid.uuid4().hex
        evidence = {"method": "output_mean_difference" if negative else "output_mean", "autograd": False,
                    "normalised": True, "raw_direction_norm": norm, "contexts": copy.deepcopy(position_summaries[name])}
        staged[direction_id] = {"state_id": engine.probe_state_id, "target": name,
                                "vector": vector.tolist(), "evidence": evidence, "provenance": provenance(engine),
                                "shape": list(module.weight.shape), "dtype": str(module.weight.dtype),
                                "weight_sha256": weight_digest(module.weight)}
        results.append({"target": name, "direction_id": direction_id, "dimension": len(vector),
                        "direction_projection": available[name]["direction_projection"], "evidence": evidence,
                        "top_columns": [{"column": i, "projection": float(projection[i]),
                                         "mean_abs_activation": float(mean_input[i]), "score": float(score[i])} for i in indices]})
    engine.probe_cache.update(staged)
    while len(engine.probe_cache) > MAX_HANDLES:
        del engine.probe_cache[next(iter(engine.probe_cache))]
    return {"probes": results, "state_id": engine.probe_state_id,
            "policy": "Positions index the full prompt plus selected continuation; negative positions count from the end. Only ordinary Linear module outputs are measured, not decoder residual-stream states. Directions are observational, not causal attribution, and handles are scoped to their source tensor. Handles expire on accepted edits or restore. Test rows are sealed."}


def selected_direction(engine, proposal):
    import torch
    check_state(engine)
    name = proposal["targets"][0]
    direction_id = proposal.get("direction_id")
    item = engine.probe_cache.get(direction_id) if isinstance(direction_id, str) else None
    if not item or item.get("state_id") != engine.probe_state_id or item.get("target") != name:
        raise ValueError("Select a measured direction handle for this tensor on the current model state")
    columns = proposal.get("columns")
    parameter = engine.editable[name]
    if (item.get("provenance") != provenance(engine) or item.get("shape") != list(parameter.shape)
            or item.get("dtype") != str(parameter.dtype) or item.get("weight_sha256") != weight_digest(parameter)):
        raise ValueError("Measured direction does not match the current checkpoint, generation policy or tensor state")
    if (not isinstance(columns, list) or not 1 <= len(columns) <= MAX_COLUMNS
            or any(type(i) is not int or not 0 <= i < parameter.shape[1] for i in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError("Choose one to 256 distinct in-range input columns")
    vector = torch.tensor(item["vector"], dtype=torch.float32, device=parameter.device)
    if vector.shape != (parameter.shape[0],) or not torch.isfinite(vector).all() or abs(float(vector.norm()) - 1.0) > 1e-5:
        raise ValueError("Measured direction has invalid dimensions, values or normalisation")
    return vector, columns, copy.deepcopy(item["evidence"])
