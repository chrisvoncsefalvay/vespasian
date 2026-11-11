#!/usr/bin/env python3
"""Fetch a revision-pinned, bounded text dataset from the Hugging Face Hub."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import task_metadata

from typing import Any, Callable, Iterable


class DatasetError(ValueError):
    """An actionable dataset or mapping error, without credentials or source data."""


def selector_tokens(row: dict[str, Any], field: str) -> tuple[str | int | None, ...]:
    """Parse dotted keys, numeric indexes and one paired-array expansion."""
    if not isinstance(field, str) or not field or field != field.strip():
        raise DatasetError("Use a nonempty field selector")
    if field in row:
        return (field,)
    tokens: list[str | int | None] = []
    for component in field.split("."):
        match = re.fullmatch(r"([^\[\]]+)((?:\[[0-9]*\])*)", component)
        if not match:
            raise DatasetError(f"Invalid field selector {field!r}. Use examples[].input or examples[0].input")
        key, suffix = match.groups()
        tokens.append(int(key) if key.isascii() and key.isdigit() else key)
        tokens.extend(None if index == "" else int(index) for index in re.findall(r"\[([0-9]*)\]", suffix))
    if tokens.count(None) > 1:
        raise DatasetError("A field selector can expand one array. Use an explicit index for deeper arrays")
    return tuple(tokens)


def _walk(value: Any, tokens: tuple[str | int | None, ...]) -> Any:
    for token in tokens:
        if isinstance(value, dict) and token is not None and str(token) in value:
            value = value[str(token)]
        elif isinstance(value, (list, tuple)) and isinstance(token, int) and 0 <= token < len(value):
            value = value[token]
        else:
            raise KeyError(tokens)
    return value


def field_value(row: dict[str, Any], field: str) -> Any:
    tokens = selector_tokens(row, field)
    if None not in tokens:
        return _walk(row, tokens)
    position = tokens.index(None)
    values = _walk(row, tokens[:position])
    if not isinstance(values, (list, tuple)):
        raise DatasetError(f"The field {field!r} does not select an array")
    return [_walk(value, tokens[position + 1:]) for value in values]


def _candidate_fields(row: dict[str, Any], role: str) -> list[str]:
    aliases = {"prompt": ("prompt", "instruction", "question", "input", "text", "content"),
               "target": ("target", "output", "response", "answer", "completion"),
               "text": ("text", "content", "document", "documents")}[role]
    candidates: list[str] = []
    visited = 0

    def visit(value: Any, prefix: str = "", depth: int = 0):
        nonlocal visited
        visited += 1
        if visited > 256 or depth > 5 or not isinstance(value, dict):
            return
        for alias in aliases:
            if alias in value:
                field = f"{prefix}.{alias}" if prefix else alias
                if role != "target" and isinstance(value[alias], (list, tuple)) and value[alias] and isinstance(value[alias][0], str):
                    field += "[]"
                if role == "target" and alias == "target" and "target_scores" in value:
                    fallback = f"{prefix}.target_scores" if prefix else "target_scores"
                    field = f"{field} ?? {fallback}"
                candidates.append(field)
        for key, child in list(value.items())[:50]:
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, dict):
                visit(child, path, depth + 1)
            elif isinstance(child, (list, tuple)) and child and isinstance(child[0], dict):
                visit(child[0], f"{path}[]", depth + 1)

    visit(row)
    return candidates


def selector_parts(row: dict[str, Any], field: str) -> list[str]:
    if field in row:
        return [field]
    parts = [part.strip() for part in field.split("??")]
    if any(not part for part in parts):
        raise DatasetError("A fallback selector needs a field on both sides of ??")
    for part in parts:
        selector_tokens(row, part)
    return parts


def resolve_field(row: dict[str, Any], explicit: str | None, role: str) -> str:
    if explicit is not None:
        parts = selector_parts(row, explicit)
        if role != "target" and len(parts) != 1:
            raise DatasetError("Use one text selector. Target fields may use ?? for a fallback")
        present = False
        for part in parts:
            tokens = selector_tokens(row, part)
            try:
                if None in tokens:
                    position = tokens.index(None)
                    values = _walk(row, tokens[:position])
                    if not isinstance(values, (list, tuple)) or not values:
                        raise DatasetError(f"The {role} field {part!r} selects an empty or invalid array")
                    value = _walk(values[0], tokens[position + 1:])
                else:
                    value = _walk(row, tokens)
            except KeyError:
                continue
            present = True
        if not present:
            raise DatasetError(f"The dataset has no {role} field {explicit!r}. Set --{role}-field")
        return explicit
    candidates = _candidate_fields(row, role)
    if candidates:
        return candidates[0]
    raise DatasetError(f"No {role} field was found. Set --{role}-field to a text field such as examples[].input")


def _pair_paths(row: dict[str, Any], prompt: str, target: str):
    if len(selector_parts(row, prompt)) != 1:
        raise DatasetError("Use one prompt selector")
    left = selector_tokens(row, prompt)
    right = [selector_tokens(row, part) for part in selector_parts(row, target)]
    for path in right:
        if left == path:
            raise DatasetError("Prompt and target fields must be different. Select their nested fields within the shared column")
        if None in left or None in path:
            if None not in left or None not in path or left[:left.index(None)] != path[:path.index(None)]:
                raise DatasetError("Prompt and target must expand the same array, for example examples[].input and examples[].target")
    return left, right


def resolve_mapping(row: dict[str, Any], prompt: str | None, target: str | None) -> tuple[str, str]:
    prompts = [resolve_field(row, prompt, "prompt")] if prompt is not None else _candidate_fields(row, "prompt")
    targets = [resolve_field(row, target, "target")] if target is not None else _candidate_fields(row, "target")
    if not prompts:
        resolve_field(row, None, "prompt")
    if not targets:
        resolve_field(row, None, "target")
    for left in prompts:
        for right in targets:
            try:
                _pair_paths(row, left, right)
            except DatasetError:
                continue
            return left, right
    _pair_paths(row, prompts[0], targets[0])
    raise DatasetError("Choose prompt and target fields from the same example")


def _text(value: Any, role: str, field: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if role == "target" and isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if role == "target" and isinstance(value, float):
        import math
        if math.isfinite(value):
            return str(value)
    suffix = " or a finite number" if role == "target" else ""
    raise DatasetError(f"The {role} field {field!r} must contain nonempty text{suffix}")


def _choice_label(index: int) -> str:
    result = ""
    while index >= 0:
        result = chr(ord("A") + index % 26) + result
        index = index // 26 - 1
    return result


def _choice_reference(value: Any, prompt: str, field: str) -> tuple[str, str, dict[str, Any]]:
    """Expose every scored option and preserve every positive binary answer."""
    def unique_keys(pairs):
        result = {}
        for key, score in pairs:
            if key in result:
                raise DatasetError(f"The score map {field!r} contains duplicate choices")
            result[key] = score
        return result

    if isinstance(value, str):
        try:
            value = json.loads(value, object_pairs_hook=unique_keys)
        except (json.JSONDecodeError, TypeError):
            raise DatasetError(f"The score map {field!r} must be a JSON object") from None
    if not isinstance(value, dict) or not value or len(value) > 256:
        raise DatasetError(f"The score map {field!r} must contain between 1 and 256 choices")
    correct = []
    options = []
    for index, (answer, score) in enumerate(value.items()):
        if not isinstance(answer, str) or not answer.strip():
            raise DatasetError(f"The score map {field!r} has an empty choice")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or score not in (0, 1):
            raise DatasetError(f"The score map {field!r} needs finite binary scores of 0 or 1")
        label = _choice_label(index)
        options.append(f"{label}. {answer}")
        if score == 1:
            correct.append(label)
    if not correct:
        raise DatasetError(f"The score map {field!r} has no correct choice")
    multiple = len(correct) > 1
    instruction = ("Select all correct options. Return a JSON array of their labels in the order shown, with no other text."
                   if multiple else "Select the correct option. Return its label only, with no other text.")
    target = json.dumps(correct, ensure_ascii=False) if multiple else correct[0]
    mapped_prompt = prompt + "\n\nChoices:\n" + "\n".join(options) + "\n\n" + instruction
    return mapped_prompt, target, {"kind": "binary_choice_scores", "source": field,
        "answer_format": "json_label_array" if multiple else "single_label",
        "choice_count": len(options), "correct_count": len(correct)}


def mapped_pairs(row: dict[str, Any], prompt: str, target: str):
    """Yield aligned examples in source order. Never form an array cross-product."""
    left, right = _pair_paths(row, prompt, target)
    target_parts = selector_parts(row, target)
    if None in left:
        position = left.index(None)
        try:
            values = _walk(row, left[:position])
        except KeyError:
            raise DatasetError("The selected example array is missing from a dataset row") from None
        if not isinstance(values, (list, tuple)) or not values:
            raise DatasetError("The selected example array must contain at least one example")
        source = ((index, item, left[position + 1:], [path[position + 1:] for path in right]) for index, item in enumerate(values))
    else:
        source = [(0, row, left, right)]
    for index, item, prompt_path, target_paths in source:
        try:
            prompt_value = _text(_walk(item, prompt_path), "prompt", prompt)
        except KeyError:
            raise DatasetError(f"The prompt field {prompt!r} is missing from a dataset example") from None
        selected = None
        for part, path in zip(target_parts, target_paths):
            try:
                value = _walk(item, path)
            except KeyError:
                continue
            if value is not None:
                selected = (part, path, value)
                break
        if selected is None:
            raise DatasetError(f"The target field {target!r} is missing or null in a dataset example")
        part, path, value = selected
        if path and path[-1] == "target_scores":
            prompt_value, target_value, transformation = _choice_reference(value, prompt_value, part)
        else:
            target_value = _text(value, "target", part)
            transformation = {"kind": "number_to_text" if not isinstance(value, str) else "field_only", "source": part}
        yield {"example_index": index, "prompt": prompt_value, "target": target_value,
               "target_source": part, "transformation": transformation}


def _validate_mode(dataset_mode: str) -> None:
    if dataset_mode not in {"paired", "corpus", "prompts"}:
        raise DatasetError("Choose dataset mode paired, corpus or prompts")


def resolve_mode_fields(row: dict[str, Any], dataset_mode: str, prompt_field: str | None,
                        target_field: str | None, text_field: str | None) -> tuple[str, str]:
    if dataset_mode == "paired":
        return resolve_mapping(row, prompt_field, target_field)
    if dataset_mode == "corpus":
        return "", resolve_field(row, text_field, "text")
    return resolve_field(row, prompt_field, "prompt"), ""


def mapped_examples(row: dict[str, Any], fields: tuple[str, str], dataset_mode: str):
    if dataset_mode == "paired":
        yield from mapped_pairs(row, *fields)
        return
    selector = fields[1] if dataset_mode == "corpus" else fields[0]
    role = "text" if dataset_mode == "corpus" else "prompt"
    if len(selector_parts(row, selector)) != 1:
        raise DatasetError("Use one text selector")
    tokens = selector_tokens(row, selector)
    try:
        if None in tokens:
            position = tokens.index(None)
            values = _walk(row, tokens[:position])
            if not isinstance(values, (list, tuple)) or not values:
                raise DatasetError("The selected text array must contain at least one example")
            selected = ((index, value, tokens[position + 1:]) for index, value in enumerate(values))
        else:
            selected = [(0, row, tokens)]
        for index, value, path in selected:
            text = _text(_walk(value, path), role, selector)
            yield {"example_index": index, "prompt": text if dataset_mode == "prompts" else "",
                   "target": text if dataset_mode == "corpus" else "", "source_field": selector,
                   "transformation": {"kind": "field_only", "source": selector}}
    except KeyError:
        raise DatasetError(f"The {role} field {selector!r} is missing from a dataset example") from None


def _runtime() -> tuple[Any, Callable[..., Iterable[dict[str, Any]]], str]:
    try:
        import datasets
        from huggingface_hub import HfApi
    except ImportError:
        raise DatasetError("Install the dataset runtime with vespasian setup") from None
    if int(datasets.__version__.split(".")[0]) < 4:
        raise DatasetError("Hugging Face dataset imports require datasets 4 or later. Run vespasian setup")
    # datasets >= 4 does not execute dataset loading scripts from repositories.
    datasets.disable_progress_bars()
    return HfApi(), datasets.load_dataset, datasets.__version__


def _validate_repo(repo: str) -> None:
    parts = repo.split("/")
    if len(parts) not in (1, 2) or len(repo) > 192 or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
        or ".." in part or "--" in part or part.endswith((".", ".git"))
        for part in parts
    ):
        raise DatasetError("Use a Hugging Face dataset identifier such as organisation/dataset")


def _preview_runtime():
    api, loader, version = _runtime()
    import datasets
    return api, loader, datasets.get_dataset_config_names, datasets.get_dataset_split_names, version


def _dtype(value: Any) -> str:
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple)):
        return "list"
    return type(value).__name__


def _columns(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    columns: dict[str, set[str]] = {}

    def visit(row, prefix="", depth=0):
        for name, value in list(row.items())[:50]:
            if len(columns) >= 256:
                return
            field = f"{prefix}.{name}" if prefix else str(name)
            columns.setdefault(field, set()).add(_dtype(value))
            if isinstance(value, dict) and depth < 5:
                visit(value, field, depth + 1)
            elif isinstance(value, (list, tuple)) and depth < 5:
                for item in value[:20]:
                    if isinstance(item, dict):
                        visit(item, f"{field}[]", depth + 1)

    for row in rows:
        visit(row)
    return [{"name": name, "dtype": " | ".join(sorted(types))} for name, types in columns.items()]


def _display_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Bound display data without changing the imported dataset or its mapping."""
    truncated: list[str] = []

    def shorten(value, path, budget, depth=0):
        budget["nodes"] -= 1
        if budget["nodes"] < 0 or depth > 6:
            truncated.append(path)
            return None
        if isinstance(value, str):
            size = min(4000, max(0, budget["characters"]))
            budget["characters"] -= min(len(value), size)
            if len(value) > size:
                truncated.append(path)
            return value[:size]
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            import math
            if math.isfinite(value):
                return value
            truncated.append(path)
            return str(value)
        if isinstance(value, dict):
            if len(value) > 50:
                truncated.append(path)
            return {str(key): shorten(item, f"{path}.{key}", budget, depth + 1)
                    for key, item in list(value.items())[:50]}
        if isinstance(value, (list, tuple)):
            if len(value) > 20:
                truncated.append(path)
            return [shorten(item, f"{path}[{index}]", budget, depth + 1)
                    for index, item in enumerate(value[:20])]
        # Binary/image/audio objects are described, never decoded or executed.
        truncated.append(path)
        return f"<{type(value).__name__}>"

    displayed = [shorten(row, f"rows[{index}]", {"characters": 24000, "nodes": 256})
                 for index, row in enumerate(rows)]
    return displayed, truncated


def prepare_task_row(row, fields, mode, task, features=None):
    value = None
    if task == "text-classification" and mode == "paired":
        for part in selector_parts(row, fields[1]):
            try:
                value = field_value(row, part)
                break
            except (KeyError, DatasetError):
                pass
    try:
        return task_metadata.prepare_row(row, fields, mode, task, features, target_value=value)
    except ValueError as error:
        raise DatasetError(str(error)) from None


def _model_task_info(api: Any, model_id: str | None):
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id):
        return None
    try:
        return api.model_info(repo_id=model_id)
    except Exception:
        return None


def preview(
    *, repo: str, split: str = "train", config: str | None = None,
    revision: str | None = None, prompt_field: str | None = None,
    target_field: str | None = None, preview_rows: int = 3,
    dataset_mode: str = "paired", text_field: str | None = None,
    hf_task: str | None = None, model_id: str | None = None,
    api: Any = None, loader: Callable[..., Iterable[dict[str, Any]]] | None = None,
    config_names: Callable[..., list[str]] | None = None,
    split_names: Callable[..., list[str]] | None = None,
    runtime_version: str | None = None,
) -> dict[str, Any]:
    """Read metadata and a few rows at one immutable Hub revision, without publishing files."""
    _validate_repo(repo)
    _validate_mode(dataset_mode)
    try:
        hf_task = task_metadata.normalize_task(hf_task)
    except ValueError as error:
        raise DatasetError(str(error)) from None
    if isinstance(preview_rows, bool) or not isinstance(preview_rows, int) or not 1 <= preview_rows <= 5:
        raise DatasetError("Preview size must be between 1 and 5 rows")
    if api is None or loader is None or config_names is None or split_names is None:
        api, loader, config_names, split_names, runtime_version = _preview_runtime()
    result: dict[str, Any] = {
        "repo": repo, "revision": None, "config": config, "split": split,
        "configs": [], "splits": [], "columns": [], "rows": [],
        "dataset_mode": dataset_mode,
        "prompt_field": prompt_field if dataset_mode != "corpus" else None,
        "target_field": target_field if dataset_mode == "paired" else None,
        "text_field": text_field if dataset_mode == "corpus" else None,
        "mapping_error": None, "error": None, "truncated": False,
        "mapped_rows": [], "mapped_rows_truncated": False,
        "truncated_fields": [], "datasets_version": runtime_version,
    }
    try:
        info = api.dataset_info(repo_id=repo, revision=revision)
    except Exception:
        result["error"] = "Could not resolve the dataset revision. Check its identifier, access and network connection"
        return result
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", sha):
        result["error"] = "The Hub returned no immutable revision for this dataset"
        return result
    result["revision"] = sha
    result["task_metadata"] = task_metadata.assess_task(info, hf_task, _model_task_info(api, model_id))
    try:
        configs = config_names(repo, revision=sha)
    except Exception:
        result["error"] = "Could not read dataset configurations. Check access and supported data files"
        return result
    if not isinstance(configs, list) or not configs or any(not isinstance(name, str) for name in configs):
        result["error"] = "The dataset has no available configurations"
        return result
    result["configs"] = configs
    if config is None:
        if len(configs) == 1:
            config = configs[0]
        elif "default" in configs:
            config = "default"
        elif "main" in configs:
            config = "main"
        else:
            result["error"] = "Choose a dataset configuration from the available configurations"
            return result
    if config not in configs:
        result["error"] = "The requested configuration is unavailable. Choose an available configuration"
        return result
    result["config"] = config
    try:
        splits = split_names(repo, config_name=config, revision=sha)
    except Exception:
        result["error"] = "Could not read dataset splits. Check access and the selected configuration"
        return result
    if not isinstance(splits, list) or any(not isinstance(name, str) for name in splits):
        result["error"] = "Could not read the available dataset splits"
        return result
    result["splits"] = splits
    if split not in splits:
        result["error"] = "The requested split is unavailable. Choose an available split"
        return result
    try:
        stream = loader(repo, name=config, split=split, revision=sha, streaming=True)
        raw_rows = list(itertools.islice(stream, preview_rows))
    except Exception:
        result["error"] = "Could not preview dataset rows. Check access and network connectivity"
        return result
    if not raw_rows:
        result["error"] = "The selected dataset split contains no rows"
        return result
    if any(not isinstance(row, dict) for row in raw_rows):
        result["error"] = "Dataset rows must be objects with named columns"
        return result
    result["columns"] = _columns(raw_rows)
    result["rows"], result["truncated_fields"] = _display_rows(raw_rows)
    result["truncated"] = bool(result["truncated_fields"])
    try:
        fields = resolve_mode_fields(raw_rows[0], dataset_mode, prompt_field, target_field, text_field)
        if dataset_mode == "corpus":
            result["text_field"] = fields[1]
        else:
            result["prompt_field"] = fields[0]
            if dataset_mode == "paired":
                result["target_field"] = fields[1]
        mapped = []
        for row_index, row in enumerate(raw_rows):
            remaining = 20 - len(mapped)
            if remaining == 0:
                result["mapped_rows_truncated"] = True
                break
            selector = fields[1] if dataset_mode == "corpus" else fields[0]
            tokens = selector_tokens(row, selector)
            available = 1
            if None in tokens:
                try:
                    selected = _walk(row, tokens[:tokens.index(None)])
                except KeyError:
                    raise DatasetError("The selected example array is missing from a preview row") from None
                if not isinstance(selected, (list, tuple)):
                    raise DatasetError("The selected example field must contain an array")
                available = len(selected)
            try:
                prepared = prepare_task_row(row, fields, dataset_mode, result["task_metadata"]["selected_task"], getattr(stream, "features", None))
            except ValueError as error:
                raise DatasetError(str(error)) from None
            for pair in itertools.islice(mapped_examples(prepared, fields, dataset_mode), remaining):
                mapped.append({"row_index": row_index, **pair})
            if available > remaining:
                result["mapped_rows_truncated"] = True
                break
        result["mapped_rows"], shortened = _display_rows(mapped)
        result["truncated_fields"].extend(path.replace("rows[", "mapped_rows[", 1) for path in shortened)
        result["truncated"] = bool(result["truncated_fields"])
    except DatasetError as error:
        # Preserve requested selectors and raw rows so the fields can be corrected.
        active_roles = (("text", text_field),) if dataset_mode == "corpus" else (("prompt", prompt_field),)
        if dataset_mode == "paired":
            active_roles += (("target", target_field),)
        for role, explicit in active_roles:
            if result[f"{role}_field"] is None:
                try:
                    result[f"{role}_field"] = resolve_field(raw_rows[0], explicit, role)
                except DatasetError:
                    pass
        result["mapping_error"] = str(error)
    return result


def fetch(
    *, repo: str, split: str, output: Path, receipt: Path, limit: int = 100,
    config: str | None = None, revision: str | None = None,
    prompt_field: str | None = None, target_field: str | None = None,
    dataset_mode: str = "paired", text_field: str | None = None,
    hf_task: str | None = None, model_id: str | None = None,
    api: Any = None, loader: Callable[..., Iterable[dict[str, Any]]] | None = None,
    config_names: Callable[..., list[str]] | None = None,
    runtime_version: str | None = None,
) -> dict[str, Any]:
    """Stream at most limit mapped examples, then atomically publish files without overwrite.

    api and loader can be injected for offline contract tests. Real requests use
    the existing Hugging Face environment/cache authentication without exposing it.
    """
    _validate_repo(repo)
    _validate_mode(dataset_mode)
    try:
        hf_task = task_metadata.normalize_task(hf_task)
    except ValueError as error:
        raise DatasetError(str(error)) from None
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100_000:
        raise DatasetError("Dataset limit must be between 1 and 100000 examples")
    if not split or not split.strip():
        raise DatasetError("Choose a nonempty dataset split")
    output, receipt = Path(output).absolute(), Path(receipt).absolute()
    if output.resolve() == receipt.resolve():
        raise DatasetError("Dataset and receipt must use different output paths")
    for path in (output, receipt):
        if path.exists() or path.is_symlink():
            raise DatasetError(f"Output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    requested_config = config
    if api is None or loader is None:
        api, loader, runtime_version = _runtime()
        if config_names is None:
            import datasets
            config_names = datasets.get_dataset_config_names
    try:
        info = api.dataset_info(repo_id=repo, revision=revision)
    except Exception:
        raise DatasetError("Could not resolve the dataset revision. Check its identifier, access and network connection") from None
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", sha):
        raise DatasetError("The Hub returned no immutable revision for this dataset")
    task = task_metadata.assess_task(info, hf_task, _model_task_info(api, model_id))
    try:
        selected_task = task_metadata.require_task(task)
    except ValueError as error:
        raise DatasetError(str(error)) from None
    if config is None and config_names is not None:
        try:
            available = config_names(repo, revision=sha)
        except Exception:
            raise DatasetError("Could not read dataset configurations. Check access and supported data files") from None
        if not isinstance(available, list) or not available or any(not isinstance(name, str) for name in available):
            raise DatasetError("The dataset has no available configurations")
        if len(available) == 1:
            config = available[0]
        elif "default" in available:
            config = "default"
        elif "main" in available:
            config = "main"
        else:
            raise DatasetError("Choose a dataset configuration with --dataset-config")
    try:
        stream = loader(repo, name=config, split=split, revision=sha, streaming=True)
    except Exception:
        raise DatasetError("Could not open the dataset. Check its configuration and split, and use a repository containing supported data files") from None

    staged: list[Path] = []
    published: list[tuple[Path, Path]] = []
    try:
        digest = hashlib.sha256()
        fields: tuple[str, str] | None = None
        columns: set[str] = set()
        count = 0
        source_rows = 0
        transformations: dict[str, int] = {}
        target_sources: dict[str, int] = {}
        with tempfile.NamedTemporaryFile(prefix=".vespasian-dataset-", dir=output.parent, delete=False) as handle:
            data_temp = Path(handle.name)
            staged.append(data_temp)
            try:
                for index, row in enumerate(stream, start=1):
                    source_rows = index
                    if not isinstance(row, dict):
                        raise DatasetError(f"Dataset row {index} is not an object")
                    if fields is None:
                        fields = resolve_mode_fields(row, dataset_mode, prompt_field, target_field, text_field)
                    columns.update(str(column) for column in row)
                    try:
                        prepared = prepare_task_row(row, fields, dataset_mode, selected_task, getattr(stream, "features", None))
                    except ValueError as error:
                        raise DatasetError(str(error)) from None
                    for pair in mapped_examples(prepared, fields, dataset_mode):
                        strategy = pair["transformation"]["kind"]
                        transformations[strategy] = transformations.get(strategy, 0) + 1
                        if "target_source" in pair:
                            target_sources[pair["target_source"]] = target_sources.get(pair["target_source"], 0) + 1
                        canonical = {"prompt": pair["prompt"], "target": pair["target"]}
                        if dataset_mode != "paired":
                            canonical["mode"] = dataset_mode
                        encoded = (json.dumps(canonical, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                        handle.write(encoded)
                        digest.update(encoded)
                        count += 1
                        if count == limit:
                            break
                    if count == limit:
                        break
            except DatasetError:
                raise
            except Exception:
                raise DatasetError("Dataset streaming failed. Check access and network connectivity") from None
            handle.flush()
            os.fsync(handle.fileno())
        if not count or fields is None:
            raise DatasetError("The selected dataset split contains no rows")
        stream_info = getattr(stream, "info", None)
        resolved_config = getattr(stream_info, "config_name", None) or config
        used_roots = {part if part in columns else re.split(r"[.\[]", part, 1)[0]
                      for field in fields if field for part in selector_parts(row, field)}
        expanded = any(None in selector_tokens(row, part)
                       for field in fields if field for part in selector_parts(row, field))
        metadata: dict[str, Any] = {
            "source": "huggingface", "repo": repo, "revision_requested": revision,
            "hf_task": selected_task, "task_metadata": task,
            "revision": sha, "config_requested": requested_config, "config": resolved_config,
            "split": split, "selection": "first_examples" if expanded else "first_rows", "limit": limit,
            "source_rows": source_rows,
            "rows": count, "limit_reached": count == limit,
            "dataset_mode": dataset_mode,
            "prompt_field": fields[0] if dataset_mode != "corpus" else None,
            "target_field": fields[1] if dataset_mode == "paired" else None,
            "text_field": fields[1] if dataset_mode == "corpus" else None,
            "mapping": ("paired_array" if dataset_mode == "paired" else "array_expansion") if expanded else "field_only",
            "transformations": transformations, "target_sources": target_sources,
            "ignored_columns": sorted(columns - used_roots),
            "output": str(output), "sha256": digest.hexdigest(),
            "datasets_version": runtime_version,
        }
        with tempfile.NamedTemporaryFile(prefix=".vespasian-receipt-", dir=receipt.parent, delete=False) as handle:
            receipt_temp = Path(handle.name)
            staged.append(receipt_temp)
            handle.write((json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if the destination exists.
        # Both temporary files live on their destination filesystem.
        for temporary, destination in ((data_temp, output), (receipt_temp, receipt)):
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise DatasetError(f"Output already exists: {destination}") from None
            published.append((temporary, destination))
        return metadata
    except BaseException:
        # Undo only this invocation's own completed links, never someone else's
        # concurrent replacement, if the two-file publication cannot complete.
        for temporary, destination in reversed(published):
            try:
                if os.path.samefile(temporary, destination):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch a pinned text dataset from Hugging Face")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--config")
    parser.add_argument("--revision")
    parser.add_argument("--prompt-field", help="Prompt selector, such as question or examples[].input")
    parser.add_argument("--target-field", help="Target selector, such as answer or examples[].target")
    parser.add_argument("--dataset-mode", choices=("paired", "corpus", "prompts"), default="paired")
    parser.add_argument("--model-id", help="Optional model metadata for task suggestions; not an architecture guarantee")
    parser.add_argument("--task", dest="hf_task", help="Hugging Face text task; required when dataset tags are missing or ambiguous")
    parser.add_argument("--text-field", help="Corpus text selector, such as text or documents[].content")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--preview", action="store_true", help="Read dataset metadata and sample rows without importing")
    parser.add_argument("--preview-rows", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        values = vars(args)
        is_preview = values.pop("preview")
        preview_rows = values.pop("preview_rows")
        # Third-party diagnostics belong to stderr; stdout is exactly one JSON object.
        with contextlib.redirect_stdout(sys.stderr):
            if is_preview:
                for key in ("output", "receipt", "limit"):
                    values.pop(key)
                result = preview(**values, preview_rows=preview_rows)
            else:
                if values["output"] is None or values["receipt"] is None:
                    raise DatasetError("Dataset import requires --output and --receipt")
                result = fetch(**values)
    except DatasetError as error:
        print(f"Vespasian: {error}", file=sys.stderr)
        return 1
    except OSError:
        message = ("Could not read the dataset preview. Check runtime permissions"
                   if is_preview else "Could not publish the dataset. Check output permissions and free space")
        print(f"Vespasian: {message}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
