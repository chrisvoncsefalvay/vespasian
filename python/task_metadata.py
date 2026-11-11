"""Shared Hub task metadata for text-generation-based behavioural editing.

Tags are suggestions, not architecture or evaluation guarantees. No network calls.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import re

TASK_CHOICES = [
    ("Text generation", "text-generation"),
    ("Text classification via generated labels", "text-classification"),
    ("Question answering via generated answers", "question-answering"),
    ("Summarization", "summarization"),
    ("Translation", "translation"),
]
SUPPORTED_TASKS = frozenset(value for _, value in TASK_CHOICES)
# Task IDs refine categories on Hub dataset cards. Only unambiguous mappings.
TASK_IDS = {
    "sentiment-classification": "text-classification",
    "topic-classification": "text-classification",
    "multi-class-classification": "text-classification",
    "multi-label-classification": "text-classification",
    "natural-language-inference": "text-classification",
    "extractive-qa": "question-answering", "closed-domain-qa": "question-answering",
    "open-domain-qa": "question-answering", "multiple-choice-qa": "question-answering",
    "abstractive-summarization": "summarization", "extractive-summarization": "summarization",
    "language-modeling": "text-generation", "dialogue-modeling": "text-generation",
    "text2text-generation": "text-generation",
}
NON_TEXT_PREFIXES = ("image-", "audio-", "video-", "automatic-speech-", "text-to-image", "text-to-audio", "object-detection", "depth-estimation")


def _get(value: Any, key: str, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return sorted({item.strip() for item in value if isinstance(item, str) and item.strip()})
    return []


def normalize_task(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("Choose a supported Hugging Face text task")
    value = value.strip()
    if value not in SUPPORTED_TASKS:
        raise ValueError("Unsupported editing task. Choose " + ", ".join(sorted(SUPPORTED_TASKS)))
    return value


def model_task_suggestions(model_info: Any = None) -> list[str]:
    tag = _get(model_info, "pipeline_tag")
    if not isinstance(tag, str):
        return []
    if tag in SUPPORTED_TASKS:
        return [tag]
    if tag in {"image-text-to-text", "text2text-generation"}:
        return ["text-generation"]
    return []


def assess_task(dataset_info: Any = None, selected_task: str | None = None,
                model_info: Any = None) -> dict[str, Any]:
    selected = normalize_task(selected_task)
    card = _get(dataset_info, "card_data") or _get(dataset_info, "cardData") or {}
    raw = set(_strings(_get(card, "task_categories")) + _strings(_get(card, "task_ids")))
    for tag in _strings(_get(dataset_info, "tags")):
        if tag.startswith(("task_categories:", "task_ids:")):
            raw.add(tag.split(":", 1)[1])
    suggestions = sorted({TASK_IDS.get(tag, tag) for tag in raw
                          if TASK_IDS.get(tag, tag) in SUPPORTED_TASKS})
    unknown = sorted(tag for tag in raw if TASK_IDS.get(tag, tag) not in SUPPORTED_TASKS)
    unsupported_only = bool(raw) and not suggestions and all(tag.startswith(NON_TEXT_PREFIXES) for tag in raw)
    error = None
    if unsupported_only and selected is None:
        error = "Dataset metadata describes a non-text task. Choose a text dataset or an explicitly prepared text-only adaptation."
    if selected is None and len(suggestions) == 1 and not unknown:
        selected = suggestions[0]
        status = "metadata_suggested"
    elif selected is not None:
        status = "user_selected"
    else:
        status = "choice_required"
    warning = "Dataset tags describe a non-text task; only the explicitly selected text adaptation will be evaluated." if unsupported_only and selected else None
    if selected is not None and suggestions and selected not in suggestions:
        warning = "Selected task differs from dataset tags. Confirm that the mapped text examples express this task."
    if selected in {"text-classification", "question-answering"}:
        adaptation = "Text generation adaptation; no classification or extractive-QA head is assumed. Verify the prompt, textual target and evaluation metric."
    else:
        adaptation = "Task tags do not establish schema, model architecture or evaluation compatibility. Verify mapped text examples."
    model_revision = _get(model_info, "sha")
    if not isinstance(model_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", model_revision):
        model_revision = None
    return {"selected_task": selected, "suggested_tasks": suggestions,
            "metadata_tasks": sorted(raw), "unknown_tags": unknown,
            "requires_choice": selected is None, "status": "incompatible" if error else status,
            "warning": warning, "error": error, "adaptation": adaptation,
            "model_pipeline_tag": _get(model_info, "pipeline_tag") if isinstance(_get(model_info, "pipeline_tag"), str) else None,
            "model_metadata_revision": model_revision,
            "model_suggested_tasks": model_task_suggestions(model_info),
            "selection_source": "user" if selected_task else ("dataset_metadata" if selected else None)}


def require_task(metadata: dict[str, Any]) -> str:
    if metadata.get("error"):
        raise ValueError(metadata["error"])
    if metadata.get("requires_choice"):
        raise ValueError("Dataset task tags are missing, unknown or ambiguous. Choose a task explicitly with --task or the task selector.")
    return metadata["selected_task"]


def prepare_row(row: dict[str, Any], fields: tuple[str, str], mode: str,
                task: str | None, features: Any = None, *, target_value: Any = None) -> dict[str, Any]:
    """Resolve a top-level ClassLabel safely; reject hidden context/label loss.

    Nested labels need an explicit textual mapping rather than guessed names.
    """
    if task == "question-answering" and mode == "paired":
        if fields[0] in {"question", "query"} and any(
                isinstance(row.get(key), str) and row[key].strip()
                for key in ("context", "passage", "document")):
            raise ValueError("This QA row has separate context. Select a prepared prompt containing both question and context; question-only mapping would discard required evidence.")
    if task != "text-classification" or mode != "paired":
        return row
    for pair in (("premise", "hypothesis"), ("sentence1", "sentence2")):
        if fields[0] in pair and all(isinstance(row.get(key), str) and row[key].strip() for key in pair):
            raise ValueError("This classification row has paired input texts. Select a prepared prompt containing both inputs rather than discarding one.")
    target = fields[1]
    value = row.get(target, target_value)
    def numeric(item):
        return (isinstance(item, (int, float)) and not isinstance(item, bool)) or (isinstance(item, (list, tuple)) and any(numeric(child) for child in item))
    if not numeric(value):
        return row
    if target not in row:
        raise ValueError("Nested numeric classification labels need an explicitly prepared textual target column; label names cannot be inferred from IDs.")
    feature = _get(features, target)
    names = _get(feature, "names")
    if names is None:
        names = _get(_get(feature, "feature"), "names")
    if not isinstance(names, (list, tuple)) or not all(isinstance(name, str) for name in names):
        raise ValueError("Numeric classification labels need ClassLabel names or an explicitly prepared textual target column; numeric IDs alone do not define the classes.")
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < len(names) for item in values):
        raise ValueError("Classification label index is outside the ClassLabel vocabulary")
    converted = [names[item] for item in values]
    replacement = converted[0] if not isinstance(value, (list, tuple)) else ", ".join(converted)
    return {**row, target: replacement}


def main(argv=None) -> int:
    """Validate an explicit local-data task without requiring Hub access."""
    import argparse
    import json
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--dataset-mode", choices=("paired", "corpus", "prompts"), default="paired")
    args = parser.parse_args(argv)
    try:
        result = assess_task(selected_task=args.task)
        require_task(result)
        result["dataset_mode"] = args.dataset_mode
    except ValueError as error:
        print(f"Vespasian: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
