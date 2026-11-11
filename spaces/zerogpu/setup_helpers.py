"""Pure setup rules shared by the form, review and run request."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import html
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples" if (ROOT / "examples").is_dir() else ROOT.parent.parent / "examples"
METRICS = [("Let the editor select", "auto"), ("Exact match", "exact_match"),
           ("Token F1", "token_f1"), ("Goal score", "semantic")]
METRIC_DESCRIPTIONS = {
    "auto": "The editor selects a compatible metric after assessing your examples and objective.",
    "exact_match": "Higher is better. Answers must match the reference after normalising case and whitespace.",
    "token_f1": "Higher is better. The score measures word overlap between each answer and its reference.",
    "semantic": "Higher is better. The editor scores each generated answer against your objective on a scale from 0 to 1.",
}
# These are the controller's accounting rates, kept in step with src/budget.rs.
EDITOR_RATES = {
    "openai": {"gpt-6.1-sol": (2., 10.), "gpt-6-astra": (10., 50.), "gpt-6.1-terra": (1., 5.), "gpt-6-luna": (.1, .5)},
    "anthropic": {"claude-sonnet-5-5": (2., 10.), "claude-opus-5-5": (4., 20.),
                  "claude-sonnet-4-6": (3., 15.), "claude-opus-4-6": (5., 25.)},
}
_BASE = {
    "model": "Qwen/Qwen3.5-0.8B", "hf_task": "text-generation", "save_format": "model", "prompt_format": "auto", "patch_backup": False, "patch_repo": "", "patch_create": False, "dataset_location": "file", "dataset_source": "",
    "dataset_config": "", "dataset_split": "train", "test_dataset_source": "",
    "test_dataset_config": "", "test_dataset_split": "test", "test_example_limit": 8,
    "prompt_field": "", "target_field": "", "text_field": "", "example_limit": 8,
    "controls": "", "budget": 10., "max_turns": None, "reasoning_effort": "low", "max_new_tokens": 128,
    "health_checks": ["copying"], "health_count": 4, "health_policy": "track",
    "health_limit_pp": 10., "save_trajectory": True,
}
PRESETS = {
    "greetings": {
        "label": "Greeting responses", "description": "Use eight paired examples with explicit edit, validation and test splits.",
        "config": {**_BASE, "dataset_file": str(EXAMPLES / "greetings.jsonl"), "target_type": "factual", "dataset_mode": "paired",
                   "prompt_field": "prompt", "target_field": "target", "metric": "exact_match",
                   "goal": "Respond to greetings with exactly Salve! while preserving answers to other questions."},
    },
    "style": {
        "label": "Reference writing", "description": "Use eight prose passages as style examples for generation without a prompt.",
        "config": {**_BASE, "dataset_file": str(EXAMPLES / "style.jsonl"), "target_type": "style", "dataset_mode": "corpus",
                   "text_field": "text", "metric": "semantic",
                   "goal": "Write original, concise prose with concrete sensory details and the restrained tone of the reference passages."},
    },
    "style-prompts": {
        "label": "Writing from prompts", "description": "Use eight prompts and a style objective to evaluate generated scenes.",
        "config": {**_BASE, "dataset_file": str(EXAMPLES / "style-prompts.jsonl"), "target_type": "style", "dataset_mode": "prompts",
                   "prompt_field": "prompt", "metric": "semantic",
                   "goal": "Write a brief, original scene for each prompt using concrete details, plain language and a restrained tone."},
    },
    "gsm8k": {
        "label": "GSM8K word problems", "description": "Retrieve sixteen public word problems from openai/gsm8k.",
        "config": {**_BASE, "dataset_location": "hub", "dataset_file": None, "dataset_source": "openai/gsm8k",
                   "dataset_config": "main", "example_limit": 16, "target_type": "factual", "dataset_mode": "paired",
                   "prompt_field": "question", "target_field": "answer", "metric": "semantic",
                   "goal": "Solve each word problem correctly, with a clear calculation and the correct final numeric answer."},
    },
}


def task_choices():
    import hub
    return list(hub.task_metadata.TASK_CHOICES)


def task_help(metadata):
    metadata = metadata or {}
    message = metadata.get("error") or metadata.get("warning")
    suggestions = metadata.get("suggested_tasks") or []
    if not message:
        message = ("Suggested from dataset metadata: " + ", ".join(suggestions) + ". Review the task and mapped examples."
                   if suggestions else "Choose a task explicitly when dataset tags are missing or ambiguous. Tags do not replace checking the mapped examples.")
    return '<p class="vs-inline-note">' + html.escape(str(message)) + '</p>'


def starter_presets():
    return [(preset["label"], key) for key, preset in PRESETS.items()]


def preset_config(name):
    if name not in PRESETS:
        raise ValueError("Choose one of the starter experiments")
    return deepcopy(PRESETS[name]["config"])


def metric_options(mode):
    return list(METRICS) if mode == "paired" else [("Goal score", "semantic")]


metric_choices = metric_options


def metric_for_mode(mode, current="auto"):
    allowed = {value for _, value in metric_options(mode)}
    return current if current in allowed else ("auto" if mode == "paired" else "semantic")


def suggested_objective(target_type, mode, reference_field="", text_field=""):
    if mode == "corpus":
        return f"Match the tone and writing style of the examples in {text_field or 'the text field'}, while writing original content."
    if mode == "prompts":
        if target_type == "style":
            return "Answer each prompt in the writing style you specify, while preserving its meaning and requested facts."
        if target_type == "format":
            return "Answer each prompt in the required output format, including every requested field."
        return "Answer each prompt accurately and satisfy the behaviour described in this objective."
    field = reference_field or "the reference field"
    if target_type == "style":
        return f"Match the tone and writing style shown in {field}, while answering each prompt accurately."
    if target_type == "format":
        return f"Follow the output format shown in {field} for every answer."
    if target_type == "classification":
        return f"Return the correct class from {field} for each input."
    if target_type == "custom":
        return f"Produce answers that satisfy the behaviour demonstrated in {field}."
    return f"Return answers that agree with the references in {field}."


objective_suggestion = suggested_objective


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalise_config(config):
    result = dict(config)
    turns = result.get("max_turns")
    if turns is None or isinstance(turns, str) and not turns.strip():
        result["max_turns"] = None
    else:
        number = _number(turns)
        if number is None or int(number) != number or not 1 <= number <= 4294967295:
            raise ValueError("Maximum edit turns must be blank or a positive whole number up to 4294967295")
        result["max_turns"] = int(number)
    policy = result.get("health_policy")
    if policy == "track":
        result["health_max_drop"] = None
    elif policy == "enforce":
        limit = _number(result.get("health_limit_pp"))
        if limit is None or not 0 <= limit <= 100:
            raise ValueError("Set the allowed health drop between 0 and 100 percentage points")
        result["health_max_drop"] = limit / 100
    elif policy is not None:
        raise ValueError("Choose whether to track health scores or enforce a regression limit")
    for key in ("input_price", "output_price"):
        value = result.get(key)
        if value is None or isinstance(value, str) and not value.strip():
            result[key] = None
        else:
            number = _number(value)
            if number is None or number <= 0:
                raise ValueError("Token prices must be positive finite amounts in USD per million tokens")
            result[key] = number
    return result


normalize_config = normalise_config


def fingerprint(config, test=False):
    """Identify data selection without storing credentials in display state."""
    shared = ("hf_task", "dataset_mode", "prompt_field", "target_field", "text_field")
    fields = (("test_dataset_source", "test_dataset_config", "test_dataset_split", "test_example_limit") if test else
              ("dataset_location", "dataset_source", "dataset_file", "dataset_config", "dataset_split", "example_limit"))
    payload = {}
    for name in (*fields, *shared):
        value = config.get(name)
        if isinstance(value, str) and not value.strip():
            value = None
        if name.endswith("example_limit") and _number(value) is not None:
            value = float(value)
        payload[name] = value
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def editor_choices(provider):
    return [(model, model) for model in EDITOR_RATES.get(provider, {})]


def editor_pricing(provider, model, input_price=None, output_price=None):
    rates = EDITOR_RATES.get(provider, {}).get(model)
    has_input = input_price is not None and not (isinstance(input_price, str) and not input_price.strip())
    has_output = output_price is not None and not (isinstance(output_price, str) and not output_price.strip())
    incoming = _number(input_price) if has_input else (rates[0] if rates else None)
    outgoing = _number(output_price) if has_output else (rates[1] if rates else None)
    return {"input_per_million": incoming, "output_per_million": outgoing, "known": rates is not None}


def choose_editor(provider, current=None, remembered=None):
    memory = dict(remembered or {})
    old_provider = memory.pop("_provider", None)
    if old_provider and current:
        memory[old_provider] = current
    value = memory.get(provider) or next(iter(EDITOR_RATES.get(provider, {})), "")
    memory["_provider"] = provider
    return value, memory


def _required(errors, config, key, message):
    if not str(config.get(key) or "").strip():
        errors[key] = message


def _range(errors, config, key, low, high, label, integer=False):
    value = _number(config.get(key))
    if value is None or not low <= value <= high or (integer and int(value) != value):
        errors[key] = f"Set {label} between {low:g} and {high:g}."


def _preview_errors(config, preview, test=False):
    prefix = "test_" if test else ""
    field = prefix + "dataset_source" if test or config.get("dataset_location", "hub") == "hub" else "dataset_file"
    label = "test dataset" if test else "dataset"
    if not preview or not preview.get("_current"):
        return {field: f"Preview the current {label} selection before continuing."}
    if preview.get("error"):
        return {field: str(preview["error"])[:500]}
    if preview.get("mapping_error"):
        mapping_field = "text_field" if config.get("dataset_mode") == "corpus" else "prompt_field"
        return {mapping_field: str(preview["mapping_error"])[:500]}
    if not preview.get("mapped_rows"):
        return {field: f"Choose {label} fields that produce at least one usable example."}
    return {}


def validate_step(config, step, preview=None, test_preview=None):
    """Return field-addressable errors in form order."""
    if step in {"review", "all"}:
        errors = {}
        for current in ("target", "behaviour", "data", "run"):
            errors.update(validate_step(config, current, preview, test_preview))
        return errors
    errors = {}
    if step == "behaviour":
        if config.get("target_type") not in {"factual", "style", "format", "classification", "custom"}:
            errors["target_type"] = "Choose the type of change you want to make."
        mode = config.get("dataset_mode", "paired")
        if mode not in {"paired", "corpus", "prompts"}:
            errors["dataset_mode"] = "Choose paired examples, reference writing or prompts."
        _required(errors, config, "goal", "Describe the behaviour you want the edited model to produce.")
        if config.get("metric", "auto") not in {value for _, value in metric_options(mode)}:
            errors["metric"] = "Choose Goal score for reference writing and prompt-only examples."
        checks = config.get("health_checks") or []
        if not isinstance(checks, list) or len(checks) > 4:
            errors["health_checks"] = "Choose up to four preservation checks."
        if checks:
            _range(errors, config, "health_count", 1, 16, "examples per preservation check", integer=True)
        health_only = {key: value for key, value in config.items() if key.startswith("health_")}
        try:
            normalise_config(health_only)
        except ValueError as error:
            errors["health_limit_pp" if config.get("health_policy") == "enforce" else "health_policy"] = str(error)
    elif step == "target":
        _required(errors, config, "model", "Choose a base model.")
    elif step == "data":
        task = config.get("hf_task")
        if task not in {value for _, value in task_choices()}:
            errors["hf_task"] = "Choose a supported task; dataset tags alone do not establish compatibility."
        metadata = (preview or {}).get("task_metadata") or {}
        if metadata.get("error"):
            errors["hf_task"] = str(metadata["error"])
        location = config.get("dataset_location", "hub")
        if location in {"file", "upload"}:
            path = config.get("dataset_file")
            if not path:
                errors["dataset_file"] = "Choose a dataset file or load a starter experiment."
            elif not Path(path).is_file():
                errors["dataset_file"] = "Choose the dataset file again."
        elif location == "hub":
            if not str(config.get("dataset_source") or "").strip():
                errors["dataset_source"] = "Choose a dataset repository or search the Hub."
            _required(errors, config, "dataset_split", "Choose the dataset split.")
        else:
            errors["dataset_location"] = "Choose Hugging Face or a dataset file."
        _range(errors, config, "example_limit", 5, 128, "the example limit", integer=True)
        if not errors:
            errors.update(_preview_errors(config, preview))
        if config.get("test_dataset_source"):
            _required(errors, config, "test_dataset_split", "Choose the held-out test split.")
            _range(errors, config, "test_example_limit", 1, 128, "the test example limit", integer=True)
            errors.update(_preview_errors(config, test_preview, test=True))
    elif step in {"run", "editor"}:
        if not isinstance(config.get("patch_backup", False), bool) or not isinstance(config.get("patch_create", False), bool):
            errors["patch_repo"] = "Choose whether to authorize private patch backup. Declarative true/false choices are required."
        if config.get("patch_backup"):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", str(config.get("patch_repo") or "")) or ".." in str(config.get("patch_repo")):
                errors["patch_repo"] = "Name the private repository that will receive your patches."
        if config.get("save_format", "model") not in {"patch", "model"}:
            errors["save_format"] = "Choose patches only or a materialised model."
        if config.get("prompt_format", "auto") not in {"auto", "chat", "plain"}:
            errors["prompt_format"] = "Choose an available prompt format."
        provider = config.get("editor")
        if provider not in EDITOR_RATES:
            errors["editor"] = "Choose OpenAI or Anthropic as the editor."
        _required(errors, config, "editor_model", "Choose an editor model available to your API account.")
        _required(errors, config, "api_key", "Add the API key for your editor account.")
        _range(errors, config, "budget", .01, 100, "the editor budget in USD")
        turns = config.get("max_turns")
        if turns is not None and not (isinstance(turns, str) and not turns.strip()):
            _range(errors, config, "max_turns", 1, 4294967295, "the maximum edit turns", integer=True)
        if config.get("reasoning_effort", "low") not in {"low", "medium", "high"}:
            errors["reasoning_effort"] = "Choose a supported reasoning effort."
        _range(errors, config, "max_new_tokens", 1, 128, "maximum generated tokens", integer=True)
        prices = editor_pricing(provider, config.get("editor_model"), config.get("input_price"), config.get("output_price"))
        for key, rate in (("input_price", prices["input_per_million"]), ("output_price", prices["output_per_million"])):
            if rate is None or rate <= 0:
                errors[key] = "Set a positive USD price per million tokens for this model."
    else:
        raise ValueError("Unknown setup step")
    return errors


def split_summary(config, preview=None, test_preview=None):
    """Use selected upload counts when present, otherwise state a bounded estimate."""
    preview = preview or {}
    counts = preview.get("counts") or preview.get("split_counts")
    exact = isinstance(counts, dict) and bool(counts)
    if exact:
        result = {name: int(counts.get(name, 0)) for name in ("edit", "validation", "test", "preserve")}
    else:
        limit = int(_number(config.get("example_limit")) or 0)
        validation = max(1, limit // 5) if limit else 0
        result = {"edit": max(0, limit - validation), "validation": validation, "test": 0, "preserve": 0}
    if config.get("test_dataset_source"):
        tested = (test_preview or {}).get("counts")
        if isinstance(tested, dict):
            result["test"] += sum(int(value) for value in tested.values())
        else:
            result["test"] += int(_number(config.get("test_example_limit")) or 0)
            exact = False
    return {**result, "total": sum(result.values()), "exact": exact,
            "label": "Selected examples" if exact else "Planned maximum examples"}


def review_html(config, preview=None, test_preview=None):
    """Render an allowlist of public fields, never credentials or input mappings."""
    esc = lambda value: html.escape(str(value if value is not None else ""), quote=True)
    counts = split_summary(config, preview, test_preview)
    mode = {"paired": "Paired examples", "corpus": "Reference writing", "prompts": "Prompts with a rubric"}.get(config.get("dataset_mode"), "")
    metric = dict((value, label) for label, value in METRICS).get(config.get("metric", "auto"), "")
    try:
        normal = normalise_config({key: value for key, value in config.items() if key.startswith("health_")})
        drop = normal.get("health_max_drop")
        policy = "Track scores" if drop is None else f"Reject a drop above {100 * float(drop):g} percentage points"
    except (ValueError, TypeError):
        policy = "Choose a valid preservation limit"
    source = Path(config["dataset_file"]).name if config.get("dataset_file") and config.get("dataset_location") in {"file", "upload"} else config.get("dataset_source", "")
    budget = _number(config.get("budget"))
    entries = [
        ("Base model", config.get("model")), ("Task", config.get("hf_task") or "Choose a task"), ("Final output", "Patches only" if config.get("save_format") == "patch" else "Materialised model and patches"), ("Prompt format", config.get("prompt_format", "auto")), ("Data", source), ("Example type", mode),
        ("Source", " · ".join(str(value) for value in (config.get("dataset_config"), config.get("dataset_split")) if value)),
        (counts["label"], f"{counts['edit']} edit · {counts['validation']} validation · {counts['test']} test" + (f" · {counts['preserve']} preserve" if counts["preserve"] else "")),
        ("Objective metric", metric), ("Preservation policy", policy),
        ("Preservation checks", ", ".join(config.get("health_checks") or []) or "None selected"),
        ("Editor model", config.get("editor_model")), ("Patch backup", config.get("patch_repo") if config.get("patch_backup") else "Local/download only; temporary Space storage"),
        ("Editor budget", f"${budget:.2f} USD" if budget is not None else "Choose a budget"),
        ("Maximum edit turns", config.get("max_turns") or "No turn cap · spending limit applies"),
        ("Reasoning effort", config.get("reasoning_effort", "low")),
        ("Answer token limit", config.get("max_new_tokens", 128)),
        ("Sequence token limit", 1024), ("Seed", 42),
        ("Editor output token limit", 2048), ("Preservation tolerance", 0),
    ]
    rows = "".join(f'<div><dt>{esc(label)}</dt><dd>{esc(value)}</dd></div>' for label, value in entries)
    return (f'<section class="vs-review-summary"><h3>Experiment details</h3><p class="vs-review-goal">{esc(config.get("goal"))}</p>'
            f'<dl>{rows}</dl><p class="vs-inline-note">The editor budget covers frontier API calls. GPU quota is accounted for separately.</p></section>')
