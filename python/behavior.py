"""Conservative, deterministic diagnostics for measured, non-test generations.

Signals describe observations, not correctness judgments or new acceptance scores.
Only a complete numeric scalar or JSON value can justify a content/format split.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import json
import re

VISIBLE_SPLITS = ("edit", "validation", "preserve")
ROW_ID_LIMIT = 8
SIGNALS = (
    "repetition", "length_truncation", "terminated", "unknown_termination",
    "empty_output", "reference_mismatch", "format_only_mismatch",
    "content_difference", "unclassified_reference_mismatch", "preservation_mismatch",
)
LIMITATIONS = (
    "Diagnostics are observations, not acceptance scores. Repetition is a conservative heuristic, "
    "not proof of an unwanted loop. Length means a generation limit, not an incorrect answer. "
    "Termination does not prove completeness. Content/format comparisons cover only whole numeric "
    "scalars or valid JSON; other mismatches remain unclassified. No semantic correctness is inferred. "
    "Preservation mismatches are reference failures, not necessarily regressions from baseline."
)
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def visible_samples(samples):
    """Canonical IDs cannot be supplied by dataset text or refer to sealed tests."""
    counters = Counter()
    for row in samples:
        split = row.get("split")
        if split not in VISIBLE_SPLITS:
            continue
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not re.fullmatch(rf"{split}:[0-9]{{1,10}}", row_id):
            row_id = f"{split}:{counters[split]}"
        counters[split] += 1
        yield row_id, row


def _repetition(text):
    # Require a substantial consecutive repeated span, not scattered n-gram reuse.
    # A width cap keeps work linear in generated output length, including long traces.
    tokens = re.findall(r"\w+|[^\w\s]", text.casefold())
    word_prefix = [0]
    for token in tokens:
        word_prefix.append(word_prefix[-1] + int(any(char.isalnum() for char in token)))
    best = None
    for width in range(1, min(128, len(tokens) // 3) + 1):
        equal = 0
        for end in range(width, len(tokens)):
            equal = equal + 1 if tokens[end] == tokens[end - width] else 0
            copies = 1 + equal // width
            span = copies * width
            if copies < 3 or span < 24:
                continue
            start = end + 1 - span
            # Punctuation-only runs and ordinary document separators are not evidence.
            words = word_prefix[end + 1] - word_prefix[start]
            if words < 12:
                continue
            if best is None or span > best["span_tokens"]:
                best = {"method": "consecutive_token_blocks", "start_token": start,
                        "block_tokens": width, "copies": copies, "span_tokens": span,
                        "total_text_tokens": len(tokens), "span_fraction": round(span / len(tokens), 4)}
    return best


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Nonfinite JSON value")

    return json.loads(text, parse_float=Decimal, parse_int=Decimal,
                      parse_constant=invalid, object_pairs_hook=pairs)


def _equal_json(left, right):
    # Python otherwise equates true with 1, and false with 0.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal_json(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_equal_json(a, b) for a, b in zip(left, right))
    return left == right


def _reference_comparison(prediction, target):
    p, t = prediction.strip(), target.strip()
    if NUMBER.fullmatch(p) and NUMBER.fullmatch(t):
        try:
            return "numeric_scalar", Decimal(p) == Decimal(t)
        except InvalidOperation:
            return None
    # Entire JSON only. Do not fish a desired answer out of explanations, reasoning
    # traces, markdown fences, or mixed prose and claim the content is correct.
    try:
        expected, actual = _strict_json(t), _strict_json(p)
        return "whole_json_value", _equal_json(actual, expected)
    except (ValueError, RecursionError, InvalidOperation):
        return None


def analyze_sample(row):
    """Return bounded evidence, without copying prompts, targets or generations."""
    prediction = row.get("prediction")
    if not isinstance(prediction, str):
        return {"signals": [], "evidence": {"status": "no_prediction"}}
    signals, evidence = [], {}
    reason = row.get("finish_reason")
    if reason == "length":
        signals.append("length_truncation")
    elif reason in ("stop", "eos", "eos_token"):
        signals.append("terminated")
    else:
        signals.append("unknown_termination")
    evidence["termination"] = {
        "finish_reason": reason if reason in ("length", "stop", "eos", "eos_token") else "unknown",
        "generated_tokens": row.get("generated_tokens") if isinstance(row.get("generated_tokens"), int) and not isinstance(row.get("generated_tokens"), bool) and row["generated_tokens"] >= 0 else None,
    }
    if not prediction.strip():
        signals.append("empty_output")
    repeated = _repetition(prediction)
    if repeated:
        signals.append("repetition")
        evidence["repetition"] = repeated
    if row.get("mode", "paired") == "paired" and row.get("exact_match") is False and isinstance(row.get("target"), str):
        signals.append("reference_mismatch")
        comparison = _reference_comparison(prediction, row["target"])
        if comparison is None:
            signals.append("unclassified_reference_mismatch")
            evidence["reference"] = {"method": "existing_exact_match", "classification": "unclassified"}
        else:
            method, equal = comparison
            signals.append("format_only_mismatch" if equal else "content_difference")
            evidence["reference"] = {"method": method, "equivalent_value": equal,
                                     "classification": "format_only" if equal else "content_difference"}
        if row.get("split") == "preserve":
            signals.append("preservation_mismatch")
    return {"signals": signals, "evidence": evidence}


def annotate_samples(samples, *, row_ids=None):
    """Copy rows and annotate non-test rows; worker may supply full-dataset IDs.

    Without row_ids, IDs are regenerated, never taken from untrusted dataset fields.
    For batched evaluation, row_ids must be parallel to samples and generated by
    the worker from its complete initialised dataset, including omitted splits.
    """
    if row_ids is not None and len(row_ids) != len(samples):
        raise ValueError("row_ids must have one entry per sample")
    # Build by position instead of trusting caller-supplied IDs or annotations.
    results = []
    counters = Counter()
    for index, row in enumerate(samples):
        result = dict(row)
        split = row.get("split")
        if split in VISIBLE_SPLITS:
            row_id = row_ids[index] if row_ids is not None else f"{split}:{counters[split]}"
            if not isinstance(row_id, str) or not re.fullmatch(rf"{split}:[0-9]{{1,10}}", row_id):
                raise ValueError("row_ids must be canonical non-test IDs matching each sample split")
            result["row_id"] = row_id
            counters[split] += 1
            result["behavior"] = analyze_sample(row)
        results.append(result)
    return results


def summarize_evaluation(samples):
    """Bounded summary measured exclusively from actual, non-test samples."""
    signals = {name: {"count": 0, "row_ids": []} for name in SIGNALS}
    splits = {split: {"count": 0, "signals": {name: 0 for name in SIGNALS}} for split in VISIBLE_SPLITS}
    reference_count = 0
    diagnosed_count = 0
    for row_id, row in visible_samples(samples):
        split = splits[row["split"]]
        split["count"] += 1
        diagnosed_count += int(isinstance(row.get("prediction"), str))
        if row.get("mode", "paired") == "paired" and isinstance(row.get("exact_match"), bool) and isinstance(row.get("target"), str):
            reference_count += 1
        for name in analyze_sample(row)["signals"]:
            signal = signals[name]
            signal["count"] += 1
            split["signals"][name] += 1
            if len(signal["row_ids"]) < ROW_ID_LIMIT:
                signal["row_ids"].append(row_id)
    for signal in signals.values():
        signal["row_ids_truncated"] = signal["count"] > len(signal["row_ids"])
    return {"schema_version": 1, "scope": "edit_validation_preserve", "sample_count": sum(s["count"] for s in splits.values()),
            "diagnosed_count": diagnosed_count, "reference_count": reference_count, "signals": signals, "by_split": splits, "limitations": LIMITATIONS}
