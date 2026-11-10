#!/usr/bin/env python3
"""Fixed health examples and deterministic scoring for model weight searches."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

SCHEMA_VERSION = 1
MMLU_REPO = "cais/mmlu"
MMLU_SUBJECTS = """abstract_algebra anatomy astronomy business_ethics clinical_knowledge college_biology college_chemistry college_computer_science college_mathematics college_medicine college_physics computer_security conceptual_physics econometrics electrical_engineering elementary_mathematics formal_logic global_facts high_school_biology high_school_chemistry high_school_computer_science high_school_european_history high_school_geography high_school_government_and_politics high_school_macroeconomics high_school_mathematics high_school_microeconomics high_school_physics high_school_psychology high_school_statistics high_school_us_history high_school_world_history human_aging human_sexuality international_law jurisprudence logical_fallacies machine_learning management marketing medical_genetics miscellaneous moral_disputes moral_scenarios nutrition philosophy prehistory professional_accounting professional_law professional_medicine professional_psychology public_relations security_studies sociology us_foreign_policy virology world_religions""".split()
LOCAL = {
    "copying": ("Copying", "exact_text"),
    "arithmetic": ("Arithmetic", "numeric_exact"),
    "json": ("JSON output", "json_exact"),
}


class HealthError(ValueError):
    """A health-check configuration or source error without credentials."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def check_definition(check: str) -> tuple[str, str]:
    if check in LOCAL:
        return LOCAL[check]
    if check.startswith("mmlu:") and check[5:] in MMLU_SUBJECTS:
        return "MMLU · " + check[5:].replace("_", " "), "choice_logprob_accuracy"
    raise HealthError("Unknown health check. Use copying, arithmetic, json or mmlu:subject")


def parse_selection(specification: str) -> dict[str, Any]:
    try:
        check, value = specification.rsplit(":", 1)
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError()
        examples = int(value)
    except (ValueError, AttributeError):
        raise HealthError("Use check:count, for example copying:20 or mmlu:anatomy:20") from None
    check_definition(check)
    if not 1 <= examples <= 1000:
        raise HealthError("Choose between 1 and 1000 examples per health check")
    return {"check": check, "examples": examples}


def validate_selections(selections: list[dict[str, Any]]) -> None:
    if not isinstance(selections, list) or not selections:
        raise HealthError("Select at least one health check")
    seen = set()
    for selection in selections:
        if not isinstance(selection, dict) or set(selection) != {"check", "examples"}:
            raise HealthError("Each health selection needs a check and example count")
        check = selection["check"]
        if not isinstance(check, str):
            raise HealthError("Health check names must be text")
        check_definition(check)
        examples = selection["examples"]
        if isinstance(examples, bool) or not isinstance(examples, int) or not 1 <= examples <= 1000:
            raise HealthError("Choose between 1 and 1000 examples per health check")
        if check in seen:
            raise HealthError("Each health check can be selected only once")
        seen.add(check)


def _random(seed: int, check: str) -> random.Random:
    digest = hashlib.sha256(f"vespasian-health-v1:{seed}:{check}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _local(check: str, count: int, seed: int) -> list[dict[str, Any]]:
    rng = _random(seed, check)
    rows = []
    seen = set()
    for index in range(count):
        if check == "copying":
            words = ["".join(rng.choice("abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(rng.randint(3, 9))) for _ in range(rng.randint(2, 5))]
            target = f"{index:04d}: " + " ".join(words)
            prompt = "Copy the following text exactly. Return only the text, with the same characters and spacing.\n\n" + target
        elif check == "arithmetic":
            while True:
                a, b = rng.randint(0, 999), rng.randint(1, 99)
                operation = rng.choice(("+", "-", "*"))
                key = (a, b, operation)
                if key not in seen:
                    seen.add(key)
                    break
            target = str(a + b if operation == "+" else a - b if operation == "-" else a * b)
            prompt = f"Calculate {a} {operation} {b}. Return only the integer answer."
        else:
            payload = {"item": f"item-{index:04d}", "count": rng.randint(0, 99), "active": bool(rng.getrandbits(1))}
            target = json.dumps(payload, separators=(",", ":"))
            prompt = f"Return one JSON object with exactly these fields: item is the string \"{payload['item']}\", count is the integer {payload['count']}, and active is {str(payload['active']).lower()}. Return JSON only."
        rows.append({"id": f"{check}:{index:04d}", "check": check, "prompt": prompt, "target": target})
    return rows


def _runtime():
    try:
        import datasets
        from huggingface_hub import HfApi
    except ImportError:
        raise HealthError("Install the dataset runtime with vespasian setup") from None
    if int(datasets.__version__.split(".")[0]) < 4:
        raise HealthError("MMLU checks need datasets 4 or later. Run vespasian setup")
    datasets.disable_progress_bars()
    return HfApi(), datasets.load_dataset


def prepare(selections: list[dict[str, Any]], seed: int, *, api=None, loader=None) -> dict[str, Any]:
    validate_selections(selections)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise HealthError("The health seed must be between 0 and 4294967295")
    revision = None
    if any(selection["check"].startswith("mmlu:") for selection in selections):
        if api is None or loader is None:
            api, loader = _runtime()
        try:
            revision = api.dataset_info(MMLU_REPO).sha
        except Exception:
            raise HealthError("Could not resolve MMLU. Check the network connection or use local health checks") from None
        if not isinstance(revision, str) or not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", revision):
            raise HealthError("MMLU did not resolve to an immutable revision")
    rows, checks = [], []
    for selection in selections:
        check, count = selection["check"], selection["examples"]
        label, metric = check_definition(check)
        if check in LOCAL:
            selected = _local(check, count, seed)
            source = {"kind": "generated", "generator": "vespasian-health-v1", "seed": seed}
        else:
            subject = check[5:]
            try:
                dataset = loader(MMLU_REPO, subject, split="test", revision=revision)
                available = len(dataset)
            except Exception:
                raise HealthError("Could not load the selected MMLU subject. Check network connectivity and dataset access") from None
            if count > available:
                raise HealthError(f"MMLU {subject} has {available} test examples. Choose at most {available}")
            indices = sorted(_random(seed, check).sample(range(available), count))
            selected = []
            for index in indices:
                row = dataset[index]
                question, choices, answer = row.get("question"), row.get("choices"), row.get("answer")
                if not isinstance(question, str) or not question.strip() or not isinstance(choices, list) or len(choices) != 4 or any(not isinstance(choice, str) or not choice.strip() for choice in choices) or isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < 4:
                    raise HealthError("MMLU returned an invalid multiple-choice example")
                prompt = "The following is a multiple choice question about " + subject.replace("_", " ") + ".\n\n" + question
                prompt += "".join(f"\n{letter}. {choice}" for letter, choice in zip("ABCD", choices)) + "\nAnswer:"
                selected.append({"id": f"{check}:test:{index}", "check": check, "prompt": prompt, "target": "ABCD"[answer], "choices": list("ABCD"), "source_index": index})
            source = {"kind": "huggingface", "repo": MMLU_REPO, "config": subject, "split": "test", "revision": revision, "available": available, "sample_indices": indices, "shots": 0, "scoring": "sum of conditional log probabilities for each answer letter without length normalisation"}
        if len({row["prompt"] for row in selected}) != len(selected):
            raise HealthError(f"The {check} health set contains duplicate prompts")
        rows.extend(selected)
        checks.append({"check": check, "label": label, "count": count, "metric": metric, "source": source})
    return {"schema_version": SCHEMA_VERSION, "seed": seed, "checks": checks, "rows": rows, "rows_sha256": hashlib.sha256(canonical(rows)).hexdigest()}


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise HealthError("Unsupported health-set format")
    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise HealthError("The health seed must be between 0 and 4294967295")
    checks, rows = manifest.get("checks"), manifest.get("rows")
    if not isinstance(checks, list) or not isinstance(rows, list):
        raise HealthError("The health set needs checks and rows")
    selections = [{"check": check.get("check"), "examples": check.get("count")} for check in checks if isinstance(check, dict)]
    if len(selections) != len(checks):
        raise HealthError("The health check definitions are invalid")
    validate_selections(selections)
    if hashlib.sha256(canonical(rows)).hexdigest() != manifest.get("rows_sha256"):
        raise HealthError("The health examples do not match their saved hash")
    counts = {selection["check"]: 0 for selection in selections}
    ids, prompts = set(), set()
    for row in rows:
        if not isinstance(row, dict) or row.get("check") not in counts:
            raise HealthError("A health example names an unknown check")
        for field in ("id", "prompt", "target"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise HealthError(f"A health example has no {field}")
        if row["id"] in ids or (row["check"], row["prompt"]) in prompts:
            raise HealthError("The health set contains duplicate examples")
        ids.add(row["id"])
        prompts.add((row["check"], row["prompt"]))
        counts[row["check"]] += 1
        if row["check"].startswith("mmlu:") and (row.get("choices") != list("ABCD") or row["target"] not in "ABCD" or len(row["target"]) != 1):
            raise HealthError("A MMLU health example has invalid choices")
        if row["check"] == "arithmetic" and not re.fullmatch(r"-?[0-9]+", row["target"]):
            raise HealthError("An arithmetic health target is invalid")
        if row["check"] == "json":
            try:
                value = json.loads(row["target"])
            except ValueError:
                raise HealthError("A JSON health target is invalid") from None
            if not isinstance(value, dict):
                raise HealthError("A JSON health target must be an object")
    for definition in checks:
        label, metric = check_definition(definition["check"])
        if counts[definition["check"]] != definition["count"] or definition.get("metric") != metric or definition.get("label") != label:
            raise HealthError("The health examples do not match their check definitions")
        source = definition.get("source")
        if not isinstance(source, dict):
            raise HealthError("A health check has no source provenance")
        check = definition["check"]
        if check in LOCAL:
            if source.get("kind") != "generated" or source.get("generator") != "vespasian-health-v1" or source.get("seed") != seed:
                raise HealthError("A local health check has invalid generator provenance")
        else:
            revision, indices, available = source.get("revision"), source.get("sample_indices"), source.get("available")
            if source.get("kind") != "huggingface" or source.get("repo") != MMLU_REPO or source.get("config") != check[5:] or source.get("split") != "test" or source.get("shots") != 0:
                raise HealthError("A MMLU health check has invalid source provenance")
            if not isinstance(revision, str) or not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", revision):
                raise HealthError("A MMLU health check has no immutable revision")
            if isinstance(available, bool) or not isinstance(available, int) or available < definition["count"] or not isinstance(indices, list) or len(indices) != definition["count"] or any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < available for index in indices) or len(set(indices)) != len(indices):
                raise HealthError("A MMLU health check has invalid source indices")
            if indices != [row.get("source_index") for row in rows if row["check"] == check]:
                raise HealthError("MMLU examples do not match their source indices")
    return manifest


def _strict_json(text: str):
    def pairs(values):
        value = {}
        for key, item in values:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value
    def constant(_):
        raise ValueError("Nonfinite JSON number")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def score(check: str, prediction: str, target: str) -> bool:
    if check == "copying":
        if prediction == target:
            return True
        if prediction.endswith("\r\n"):
            prediction = prediction[:-2]
        elif prediction.endswith("\n"):
            prediction = prediction[:-1]
        return prediction == target
    if check == "arithmetic":
        value = prediction.strip()
        try:
            return bool(re.fullmatch(r"[+-]?[0-9]+", value)) and int(value) == int(target)
        except ValueError:
            return False
    if check == "json":
        try:
            actual, expected = _strict_json(prediction), _strict_json(target)
            return isinstance(actual, dict) and canonical(actual) == canonical(expected)
        except (ValueError, TypeError):
            return False
    if check.startswith("mmlu:"):
        return prediction == target
    raise HealthError("Unknown health scorer")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare fixed model health examples")
    parser.add_argument("--selection", action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = prepare([parse_selection(value) for value in args.selection], args.seed)
        encoded = canonical(result)
        with args.output.open("xb") as destination:
            destination.write(encoded + b"\n")
        summary = {"schema_version": SCHEMA_VERSION, "seed": args.seed, "count": len(result["rows"]), "checks": result["checks"], "rows_sha256": result["rows_sha256"], "sha256": hashlib.sha256(encoded + b"\n").hexdigest()}
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 0
    except HealthError as error:
        print(f"Vespasian: {error}", file=sys.stderr)
    except OSError:
        print("Vespasian: Could not save the health examples. Choose a new run directory with available space", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
