#!/usr/bin/env python3
"""Publish a verified Vespasian checkpoint with a curated, hashed file set."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Callable


class PublishError(ValueError):
    """A publishing failure with credentials and local source data omitted."""


_FILES = {
    "config.json", "generation_config.json", "model.safetensors",
    "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "tokenizer.model", "spiece.model",
    "vocab.json", "vocab.txt", "merges.txt", "chat_template.jinja",
    "processor_config.json", "preprocessor_config.json", "image_processor_config.json",
    "video_processor_config.json", "video_preprocessor_config.json",
}
_SCORES = {"exact_match", "token_f1", "numeric_match", "semantic", "loss", "count"}
_SPLITS = {"edit", "validation", "preserve", "test"}
_SOURCE_MODELS = {"Qwen/Qwen3.5-0.8B", "google/gemma-3-1b-it"}


def _repo_id(repo: str) -> None:
    parts = repo.split("/")
    if len(parts) != 2 or len(repo) > 192 or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
        or ".." in part or "--" in part or part.endswith((".", ".git"))
        for part in parts
    ):
        raise PublishError("Use a model repository ID such as owner/model")


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PublishError(f"The run is missing a regular {path.name} file")
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            raise PublishError(f"The run's {path.name} is too large")
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise PublishError(f"Could not read the run's {path.name}") from None
    if not isinstance(result, dict):
        raise PublishError(f"The run's {path.name} must be a JSON object")
    return result


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(name: Any) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise PublishError("Checkpoint manifest contains an invalid path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in name.split("/")):
        raise PublishError("Checkpoint manifest contains an unsafe path")
    allowed = name in _FILES or re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name)
    allowed = allowed or re.fullmatch(r"chat_templates/[A-Za-z0-9_.-]+\.jinja", name)
    if not allowed:
        raise PublishError("Checkpoint manifest contains a file outside the model-file allowlist")
    return name


def _absolute_path(value: Any) -> bool:
    return isinstance(value, str) and (
        value.startswith(("/", "~/", "\\\\")) or bool(re.match(r"^[A-Za-z]:[\\/]", value))
    )


def _remove_path_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _remove_path_metadata(item) for key, item in value.items()
                if not (_absolute_path(item) and key != "content")}
    if isinstance(value, list):
        return [_remove_path_metadata(item) for item in value]
    return value


def _metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublishError("The completed run has no aggregate measurements")
    output = {}
    for split, measurements in value.items():
        if split not in _SPLITS or not isinstance(measurements, dict):
            continue
        filtered = {}
        for name in _SCORES:
            if name not in measurements:
                continue
            number = measurements[name]
            if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
                raise PublishError("The run contains invalid aggregate measurements")
            if number < 0 or (name not in {"loss", "count"} and number > 1):
                raise PublishError("The run contains out-of-range aggregate measurements")
            if name == "count" and (not isinstance(number, int) or number < 1):
                raise PublishError("The run contains an invalid example count")
            filtered[name] = number
        if filtered:
            output[split] = filtered
    return output


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublishError("The run contains an invalid turn count")
    return value


def _controls(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("passed"), bool):
        raise PublishError("The run contains invalid behavioural control measurements")
    if value.get("scope") != "supplied_examples" or value.get("status") not in {
        "unchanged", "not_exercised", "retained", "regressed"
    }:
        raise PublishError("The run contains an invalid behavioural control status")
    result = {"passed": value["passed"], "status": value["status"], "scope": value["scope"]}
    for name in ("checked", "applicable", "unchanged"):
        number = value.get(name)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise PublishError("The run contains an invalid behavioural control count")
        result[name] = number
    if result["applicable"] > result["checked"]:
        raise PublishError("The run contains inconsistent behavioural control counts")
    retention = value.get("retention")
    if retention is not None and (
        isinstance(retention, bool) or not isinstance(retention, (float, int))
        or not math.isfinite(retention) or not 0 <= retention <= 1
    ):
        raise PublishError("The run contains an invalid behavioural retention score")
    result["retention"] = retention
    return result


def _health(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PublishError("The run contains invalid health measurements")
    output = {}
    metrics = {"copying": "exact_text", "arithmetic": "numeric_exact", "json": "json_exact"}
    for check, row in value.items():
        expected = metrics.get(check)
        if expected is None and re.fullmatch(r"mmlu:[a-z_]+", check):
            expected = "choice_logprob_accuracy"
        if expected is None or not isinstance(row, dict) or row.get("metric") != expected:
            raise PublishError("The run contains an unknown health measurement")
        score, count = row.get("score"), row.get("count")
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise PublishError("The run contains an invalid health score")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 1000:
            raise PublishError("The run contains an invalid health example count")
        output[check] = {"score": score, "count": count, "metric": expected}
    return output


def _hex(value: Any, lengths=(64,)) -> str | None:
    if isinstance(value, str) and len(value) in lengths and re.fullmatch(r"[0-9a-fA-F]+", value):
        return value.lower()
    return None


@dataclass
class Prepared:
    temporary: tempfile.TemporaryDirectory
    directory: Path
    files: list[dict[str, Any]]
    summary: dict[str, Any]

    def close(self):
        self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


_CARD_START = "<!-- vespasian:model-card:start -->"
_CARD_END = "<!-- vespasian:model-card:end -->"
_PROJECT = "https://chrisvoncsefalvay.github.io/vespasian/"
# V mark and colours from site/assets/favicon.svg.
_BADGE = '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="36" viewBox="0 0 240 36" role="img" aria-label="Created with Vespasian"><rect width="240" height="36" rx="6" fill="#14181b"/><path d="M9 9 19 27 29 9M15 9l4 8 4-8" fill="none" stroke="#d5a278" stroke-width="2"/><text x="39" y="23" font-family="sans-serif" font-size="15" fill="#f5eee7">Created with Vespasian</text></svg>'

def _card_base_models(card: str) -> set[str]:
    """Read only frontmatter, with safe YAML parsing (a Hub runtime dependency)."""
    match = re.match(r"\A(?:\ufeff)?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", card, re.DOTALL)
    if match is None:
        if card.lstrip("\ufeff").startswith("---"):
            raise PublishError("Existing model card frontmatter cannot be verified; review it before uploading")
        return set()
    try:
        import yaml
        metadata = yaml.safe_load(match.group(1))
        if metadata is None:
            return set()
        if not isinstance(metadata, dict):
            raise ValueError("Not a mapping")
        value = metadata.get("base_model")
        if value is None:
            return set()
        values = value if isinstance(value, list) else [value]
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError("Invalid base model metadata")
        return set(values)
    except Exception:
        raise PublishError("Model card base_model metadata cannot be verified; review it before uploading") from None


def _merge_card(existing: str, generated: str) -> str:
    """Preserve author metadata/prose; replace only our explicitly owned section."""
    existing_bases = _card_base_models(existing)
    generated_bases = _card_base_models(generated)
    if existing_bases and existing_bases != generated_bases:
        raise PublishError("Existing model card base_model metadata does not match the checkpoint's verified base model. Choose a different repository or review and correct the existing card before uploading")
    section = generated[generated.index(_CARD_START):]
    if _CARD_START not in existing and _CARD_END not in existing:
        return existing + "\n\n" + section
    if existing.count(_CARD_START) != 1 or existing.count(_CARD_END) != 1:
        raise PublishError("Existing model card has ambiguous Vespasian sections; review it before uploading")
    start, end = existing.index(_CARD_START), existing.index(_CARD_END)
    if end < start:
        raise PublishError("Existing model card has malformed Vespasian section markers")
    return existing[:start] + section.rstrip("\n") + existing[end + len(_CARD_END):]

def _download_card(api: Any, repo: str, revision: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=repo, filename="README.md", repo_type="model",
                                    revision=revision, token=api.token))
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Oversized card")
        return path.read_bytes().decode("utf-8")
    except Exception:
        raise PublishError("Could not read the existing model card; upload stopped to preserve it") from None


def prepare(run: Path, repo: str) -> Prepared:
    """Create a private upload snapshot after verifying the complete local receipt."""
    _repo_id(repo)
    run = Path(run).absolute()
    report = _read_json(run / "report.json")
    checkpoint = _read_json(run / "checkpoint.json")
    model = _read_json(run / "model.json")
    settings = _read_json(run / "run.json")
    if report.get("status") != "complete" or report.get("reload_verified") is not True:
        raise PublishError("Publish requires a completed run with a verified checkpoint reload")
    if checkpoint.get("source_unchanged") is not True:
        raise PublishError("Publish requires verified unchanged source weights")
    metric = report.get("metric")
    if metric not in {"exact_match", "token_f1", "numeric_match", "semantic"}:
        raise PublishError("The run uses an unsupported objective metric")
    root = run / "checkpoint"
    if root.is_symlink() or not root.is_dir():
        raise PublishError("The run has no regular checkpoint directory")
    entries = checkpoint.get("files")
    if not isinstance(entries, list) or not entries:
        raise PublishError("The checkpoint receipt has no file hashes")
    summary = {
        "status": "complete", "metric": metric,
        "turns": _count(report.get("turns")),
        "accepted_turns": _count(report.get("accepted_turns")),
        "baseline_metrics": _metrics(report.get("baseline_metrics")),
        "best_metrics": _metrics(report.get("best_metrics")),
        "final_metrics": _metrics(report.get("final_metrics")),
        "reload_verified": True, "source_unchanged": True,
        "final_scoring_complete": report.get("final_scoring_error") is None,
    }
    for name in ("baseline_health", "best_health", "final_health"):
        values = _health(report.get(name))
        if values is not None:
            summary[name] = values
    controls = _controls(report.get("controls"))
    if controls is not None:
        summary["controls"] = controls
    if summary["accepted_turns"] > summary["turns"]:
        raise PublishError("The run's retained-turn count exceeds its proposals")
    temporary = tempfile.TemporaryDirectory(prefix="vespasian-publish-")
    destination = Path(temporary.name)
    source_hashes = []
    names = set()
    try:
        for entry in entries:
            if not isinstance(entry, dict):
                raise PublishError("The checkpoint manifest is invalid")
            name = _safe_file(entry.get("path"))
            expected = _hex(entry.get("sha256"))
            if name in names or expected is None:
                raise PublishError("The checkpoint manifest has duplicate paths or invalid hashes")
            names.add(name)
            source = root / name
            for candidate in (source, *source.parents):
                if candidate == root:
                    break
                if candidate.is_symlink():
                    raise PublishError("Checkpoint files must not contain symbolic links")
            if not source.is_file() or not source.resolve().is_relative_to(root.resolve()):
                raise PublishError("A checkpoint file is missing or outside the run directory")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with source.open("rb") as handle, target.open("xb") as staged:
                for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    staged.write(block)
                    digest.update(block)
                    size += len(block)
            if digest.hexdigest() != expected or ("bytes" in entry and entry["bytes"] != size):
                raise PublishError("A checkpoint file no longer matches its saved hash")
            source_hashes.append({"path": name, "sha256": expected, "bytes": size})
            if name.endswith("config.json"):
                original = json.loads(target.read_text(encoding="utf-8"))
                public = _remove_path_metadata(original)
                if public != original:
                    target.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if "config.json" not in names or not any(name.endswith(".safetensors") for name in names):
            raise PublishError("The checkpoint needs a model configuration and safetensors weights")
        if not names.intersection({"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt"}):
            raise PublishError("The checkpoint has no tokenizer vocabulary")
        # A shard index must refer only to verified files in this exact snapshot.
        if "model.safetensors.index.json" in names:
            index = json.loads((destination / "model.safetensors.index.json").read_text())
            weights = index.get("weight_map")
            if not isinstance(weights, dict) or any(name not in names for name in weights.values()):
                raise PublishError("The model shard index references an unverified weight file")
        base_model = model.get("model") if model.get("model") in _SOURCE_MODELS else None
        version = settings.get("version")
        if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?", version):
            version = None
        target_categories = {"factual": "Factual answers", "style": "Style and tone",
                             "format": "Response format", "classification": "Classification",
                             "custom": "Custom behaviour"}
        target_category = target_categories.get(settings.get("target_type"), "Not reported")
        tables = model.get("token_tables")
        table_info = None
        if isinstance(tables, list) and all(isinstance(t, dict) and isinstance(t.get("tied"), bool) for t in tables):
            table_info = {"available_tables": len(tables), "tied_input_output_tables": sum(t["tied"] for t in tables)}
        hf_task = settings.get("hf_task")
        if hf_task not in {"text-generation", "text-classification", "question-answering", "summarization", "translation"}:
            hf_task = None
        provenance = {
            "hf_task": hf_task,
            "vespasian_version": version,
            "target_category": target_category,
            "token_table_catalogue": table_info,
            "base_model": base_model,
            "source_revision": _hex(model.get("revision"), (40, 64)),
            "dataset_sha256": _hex(settings.get("dataset_sha256")),
            "dataset_mode": settings.get("dataset_mode", "paired") if settings.get("dataset_mode", "paired") in {"paired", "corpus", "prompts"} else None,
            "source_checkpoint_files": source_hashes,
            "published_checkpoint_files": [
                {"path": name, "sha256": _sha(destination / name), "bytes": (destination / name).stat().st_size}
                for name in sorted(names)
            ],
        }
        (destination / "vespasian-results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (destination / "vespasian-provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        published_config = _read_json(destination / "config.json")
        pipeline_tag = "image-text-to-text" if published_config.get("model_type") == "qwen3_5" else "text-generation"
        header = ["---", "library_name: transformers", f"pipeline_tag: {pipeline_tag}", "tags:", "- vespasian", "- model-editing"]
        if base_model:
            header.append(f"base_model: {json.dumps(base_model)}")
        header += ["---", "", _CARD_START, "", f"[![Created with Vespasian](vespasian-badge.svg)]({_PROJECT})", "", f"# {repo.split('/')[-1]}", "", "A checkpoint selected by Vespasian's direct weight search.", "",
                   f"The search retained {summary['accepted_turns']} changes across {summary['turns']} proposals. The saved checkpoint was reloaded and verified.", "",
                   "| Split | Baseline examples | Baseline score | Final examples | Final score |", "| --- | ---: | ---: | ---: | ---: |"]
        for split in ("edit", "validation", "preserve", "test"):
            values = summary["final_metrics"].get(split, {})
            baseline = summary["baseline_metrics"].get(split, {})
            before = f"{baseline[metric]:.6f}" if metric in baseline else "not reported"
            after = f"{values[metric]:.6f}" if metric in values else "not reported"
            header.append(f"| {split} | {baseline.get('count', 'not reported')} | {before} | {values.get('count', 'not reported')} | {after} |")
        header += ["", f"Objective: `{metric}`.", "", "[Measurements](vespasian-results.json) · [Provenance and hashes](vespasian-provenance.json)", ""]
        if controls is not None:
            retention = "not measured" if controls["retention"] is None else f"{controls['retention']:.6f}"
            header += [f"Behavioural controls on supplied examples: `{controls['status']}`.", "",
                       f"Checked {controls['checked']} changed answers. Applicable: {controls['applicable']}. Unchanged: {controls['unchanged']}. Retention: {retention}.", ""]
        if summary.get("final_health"):
            header += ["Health checks on the fixed monitoring examples:", "", "| Check | Examples | Baseline | Final |", "| --- | ---: | ---: | ---: |"]
            for check, values in sorted(summary["final_health"].items()):
                baseline = summary.get("baseline_health", {}).get(check, {}).get("score")
                before = f"{baseline:.6f}" if baseline is not None else "—"
                header.append(f"| {check} | {values['count']} | {before} | {values['score']:.6f} |")
            header.append("")
        header += ["## Intended use and edit process", "",
                   f"Experimental text generation with a targeted behavioural edit. Behaviour category: {target_category}. The private user objective, prompts, examples and editor conversation are deliberately not published.",
                   f"Dataset mode: {provenance['dataset_mode'] or 'not reported'}. Hugging Face task (text adaptation): {hf_task or 'not reported'}. The objective score is {metric}; it is not a general capability or safety rating.",
                   "Reported evaluations use text only; retaining a multimodal architecture does not verify its other capabilities.",
                   "Vespasian assesses the starting model, proposes bounded direct weight changes, measures candidate behaviour, retains selected changes and reloads the saved checkpoint for verification. Proposal counts do not imply independent improvements.",
                   "Token-row editing, when used, changes existing vocabulary rows rather than adding tokens. Shared input/output weights can make a single row edit affect both roles. No raw token selections are published.",
                   "", "## Provenance", "",
                   f"Base model: {base_model or 'not publicly identified'}.",
                   f"Source revision: {provenance['source_revision'] or 'not reported'}.",
                   f"Vespasian version: {version or 'not reported'}.",
                   "", "## Evaluation limits", "",
                   "Reload verification confirms checkpoint reload consistency, not broad correctness or safety. Missing measurements are not zero scores. Validation selection can overfit; held-out test and preservation results above are only evidence for the supplied examples.",
                   "Final scoring completed." if summary["final_scoring_complete"] else "WARNING: final scoring was incomplete; available measurements must not be treated as a complete final evaluation.",
                   "Health checks and controls, when reported, cover only the monitoring examples. This card does not establish suitability for consequential decisions or production deployment.",
                   "Review the base model's licence, restrictions and documentation before use. No new licence or permission is inferred by this card.",
                   "", _CARD_END, ""]
        if table_info is not None:
            header[-2:-2] = [f"Inspected token-table catalogue: {table_info['available_tables']} available tables, including {table_info['tied_input_output_tables']} tied input/output tables. This describes available edit targets, not evidence that a token-row edit was applied.", ""]
        (destination / "vespasian-badge.svg").write_text(_BADGE + "\n", encoding="utf-8")
        (destination / "README.md").write_bytes("\n".join(header).encode("utf-8"))
        files = [{"path": str(path.relative_to(destination)), "sha256": _sha(path), "bytes": path.stat().st_size}
                 for path in sorted(destination.rglob("*")) if path.is_file()]
        return Prepared(temporary, destination, files, summary)
    except BaseException:
        temporary.cleanup()
        raise


def _runtime():
    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    except ImportError:
        raise PublishError("Install the Hugging Face runtime with vespasian setup") from None
    return HfApi(), CommitOperationAdd, CommitOperationDelete


def _status(error: Exception) -> int | None:
    return getattr(getattr(error, "response", None), "status_code", None)


def _info(api: Any, repo: str):
    try:
        return api.model_info(repo_id=repo)
    except Exception as error:
        if _status(error) == 404:
            return None
        raise PublishError("Could not inspect the model repository. Check HF_TOKEN, repository access and network connectivity") from None


def _visibility(info: Any, private: bool):
    actual = getattr(info, "private", None)
    if not isinstance(actual, bool) or actual != private:
        raise PublishError("The existing repository visibility does not match the requested visibility")


def _obsolete_weights(api: Any, repo: str, info: Any, new_names: set[str], prepared: Prepared) -> list[str]:
    """Remove only conflicting standard weight files, pinned to the inspected head."""
    parent = _hex(getattr(info, "sha", None), (40, 64))
    try:
        names = api.list_repo_files(repo_id=repo, repo_type="model", revision=parent)
    except Exception:
        raise PublishError("Could not inspect the existing model files. Check repository access and network connectivity") from None
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise PublishError("The Hub did not return a valid model file listing")
    if "README.md" in names:
        if parent is None:
            raise PublishError("Cannot safely preserve the existing model card without a verified repository revision")
        existing = _download_card(api, repo, parent)
        path = prepared.directory / "README.md"
        path.write_bytes(_merge_card(existing, path.read_bytes().decode("utf-8")).encode("utf-8"))
        for item in prepared.files:
            if item["path"] == "README.md":
                item.update(sha256=_sha(path), bytes=path.stat().st_size)
    standard = {"model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"}
    def weight(name):
        return name in standard or bool(re.fullmatch(r"(?:model-\d{5}-of-\d{5}\.safetensors|pytorch_model-\d{5}-of-\d{5}\.bin)", name))
    return sorted(name for name in names if name not in new_names and weight(name))


def publish(*, run: Path, repo: str, visibility: str, create: bool = False,
            api: Any = None, operation: Callable[..., Any] | None = None,
            delete_operation: Callable[..., Any] | None = None) -> dict[str, Any]:
    _repo_id(repo)
    if visibility not in {"private", "public"}:
        raise PublishError("Choose private or public repository visibility")
    private = visibility == "private"
    with prepare(run, repo) as prepared:
        if api is None or operation is None:
            api, operation, delete_operation = _runtime()
        info = _info(api, repo)
        created = False
        if info is None:
            if not create:
                raise PublishError("The model repository does not exist. Add --create to create it")
            try:
                api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=False)
                created = True
            except Exception as error:
                if _status(error) != 409:
                    raise PublishError("Could not create the model repository. Check HF_TOKEN permissions and the repository name") from None
            info = _info(api, repo)
            if info is None:
                raise PublishError("The model repository could not be verified after creation")
        _visibility(info, private)
        # Check visibility again immediately before creating the atomic file commit.
        info = _info(api, repo)
        if info is None:
            raise PublishError("The model repository disappeared before upload")
        _visibility(info, private)
        obsolete = _obsolete_weights(api, repo, info, {item["path"] for item in prepared.files}, prepared)
        operations = [operation(path_in_repo=item["path"], path_or_fileobj=prepared.directory / item["path"])
                      for item in prepared.files]
        if obsolete:
            if delete_operation is None:
                from huggingface_hub import CommitOperationDelete
                delete_operation = CommitOperationDelete
            operations.extend(delete_operation(path_in_repo=name, is_folder=False) for name in obsolete)
        kwargs = {"repo_id": repo, "repo_type": "model", "operations": operations,
                  "commit_message": "Publish Vespasian checkpoint"}
        parent = _hex(getattr(info, "sha", None), (40, 64))
        if parent:
            kwargs["parent_commit"] = parent
        try:
            committed = api.create_commit(**kwargs)
        except Exception:
            prefix = "The repository was created, but checkpoint publication did not complete" if created else "Checkpoint publication did not complete"
            raise PublishError(f"{prefix}. Check HF_TOKEN write access, repository state and network connectivity") from None
        oid = _hex(getattr(committed, "oid", None), (40, 64))
        if oid is None:
            raise PublishError("The Hub did not return a valid publication commit")
        return {"repo": repo, "private": private, "created": created, "commit": oid,
                "repo_url": f"https://huggingface.co/{repo}", "files": prepared.files, "deleted_files": obsolete}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Publish a verified Vespasian checkpoint")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--visibility", choices=("private", "public"), required=True)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = publish(**vars(args))
    except PublishError as error:
        print(f"Vespasian: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError):
        print("Vespasian: Could not prepare the verified checkpoint for publication", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
