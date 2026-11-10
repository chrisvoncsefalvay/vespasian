#!/usr/bin/env python3
"""Validate and publish an explicitly selected Vespasian editor trajectory."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable


SCHEMA = "vespasian.trajectory"
VERSION = 1
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_FILE_BYTES = 512 * 1024 * 1024
KINDS = {"run_start", "editor_request", "editor_response", "editor_error", "baseline", "turn", "health", "event", "run_end"}
FIELDS = {"schema", "version", "sequence", "timestamp_unix_ms", "type", "payload"}
METADATA = {"run_id", "version", "model", "model_requested", "revision", "model_revision", "goal", "controls",
            "editor", "editor_model", "budget_usd", "seed", "metric", "dataset_sha256", "dataset_mode", "target_type",
            "health_checks", "health_manifest", "health_max_drop", "max_turns", "budget_mode"}
SECRET_ENV = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_TOKEN",
              "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
TOKEN_PATTERN = re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,})\b")


class TrajectoryError(ValueError):
    """An export error that does not echo credentials or trajectory contents."""


def _repo_id(repo: str) -> None:
    parts = repo.split("/")
    if len(parts) != 2 or len(repo) > 192 or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
        or ".." in part or "--" in part or part.endswith((".", ".git"))
        for part in parts
    ):
        raise TrajectoryError("Use a dataset repository ID such as owner/trajectories")


def _credential_key(key: str) -> bool:
    key = key.lower().replace("-", "_")
    return key in {"hf_token", "token", "api_key", "authorization", "password", "secret", "access_token",
                   "refresh_token", "headers", "environment", "env", "hf_token_path", "huggingface_token",
                   "hugging_face_hub_token", "huggingfacehub_api_token"} or key.endswith(("_api_key", "_access_token"))


def _absolute_path(value: Any) -> bool:
    return isinstance(value, str) and (
        value.startswith(("/", "~/", "\\\\")) or bool(re.match(r"^[A-Za-z]:[\\/]", value))
    )


def _private_path_key(key: str) -> bool:
    return key in {"path", "output", "checkpoint", "source_path", "dataset_source", "python", "file",
                   "cwd", "working_directory", "model", "model_requested"} or key.endswith("_path")


def _safe_payload(value: Any, secrets: list[str], depth: int = 0) -> None:
    if depth > 64:
        raise TrajectoryError("The trajectory contains an excessively nested record")
    if isinstance(value, dict):
        for key, item in value.items():
            if _credential_key(key):
                raise TrajectoryError("The trajectory contains a credential field. Redact it before upload")
            if _private_path_key(key) and _absolute_path(item):
                raise TrajectoryError("The trajectory contains a local path field. Redact it before upload")
            _safe_payload(item, secrets, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _safe_payload(item, secrets, depth + 1)
    elif isinstance(value, str):
        if any(secret in value for secret in secrets) or TOKEN_PATTERN.search(value):
            raise TrajectoryError("The trajectory contains an access token. Redact it before upload")
    elif isinstance(value, float) and not math.isfinite(value):
        raise TrajectoryError("The trajectory contains a nonfinite number")


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _constant(_):
    raise TrajectoryError("The trajectory contains a nonfinite number")


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise TrajectoryError("The trajectory contains duplicate object fields")
        value[key] = item
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def prepare(run: Path) -> Prepared:
    """Snapshot complete JSONL records without inferring success after a crash."""
    run = Path(run).absolute()
    if run.is_symlink() or not run.is_dir():
        raise TrajectoryError("Choose a regular run directory containing trajectory.jsonl")
    source = run / "trajectory.jsonl"
    if source.is_symlink() or not source.is_file():
        raise TrajectoryError("The run has no regular trajectory.jsonl file")
    if source.stat().st_size > MAX_FILE_BYTES:
        raise TrajectoryError("The editor trajectory exceeds its file-size limit")
    temporary = tempfile.TemporaryDirectory(prefix="vespasian-trajectory-")
    directory = Path(temporary.name)
    destination = directory / "trajectory.jsonl"
    secrets = [value for key in SECRET_ENV if (value := os.environ.get(key))]
    sequence = 0
    total_bytes = 0
    previous_timestamp = 0
    first_timestamp = None
    metadata = None
    end = None
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle, destination.open("xb") as output:
            while raw := handle.readline(MAX_RECORD_BYTES + 2):
                total_bytes += len(raw)
                if len(raw) > MAX_RECORD_BYTES + 1 or total_bytes > MAX_FILE_BYTES:
                    raise TrajectoryError("The editor trajectory exceeds its file-size limit")
                if not raw.endswith(b"\n"):
                    raise TrajectoryError("The final trajectory record is incomplete")
                try:
                    record = json.loads(raw.decode("utf-8"), parse_constant=_constant, object_pairs_hook=_object)
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                    raise TrajectoryError("The trajectory contains an invalid JSON record") from None
                if not isinstance(record, dict) or set(record) != FIELDS:
                    raise TrajectoryError("The trajectory record has unsupported fields")
                if record["schema"] != SCHEMA or type(record["version"]) is not int or record["version"] != VERSION:
                    raise TrajectoryError("The trajectory uses an unsupported schema version")
                if not _integer(record["sequence"]) or record["sequence"] != sequence:
                    raise TrajectoryError("The trajectory records are out of order")
                timestamp = record["timestamp_unix_ms"]
                if not _integer(timestamp) or timestamp < previous_timestamp:
                    raise TrajectoryError("The trajectory contains invalid timestamps")
                kind, payload = record["type"], record["payload"]
                if not isinstance(kind, str) or kind not in KINDS or not isinstance(payload, dict):
                    raise TrajectoryError("The trajectory contains an unsupported record type or payload")
                if end is not None:
                    raise TrajectoryError("The trajectory contains records after its completion")
                if sequence == 0:
                    if kind != "run_start" or set(payload) - METADATA:
                        raise TrajectoryError("The trajectory must begin with supported run metadata")
                    run_id = payload.get("run_id")
                    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id):
                        raise TrajectoryError("The trajectory has no valid run identifier")
                    metadata = payload
                    first_timestamp = timestamp
                elif kind == "run_start":
                    raise TrajectoryError("The trajectory contains more than one run start")
                if kind == "run_end":
                    if set(payload) != {"status", "result"} or payload["status"] not in {"complete", "failed", "cancelled"}:
                        raise TrajectoryError("The trajectory has an invalid completion status")
                    end = payload["status"]
                _safe_payload(payload, secrets)
                output.write(raw)
                digest.update(raw)
                sequence += 1
                previous_timestamp = timestamp
        if metadata is None:
            raise TrajectoryError("The trajectory file is empty")
        sha256 = digest.hexdigest()
        trajectory_name = f"trajectories/{sha256}.jsonl"
        manifest_name = f"manifests/{sha256}.json"
        target = directory / trajectory_name
        target.parent.mkdir()
        destination.rename(target)
        summary = {
            "schema": SCHEMA, "version": VERSION, "run_id": metadata["run_id"],
            "status": end or "incomplete", "records": sequence,
            "started_unix_ms": first_timestamp, "last_record_unix_ms": previous_timestamp,
            "trajectory": {"path": trajectory_name, "sha256": sha256, "bytes": total_bytes},
        }
        manifest = directory / manifest_name
        manifest.parent.mkdir()
        manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = [{"path": name, "sha256": _hash(directory / name), "bytes": (directory / name).stat().st_size}
                 for name in (trajectory_name, manifest_name)]
        return Prepared(temporary, directory, files, summary)
    except BaseException:
        temporary.cleanup()
        raise


def _runtime():
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        raise TrajectoryError("Install the Hugging Face runtime with vespasian setup") from None
    return HfApi(), CommitOperationAdd


def _status(error: Exception) -> int | None:
    return getattr(getattr(error, "response", None), "status_code", None)


def _info(api: Any, repo: str, revision: str | None = None):
    try:
        kwargs = {"repo_id": repo}
        if revision:
            kwargs["revision"] = revision
        return api.dataset_info(**kwargs)
    except Exception as error:
        if _status(error) == 404:
            return None
        raise TrajectoryError("Could not inspect the dataset repository. Check your Hugging Face login and repository access") from None


def _visibility(info: Any, private: bool):
    actual = getattr(info, "private", None)
    if not isinstance(actual, bool) or actual != private:
        raise TrajectoryError("The existing dataset repository visibility does not match the requested visibility")


def _commit(value: Any) -> str | None:
    if isinstance(value, str) and len(value) in {40, 64} and re.fullmatch(r"[0-9a-fA-F]+", value):
        return value.lower()
    return None


def publish(*, run: Path, repo: str, visibility: str, create: bool = False,
            api: Any = None, operation: Callable[..., Any] | None = None) -> dict[str, Any]:
    _repo_id(repo)
    if visibility not in {"private", "public"}:
        raise TrajectoryError("Choose private or public dataset visibility")
    private = visibility == "private"
    with prepare(run) as prepared:
        if api is None or operation is None:
            api, operation = _runtime()
        info = _info(api, repo)
        created = False
        if info is None:
            if not create:
                raise TrajectoryError("The dataset repository does not exist. Add --create to create it")
            try:
                api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=False)
                created = True
            except Exception as error:
                if _status(error) != 409:
                    raise TrajectoryError("Could not create the dataset repository. Check your Hugging Face write access and repository name") from None
            info = _info(api, repo)
            if info is None:
                raise TrajectoryError("The dataset repository could not be verified after creation")
        _visibility(info, private)
        # Pin the file listing and atomic update to the last inspected head.
        info = _info(api, repo)
        if info is None:
            raise TrajectoryError("The dataset repository disappeared before upload")
        _visibility(info, private)
        parent = _commit(getattr(info, "sha", None))
        if parent is None:
            raise TrajectoryError("The Hub did not return a valid dataset revision")
        try:
            names = api.list_repo_files(repo_id=repo, repo_type="dataset", revision=parent)
        except Exception:
            raise TrajectoryError("Could not inspect the existing dataset files. Check repository access") from None
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise TrajectoryError("The Hub did not return a valid dataset file listing")
        if "README.md" not in names:
            readme = prepared.directory / "README.md"
            readme.write_text("""---
tags:
- vespasian
- model-editing
- editor-trajectories
---

# Vespasian editor trajectories

Each file in `trajectories/` records one run's observable editor exchanges and measured weight-search steps. Matching files in `manifests/` contain the content hash, record count and completion status.

Records use the `vespasian.trajectory` schema, version 1. Each JSONL row has a sequence number, timestamp, record type and payload. Editor requests include the supplied context and response schema. Responses contain the editor's returned structured output and recorded usage. Turn records include the measured objective and health scores when enabled.

Trajectories may contain the supplied prompts, reference answers and model outputs. They do not contain hidden model reasoning. A run without a final record is marked `incomplete`. Failed and cancelled runs keep their recorded status.
""", encoding="utf-8")
            prepared.files.append({"path": "README.md", "sha256": _hash(readme), "bytes": readme.stat().st_size})
        operations = [operation(path_in_repo=item["path"], path_or_fileobj=prepared.directory / item["path"])
                      for item in prepared.files]
        try:
            committed = api.create_commit(repo_id=repo, repo_type="dataset", operations=operations,
                commit_message="Add Vespasian editor trajectory", parent_commit=parent)
        except Exception:
            raise TrajectoryError("Trajectory upload did not complete. Check write access, repository state and network connectivity") from None
        oid = _commit(getattr(committed, "oid", None))
        if oid is None:
            raise TrajectoryError("The Hub did not return a valid trajectory commit")
        try:
            readback = _info(api, repo, oid)
            if readback is None or _commit(getattr(readback, "sha", None)) != oid:
                raise TrajectoryError("The uploaded dataset revision could not be found")
            _visibility(readback, private)
            for item in prepared.files:
                downloaded = Path(api.hf_hub_download(repo_id=repo, filename=item["path"], repo_type="dataset",
                    revision=oid, force_download=True, local_dir=prepared.directory / "readback"))
                if downloaded.stat().st_size != item["bytes"] or _hash(downloaded) != item["sha256"]:
                    raise TrajectoryError("An uploaded trajectory file failed its hash check")
        except Exception:
            raise TrajectoryError(f"Trajectory committed as {oid}, but readback verification failed. The local file is saved") from None
        return {"version": VERSION, "repo": repo, "private": private, "created": created,
                "commit": oid, "repo_url": f"https://huggingface.co/datasets/{repo}",
                "verified": True, "status": prepared.summary["status"], "records": prepared.summary["records"],
                "files": prepared.files}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Upload an editor trajectory to a Hugging Face dataset repository")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--visibility", choices=("private", "public"), required=True)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = publish(**vars(args))
    except TrajectoryError as error:
        print(f"Vespasian: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RecursionError):
        print("Vespasian: Could not prepare the editor trajectory for upload", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
