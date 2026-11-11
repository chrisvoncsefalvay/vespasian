"""Account-scoped Hub access and the CLI's canonical dataset mapping."""

from __future__ import annotations

import hashlib
import html
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core" if (ROOT / "core").is_dir() else ROOT.parent.parent / "python"
DATA_FIELDS = {"dataset_source", "dataset_config", "dataset_split", "dataset_mode", "prompt_field",
               "target_field", "text_field", "example_limit", "test_dataset_source",
               "test_dataset_config", "test_dataset_split", "test_example_limit"}


def core_module(name):
    spec = importlib.util.spec_from_file_location(f"vespasian_core_{name}", CORE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dataset = core_module("dataset")


def token_for(oauth_token=None, override=""):
    """False explicitly disables implicit use of the Space owner's login."""
    return (override or "").strip() or getattr(oauth_token, "token", None) or False


def services(token=False, cache_dir=None):
    import datasets
    from huggingface_hub import HfApi
    download = datasets.DownloadConfig(cache_dir=cache_dir, token=token or False)
    return {
        "api": HfApi(token=token or False),
        "loader": lambda *a, **kw: datasets.load_dataset(*a, **kw, token=token or False, cache_dir=cache_dir, download_config=download),
        "config_names": lambda *a, **kw: datasets.get_dataset_config_names(*a, **kw, token=token or False, download_config=download),
        "split_names": lambda *a, **kw: datasets.get_dataset_split_names(*a, **kw, token=token or False, cache_dir=cache_dir, download_config=download),
        "runtime_version": datasets.__version__,
    }


def repository(value: str) -> str:
    value = (value or "").strip().removeprefix("hf://").removeprefix("https://huggingface.co/datasets/").rstrip("/")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value) or ".." in value:
        raise ValueError("Choose a dataset repository such as openai/gsm8k")
    return value


def account_datasets(token=False, api=None):
    if not token:
        return None, []
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    who = api.whoami()
    owners = [who["name"], *[org["name"] for org in who.get("orgs", [])]]
    choices = set()
    for owner in owners:
        for info in api.list_datasets(author=owner, limit=100, sort="last_modified"):
            choices.add(info.id)
    return who["name"], sorted(choices, key=str.casefold)


def search_dataset_results(query, token=False, limit=12, api=None):
    """Search public datasets with the visitor's explicit authentication context."""
    query = str(query or "").strip()[:120]
    if len(query) < 2:
        return []
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 25:
        raise ValueError("Choose between 1 and 25 search results")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token or False)
    try:
        found = api.list_datasets(search=query, limit=limit, sort="downloads")
        results = []
        seen = set()
        for info in itertools.islice(found, limit):
            identifier = getattr(info, "id", None)
            if not isinstance(identifier, str) or identifier in seen or getattr(info, "private", False):
                continue
            seen.add(identifier)
            downloads = getattr(info, "downloads", None)
            results.append({"id": identifier, "private": False, "gated": bool(getattr(info, "gated", False)),
                            "downloads": downloads if isinstance(downloads, int) and not isinstance(downloads, bool) else None})
        return results
    except Exception:
        raise ValueError("Could not search the Hub. Try again or enter a dataset repository directly") from None


def search_datasets(query, token=False, limit=12, api=None):
    return [(f"{item['id']} · {'Access request required' if item['gated'] else 'Public'}", item["id"])
            for item in search_dataset_results(query, token, limit, api)]


def _preview_local(config, token=False):
    with tempfile.TemporaryDirectory(prefix="vs-preview-") as cache:
        return dataset.preview(
            repo=repository(config.get("dataset_source", "")),
            config=config.get("dataset_config") or None,
            split=config.get("dataset_split") or "train",
            dataset_mode=config.get("dataset_mode", "paired"),
            prompt_field=config.get("prompt_field") or None,
            target_field=config.get("target_field") or None,
            text_field=config.get("text_field") or None,
            preview_rows=3, **services(token, cache),
        )


def _display(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return html.escape(text[:12000])


def field_choices(result):
    """Suggest bounded selectors from observed rows, including shared columns."""
    found = {}

    def add(path, value):
        if path and len(found) < 120:
            snippet = " ".join((value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)).split())
            found.setdefault(path, snippet[:64] + ("…" if len(snippet) > 64 else ""))

    def visit(value, path="", depth=0, expanded=False):
        if depth > 6 or len(found) >= 120:
            return
        if isinstance(value, dict):
            if path.endswith("target_scores"):
                add(path, value)
            for key, child in list(value.items())[:40]:
                visit(child, f"{path}.{key}" if path else str(key), depth + 1, expanded)
        elif isinstance(value, (list, tuple)):
            if not value:
                return
            if not expanded:
                visit(value[0], f"{path}[]", depth + 1, True)
            for index, child in enumerate(value[:3]):
                visit(child, f"{path}[{index}]", depth + 1, expanded)
        elif not isinstance(value, bool):
            add(path, value)

    for row in result.get("rows", [])[:3]:
        if isinstance(row, dict) and isinstance(row.get("row"), dict):
            row = row["row"]
        visit(row)
    for role in ("prompt_field", "target_field", "text_field"):
        if result.get(role) and result[role] not in found:
            found[result[role]] = "Detected mapping"
    return [(f"{path} · {sample}" if sample else path, path) for path, sample in found.items()]


schema_choices = field_choices


def stale_preview_html(role="Edit/validation"):
    return (f'<section class="vs-preview-stale" role="status"><span class="vs-kicker">{_display(role)} data</span>'
            '<h3>Refresh this preview</h3><p>The data selection changed. Preview it to confirm the fields and examples.</p></section>')


def preview_html(result, role="Edit/validation"):
    rows = result.get("rows", [])
    mapped = result.get("mapped_rows", [])
    parts = [f'<section class="vs-data-preview" data-preview-role="{_display(role)}">']
    parts.append(f'<header><span class="vs-kicker">{_display(role)} data</span><strong>{_display(result.get("repo", "Dataset"))}</strong><span>{_display(result.get("config") or "default")} · {_display(result.get("split", ""))} · {_display(str(result.get("revision", ""))[:8])}</span></header>')
    if result.get("mapping_error") or result.get("error"):
        parts.append(f'<p class="vs-preview-note">{_display(result.get("mapping_error") or result["error"])}</p>')
    for index, row in enumerate(rows[:3]):
        if isinstance(row, dict) and "row" in row and isinstance(row["row"], dict):
            row = row["row"]
        parts.append(f'<details {"open" if index == 0 else ""}><summary>Record {index + 1}</summary>')
        examples = [item for position, item in enumerate(mapped) if isinstance(item, dict) and item.get("row_index", position) == index]
        for number, item in enumerate(examples):
            if len(examples) > 1:
                parts.append(f'<p class="vs-kicker">Example {number + 1}</p>')
            corpus = result.get("dataset_mode") == "corpus" or item.get("mode") == "corpus"
            for field in ("prompt", "target", "text"):
                if item.get(field):
                    label = ("Text" if corpus else "Reference") if field == "target" else field.title()
                    parts.append(f'<div class="vs-data-mapped"><h4>{label}</h4><pre>{_display(item[field])}</pre></div>')
        if isinstance(row, dict):
            parts.append(f'<details class="vs-raw-record" {"" if examples else "open"}><summary>Raw record</summary>')
            for field, value in row.items():
                parts.append(f'<div class="vs-data-field"><h4>{_display(field)}</h4><pre>{_display(value)}</pre></div>')
            parts.append('</details>')
        parts.append('</details>')
    parts.append('</section>')
    return "".join(parts)


def _fetch_local(config, directory: Path, token=False, *, test=False):
    prefix = "test_" if test else ""
    name = "test" if test else "training"
    with tempfile.TemporaryDirectory(prefix="vs-import-") as cache:
        runtime = services(token, cache)
        runtime.pop("split_names")
        return dataset.fetch(
            repo=repository(config.get(prefix + "dataset_source", "")),
            config=config.get(prefix + "dataset_config") or None,
            split=config.get(prefix + "dataset_split") or ("test" if test else "train"),
            output=directory / f"{name}.jsonl", receipt=directory / f"{name}-source.json",
            limit=int(config.get("test_example_limit" if test else "example_limit", 16)),
            dataset_mode=config.get("dataset_mode", "paired"),
            prompt_field=config.get("prompt_field") or None,
            target_field=config.get("target_field") or None,
            text_field=config.get("text_field") or None, **runtime,
        )


def _dataset_subprocess(kind, config, token=False, directory=None, test=False):
    # datasets also caches Hub cards outside its data cache. A separate process
    # sets every cache root before importing either library, without changing
    # another visitor's environment or inheriting the Space owner's credentials.
    with tempfile.TemporaryDirectory(prefix="vs-hub-") as cache:
        allowed = {"PATH", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"}
        env = {key: value for key, value in os.environ.items() if key in allowed}
        env.update(HOME=cache, HF_HOME=cache, HF_HUB_CACHE=str(Path(cache) / "hub"),
                   HF_DATASETS_CACHE=str(Path(cache) / "datasets"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                   HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_TELEMETRY="1")
        if token:
            env["HF_TOKEN"] = token
        payload = {"config": {key: value for key, value in config.items() if key in DATA_FIELDS},
                   "directory": str(directory) if directory else None, "test": test}
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--dataset-operation", kind],
                                    input=json.dumps(payload), env=env, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            raise ValueError("Dataset retrieval timed out. Reduce the example limit or try again") from None
        try:
            response = json.loads(result.stdout)
        except ValueError:
            raise ValueError("Dataset retrieval stopped. Check the repository and try again") from None
        if result.returncode or not response.get("ok"):
            raise ValueError(response.get("error", "Dataset retrieval stopped"))
        return response["result"]


def preview(config, token=False):
    return _dataset_subprocess("preview", config, token)


def fetch_source(config, directory: Path, token=False, *, test=False):
    return _dataset_subprocess("fetch", config, token, directory, test)


def _upload_records(path: str, limit: int) -> list[dict]:
    source = Path(path)
    if source.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Choose a dataset file smaller than 20 MB")
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        with source.open() as handle:
            raw = [json.loads(line) for line in itertools.islice((line for line in handle if line.strip()), limit)]
    elif suffix == ".json":
        raw = json.loads(source.read_text())
        if isinstance(raw, dict):
            raw = raw.get("data", [raw])
        raw = raw[:limit]
    elif suffix == ".txt":
        raw = [{"text": line} for line in source.read_text().splitlines() if line.strip()][:limit]
    elif suffix in {".csv", ".parquet"}:
        import datasets
        stream = datasets.load_dataset("csv" if suffix == ".csv" else "parquet", data_files=str(source), split="train", streaming=True, token=False)
        raw = list(stream.take(limit))
    else:
        raise ValueError("Choose a JSONL, JSON, CSV, Parquet or text file")
    if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
        raise ValueError("Each dataset record must contain named fields")
    return raw


def _map_upload_records(raw, config):
    limit = int(config.get("example_limit", 16))
    rows = []
    mode = config.get("dataset_mode", "paired")
    for row in raw:
        fields = dataset.resolve_mode_fields(row, mode, config.get("prompt_field") or None,
                                             config.get("target_field") or None, config.get("text_field") or None)
        for item in dataset.mapped_examples(row, fields, mode):
            canonical = {"prompt": item["prompt"], "target": item["target"]}
            if mode != "paired":
                canonical["mode"] = mode
            if row.get("split"):
                canonical["split"] = row["split"]
            rows.append(canonical)
            if len(rows) == limit:
                return rows
    return rows


def read_upload(path: str, config) -> list[dict]:
    return _map_upload_records(_upload_records(path, int(config.get("example_limit", 16))), config)


def preview_upload(path: str, config):
    """Preview raw schema and mapped examples using the same importer as runs."""
    raw = _upload_records(path, int(config.get("example_limit", 16)))
    mode = config.get("dataset_mode", "paired")
    result = {"repo": Path(path).name, "config": None, "configs": [], "split": "uploaded records", "splits": [],
              "dataset_mode": mode, "rows": raw[:3], "mapped_rows": [], "error": None, "mapping_error": None,
              "prompt_field": config.get("prompt_field") or None, "target_field": config.get("target_field") or None,
              "text_field": config.get("text_field") or None}
    if not raw:
        result["error"] = "The dataset file contains no examples"
        return result
    try:
        rows = _map_upload_records(raw, config)
        assigned = assign_splits(rows)
        result["counts"] = {name: sum(row.get("split") == name for row in assigned)
                            for name in ("edit", "validation", "test", "preserve")}
        result["total_rows"] = len(rows)
        fields = dataset.resolve_mode_fields(raw[0], mode, config.get("prompt_field") or None,
                                            config.get("target_field") or None, config.get("text_field") or None)
        if mode == "corpus":
            result["text_field"] = fields[1]
        else:
            result["prompt_field"] = fields[0]
            if mode == "paired":
                result["target_field"] = fields[1]
        for index, row in enumerate(raw[:3]):
            for mapped in dataset.mapped_examples(row, fields, mode):
                result["mapped_rows"].append({"row_index": index, **mapped})
                if len(result["mapped_rows"]) >= 20:
                    return result
    except (ValueError, KeyError, TypeError) as error:
        result["mapping_error"] = str(error)[:1000]
    return result


def assign_splits(rows, test_rows=()):
    """Match the core's hash split and reject cross-source identity overlap."""
    if len(rows) < 5 and not all(row.get("split") for row in rows):
        raise ValueError("Choose at least five examples for edit and validation splits")
    rows = [dict(row) for row in rows]
    test_rows = [dict(row, split="test") for row in test_rows]
    if any(row.get("split") for row in rows) and not all(row.get("split") for row in rows):
        raise ValueError("Set a split on every uploaded row, or leave all splits empty")
    identity = lambda row: row["target"] if row.get("mode") == "corpus" else row["prompt"]
    seen = set()
    for row in [*rows, *test_rows]:
        key = " ".join(identity(row).split()).lower()
        if key in seen:
            raise ValueError("Training and test examples must have distinct prompts or corpus texts")
        seen.add(key)
    if rows and not rows[0].get("split"):
        order = sorted(range(len(rows)), key=lambda index: hashlib.sha256(identity(rows[index]).encode()).hexdigest())
        validation = max(1, len(rows) // 5)
        for position, index in enumerate(order):
            rows[index]["split"] = "validation" if position < validation else "edit"
    return [*rows, *test_rows]


def prepare_data(config, directory: Path, token=False):
    if config.get("dataset_location") in {"file", "upload"} or (config.get("dataset_file") and not config.get("dataset_source")):
        rows = read_upload(config["dataset_file"], config)
        source = {"source": "upload", "rows": len(rows)}
        (directory / "training-source.json").write_text(json.dumps(source, indent=2))
    else:
        fetch_source(config, directory, token)
        rows = [json.loads(line) for line in (directory / "training.jsonl").read_text().splitlines()]
    test_rows = []
    if config.get("test_dataset_source"):
        fetch_source(config, directory, token, test=True)
        test_rows = [json.loads(line) for line in (directory / "test.jsonl").read_text().splitlines()]
    combined = assign_splits(rows, test_rows)
    output = directory / "input.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined))
    return output


def publish_run(run: Path, *, kind: str, repo: str, visibility: str, create: bool, token):
    if not token:
        raise ValueError("Sign in with Hugging Face to save to your account")
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    module = core_module("publish" if kind == "model" else "trajectory")
    extra = {"delete_operation": CommitOperationDelete} if kind == "model" else {}
    return module.publish(run=run, repo=repo, visibility=visibility, create=create,
                          api=HfApi(token=token), operation=CommitOperationAdd, **extra)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3 or sys.argv[1] != "--dataset-operation":
            raise ValueError("Choose a dataset operation")
        payload = json.load(sys.stdin)
        token = os.environ.get("HF_TOKEN") or False
        if sys.argv[2] == "preview":
            result = _preview_local(payload["config"], token)
        elif sys.argv[2] == "fetch":
            result = _fetch_local(payload["config"], Path(payload["directory"]), token, test=payload["test"])
        else:
            raise ValueError("Unknown dataset operation")
        response = {"ok": True, "result": result}
    except Exception as error:
        message = str(error)
        if os.environ.get("HF_TOKEN"):
            message = message.replace(os.environ["HF_TOKEN"], "[redacted]")
        response = {"ok": False, "error": message[:1000]}
    print(json.dumps(response, ensure_ascii=False), flush=True)
    # All importer output files are closed. Avoid running native streaming-reader
    # finalisers while their background threads are shutting down. The parent
    # owns cache cleanup and the OS reclaims this short-lived reader process.
    os._exit(0)
