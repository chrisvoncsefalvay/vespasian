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
        for info in api.list_datasets(author=owner, limit=100, sort="lastModified", direction=-1):
            choices.add(info.id)
    return who["name"], sorted(choices, key=str.casefold)


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


def preview_html(result):
    rows = result.get("rows", [])
    mapped = result.get("mapped_rows", [])
    parts = ['<section class="vs-data-preview">']
    parts.append(f'<header><strong>{_display(result.get("repo", "Dataset"))}</strong><span>{_display(result.get("config") or "default")} · {_display(result.get("split", ""))} · {_display(str(result.get("revision", ""))[:8])}</span></header>')
    if result.get("mapping_error") or result.get("error"):
        parts.append(f'<p class="vs-preview-note">{_display(result.get("mapping_error") or result["error"])}</p>')
    for index, row in enumerate(rows[:3]):
        if isinstance(row, dict) and "row" in row and isinstance(row["row"], dict):
            row = row["row"]
        parts.append(f'<details {"open" if index == 0 else ""}><summary>Record {index + 1}</summary>')
        if isinstance(row, dict):
            for field, value in row.items():
                parts.append(f'<div class="vs-data-field"><h4>{_display(field)}</h4><pre>{_display(value)}</pre></div>')
        for item in (item for position, item in enumerate(mapped) if isinstance(item, dict) and item.get("row_index", position) == index):
            for field in ("prompt", "target", "text"):
                if item.get(field):
                    label = "Reference" if field == "target" else field.title()
                    parts.append(f'<div class="vs-data-mapped"><h4>{label}</h4><pre>{_display(item[field])}</pre></div>')
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


def read_upload(path: str, config) -> list[dict]:
    source = Path(path)
    if source.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Choose a dataset file smaller than20MB")
    suffix = source.suffix.lower()
    limit = int(config.get("example_limit", 16))
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
    rows = []
    mode = config.get("dataset_mode", "paired")
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("Each dataset record must contain named fields")
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
