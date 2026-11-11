"""Private, durable exact-value patches. No model, dataset or provider calls here.

A manifest is cumulative relative to a verified pristine base. Row edits store
only selected rows; other edits store complete changed tensors. Values are the
actual stored dtype, not an approximate low-rank factor or rounded delta.
"""
from __future__ import annotations

import hashlib
import contextlib
import shutil
import sys
import json
import os
from pathlib import Path
import re
import tempfile
import uuid
import zipfile


class PatchError(ValueError):
    pass


_BADGE = '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="36" viewBox="0 0 240 36" role="img" aria-label="Created with Vespasian"><rect width="240" height="36" rx="6" fill="#14181b"/><path d="M9 9 19 27 29 9M15 9l4 8 4-8" fill="none" stroke="#d5a278" stroke-width="2"/><text x="39" y="23" font-family="sans-serif" font-size="15" fill="#f5eee7">Created with Vespasian</text></svg>'

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path):
    # Windows does not support opening directory descriptors this way. Files
    # are still flushed before atomic replace; directory-entry power-loss
    # durability is platform-dependent there (unlike the POSIX fsync path).
    if sys.platform == "win32":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def _atomic(path, data):
    path = Path(path)
    _directory(path.parent)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _read(path):
    try:
        path = Path(path)
        if path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
            raise PatchError("Unsafe patch metadata file")
        return json.loads(path.read_bytes())
    except Exception:
        raise PatchError("Patch metadata is missing or invalid") from None


def _base(engine):
    engine.ready()
    files = engine.source_files
    if not files or any(not isinstance(row.get("sha256"), str) or not _HASH.fullmatch(row["sha256"])
                        or type(row.get("bytes")) is not int or row["bytes"] < 0 for row in files):
        raise PatchError("A verified source checkpoint manifest is required for replayable patches")
    # Private local paths and source file names are deliberately not exported.
    source = sorted((row.get("path"), row["sha256"], row["bytes"]) for row in files)
    layout = {"parameters": [(name, list(value.shape), str(value.dtype)) for name, value in engine.model.named_parameters()],
              "token_tables": [{key: row.get(key) for key in ("name", "aliases", "roles", "tied", "shape")} for row in engine.token_table_info]}
    revision = Path(engine.source_dir).name if engine.source_dir else ""
    floating = sorted({str(value.dtype).removeprefix("torch.") for value in engine.model.parameters() if value.is_floating_point()})
    requested = getattr(engine, "load_dtype", None)
    load_dtype = str(requested).removeprefix("torch.") if requested is not None else (floating[0] if len(floating) == 1 else None)
    return {"source_sha256": _sha(_json(source)), "source_file_count": len(source),
            "layout_sha256": _sha(_json(layout)), "floating_dtypes": floating, "load_dtype": load_dtype,
            "revision": revision if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision) else None}


def _compatible(expected, actual):
    return all(expected.get(key) == actual.get(key) for key in ("source_sha256", "source_file_count", "layout_sha256"))


def _write_manifest(directory, manifest):
    data = _json(manifest)
    digest = _sha(data)
    path = Path(directory) / "manifests" / (digest + ".json")
    if path.exists():
        if path.read_bytes() != data:
            raise PatchError("Existing content-addressed manifest is corrupt")
    else:
        _atomic(path, data)
    return digest


def _manifest(directory, digest):
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise PatchError("Invalid patch manifest reference")
    path = Path(directory) / "manifests" / (digest + ".json")
    try:
        if path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
            raise PatchError("Unsafe patch manifest file")
        data = path.read_bytes()
    except OSError:
        raise PatchError("Referenced patch manifest is missing") from None
    if _sha(data) != digest:
        raise PatchError("Patch manifest checksum mismatch")
    try:
        value = json.loads(data)
    except Exception:
        raise PatchError("Invalid patch manifest") from None
    if not isinstance(value, dict) or set(value) != {"version", "encoding", "base", "turn", "status", "tensors"}:
        raise PatchError("Unexpected patch manifest fields")
    if value.get("version") != 1 or value.get("encoding") != "exact-after-values" or not isinstance(value.get("tensors"), list):
        raise PatchError("Unsupported patch format")
    base = value.get("base")
    required = {"source_sha256", "source_file_count", "layout_sha256", "floating_dtypes", "revision"}
    if not isinstance(base, dict) or not required <= set(base) or set(base) - required - {"model", "load_dtype"}:
        raise PatchError("Unexpected base metadata fields")
    if any(not isinstance(base[key], str) or not _HASH.fullmatch(base[key]) for key in ("source_sha256", "layout_sha256")):
        raise PatchError("Invalid base fingerprints")
    if type(base["source_file_count"]) is not int or base["source_file_count"] < 1:
        raise PatchError("Invalid base file count")
    if not isinstance(base["floating_dtypes"], list) or not base["floating_dtypes"] or any(item not in {"float16", "bfloat16", "float32", "float64"} for item in base["floating_dtypes"]):
        raise PatchError("Unsupported base floating dtypes")
    if base.get("load_dtype") is not None and base["load_dtype"] not in {"float16", "bfloat16", "float32", "float64"}:
        raise PatchError("Unsupported requested loading dtype")
    if base["revision"] is not None and not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", str(base["revision"])):
        raise PatchError("Invalid base revision")
    if "model" in base and (not isinstance(base["model"], str) or not _REPO.fullmatch(base["model"]) or ".." in base["model"]):
        raise PatchError("Invalid base model identifier")
    for row in value["tensors"]:
        if not isinstance(row, dict) or set(row) != {"name", "kind", "shape", "dtype", "ids", "sha256", "bytes"}:
            raise PatchError("Unexpected patch tensor metadata fields")
        if not isinstance(row["name"], str) or not row["name"] or len(row["name"]) > 2048 or row["kind"] not in {"rows", "tensor"}:
            raise PatchError("Invalid patch tensor name or kind")
        if not isinstance(row["sha256"], str) or not _HASH.fullmatch(row["sha256"]) or type(row["bytes"]) is not int or row["bytes"] < 1:
            raise PatchError("Invalid tensor blob identity")
        if row["dtype"] not in {"float16", "bfloat16", "float32", "float64"} or not isinstance(row["shape"], list) or not row["shape"] or any(type(n) is not int or n < 1 for n in row["shape"]):
            raise PatchError("Invalid tensor shape or dtype")
    if type(value.get("turn")) is not int or value["turn"] < 0 or value.get("status") not in {"baseline", "candidate", "accepted", "rejected"}:
        raise PatchError("Invalid patch state")
    return value


def _index(directory):
    value = _read(Path(directory) / "index.json")
    if not isinstance(value, dict) or set(value) != {"version", "journal_id", "base", "latest", "best"} or value.get("version") != 1 or not re.fullmatch(r"[0-9a-f]{32}", str(value.get("journal_id", ""))):
        raise PatchError("Invalid patch journal")
    latest, best = _manifest(directory, value.get("latest")), _manifest(directory, value.get("best"))
    if latest["base"] != value.get("base") or best["base"] != value.get("base"):
        raise PatchError("Patch journal base identities disagree")
    if best["status"] not in {"baseline", "accepted"}:
        raise PatchError("Best patch is not a retained checkpoint")
    return value


def initialize(engine, directory, base_model=None):
    """Create baseline latest/best pointers before the first edit; idempotent."""
    directory = _directory(directory)
    base = _base(engine)
    if base_model is not None:
        if not isinstance(base_model, str) or not _REPO.fullmatch(base_model) or ".." in base_model:
            raise PatchError("Use a Hub model ID or omit the private local model path")
        base["model"] = base_model
    if (directory / "index.json").exists():
        existing = _index(directory)
        if not _compatible(existing["base"], base):
            raise PatchError("Journal belongs to a different base checkpoint")
        return describe(directory)
    if engine.changed or engine.pending or engine.row_changed or engine.row_pending:
        raise PatchError("Initialize the patch journal on a pristine base before editing")
    manifest = {"version": 1, "encoding": "exact-after-values", "base": base,
                "turn": 0, "status": "baseline", "tensors": []}
    digest = _write_manifest(directory, manifest)
    _atomic(directory / "index.json", _json({"version": 1, "journal_id": uuid.uuid4().hex,
                                             "base": base, "latest": digest, "best": digest}))
    return describe(directory)


def _blob(directory, digest):
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise PatchError("Invalid patch tensor reference")
    path = Path(directory) / "blobs" / (digest + ".safetensors")
    try:
        if path.is_symlink():
            raise PatchError("Symlinked tensor blobs are not allowed")
        data = path.read_bytes()
    except OSError:
        raise PatchError("Patch tensor file is missing") from None
    if _sha(data) != digest:
        raise PatchError("Patch tensor checksum mismatch")
    return data


def record_candidate(engine, directory, turn):
    """Persist the current candidate before evaluation, without advancing best."""
    from safetensors.torch import save
    import torch
    directory = Path(directory)
    index = _index(directory)
    if not _compatible(index["base"], _base(engine)):
        raise PatchError("Candidate base identity does not match the journal")
    previous = _manifest(directory, index["latest"])
    if type(turn) is not int or turn <= previous["turn"]:
        raise PatchError("A new candidate needs a strictly increasing turn number")
    if not engine.pending and not engine.row_pending:
        raise PatchError("No candidate patch is pending")
    entries = []
    names = sorted(engine.changed | set(engine.pending))
    rows = {name: set(ids) for name, ids in engine.row_changed.items()}
    for name, pending in engine.row_pending.items():
        rows.setdefault(name, set()).update(pending["ids"])
    if set(names) & set(rows):
        raise PatchError("A physical tensor cannot be stored twice")
    for name in names + sorted(rows):
        is_rows = name in rows
        parameter = (engine.token_tables if is_rows else engine.editable).get(name)
        if parameter is None:
            raise PatchError("A changed tensor is no longer an eligible edit target")
        ids = sorted(rows[name]) if is_rows else None
        values = parameter.detach()
        if ids is not None:
            values = values.index_select(0, torch.tensor(ids, dtype=torch.long, device=parameter.device))
        values = values.cpu().contiguous().clone()
        if not torch.isfinite(values).all():
            raise PatchError("Nonfinite values cannot be persisted")
        data = save({"values": values})
        digest = _sha(data)
        blob = directory / "blobs" / (digest + ".safetensors")
        if blob.exists():
            _blob(directory, digest)
        else:
            _atomic(blob, data)
        entries.append({"name": name, "kind": "rows" if is_rows else "tensor", "shape": list(parameter.shape),
                        "dtype": str(parameter.dtype).removeprefix("torch."), "ids": ids,
                        "sha256": digest, "bytes": len(data)})
    manifest = {"version": 1, "encoding": "exact-after-values", "base": index["base"],
                "turn": turn, "status": "candidate", "tensors": entries}
    index["latest"] = _write_manifest(directory, manifest)
    _atomic(directory / "index.json", _json(index))
    return describe(directory)


def finalize(directory, turn, accepted):
    """Commit a measured decision atomically; rejected latest never replaces best."""
    directory = Path(directory)
    if type(accepted) is not bool:
        raise PatchError("The measured acceptance decision must be a boolean")
    index = _index(directory)
    latest = _manifest(directory, index["latest"])
    if latest["turn"] != turn:
        raise PatchError("The latest persisted candidate does not match this turn")
    status = "accepted" if accepted else "rejected"
    if latest["status"] == status:
        return describe(directory)
    if latest["status"] != "candidate":
        raise PatchError("This candidate already has a different decision")
    latest["status"] = status
    index["latest"] = _write_manifest(directory, latest)
    if accepted:
        index["best"] = index["latest"]
    _atomic(directory / "index.json", _json(index))
    return describe(directory)


def describe(directory):
    index = _index(directory)
    result = {"version": 1, "encoding": "exact-after-values", "journal_id": index["journal_id"], "base": index["base"]}
    for which in ("latest", "best"):
        manifest = _manifest(directory, index[which])
        result[which] = {"manifest_sha256": index[which], "turn": manifest["turn"], "status": manifest["status"],
                         "tensor_count": len(manifest["tensors"]), "tensor_bytes": sum(row["bytes"] for row in manifest["tensors"])}
    return result


def restore(engine, directory, which="best"):
    """Replay exact values onto a freshly loaded matching base, never an edited model."""
    from safetensors.torch import load
    import torch
    if which not in {"best", "latest"}:
        raise PatchError("Choose best or latest")
    index = _index(directory)
    if not _compatible(index["base"], _base(engine)):
        raise PatchError("Patch base checkpoint or tensor layout/dtype does not match")
    if engine.changed or engine.pending or engine.row_changed or engine.row_pending:
        raise PatchError("Restore requires a freshly loaded, unedited base model")
    manifest = _manifest(directory, index[which])
    prepared, seen = [], set()
    for row in manifest["tensors"]:
        name, kind = row.get("name"), row.get("kind")
        if not isinstance(name, str) or name in seen or kind not in {"rows", "tensor"}:
            raise PatchError("Duplicate or invalid canonical tensor")
        seen.add(name)
        parameter = (engine.token_tables if kind == "rows" else engine.editable).get(name)
        if parameter is None or list(parameter.shape) != row.get("shape") or str(parameter.dtype).removeprefix("torch.") != row.get("dtype"):
            raise PatchError("Patch tensor metadata does not match the loaded base")
        ids = row.get("ids")
        if kind == "rows":
            if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 or i >= parameter.shape[0] for i in ids) or ids != sorted(set(ids)):
                raise PatchError("Invalid sparse row selectors")
        elif ids is not None:
            raise PatchError("Dense patches cannot contain row selectors")
        data = _blob(directory, row.get("sha256"))
        if row.get("bytes") != len(data):
            raise PatchError("Patch tensor size mismatch")
        try:
            tensors = load(data)
        except Exception:
            raise PatchError("Invalid patch tensor encoding") from None
        if set(tensors) != {"values"}:
            raise PatchError("Unexpected patch tensor entries")
        values = tensors["values"]
        shape = [len(ids), *parameter.shape[1:]] if kind == "rows" else list(parameter.shape)
        if list(values.shape) != shape or values.dtype != parameter.dtype or not torch.isfinite(values).all():
            raise PatchError("Invalid patch tensor values")
        original = (parameter.detach() if ids is None else parameter.detach().index_select(0, torch.tensor(ids, dtype=torch.long, device=parameter.device))).cpu().clone()
        prepared.append((name, parameter, ids, values, original))
    try:
        with torch.no_grad():
            for name, parameter, ids, values, original in prepared:
                if ids is None:
                    parameter.copy_(values.to(parameter.device))
                else:
                    parameter.index_copy_(0, torch.tensor(ids, device=parameter.device), values.to(parameter.device))
    except BaseException:
        with torch.no_grad():
            for _, parameter, ids, _, original in prepared:
                if ids is None:
                    parameter.copy_(original.to(parameter.device))
                else:
                    parameter.index_copy_(0, torch.tensor(ids, device=parameter.device), original.to(parameter.device))
        raise
    engine.changed.update(name for name, _, ids, _, _ in prepared if ids is None)
    engine.row_changed.update({name: set(ids) for name, _, ids, _, _ in prepared if ids is not None})
    engine.model.requires_grad_(False)
    engine.model.zero_grad(set_to_none=True)
    return {"restored": which, "turn": manifest["turn"], "status": manifest["status"],
            "manifest_sha256": index[which], "tensor_count": len(prepared), "base_verified": True}


def _files(directory):
    directory = Path(directory)
    index = _index(directory)
    names = {"index.json"}
    for digest in {index["latest"], index["best"]}:
        manifest = _manifest(directory, digest)
        names.add(f"manifests/{digest}.json")
        for row in manifest["tensors"]:
            _blob(directory, row.get("sha256"))
            names.add(f'blobs/{row["sha256"]}.safetensors')
    return sorted(names)


def _export_snapshot(directory, destination):
    """Export only the current latest/best and referenced blobs, not historical edits."""
    directory, destination = Path(directory), Path(destination)
    names = _files(directory)
    _directory(destination.parent)
    fd, temporary = tempfile.mkstemp(prefix=".patch-bundle-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_STORED) as archive:
                for name in names:
                    archive.write(directory / name, name)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(destination), "bytes": destination.stat().st_size,
            "sha256": _file_sha(destination), "files": len(names), **describe(directory)}


def import_bundle(bundle, directory):
    """Import a safe standalone bundle into a new directory; no model is loaded."""
    directory = Path(directory)
    if directory.exists():
        raise PatchError("Choose a new directory for the imported patch bundle")
    _directory(directory.parent)
    with tempfile.TemporaryDirectory(prefix=".patch-import-", dir=directory.parent) as temporary:
        root = Path(temporary) / "journal"
        root.mkdir(mode=0o700)
        try:
            with zipfile.ZipFile(bundle) as archive:
                infos = archive.infolist()
                if len(infos) > 10000 or sum(info.file_size for info in infos) > 32 * 1024**3:
                    raise PatchError("Patch archive exceeds supported limits")
                names = [info.filename for info in infos]
                if len(names) != len(set(names)):
                    raise PatchError("Duplicate patch archive paths")
                for info in infos:
                    if not re.fullmatch(r"index\.json|manifests/[0-9a-f]{64}\.json|blobs/[0-9a-f]{64}\.safetensors", info.filename):
                        raise PatchError("Unexpected patch archive path")
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise PatchError("Symlinks are not allowed in patch archives")
                    _atomic(root / info.filename, archive.read(info))
            if set(_files(root)) != set(names):
                raise PatchError("Patch bundle contains unreferenced or missing files")
            os.replace(root, directory)
            _sync_directory(directory.parent)
        except (zipfile.BadZipFile, OSError):
            raise PatchError("Could not import the patch archive") from None
    return describe(directory)


def _publish_snapshot(directory, repo, token, *, consent=False, create=False, api=None, operation=None, download=None, previous=None):
    """Explicit private-only sync. Never uses implicit owner/login credentials."""
    if consent is not True:
        raise PatchError("Explicit private patch persistence consent is required")
    if not isinstance(token, str) or not token.strip():
        raise PatchError("A session Hugging Face token is required; implicit credentials are disabled")
    if not isinstance(repo, str) or not _REPO.fullmatch(repo) or ".." in repo:
        raise PatchError("Choose the authorized owner/repository destination")
    directory = Path(directory)
    names = _files(directory)
    index = _index(directory)
    if api is not None and hasattr(api, "token") and api.token != token:
        raise PatchError("The API client must use the supplied session credentials, not an owner login")
    if api is None or operation is None or download is None:
        from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
        api = api or HfApi(token=token)
        operation = operation or CommitOperationAdd
        download = download or hf_hub_download
    try:
        created = False
        try:
            info = api.repo_info(repo_id=repo, repo_type="model")
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status != 404 or not create:
                raise PatchError("Could not verify the private patch repository") from None
            api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=False)
            created = True
            info = api.repo_info(repo_id=repo, repo_type="model")
        if getattr(info, "private", None) is not True:
            raise PatchError("Patch persistence requires a verified private repository")
        # Recheck immediately before the atomic commit; never change visibility.
        info = api.repo_info(repo_id=repo, repo_type="model")
        revision = getattr(info, "sha", None)
        if getattr(info, "private", None) is not True or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
            raise PatchError("Private repository revision could not be verified")
        prefix = f'vespasian-patches/{index["journal_id"]}'
        if created:
            root = _directory(directory / "created-repository-card")
            lines = ["---", "tags:", "- vespasian", "- model-editing", "- patches"]
            if index["base"].get("model"):
                lines.append("base_model: " + json.dumps(index["base"]["model"]))
            lines += ["---", "", "[![Created with Vespasian](vespasian-badge.svg)](https://chrisvoncsefalvay.github.io/vespasian/)",
                      "", "# Private Vespasian patch storage", "",
                      "This repository stores exact-value patches, not a standalone model checkpoint.",
                      "Restore the best patch onto its matching verified base model before use. Latest may be pending or rejected; best is the retained state.",
                      "Patch manifests record source fingerprints, tensor shapes, dtypes, sparse row selectors and measured acceptance status. They do not establish broad model quality or safety.",
                      "Private objectives, prompts, dataset rows and provider credentials are not included. No benchmark or improvement claim is made by this storage card.",
                      "Review the base model's licence and restrictions. No new licence is inferred.", ""]
            _atomic(root / "README.md", "\n".join(lines).encode())
            _atomic(root / "vespasian-badge.svg", _BADGE.encode())
            names += ["created-repository-card/README.md", "created-repository-card/vespasian-badge.svg"]
        def remote_name(name):
            return name.removeprefix("created-repository-card/") if name.startswith("created-repository-card/") else f"{prefix}/{name}"
        hashes = {name: _file_sha(directory / name) for name in names}
        reusable = isinstance(previous, dict) and previous.get("repo") == repo and previous.get("revision") == revision and previous.get("prefix") == prefix
        changed = [name for name in names if not reusable or previous.get("file_hashes", {}).get(name) != hashes[name]]
        if not changed:
            return {"repo": repo, "revision": revision, "private": True, "verified": True, "prefix": prefix, "files": len(names), "uploaded_files": 0, "unchanged": True, "file_hashes": hashes}
        actions = [operation(path_in_repo=remote_name(name), path_or_fileobj=str(directory / name)) for name in changed]
        result = api.create_commit(repo_id=repo, repo_type="model", operations=actions,
                                   commit_message="Persist Vespasian latest and best patches", parent_commit=revision)
        commit = getattr(result, "oid", None)
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
            raise PatchError("Patch commit outcome is uncertain; inspect the repository before retrying")
        if getattr(api.repo_info(repo_id=repo, repo_type="model", revision=commit), "private", None) is not True:
            raise PatchError("Patch commit was accepted but private visibility could not be verified")
        with tempfile.TemporaryDirectory(prefix="vespasian-private-readback-") as cache:
            for name in changed:
                path = download(repo_id=repo, filename=remote_name(name), repo_type="model", revision=commit, token=token, cache_dir=cache)
                if _file_sha(path) != _file_sha(directory / name):
                    raise PatchError("Patch commit was accepted but readback verification failed")
        return {"repo": repo, "revision": commit, "private": True, "verified": True, "prefix": prefix, "files": len(names), "uploaded_files": len(changed), "unchanged": False, "file_hashes": hashes}
    except PatchError:
        raise
    except Exception:
        raise PatchError("Private patch persistence failed; local patches remain available. Check the repository before retrying an uncertain upload") from None



@contextlib.contextmanager
def _snapshot(directory):
    """Freeze one atomic pointer view; never export a mixture of journal turns."""
    directory = Path(directory)
    before = (directory / "index.json").read_bytes()
    names = _files(directory)
    with tempfile.TemporaryDirectory(prefix="vespasian-patch-snapshot-") as temp:
        root = Path(temp)
        for name in names:
            _directory((root / name).parent)
            shutil.copyfile(directory / name, root / name)
            os.chmod(root / name, 0o600)
        if (root / "index.json").read_bytes() != before or (directory / "index.json").read_bytes() != before:
            raise PatchError("The patch journal changed while preparing the snapshot; retry after the current turn")
        _files(root)
        yield root


def export_bundle(directory, destination):
    with _snapshot(directory) as snapshot:
        return _export_snapshot(snapshot, destination)


def publish_private(directory, repo, token, *, consent=False, create=False, api=None, operation=None, download=None):
    if consent is not True:
        raise PatchError("Explicit private patch persistence consent is required")
    if not isinstance(token, str) or not token.strip():
        raise PatchError("A session Hugging Face token is required; implicit credentials are disabled")
    directory = Path(directory)
    receipt_path = directory / ".backup.json"
    try:
        previous = _read(receipt_path) if receipt_path.exists() else None
    except PatchError:
        previous = None
    with _snapshot(directory) as snapshot:
        result = _publish_snapshot(snapshot, repo, token, consent=consent, create=create,
                                   api=api, operation=operation, download=download, previous=previous)
    _atomic(receipt_path, _json(result))
    return {key: value for key, value in result.items() if key != "file_hashes"}


def materialize(directory, model=None, *, revision=None, output, device="cpu"):
    """Materialize a verified best patch from its base without private dataset rows."""
    import torch
    from worker import Engine, load_model_components
    index = _index(directory)
    identifier = model or index["base"].get("model")
    if not identifier:
        raise PatchError("Supply the original base model path or Hub ID")
    revision = revision or index["base"].get("revision")
    if device not in {"cpu", "cuda", "auto"}:
        raise PatchError("Choose cpu, cuda or auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if Path(identifier).is_dir():
        source = Path(identifier).resolve()
    else:
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download(identifier, "config.json", revision=revision)).parent
        revision = source.name
    # Match the patch's stored dtype, which may differ from a new run's default.
    manifest = _manifest(directory, index["best"])
    dtypes = set(index["base"].get("floating_dtypes", []))
    dtype_name = index["base"].get("load_dtype")
    if dtype_name is None:
        if len(dtypes) != 1:
            raise PatchError("This older mixed-dtype patch lacks its original loader dtype; use the matching application runtime for replay")
        dtype_name = next(iter(dtypes))
    if dtype_name not in {"float16", "bfloat16", "float32", "float64"}:
        raise PatchError("Unsupported base floating-point dtype")
    dtype = getattr(torch, dtype_name)
    engine = Engine()
    engine.model, engine.tokenizer, engine.processor = load_model_components(identifier, revision=revision, device=device, dtype=dtype)
    engine.model.eval().requires_grad_(False)
    engine.load_dtype = dtype_name
    engine.source_dir = source
    engine.source_files = engine.manifest(source)
    engine.register_editable()
    receipt = restore(engine, directory, "best")
    receipt["checkpoint"] = engine.save({"path": str(output)})
    return receipt



def dispatch(engine, request):
    """Shared worker protocol. The caller scopes paths to its run directory."""
    command = request.get("command")
    directory = request.get("path")
    if not isinstance(directory, str) or not directory:
        raise PatchError("A local patch journal path is required")
    if command == "patch_init":
        return initialize(engine, directory, request.get("base_model"))
    if command == "patch_candidate":
        return record_candidate(engine, directory, request.get("turn"))
    if command == "patch_finalize":
        return finalize(directory, request.get("turn"), request.get("accepted"))
    if command == "patch_restore":
        return restore(engine, directory, request.get("which", "best"))
    if command == "patch_export":
        destination = request.get("destination")
        if not isinstance(destination, str) or not destination:
            raise PatchError("A bundle destination is required")
        return export_bundle(directory, destination)
    if command == "patch_describe":
        return describe(directory)
    raise PatchError("Unknown patch command")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Persist or replay Vespasian exact-value patches")
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup")
    backup.add_argument("--directory", required=True)
    backup.add_argument("--repo", required=True)
    backup.add_argument("--authorized", action="store_true")
    backup.add_argument("--create", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--directory", required=True)
    export.add_argument("--output", required=True)
    replay = commands.add_parser("materialize")
    source = replay.add_mutually_exclusive_group(required=True)
    source.add_argument("--directory")
    source.add_argument("--bundle")
    replay.add_argument("--model")
    replay.add_argument("--revision")
    replay.add_argument("--output", required=True)
    replay.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.command == "backup":
            result = publish_private(args.directory, args.repo, os.environ.get("HF_TOKEN"),
                                     consent=args.authorized, create=args.create)
        elif args.command == "export":
            result = export_bundle(args.directory, args.output)
        elif args.bundle:
            with tempfile.TemporaryDirectory(prefix="vespasian-patch-replay-") as temp:
                directory = Path(temp) / "journal"
                import_bundle(args.bundle, directory)
                result = materialize(directory, args.model, revision=args.revision, output=args.output, device=args.device)
        else:
            result = materialize(args.directory, args.model, revision=args.revision, output=args.output, device=args.device)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (PatchError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
