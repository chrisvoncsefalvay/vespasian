#!/usr/bin/env python3
"""Stage and publish the ZeroGPU app independently of the Cargo package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[2]
TARGET = "x86_64-unknown-linux-gnu"
SPACE_FILES = (
    "README.md",
    "requirements.txt",
    "app.py",
    "ui.py",
    "dashboard.py",
    "setup_helpers.py",
    "runtime.py",
    "gpu.py",
    "bridge.py",
    "python_bridge.py",
    "hub.py",
)
CORE_FILES = ("worker.py", "dataset.py", "task_metadata.py", "patches.py", "health.py", "publish.py", "trajectory.py")
PROVENANCE = "PROVENANCE.json"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return path.read_bytes()


def tree_files(directory: Path) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Expected a directory: {directory}")
    result = []
    for path in sorted(directory.rglob("*")):
        parts = path.relative_to(directory).parts
        if any(part.startswith(".") or part == "__pycache__" for part in parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported in the bundle: {path}")
        if path.is_file():
            result.append(path)
    return result


def validate_binary(data: bytes) -> None:
    # ELF64, little endian, System V/Linux ABI, x86-64; rejects a host ARM build.
    if (
        len(data) < 64
        or data[:6] != b"\x7fELF\x02\x01"
        or data[7] not in {0, 3}
        or int.from_bytes(data[18:20], "little") != 62
    ):
        raise ValueError("The Space requires a Linux x86-64 ELF controller")


def build_bundle(root: Path, binary: Path, output: Path, *, allow_dirty: bool = False) -> dict:
    root, binary, output = root.resolve(), binary.absolute(), output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists; choose a fresh directory: {output}")
    source_status = git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "Cargo.toml",
        "Cargo.lock",
        "src",
        "python",
        "examples",
        "spaces/zerogpu",
        "LICENSE",
    )
    if source_status and not allow_dirty:
        raise ValueError("Source files have uncommitted changes; commit them or use --allow-dirty locally")

    app = root / "spaces" / "zerogpu"
    sources = {name: app / name for name in SPACE_FILES}
    if (app / "packages.txt").exists():
        sources["packages.txt"] = app / "packages.txt"
    if (app / "assets").exists():
        for path in tree_files(app / "assets"):
            sources[path.relative_to(app).as_posix()] = path
    sources.update({f"core/{name}": root / "python" / name for name in CORE_FILES})
    for path in tree_files(root / "examples"):
        sources[path.relative_to(root).as_posix()] = path
    sources["LICENSE"] = root / "LICENSE"

    payload = {name: source_bytes(path) for name, path in sources.items()}
    payload["bin/vespasian"] = source_bytes(binary)
    validate_binary(payload["bin/vespasian"])
    source_paths = set(sources.values()) | {root / "Cargo.toml", root / "Cargo.lock"}
    source_paths.update(tree_files(root / "src"))
    manifest = {
        "schema_version": 1,
        "source_commit": git(root, "rev-parse", "HEAD"),
        "source_tree": git(root, "rev-parse", "HEAD^{tree}"),
        "source_dirty": bool(source_status),
        "version": tomllib.loads(source_bytes(root / "Cargo.toml").decode())["package"]["version"],
        "target": TARGET,
        "source_sha256": {
            path.relative_to(root).as_posix(): digest(source_bytes(path))
            for path in sorted(source_paths)
        },
        "files": {
            name: {
                "sha256": digest(data),
                "size": len(data),
                "executable": name == "bin/vespasian"
                or bool(sources[name].stat().st_mode & stat.S_IXUSR),
            }
            for name, data in sorted(payload.items())
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, data in payload.items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            destination.chmod(0o755 if manifest["files"][name]["executable"] else 0o644)
        (staging / PROVENANCE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify_bundle(staging)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest


def manifest_files(manifest: dict) -> set[str]:
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported bundle provenance")
    names = set(manifest["files"])
    for name in names:
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or path.as_posix() != name
            or any(part.startswith(".") for part in path.parts)
            or "\\" in name
            or name == PROVENANCE
        ):
            raise ValueError(f"Invalid bundle path: {name!r}")
    return names


def verify_bundle(bundle: Path) -> dict:
    manifest = json.loads(source_bytes(bundle / PROVENANCE))
    expected = manifest_files(manifest)
    actual = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in bundle: {path}")
        if path.is_file():
            actual.add(path.relative_to(bundle).as_posix())
    if actual != expected | {PROVENANCE}:
        raise ValueError(f"Bundle inventory differs from provenance: {sorted(actual ^ (expected | {PROVENANCE}))}")
    for name in expected:
        data = source_bytes(bundle / name)
        details = manifest["files"][name]
        if digest(data) != details["sha256"] or len(data) != details["size"]:
            raise ValueError(f"Bundle hash mismatch: {name}")
    validate_binary(source_bytes(bundle / "bin" / "vespasian"))
    return manifest


def hub_tools():
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    return HfApi, CommitOperationAdd, CommitOperationDelete, hf_hub_download, EntryNotFoundError


def upload_bundle(bundle: Path, repo_id: str) -> str:
    manifest = verify_bundle(bundle)
    if manifest["source_dirty"]:
        raise ValueError("Only a bundle built from committed source can be deployed")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required for Space deployment")
    HfApi, Add, Delete, download, Missing = hub_tools()
    api = HfApi(token=token)
    info = api.space_info(repo_id=repo_id)
    if info.sdk != "gradio":
        raise ValueError("The destination must be an existing Gradio Space")
    runtime = api.get_space_runtime(repo_id=repo_id)
    if (runtime.requested_hardware or runtime.hardware) != "zero-a10g":
        raise ValueError("The destination must already use ZeroGPU hardware")
    previous = set()
    try:
        previous_path = download(repo_id, PROVENANCE, repo_type="space", revision=info.sha, token=token)
    except Missing:
        pass
    else:
        previous = manifest_files(json.loads(Path(previous_path).read_text()))
    current = manifest_files(manifest)
    stale = sorted(previous - current)
    if stale:
        print("Removing obsolete generated files: " + ", ".join(stale))
    operations = [Delete(path_in_repo=name) for name in stale]
    operations.extend(
        Add(path_in_repo=name, path_or_fileobj=str(bundle / name))
        for name in sorted(current | {PROVENANCE})
    )
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="space",
        revision="main",
        parent_commit=info.sha,
        operations=operations,
        commit_message=f"Deploy Vespasian {manifest['version']} ({manifest['source_commit'][:12]})",
    )
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Stage a fresh Space bundle")
    build.add_argument("--binary", type=Path, default=ROOT / "target" / TARGET / "release" / "vespasian")
    build.add_argument("--output", type=Path, default=ROOT / "dist" / "zerogpu-space")
    build.add_argument("--allow-dirty", action="store_true", help="Local inspection only; prevents uploading this bundle")
    verify = commands.add_parser("verify", help="Check a staged bundle against its hashes")
    verify.add_argument("bundle", type=Path)
    upload = commands.add_parser("upload", help="Deploy a clean bundle to an existing ZeroGPU Space")
    upload.add_argument("bundle", type=Path)
    upload.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    if args.command == "build":
        manifest = build_bundle(ROOT, args.binary, args.output, allow_dirty=args.allow_dirty)
        print(f"Staged {len(manifest['files'])} files at {args.output} from {manifest['source_commit']}")
    elif args.command == "verify":
        manifest = verify_bundle(args.bundle)
        print(f"Verified {len(manifest['files'])} files from {manifest['source_commit']}")
    else:
        print(f"Space commit: {upload_bundle(args.bundle, args.repo_id)}")


if __name__ == "__main__":
    main()
