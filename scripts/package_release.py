#!/usr/bin/env python3
"""Package the native executable with examples and documentation."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tarfile
import time
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "x86_64-unknown-linux-gnu": ("elf", 62),
    "aarch64-unknown-linux-gnu": ("elf", 183),
    "aarch64-apple-darwin": ("macho", 0x0100000C),
    "x86_64-pc-windows-msvc": ("pe", 0x8664),
    "aarch64-pc-windows-msvc": ("pe", 0xAA64),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_binary(data, target):
    kind, machine = None, None
    if data[:4] == b"\x7fELF" and len(data) >= 20:
        kind = "elf"
        machine = struct.unpack_from(("<" if data[5] == 1 else ">") + "H", data, 18)[0]
        if data[4] != 2:
            raise ValueError("Executable must be 64-bit")
    elif data[:4] == b"\xcf\xfa\xed\xfe" and len(data) >= 8:
        kind, machine = "macho", struct.unpack_from("<I", data, 4)[0]
    elif data[:2] == b"MZ" and len(data) >= 64:
        offset = struct.unpack_from("<I", data, 60)[0]
        if len(data) >= offset + 6 and data[offset:offset + 4] == b"PE\0\0":
            kind, machine = "pe", struct.unpack_from("<H", data, offset + 4)[0]
    if (kind, machine) != TARGETS[target]:
        raise ValueError(f"Executable architecture does not match {target}")


def archive_readme(version):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    raw = f"https://raw.githubusercontent.com/chrisvoncsefalvay/vespasian/v{version}/"
    blob = f"https://github.com/chrisvoncsefalvay/vespasian/blob/v{version}/"
    def link(match):
        image, label, path = match.groups()
        if path.startswith(("http:", "https:", "#", "mailto:")):
            return match[0]
        return f"{image}[{label}]({raw if image else blob}{path})"
    text = re.sub(r"(!?)\[([^\]]*)\]\(([^)]+)\)", link, text)
    return re.sub(r'(src=["\'])(docs/[^"\']+)', lambda m: m[1] + raw + m[2], text).encode()


def package(binary, target, output):
    version = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
    binary = binary.resolve(strict=True)
    data = binary.read_bytes()
    verify_binary(data, target)
    if subprocess.check_output([str(binary), "--version"], text=True).strip() != f"vespasian {version}":
        raise ValueError("Executable version differs from Cargo.toml")
    subprocess.run([str(binary), "--help"], check=True, stdout=subprocess.DEVNULL)
    prefix, stem = f"vespasian-{version}", f"vespasian-{version}-{target}"
    executable = "vespasian.exe" if "windows" in target else "vespasian"
    payload = {executable: data, "README.md": archive_readme(version), "LICENSE": (ROOT / "LICENSE").read_bytes()}
    for name in ("CHANGELOG.md", "CONTRIBUTING.md", "docs/RELEASING.md"):
        if (ROOT / name).is_file():
            payload[name] = (ROOT / name).read_bytes()
    for path in sorted((ROOT / "examples").glob("*.jsonl")):
        payload[f"examples/{path.name}"] = path.read_bytes()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (stem + (".zip" if "windows" in target else ".tar.gz"))
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    with archive.open("xb") as destination:
        if "windows" in target:
            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for name, data in sorted(payload.items()):
                    item = zipfile.ZipInfo(f"{prefix}/{name}", time.gmtime(max(epoch, 315532800))[:6])
                    item.compress_type = zipfile.ZIP_DEFLATED
                    item.external_attr = (0o755 if name == executable else 0o644) << 16
                    bundle.writestr(item, data)
        else:
            with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as bundle:
                    for name, data in sorted(payload.items()):
                        item = tarfile.TarInfo(f"{prefix}/{name}")
                        item.size, item.mtime = len(data), epoch
                        item.mode = 0o755 if name == executable else 0o644
                        bundle.addfile(item, io.BytesIO(data))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    manifest = {"version": version, "target": target, "commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "archive": {"file": archive.name, "sha256": sha256(archive.read_bytes()), "bytes": archive.stat().st_size},
                "files": [{"path": f"{prefix}/{name}", "sha256": sha256(data), "bytes": len(data)} for name, data in sorted(payload.items())]}
    (output / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / f"{archive.name}.sha256").write_text(f"{manifest['archive']['sha256']}  {archive.name}\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/release")
    args = parser.parse_args()
    binary = args.binary or ROOT / "target" / args.target / "release" / ("vespasian.exe" if "windows" in args.target else "vespasian")
    print(json.dumps(package(binary, args.target, args.output), indent=2))
