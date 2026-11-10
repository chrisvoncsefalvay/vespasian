#!/usr/bin/env python3
"""Collect platform archives and the crate into one release manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tomllib

from package_release import ROOT, TARGETS


def release_version(tag):
    version = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
    if tag != f"v{version}" or not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", tag):
        raise ValueError("Release tag must match the Cargo package version")
    return version


def collect(directory, tag):
    version = release_version(tag)
    assets, commits = [], set()
    for target in TARGETS:
        manifest = json.loads((directory / f"vespasian-{version}-{target}.json").read_text(encoding="utf-8"))
        if manifest["version"] != version or manifest["target"] != target:
            raise ValueError(f"Mismatched platform manifest for {target}")
        commits.add(manifest["commit"])
        asset = manifest["archive"]
        path = directory / asset["file"]
        extension = ".zip" if "windows" in target else ".tar.gz"
        if asset["file"] != f"vespasian-{version}-{target}{extension}" or hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"Archive checksum failed for {target}")
        assets.append({"target": target, **asset})
    if len(commits) != 1 or None in commits:
        raise ValueError("Platform archives must come from the same Git commit")
    crate = directory / f"vespasian-{version}.crate"
    with tarfile.open(crate) as bundle:
        vcs = bundle.extractfile(f"vespasian-{version}/.cargo_vcs_info.json")
        if vcs is None or json.load(vcs)["git"]["sha1"] not in commits:
            raise ValueError("The crate must come from the same Git commit as the platform archives")
    assets.append({"target": "crate", "file": crate.name, "sha256": hashlib.sha256(crate.read_bytes()).hexdigest(), "bytes": crate.stat().st_size})
    result = {"version": version, "tag": tag, "commit": commits.pop(), "assets": assets}
    (directory / "release.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (directory / "SHA256SUMS.txt").write_text("".join(f"{item['sha256']}  {item['file']}\n" for item in sorted(assets, key=lambda item: item["file"])), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "dist/release")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--check-version", action="store_true")
    args = parser.parse_args()
    print(release_version(args.tag) if args.check_version else json.dumps(collect(args.directory, args.tag), indent=2))
