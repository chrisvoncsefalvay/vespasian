"""Release payload checks, with no publishing calls."""
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from package_release import ROOT, TARGETS, verify_binary, archive_readme
import release_manifest


class ReleaseTests(unittest.TestCase):
    def test_binary_architecture_must_match_the_asset_name(self):
        for target, (kind, machine) in TARGETS.items():
            data = bytearray(128)
            if kind == "elf":
                data[:6] = b"\x7fELF\x02\x01"
                struct.pack_into("<H", data, 18, machine)
            elif kind == "macho":
                data[:4] = b"\xcf\xfa\xed\xfe"
                struct.pack_into("<I", data, 4, machine)
            else:
                data[:2] = b"MZ"
                struct.pack_into("<I", data, 60, 64)
                data[64:68] = b"PE\0\0"
                struct.pack_into("<H", data, 68, machine)
            verify_binary(data, target)
            for other in TARGETS:
                if other != target:
                    with self.assertRaises(ValueError):
                        verify_binary(data, other)

    def test_release_version_rejects_mismatched_or_injected_tags(self):
        for tag in ("main", "v0.0.0", "v0.1.5; echo wrong", "../../release"):
            with self.assertRaises(ValueError):
                release_manifest.release_version(tag)

    def test_archive_readme_points_to_versioned_remote_media(self):
        with patch.object(Path, "read_text", return_value='![Demo](docs/media/demo.gif) [Guide](docs/guide.md) [Anchor](#install) <img src="docs/social-card.png">'):
            text = archive_readme("0.1.5").decode()
        self.assertIn("https://raw.githubusercontent.com/chrisvoncsefalvay/vespasian/v0.1.5/docs/media/demo.gif", text)
        self.assertIn("https://github.com/chrisvoncsefalvay/vespasian/blob/v0.1.5/docs/guide.md", text)
        self.assertIn("[Anchor](#install)", text)
        self.assertIn('src="https://raw.githubusercontent.com/', text)

    def test_missing_platform_asset_prevents_a_partial_release(self):
        version = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                release_manifest.collect(Path(directory), f"v{version}")
            self.assertFalse((Path(directory) / "release.json").exists())

    def test_release_collection_verifies_every_archive_and_the_crate_commit(self):
        version = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target in TARGETS:
                stem = f"vespasian-{version}-{target}"
                archive = root / (stem + (".zip" if "windows" in target else ".tar.gz"))
                archive.write_bytes(b"test archive")
                (root / f"{stem}.json").write_text(json.dumps({
                    "version": version, "target": target, "commit": "a" * 40,
                    "archive": {"file": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size},
                }), encoding="utf-8")
            crate = root / f"vespasian-{version}.crate"
            for commit in ("b" * 40, "a" * 40):
                with tarfile.open(crate, "w:gz") as bundle:
                    data = json.dumps({"git": {"sha1": commit}}).encode()
                    info = tarfile.TarInfo(f"vespasian-{version}/.cargo_vcs_info.json")
                    info.size = len(data)
                    bundle.addfile(info, io.BytesIO(data))
                if commit.startswith("b"):
                    with self.assertRaisesRegex(ValueError, "same Git commit"):
                        release_manifest.collect(root, f"v{version}")
                else:
                    result = release_manifest.collect(root, f"v{version}")
                    self.assertEqual(len(result["assets"]), len(TARGETS) + 1)
                    self.assertEqual(len((root / "SHA256SUMS.txt").read_text().splitlines()), len(TARGETS) + 1)
            archive.write_bytes(b"modified archive")
            with self.assertRaisesRegex(ValueError, "checksum failed"):
                release_manifest.collect(root, f"v{version}")


if __name__ == "__main__":
    unittest.main()
