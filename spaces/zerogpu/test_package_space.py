"""Bundle and deployment contract tests; no model loads or remote writes."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import package_space as package


class BundleFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "checkout"
        self.root.mkdir()
        self.status = ""
        self.commit = "a" * 40
        self.tree = "b" * 40
        self.write("Cargo.toml", '[package]\nname = "vespasian"\nversion = "1.2.3"\n')
        self.write("Cargo.lock", "# fixture lockfile\n")
        self.write("src/main.rs", 'fn main() { println!("fixture"); }\n')
        self.write("LICENSE", "Test licence\n")
        for name in package.SPACE_FILES:
            self.write(f"spaces/zerogpu/{name}", f"fixture for {name}\n")
        (self.root / "spaces/zerogpu/python_bridge.py").chmod(0o755)
        for name in package.CORE_FILES:
            self.write(f"python/{name}", f"# production {name}\n")
        self.write("spaces/zerogpu/assets/style.css", "body { color: white; }\n")
        self.write("examples/sample.jsonl", '{"prompt": "Hello", "target": "Hi"}\n')
        # A synthetic ELF header exercises target validation; it is never run.
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        header[18:20] = (62).to_bytes(2, "little")
        self.binary = self.root / "controller"
        self.binary.write_bytes(header)
        self.output = Path(self.temporary.name) / "bundle"
        self.git = patch.object(package, "git", side_effect=self.fake_git)
        self.git.start()
        self.addCleanup(self.git.stop)

    def fake_git(self, root, *args):
        self.assertEqual(root, self.root)
        if args[0] == "status":
            return self.status
        return self.tree if args[-1] == "HEAD^{tree}" else self.commit

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def build(self, **kwargs):
        return package.build_bundle(self.root, self.binary, self.output, **kwargs)


class BundleTests(BundleFixture, unittest.TestCase):
    def test_stages_production_files_and_excludes_helpers_and_caches(self):
        for relative in (
            "spaces/zerogpu/test_runtime.py",
            "spaces/zerogpu/package_space.py",
            "spaces/zerogpu/DEPLOYMENT.md",
            "spaces/zerogpu/deploy.py",
            "spaces/zerogpu/assets/__pycache__/cached.pyc",
            "spaces/zerogpu/assets/.env",
            "python/test_worker.py",
            "examples/__pycache__/cached.pyc",
            "examples/.private",
        ):
            self.write(relative, "excluded\n")
        manifest = self.build()
        expected = (
            set(package.SPACE_FILES)
            | {f"core/{name}" for name in package.CORE_FILES}
            | {"assets/style.css", "examples/sample.jsonl", "LICENSE", "bin/vespasian"}
        )
        self.assertEqual(set(manifest["files"]), expected)
        self.assertEqual(package.verify_bundle(self.output), manifest)
        for name in package.CORE_FILES:
            self.assertEqual((self.output / "core" / name).read_bytes(), (self.root / "python" / name).read_bytes())
        self.assertTrue((self.output / "python_bridge.py").stat().st_mode & stat.S_IXUSR)
        self.assertTrue((self.output / "bin/vespasian").stat().st_mode & stat.S_IXUSR)

    def test_bundle_contains_complete_starter_data_and_provenance(self):
        source = Path(__file__).resolve().parents[2] / "examples"
        manifest = json.loads((source / "starter-manifest.json").read_text(encoding="utf-8"))
        names = ["starter-manifest.json", "STARTERS.md"] + [item["file"] for item in manifest["starters"]]
        for name in names:
            self.write("examples/" + name, (source / name).read_text(encoding="utf-8"))
        bundle = self.build()
        for name in names:
            self.assertIn("examples/" + name, bundle["files"])
            self.assertEqual((self.output / "examples" / name).read_bytes(), (self.root / "examples" / name).read_bytes())

    def test_records_commit_version_and_content_hashes_deterministically(self):
        manifest = self.build()
        self.assertEqual(manifest["source_commit"], self.commit)
        self.assertEqual(manifest["source_tree"], self.tree)
        self.assertEqual(manifest["version"], "1.2.3")
        self.assertFalse(manifest["source_dirty"])
        for name, details in manifest["files"].items():
            data = (self.output / name).read_bytes()
            self.assertEqual(details["sha256"], sha256(data).hexdigest())
            self.assertEqual(details["size"], len(data))
        for name, expected_hash in manifest["source_sha256"].items():
            self.assertEqual(expected_hash, sha256((self.root / name).read_bytes()).hexdigest())
        second = self.output.with_name("second")
        package.build_bundle(self.root, self.binary, second)
        self.assertEqual((self.output / package.PROVENANCE).read_bytes(), (second / package.PROVENANCE).read_bytes())

    def test_optional_system_packages_are_included(self):
        self.write("spaces/zerogpu/packages.txt", "git\n")
        self.assertIn("packages.txt", self.build()["files"])

    def test_rejects_dirty_source_unless_explicitly_requested(self):
        self.status = " M python/worker.py"
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertTrue(self.build(allow_dirty=True)["source_dirty"])
        with self.assertRaisesRegex(ValueError, "committed source"):
            package.upload_bundle(self.output, "test/space")

    def test_rejects_arm64_and_non_elf_controllers(self):
        for value in (b"#!/bin/sh\n", b"not an executable" + bytes(64)):
            self.binary.write_bytes(value)
            with self.assertRaisesRegex(ValueError, "Linux x86-64"):
                self.build()
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        header[18:20] = (183).to_bytes(2, "little")
        self.binary.write_bytes(header)
        with self.assertRaisesRegex(ValueError, "Linux x86-64"):
            self.build()

    def test_keeps_existing_output_untouched(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing work")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build()
        self.assertEqual(marker.read_text(), "existing work")

    def test_rejects_missing_required_runtime(self):
        (self.root / "spaces/zerogpu/gpu.py").unlink()
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_rejects_symlinked_assets(self):
        (self.root / "spaces/zerogpu/assets/leak").symlink_to(self.root / "Cargo.toml")
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            self.build()

    def test_detects_modified_and_unexpected_bundle_files(self):
        self.build()
        app = self.output / "app.py"
        original = app.read_bytes()
        app.write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            package.verify_bundle(self.output)
        app.write_bytes(original)
        (self.output / "unexpected.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            package.verify_bundle(self.output)

    def test_rejects_unsafe_manifest_paths(self):
        for name in ("../secret", "/etc/passwd", "assets/../secret", ".git/config", "a\\b", "a//b", "./app.py", package.PROVENANCE):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Invalid bundle path"):
                package.manifest_files({"schema_version": 1, "files": {name: {}}})


class UploadTests(BundleFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.manifest = self.build()
        self.api = Mock()
        self.api.space_info.return_value = SimpleNamespace(sdk="gradio", sha="c" * 40)
        self.api.get_space_runtime.return_value = SimpleNamespace(hardware="zero-a10g", requested_hardware="zero-a10g")
        self.api.create_commit.return_value = SimpleNamespace(oid="d" * 40)
        self.client = Mock(return_value=self.api)
        self.download = Mock()
        self.previous = Path(self.temporary.name) / "previous.json"
        self.previous.write_text(json.dumps(self.manifest))
        self.download.return_value = str(self.previous)

        class Missing(Exception):
            pass

        self.Missing = Missing
        self.tools = patch.object(
            package,
            "hub_tools",
            return_value=(
                self.client,
                lambda **kwargs: SimpleNamespace(operation="add", **kwargs),
                lambda **kwargs: SimpleNamespace(operation="delete", **kwargs),
                self.download,
                Missing,
            ),
        )
        self.tools.start()
        self.addCleanup(self.tools.stop)
        self.token = patch.dict(os.environ, {"HF_TOKEN": "test-token"})
        self.token.start()
        self.addCleanup(self.token.stop)

    def test_upload_is_one_revision_checked_space_commit(self):
        self.assertEqual(package.upload_bundle(self.output, "test/space"), "d" * 40)
        kwargs = self.api.create_commit.call_args.kwargs
        self.assertEqual(kwargs["repo_type"], "space")
        self.assertEqual(kwargs["repo_id"], "test/space")
        self.assertEqual(kwargs["revision"], "main")
        self.assertEqual(kwargs["parent_commit"], "c" * 40)
        self.assertEqual({op.path_in_repo for op in kwargs["operations"]}, set(self.manifest["files"]) | {package.PROVENANCE})
        self.api.create_commit.assert_called_once()

    def test_deletes_only_previous_generated_paths_missing_from_new_bundle(self):
        previous = dict(self.manifest)
        previous["files"] = dict(previous["files"], **{"assets/obsolete.css": {}})
        self.previous.write_text(json.dumps(previous))
        package.upload_bundle(self.output, "test/space")
        operations = self.api.create_commit.call_args.kwargs["operations"]
        self.assertEqual([op.path_in_repo for op in operations if op.operation == "delete"], ["assets/obsolete.css"])

    def test_first_upload_never_deletes_unrelated_remote_files(self):
        self.download.side_effect = self.Missing
        package.upload_bundle(self.output, "test/space")
        operations = self.api.create_commit.call_args.kwargs["operations"]
        self.assertTrue(all(op.operation == "add" for op in operations))

    def test_rejects_other_hardware_or_sdk_without_writes(self):
        self.api.get_space_runtime.return_value.requested_hardware = "a10g-large"
        with self.assertRaisesRegex(ValueError, "ZeroGPU"):
            package.upload_bundle(self.output, "test/space")
        self.api.get_space_runtime.return_value.requested_hardware = "zero-a10g"
        self.api.space_info.return_value.sdk = "docker"
        with self.assertRaisesRegex(ValueError, "Gradio"):
            package.upload_bundle(self.output, "test/space")
        self.api.create_commit.assert_not_called()

    def test_requires_token_without_writes(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            package.upload_bundle(self.output, "test/space")
        self.api.create_commit.assert_not_called()


class CargoPackagingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cargo"), "cargo is required to inspect the actual crate file list")
    @unittest.skipIf(os.environ.get("VESPASIAN_SKIP_CARGO_PACKAGE_TEST") == "1", "Cargo is in use by another local check")
    def test_space_assets_are_excluded_from_the_crate(self):
        result = subprocess.run(
            ["cargo", "package", "--list", "--allow-dirty", "--locked", "--offline"],
            cwd=package.ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        names = result.stdout.splitlines()
        self.assertIn("python/worker.py", names)
        self.assertIn("src/main.rs", names)
        self.assertFalse([name for name in names if name.startswith(("spaces/", ".github/"))])


if __name__ == "__main__":
    unittest.main()
