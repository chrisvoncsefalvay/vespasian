#!/usr/bin/env python3
"""Build the self-contained GitHub Pages site with the current crate version."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist/site"


def build(output: Path, allow_missing_media: bool = False) -> dict:
    output = output.resolve()
    source = ROOT / "site"
    if output == ROOT or output == source or source in output.parents or output in source.parents:
        raise ValueError("Choose a separate output directory")
    settings = tomllib.loads((ROOT / "Cargo.toml").read_text())
    version = settings["package"]["version"]
    media = ROOT / "docs/media"
    required = [f"{name}.{extension}" for name in ("setup", "will-climb") for extension in ("cast", "gif")]
    required += [f"{name}-poster.png" for name in ("setup", "will-climb")]
    missing = [name for name in required if not (media / name).is_file()]
    if missing and not allow_missing_media:
        raise ValueError("Record the terminal sessions first: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vespasian-site-", dir=output.parent) as temporary:
        staged = Path(temporary) / "site"
        shutil.copytree(source, staged)
        for path in staged.rglob("*"):
            if path.is_file() and path.suffix in {".html", ".js", ".css", ".json", ".xml", ".txt"}:
                content = path.read_text()
                if "{{VERSION}}" in content:
                    path.write_text(content.replace("{{VERSION}}", version))
        if media.exists():
            destination = staged / "media"
            destination.mkdir()
            for name in required:
                if (media / name).is_file():
                    shutil.copy2(media / name, destination / name)
        card = ROOT / "docs/social-card.png"
        if not card.is_file():
            raise ValueError("Generate docs/social-card.png before building the site")
        shutil.copy2(card, staged / "assets/social-card.png")
        (staged / ".nojekyll").touch()
        manifest = {"version": version, "files": {str(path.relative_to(staged)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(staged.rglob("*")) if path.is_file()}}
        (staged / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            if not (output / "build.json").is_file():
                raise ValueError(f"Output directory was not created by this builder: {output}")
            shutil.rmtree(output)
        shutil.move(str(staged), output)
    return {"version": version, "output": str(output), "files": len(manifest["files"]), "missing_media": missing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-missing-media", action="store_true", help="Build the page while recordings are being prepared")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.allow_missing_media), indent=2))
