#!/usr/bin/env python3
"""Build the self-contained GitHub Pages site with the current crate version."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
from urllib.parse import unquote, urljoin, urlsplit
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist/site"


class Page(HTMLParser):
    """Collect the static information used by crawlers and link previews."""

    def __init__(self, text: str):
        super().__init__()
        self.ids = set()
        self.links = []
        self.meta = {}
        self.canonical = []
        self.schema = []
        self.headings = 0
        self.title = ""
        self.capture = None
        self.in_head = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "head":
            self.in_head = True
        if "id" in attrs:
            if attrs["id"] in self.ids:
                raise ValueError(f"Duplicate HTML id: {attrs['id']}")
            self.ids.add(attrs["id"])
        self.links.extend(attrs[key] for key in ("href", "src") if attrs.get(key))
        if tag == "meta":
            key = attrs.get("name", attrs.get("property", ""))
            self.meta[key] = attrs.get("content", "")
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical.append(attrs.get("href"))
        if tag == "h1":
            self.headings += 1
        if tag == "title" and self.in_head:
            self.capture = "title"
        elif tag == "script" and attrs.get("type") == "application/ld+json":
            self.capture = "schema"
            self.schema.append("")

    def handle_endtag(self, tag):
        if tag == "head":
            self.in_head = False
        if tag in {"title", "script"}:
            self.capture = None

    def handle_data(self, text):
        if self.capture == "title":
            self.title += text
        elif self.capture == "schema":
            self.schema[-1] += text


def validate_site(directory: Path, homepage: str, missing_media=()) -> dict:
    """Reject broken discovery metadata and local links before deployment."""
    pages = {}
    for path in directory.rglob("*.html"):
        content = path.read_text(encoding="utf-8")
        if "{{" in content:
            raise ValueError(f"Unresolved template value in {path.name}")
        pages[path.resolve()] = Page(content)
    index = pages[(directory / "index.html").resolve()]
    if index.canonical != [homepage] or index.headings != 1 or not index.title.strip():
        raise ValueError("The homepage needs one canonical URL, one H1 and a title")
    for key in ("description", "og:title", "og:description", "og:image", "twitter:card", "twitter:image"):
        if not index.meta.get(key):
            raise ValueError(f"Missing homepage metadata: {key}")
    if index.meta.get("og:url") != homepage:
        raise ValueError("The Open Graph URL must match the canonical URL")
    if "noindex" in index.meta.get("robots", "").lower():
        raise ValueError("The homepage must be indexable")
    if not index.schema:
        raise ValueError("Missing application structured data")
    for data in index.schema:
        json.loads(data)
    base = urlsplit(homepage)
    checked = 0
    for path, page in pages.items():
        relative = path.relative_to(directory.resolve()).as_posix()
        page_url = urljoin(homepage, "" if relative == "index.html" else relative)
        links = page.links + [value for key, value in page.meta.items() if key in {"og:image", "twitter:image"}]
        for link in links:
            target = urlsplit(urljoin(page_url, link))
            if target.netloc != base.netloc or not target.path.startswith(base.path):
                continue
            local = directory / unquote(target.path[len(base.path):])
            if local.is_dir():
                local /= "index.html"
            if not local.is_file():
                if local.parent == directory / "media" and local.name in missing_media:
                    continue
                raise ValueError(f"Broken local link in {relative}: {link}")
            if target.fragment and local.resolve() in pages:
                if unquote(target.fragment) not in pages[local.resolve()].ids:
                    raise ValueError(f"Missing anchor in {relative}: {link}")
            checked += 1
    sitemap = ET.parse(directory / "sitemap.xml")
    urls = [item.text for item in sitemap.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    if homepage not in urls:
        raise ValueError("The sitemap must include the canonical homepage")
    return {"pages": len(pages), "local_links": checked, "sitemap_urls": len(urls)}


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
        validation = validate_site(staged, settings["package"]["homepage"], missing)
        manifest = {"version": version, "files": {str(path.relative_to(staged)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(staged.rglob("*")) if path.is_file()}}
        (staged / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            if not (output / "build.json").is_file():
                raise ValueError(f"Output directory was not created by this builder: {output}")
            shutil.rmtree(output)
        shutil.move(str(staged), output)
    return {"version": version, "output": str(output), "files": len(manifest["files"]), "missing_media": missing, "validation": validation}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-missing-media", action="store_true", help="Build the page while recordings are being prepared")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.allow_missing_media), indent=2))
