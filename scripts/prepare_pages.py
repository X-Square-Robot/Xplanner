"""Stage the homepage and its linked assets for GitHub Pages publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlsplit


def stage(source: Path, repository: Path) -> None:
    html = (source / "index.html").read_text()
    pending = {"index.html", "style.css", "page.js", "data/demos.json",
               "data/results.json", "data/benchmark-cases.json"}
    pending.update(re.findall(r'(?:href|src|content)="((?:assets|media|data)/[^"?]+)', html))
    copied = set()

    def references(value):
        if isinstance(value, dict):
            for child in value.values():
                references(child)
        elif isinstance(value, list):
            for child in value:
                references(child)
        elif isinstance(value, str) and value.startswith(("assets/", "media/", "data/")):
            pending.add(urlsplit(value).path)

    while pending:
        relative = pending.pop()
        if relative in copied:
            continue
        path = source / relative
        if not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe homepage asset: {relative}")
        if path.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError(f"Asset exceeds GitHub's regular file limit: {relative}")
        target = repository / "docs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.add(relative)
        if path.suffix == ".json":
            references(json.loads(path.read_text()))

    # The repository currently publishes main:/ via the built-in Pages workflow.
    # Keep docs/index.html directly usable and provide the same page at the root.
    root_html = html.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n  <base href="./docs/">', 1)
    root_html = root_html.replace('href="#', 'href="/Xplanner/#')
    (repository / "index.html").write_text(root_html)
    (repository / ".nojekyll").touch()
    (repository / "docs/.nojekyll").touch()
    size = sum((source / item).stat().st_size for item in copied)
    print(f"Staged {len(copied)} linked homepage files ({size / 1024**2:.1f} MiB), with root and docs entry points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("docs"))
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    stage(args.source, args.repository)
