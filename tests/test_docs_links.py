"""Every internal link in the documentation set must resolve.

docs/ cross-references each other by section anchor (~19k words, dozens of links), and an anchor is a
rendering convention rather than something the tooling checks, so a renamed heading silently produces a
link that goes nowhere. This test resolves file targets and anchors the way GitHub's slugger does, plus
the explicit <a name="..."> anchors the longer headings use because their em dashes make the generated id
hard to predict.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
EXPLICIT_ANCHOR = re.compile(r'<a name="([^"]+)"></a>')


def github_slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped, spaces become hyphens.

    Only the ASCII subset matters here; alphanumerics are kept, everything that is not a letter, digit,
    space or hyphen is deleted, and deleting an em dash leaves its two surrounding spaces as "--".
    """
    text = heading.strip().lower()
    kept = [c for c in text if c.isalnum() or c in " -_"]
    return "".join(kept).replace(" ", "-")


def documentation_files() -> list[Path]:
    files = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "ctde_arena" / "README.md"]
    files += sorted((PROJECT_ROOT / "docs").glob("*.md"))
    return [f for f in files if f.is_file()]


def anchors_of(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    found = {github_slug(m.group(2)) for m in HEADING.finditer(text)}
    found |= set(EXPLICIT_ANCHOR.findall(text))
    return found


def test_documentation_set_is_found() -> None:
    files = documentation_files()
    assert len(files) >= 14, f"expected the indexed docs plus both READMEs, found {len(files)}"


def test_internal_file_links_resolve() -> None:
    broken = []
    for path in documentation_files():
        for target in LINK.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            relative = target.split("#", 1)[0]
            if not relative:
                continue
            resolved = (path.parent / relative).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(PROJECT_ROOT)}: {target}")
    assert not broken, "links point at files that do not exist:\n" + "\n".join(broken)


def test_internal_anchor_links_resolve() -> None:
    cache: dict[Path, set[str]] = {}
    broken = []
    for path in documentation_files():
        for target in LINK.findall(path.read_text(encoding="utf-8")):
            if "#" not in target or target.startswith(("http://", "https://", "mailto:")):
                continue
            relative, fragment = target.split("#", 1)
            resolved = (path.parent / relative).resolve() if relative else path.resolve()
            if not resolved.exists():
                continue  # reported by test_internal_file_links_resolve
            if resolved not in cache:
                cache[resolved] = anchors_of(resolved)
            if fragment not in cache[resolved]:
                broken.append(f"{path.relative_to(PROJECT_ROOT)}: {target}")
    assert not broken, "links point at headings that do not exist:\n" + "\n".join(broken)


def test_explicit_anchors_are_unique_per_file() -> None:
    duplicates = []
    for path in documentation_files():
        names = EXPLICIT_ANCHOR.findall(path.read_text(encoding="utf-8"))
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            duplicates.append(f"{path.relative_to(PROJECT_ROOT)}: {dupes}")
    assert not duplicates, "an <a name> used twice makes one of them unreachable:\n" + "\n".join(duplicates)
