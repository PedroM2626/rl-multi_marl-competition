"""Gate: the JAX version this repository names must be one version, everywhere.

A version string in a requirements file or a chapter is a provenance claim: it says which build produced the
numbers or which build a reader must install to get them. When the runtime moved to JAX the pin was written
from the version some *other* interpreter on the host happened to carry, and every prose mention in docs/ and
both READMEs, plus the wheel the Rust benchmark needs, disagreed with it. Nothing failed, because the tests
pass on either JAX. These checks make the disagreement fail: one version across pins, prose and the workflow,
and -- when the running interpreter can import JAX at all -- the installed build equal to the pin.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prose wraps, so whitespace is collapsed before matching. The pattern accepts a pin (`jax==0.4.38`), a
# prose mention (`JAX 0.4.38`, `` `jax`/`jaxlib` 0.4.38 ``) or a path-like separator, and only ever the three
# dotted components a real JAX release has.
JAX_VERSION = re.compile(r"jax(?:lib)?[^0-9\n]{0,8}(\d+\.\d+\.\d+)", re.IGNORECASE)

SOURCES = (
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "ctde_arena" / "requirements.txt",
    # docs/11 is deliberately absent: the audit is the log of superseded values, and an entry that corrects a
    # pin has to name the wrong one to say so. Every other file that states the environment is scanned.
    *[path for path in sorted((PROJECT_ROOT / "docs").glob("*.md")) if path.name != "11_code_audit.md"],
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "ctde_arena" / "README.md",
    PROJECT_ROOT / ".github" / "workflows" / "tests.yml",
)


def _mentions(path: Path) -> dict[str, list[int]]:
    text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))
    found: dict[str, list[int]] = {}
    for match in JAX_VERSION.finditer(text):
        found.setdefault(match.group(1), []).append(text[: match.start()].count(" "))
    return found


def test_every_named_jax_version_is_the_same_one() -> None:
    versions: dict[str, list[str]] = {}
    for path in SOURCES:
        if not path.is_file():
            continue
        for version in _mentions(path):
            versions.setdefault(version, []).append(path.relative_to(PROJECT_ROOT).as_posix())
    assert len(versions) == 1, (
        "the requirements pins, the chapters and the CI workflow must name one JAX build; today they name "
        + "; ".join(f"{version} in {files}" for version, files in sorted(versions.items()))
    )


def _pins(path: Path) -> dict[str, str]:
    # `core.autocrlf` means a Windows reader sees CRLF line endings in the same committed file that Linux
    # checks out with LF, so an anchor must allow the carriage return or the pin parses only on one host.
    text = path.read_text(encoding="utf-8")
    return dict(re.findall(r"^(jax|jaxlib)==(\d+\.\d+\.\d+)\r?$", text, re.M))


def test_a_crlf_checkout_of_the_same_pin_parses(tmp_path: Path) -> None:
    path = tmp_path / "requirements.txt"
    path.write_bytes(b"numpy==2.2.6\r\njax==0.4.38\r\njaxlib==0.4.38\r\n")
    assert _pins(path) == {"jax": "0.4.38", "jaxlib": "0.4.38"}


def test_both_trees_pin_jax_and_jaxlib_to_that_documented_version() -> None:
    documented = next(iter(_mentions(PROJECT_ROOT / "docs" / "09_reproducibility.md")))
    for relative in ("requirements.txt", "ctde_arena/requirements.txt"):
        pins = _pins(PROJECT_ROOT / relative)
        assert set(pins) == {"jax", "jaxlib"}, f"{relative} must pin both jax and jaxlib, found {sorted(pins)}"
        assert set(pins.values()) == {documented}, (
            f"{relative} pins {pins} while docs/ names {documented}: the pin and the prose describe "
            "different environments, and only one of them produced the numbers"
        )


def test_an_environment_that_runs_the_runtime_runs_the_pinned_build() -> None:
    try:
        import jax
    except ImportError:
        return  # CI for the measured lock and any interpreter without the runtime: nothing to compare.
    pin = _pins(PROJECT_ROOT / "requirements.txt")["jax"]
    assert jax.__version__ == pin, (
        f"this interpreter runs JAX {jax.__version__} while requirements.txt pins {pin}, so a measurement "
        "made here is not the measurement the pin claims"
    )
