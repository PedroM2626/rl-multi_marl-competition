"""Gates for the dependency-audit script's two pure functions.

The OSV call itself cannot be tested offline, and a network stub would only prove that the stub works. What
can fail silently, and therefore what is pinned here, is the input side and the reading side: a pin that is
dropped by the parser is never queried and still shows up as "clean", and a severity that is misread turns a
CRITICAL into a footnote. Both are checked against hand-built fixtures with known answers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_da_{name}", PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load("audit_dependencies")


def test_parse_pins_keeps_every_real_pin_and_drops_the_rest(tmp_path: Path) -> None:
    path = tmp_path / "requirements.txt"
    path.write_text(
        "# a comment\n"
        "\n"
        "numpy==1.26.4\n"
        "ImageIO==2.37.0\n"
        "  torch==2.5.1+cu121  \n"
        "-e .\n"
        "--extra-index-url https://download.pytorch.org/whl/cu121\n"
        "torch @ https://example.invalid/torch.whl\n"
        "python-dotenv==1.0.1\n",
        encoding="utf-8")
    assert audit.parse_pins(path) == [
        ("numpy", "1.26.4"), ("imageio", "2.37.0"), ("torch", "2.5.1+cu121"), ("python-dotenv", "1.0.1")]


def test_parse_pins_refuses_to_audit_a_range(tmp_path: Path) -> None:
    """An unpinned requirement has no OSV answer; silently skipping it would report it as clean."""
    path = tmp_path / "requirements.txt"
    path.write_text("numpy>=1.26\n", encoding="utf-8")
    for content in ("numpy>=1.26\n", "numpy==1.26.*\n", "# only a comment\n"):
        path.write_text(content, encoding="utf-8")
        try:
            audit.parse_pins(path)
        except SystemExit:
            continue
        raise AssertionError(f"{content!r} was auditable, and should not have been")


def test_the_measured_lock_is_fully_parsable() -> None:
    """The artefact the study was measured in has to be the script's own default input."""
    pins = audit.parse_pins(PROJECT_ROOT / "requirements-measured.lock.txt")
    assert len(pins) == 34, len(pins)
    assert ("torch", "2.5.1+cu121") in pins
    assert len({name for name, _ in pins}) == len(pins), "a duplicated package would be queried twice"


def test_severity_picks_the_highest_recorded_level() -> None:
    vuln = {"severity": [{"type": "ECOSYSTEM", "score": [{"severity": "MODERATE"}, {"severity": "CRITICAL"}]}]}
    assert audit.severity_of(vuln) == "CRITICAL"
    assert audit.severity_of({"severity": [{"type": "ECOSYSTEM", "score": [{"severity": "low"}]}]}) == "LOW"
    assert audit.severity_of({}) == "UNSPECIFIED"
    assert audit.severity_of({"database_specific": {"severity": "HIGH"}}) == "HIGH"
