"""The campaign-runtime guard and the provenance every run summary now records.

The repository's own history is the reason this gate exists: .venv-check held jax 0.6.2 without the Rust
extension while the measured runtime was jax 0.4.38 with it, and the older campaign scripts preferred the
former. A run artifact has to say which interpreter produced it, and a campaign has to refuse the wrong one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena import runtime


def test_jax_pin_is_the_one_requirements_declares() -> None:
    text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    match = re.search(r"^jax==([^\s#]+)", text, re.MULTILINE)
    assert match is not None, "requirements.txt no longer pins jax"
    assert runtime.jax_pin(PROJECT_ROOT) == match.group(1)


def test_provenance_records_the_interpreter() -> None:
    info = runtime.runtime_provenance(PROJECT_ROOT)
    assert set(info) == {"python", "jax", "numpy", "jax_pin", "rust_extension"}
    assert info["jax"] and info["jax_pin"], "the provenance must name both the running JAX and its pin"
    assert isinstance(info["rust_extension"], bool)
    assert info["python"].count(".") >= 1 and info["numpy"]


def test_require_rejects_a_different_jax(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime,
        "runtime_provenance",
        lambda root=None: {"jax": "0.0.0", "jax_pin": "0.4.38", "rust_extension": True},
    )
    with pytest.raises(RuntimeError, match="jax"):
        runtime.require_measured_runtime(PROJECT_ROOT)


def test_require_rejects_a_missing_rust_extension(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime,
        "runtime_provenance",
        lambda root=None: {"jax": "0.4.38", "jax_pin": "0.4.38", "rust_extension": False},
    )
    with pytest.raises(RuntimeError, match="Rust"):
        runtime.require_measured_runtime(PROJECT_ROOT)
