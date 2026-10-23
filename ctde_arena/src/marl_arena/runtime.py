"""Runtime provenance and the campaign-runtime guard.

A campaign measures the runtime its documentation describes: the JAX build pinned in requirements.txt and
the Rust extension built from native/arena_core. The CI-verification environment (.venv-check) has held a
different JAX build without that extension, and a campaign launched through it would report numbers no
number in docs/ describes (A-45, A-48, docs/09 section 9.5). This module is the one place that decides
whether an interpreter qualifies, and the values every run summary records.
"""

from __future__ import annotations

import platform
import re
from pathlib import Path
from typing import Any


# src/marl_arena/runtime.py -> the experiment tree that owns this package.
DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def _root(root: Path | None) -> Path:
    return DEFAULT_ROOT if root is None else Path(root).resolve()


def jax_pin(root: Path | None = None) -> str:
    """The JAX version requirements.txt pins for this tree."""
    requirements = (_root(root) / "requirements.txt").read_text(encoding="utf-8")
    match = re.search(r"^jax==([^\s#]+)", requirements, re.MULTILINE)
    if match is None:
        raise RuntimeError("requirements.txt no longer pins jax")
    return match.group(1).strip()


def runtime_provenance(root: Path | None = None) -> dict[str, Any]:
    """What this interpreter actually is; recorded in every run summary and checked by the guard."""
    tree = _root(root)
    import jax
    import numpy as np

    from marl_arena.systems import simulation as simulation_module

    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "jax_pin": jax_pin(tree),
        "rust_extension": simulation_module._arena_native is not None,
    }


def require_measured_runtime(root: Path | None = None) -> dict[str, Any]:
    """Refuse to launch a campaign on an interpreter that is not the measured runtime."""
    info = runtime_provenance(root)
    if info["jax"] != info["jax_pin"]:
        raise RuntimeError(
            f"campaign runtime has jax {info['jax']}, but requirements.txt pins {info['jax_pin']}"
        )
    if not info["rust_extension"]:
        raise RuntimeError(
            "campaign runtime has no Rust extension; build native/arena_core with maturin before measuring"
        )
    return info
