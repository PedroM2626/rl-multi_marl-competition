"""Checkpoint writes are atomic: a crash must not truncate a versioned policy file.

A campaign saves over the committed archive every RL_SAVE_EVERY_STEPS. Before this gate, save_checkpoint
opened the destination directly, so an interrupted write left a partial .npz where a loadable policy used
to be -- and the file is the evidence. The write now goes to a sibling temporary and is renamed into place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.rl.actions import load_checkpoint, save_checkpoint
from marl_arena.rl.networks import ActorNetwork


def _payload() -> dict:
    actor = ActorNetwork(9, 8, hidden_dim=16, seed=5)
    return {"paradigm": "DTE", "actor": actor.state_dict()}


def test_save_checkpoint_leaves_no_temporary_behind(tmp_path: Path) -> None:
    path = tmp_path / "policy.npz"
    save_checkpoint(path, _payload())

    assert path.is_file()
    assert load_checkpoint(path)["paradigm"] == "DTE"
    assert not [item for item in tmp_path.iterdir() if item.suffix == ".tmp"], "a completed save left scratch"


def test_a_failed_save_preserves_the_previous_checkpoint(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "policy.npz"
    save_checkpoint(path, _payload())
    before = path.read_bytes()

    def explode(handle, **arrays) -> None:
        handle.write(b"partial archive")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(np, "savez_compressed", explode)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        save_checkpoint(path, _payload())

    assert path.read_bytes() == before, "the interrupted save replaced the evidence"
    assert not [item for item in tmp_path.iterdir() if item.suffix == ".tmp"], "the failed save left scratch"
