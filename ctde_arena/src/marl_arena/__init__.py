"""Experiment 2's binding to the shared arena engine.

This tree used to carry a fork of the engine, which meant every fix had to be applied twice (audit item
A-11). It now carries only this shim: the package's import path points at the repository's single
``src/marl_arena``, and ``PROJECT_DIR`` names this tree, so ``marl_arena.config`` loads
``ctde_arena/.env`` -- which selects the CTDE-VD / CTDE-CAC / CTDE-Comm cycle -- and roots ``data/``
here. Scripts and tests keep inserting ``ctde_arena/src`` into ``sys.path`` exactly as before.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
__path__ = [str(PROJECT_DIR.parent / "src" / "marl_arena")]

__all__ = []
