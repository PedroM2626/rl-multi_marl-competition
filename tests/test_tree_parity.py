"""Guard the shape of the two experiment trees.

ctde_arena/ used to be a fork of the root engine, so a fix to the shared physics, metrics or controller
code had to be applied twice (A-11). The engine now exists once, in src/marl_arena; ctde_arena/src holds
only a shim that points its import path there. What remains duplicated is the per-tree tooling -- entry
points and tests that each tree runs against its own .env -- and this test asserts that it stays
byte-identical, so drift fails the suite instead of quietly producing two different instruments.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OTHER = PROJECT_ROOT / "ctde_arena"

# Files that must stay byte-identical across the two trees.
IDENTICAL = [
    "main.py",
    "scripts/plot_metrics.py",
    "scripts/random_baseline.py",
    "scripts/render_trailer.py",
    "scripts/run_experiment.py",
    "scripts/eval_readout.py",
    "scripts/eval_regimes.py",
    "scripts/convert_torch_checkpoints.py",
    "scripts/benchmark_runtime.py",
    "tests/test_collision.py",
    "tests/test_render.py",
    "tests/test_arena_config.py",
    "tests/test_readout_parity.py",
    "tests/test_policy_batching.py",
    "tests/test_native_parity.py",
    "tests/test_dashboard.py",
    "tests/test_checkpoint_atomicity.py",
    "tests/test_runtime_provenance.py",
]

# Files that legitimately differ: the MLflow integration of experiment 2's training entry point.
EXPECTED_DIFFERENT = [
    "scripts/train_rl.py",                            # MLflow integration
]


def test_shared_files_have_not_drifted() -> None:
    drifted = [
        rel
        for rel in IDENTICAL
        if (PROJECT_ROOT / rel).read_bytes() != (OTHER / rel).read_bytes()
    ]
    assert not drifted, f"shared files diverged between the trees, fix by copying: {drifted}"


def test_declared_divergences_are_real() -> None:
    stale = [
        rel
        for rel in EXPECTED_DIFFERENT
        if (PROJECT_ROOT / rel).read_bytes() == (OTHER / rel).read_bytes()
    ]
    assert not stale, f"these files are now identical and can move to IDENTICAL: {stale}"


def test_experiment_two_carries_no_engine_of_its_own() -> None:
    """The fork is gone: ctde_arena/src is one shim file, and there is one Rust crate."""
    files = sorted(p.relative_to(OTHER).as_posix() for p in (OTHER / "src").rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts)
    assert files == ["src/marl_arena/__init__.py"], files
    assert not (OTHER / "native").exists()


def test_each_tree_binds_the_shared_engine_to_its_own_configuration() -> None:
    """Importing through either tree loads the root engine files and that tree's .env."""
    probe = (
        "import marl_arena, marl_arena.config as c, marl_arena.systems.match_variant as m;"
        "print(marl_arena.PROJECT_DIR.name, c.__file__, ','.join(m.PARADIGM_CYCLE))"
    )
    seen = {}
    for tree in (PROJECT_ROOT, OTHER):
        env = {key: value for key, value in os.environ.items() if key not in {"PARADIGM_CYCLE", "ARENA_DATA_DIR"}}
        out = subprocess.run(
            [sys.executable, "-c", f"import sys; sys.path.insert(0, r'{tree / 'src'}'); {probe}"],
            cwd=tree, env=env, capture_output=True, text=True, check=True,
        ).stdout.split()
        assert Path(out[1]).resolve() == (PROJECT_ROOT / "src/marl_arena/config.py").resolve()
        seen[out[0]] = out[2]
    assert seen == {PROJECT_ROOT.name: "CTE,DTE,CTDE", "ctde_arena": "CTDE-VD,CTDE-CAC,CTDE-Comm"}
