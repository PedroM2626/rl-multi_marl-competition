"""Guard against silent drift between the two experiment trees.

ctde_arena/ is a fork of the root engine rather than an import of it, so a fix applied to the shared
physics, metrics or controller code has to be applied twice. This test asserts which files are meant to
be byte-identical and which are allowed to differ, so drift in the shared ones fails the suite instead
of quietly producing two different arenas.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OTHER = PROJECT_ROOT / "ctde_arena"

# Files that must stay byte-identical across the two trees.
IDENTICAL = [
    "main.py",
    "src/marl_arena/__init__.py",
    "src/marl_arena/config.py",
    "src/marl_arena/models.py",
    "src/marl_arena/controllers/__init__.py",
    "src/marl_arena/controllers/base.py",
    "src/marl_arena/controllers/batching.py",
    "src/marl_arena/controllers/rl_controller.py",
    "src/marl_arena/rl/__init__.py",
    "src/marl_arena/rl/actions.py",
    "src/marl_arena/rl/buffer.py",
    "src/marl_arena/rl/networks.py",
    "src/marl_arena/rl/ppo.py",
    "src/marl_arena/runtime.py",
    "src/marl_arena/systems/__init__.py",
    "src/marl_arena/systems/metrics.py",
    "src/marl_arena/systems/plotting.py",
    "src/marl_arena/systems/simulation.py",
    "native/arena_core/Cargo.toml",
    "native/arena_core/Cargo.lock",
    "native/arena_core/pyproject.toml",
    "native/arena_core/src/lib.rs",
    "src/marl_arena/ui/__init__.py",
    "src/marl_arena/ui/dashboard.py",
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

# Files that legitimately differ: the paradigm cycle and the MLflow integration. The engine, the
# JAX optimizer, the Rust kernels, the renderer and the arena are shared verbatim.
EXPECTED_DIFFERENT = [
    "src/marl_arena/systems/match_variant.py",        # PARADIGM_CYCLE only
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


def test_match_variant_differs_only_in_the_paradigm_cycle() -> None:
    """The arena definition is shared; only the list of paradigms being compared may differ."""
    import difflib

    left = (PROJECT_ROOT / "src/marl_arena/systems/match_variant.py").read_text(encoding="utf-8").splitlines()
    right = (OTHER / "src/marl_arena/systems/match_variant.py").read_text(encoding="utf-8").splitlines()
    changed = [
        line
        for line in difflib.unified_diff(left, right, lineterm="", n=0)
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    assert len(changed) == 2, f"unexpected divergence in match_variant.py: {changed}"
    assert all("PARADIGM_CYCLE" in line for line in changed)
