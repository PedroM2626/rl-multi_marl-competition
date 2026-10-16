"""The re-evaluation harness must be the study's own held-out protocol, in every detail that matters.

`scripts/eval_readout.py` compares two readouts of the *same* weights, so anything that differs between it and
`run_experiment.py::evaluate` other than the action rule becomes part of the measured difference: a different
seed offset, domain randomisation left on, one stat accumulated differently, and the greedy-versus-sampled
contrast would be measuring the harness. The list below is the protocol written as assertions rather than as
prose, checked against both source bodies, together with the behaviour of the temperature switch itself.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _source(name: str) -> str:
    return (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")


def _load(name: str):
    import importlib.util

    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_rp_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Every one of these appears in run_experiment.py's held-out evaluation and must appear verbatim in the
# re-evaluation harness too. They are the measurement, not the presentation.
PROTOCOL = [
    "ArenaSimulation(seed=seed, domain_randomization=False, load_checkpoints=True)",
    "set_rl_training(sim.controllers, False)",
    "while not sim.step(dt):",
    "result = sim.finish_match()",
    "sim.reset_match()",
    'stats["wins"] += row["winner"]',
    'stats["eliminations"] += row["eliminations"]',
    'stats["survival_sum"] += row["mean_survival_time"]',
    'stats["hits"] += row["shots_hit"]',
    'stats["misses"] += row["shots_missed"]',
    '"win_rate": stats["wins"] / n',
    '"eliminations_per_match": stats["eliminations"] / n',
    '"mean_survival_time": stats["survival_sum"] / n',
    '"shots_per_match": total_shots / n',
    '"shot_accuracy": stats["hits"] / max(total_shots, 1)',
]


def test_the_readout_harness_repeats_the_study_protocol_line_for_line() -> None:
    study = _source("run_experiment.py")
    readout = _source("eval_readout.py")
    missing = [line for line in PROTOCOL if line not in readout]
    assert not missing, f"eval_readout.py no longer performs: {missing}"
    # ... and the protocol list is not stale: everything in it is still what run_experiment does.
    absent_in_study = [line for line in PROTOCOL if line not in study]
    assert not absent_in_study, f"the checklist cites something run_experiment.py no longer does: {absent_in_study}"


def test_the_seed_offset_is_the_study_s() -> None:
    """`evaluate(args.eval_matches, args.seed + 9000)` is the published held-out stream."""
    assert "args.seed + 9000" in _source("run_experiment.py")
    readout = _source("eval_readout.py")
    assert 'default=9000' in readout, "the offset must default to the study's, not to something new"
    assert "evaluate(args.matches, seed + args.seed_offset)" in readout


def test_temperature_zero_is_argmax_and_positive_samples() -> None:
    from marl_arena.config import CONFIG
    from marl_arena.controllers import rl_controller

    class Probe:
        _greedy_action = rl_controller.RLTeamController._greedy_action
        _eval_action = rl_controller.RLTeamController._eval_action

    original = CONFIG.eval_temperature
    try:
        rl_controller.CONFIG = dataclasses.replace(CONFIG, eval_temperature=0.0)
        logits = torch.tensor([[0.1, 5.0, 0.2, 0.0]])
        assert Probe()._eval_action(logits) == 1, "T=0 must be the argmax the study reported"

        rl_controller.CONFIG = dataclasses.replace(CONFIG, eval_temperature=1e6)
        # A flat distribution over four actions: with enough draws every action has to appear.
        seen = {Probe()._eval_action(logits) for _ in range(400)}
        assert seen == {0, 1, 2, 3}, seen

        rl_controller.CONFIG = dataclasses.replace(CONFIG, eval_temperature=1.0)
        torch.manual_seed(7)
        peaked = torch.tensor([[0.0, 9.0, 0.0, 0.0]])
        assert sum(Probe()._eval_action(peaked) == 1 for _ in range(200)) > 190, \
            "temperature 1 on a 9-logit margin must still overwhelmingly take the best action"
    finally:
        rl_controller.CONFIG = CONFIG
        assert CONFIG.eval_temperature == original


def test_the_default_temperature_is_the_published_readout() -> None:
    """A fresh process must read greedy, or every committed number describes a different arena."""
    from marl_arena.config import CONFIG

    assert CONFIG.eval_temperature == 0.0
    config_source = (PROJECT_ROOT / "src" / "marl_arena" / "config.py").read_text(encoding="utf-8")
    assert 'os.getenv("EVAL_TEMPERATURE", "0")' in config_source


def test_paradigms_come_from_the_checkpoint_names_when_there_is_no_run_summary() -> None:
    """The versioned sets carry no `run_summary.json`, so the file name is the only slot record.

    Checked against real committed checkpoints rather than a fixture, because the whole point is that the
    names on disk are what a clone has: seed 1 of experiment 1 rotates to `paradigm_rotation = 1`, which puts
    DTE in Team 1, and experiment 2's seed 1 rotates to 1 as well, where Team 1 is CTDE-CAC.
    """
    module = _load("eval_readout")
    root = PROJECT_ROOT.parent if PROJECT_ROOT.name == "ctde_arena" else PROJECT_ROOT
    replicates = root / "data" / "training_replicates"
    if not (replicates / "exp1_seed1").is_dir():
        pytest.skip("the versioned replicates are not in this checkout")
    for label, expected in (
            ("exp1_seed1", {"Team 1": "DTE", "Team 2": "CTDE", "Team 3": "CTE"}),
            ("exp1_seed9", {"Team 1": "CTE", "Team 2": "DTE", "Team 3": "CTDE"}),
            ("exp2_seed1", {"Team 1": "CTDE-CAC", "Team 2": "CTDE-Comm", "Team 3": "CTDE-VD"})):
        got = module.paradigms_from_files(label.split("_")[0], sorted((replicates / label).glob("*.pt")))
        assert got == expected, (label, got)
    assert module.paradigms_from_files("exp1", []) == {}
    with pytest.raises(SystemExit):
        module.paradigms_from_files("exp1", [Path("team_1_bogus.pt")])
    with pytest.raises(SystemExit):
        module.paradigms_from_files("exp1", [Path("policy_cte.pt")])


def test_a_replicate_is_evaluated_by_the_tree_that_trained_it(tmp_path: Path) -> None:
    """Loading experiment 2's weights through experiment 1's controller can succeed and be wrong."""
    other = "exp2" if PROJECT_ROOT.name != "ctde_arena" else "exp1"
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "eval_readout.py"), "--temperature", "1",
         "--experiments", other, "--seeds", "1", "--matches", "1", "--out", str(tmp_path / "x")],
        capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert result.returncode != 0, result.stdout[-400:]
    assert "must be evaluated from" in result.stdout + result.stderr, result.stdout + result.stderr


def test_the_readout_files_are_identical_in_both_trees() -> None:
    """This tree and the other one run the same harness; the copy must not drift."""
    other_root = PROJECT_ROOT.parent if PROJECT_ROOT.name == "ctde_arena" else PROJECT_ROOT / "ctde_arena"
    here = PROJECT_ROOT / "scripts" / "eval_readout.py"
    there = other_root / "scripts" / "eval_readout.py"
    assert here.is_file() and there.is_file(), f"{there} is missing"
    assert here.read_bytes() == there.read_bytes(), f"{here} and {there} differ; they are parity files"
