"""The vectorised evaluator: one arena is the published path exactly, and training is refused."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.config import CONFIG  # noqa: E402
from marl_arena.controllers.rl_controller import set_rl_training  # noqa: E402
from marl_arena.systems.simulation import ArenaSimulation  # noqa: E402
from marl_arena.systems.vector import VecArena  # noqa: E402


def _frozen_sim(seed: int) -> ArenaSimulation:
    sim = ArenaSimulation(seed=seed, domain_randomization=False, load_checkpoints=False)
    set_rl_training(sim.controllers, False)
    return sim


def _rows(result) -> list:
    return [sorted(row.items()) for row in result.team_rows]


def test_one_arena_reproduces_the_sequential_evaluation_exactly() -> None:
    sequential = _frozen_sim(21)
    expected = []
    for _ in range(3):
        while not sequential.step(CONFIG.sim_step_dt):
            pass
        expected.append(sequential.finish_match())
        sequential.reset_match()
    got = VecArena([_frozen_sim(21)]).play(3, CONFIG.sim_step_dt)
    assert [_rows(r) for r in got] == [_rows(r) for r in expected]


def test_several_arenas_share_one_policy_and_play_their_quota() -> None:
    vec = VecArena([_frozen_sim(21 + 3 * index) for index in range(3)])
    assert all(sim.controllers is vec.controllers for sim in vec.sims)
    results = vec.play(5, CONFIG.sim_step_dt)
    assert len(results) == 5
    assert [sim.match_index - 1 for sim in vec.sims] == [2, 2, 1]


def test_a_training_controller_is_refused() -> None:
    sim = ArenaSimulation(seed=21, domain_randomization=False, load_checkpoints=False)
    with pytest.raises(RuntimeError):
        VecArena([sim]).step(CONFIG.sim_step_dt)
