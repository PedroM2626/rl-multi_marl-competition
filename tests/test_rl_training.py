from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ["RL_TRAIN_TOTAL_STEPS"] = "50"
os.environ["MATCH_DURATION_SECONDS"] = "5"
os.environ["DOMAIN_RANDOMIZATION"] = "true"

from marl_arena.config import CONFIG
from marl_arena.controllers.rl_controller import RLTeamController, save_rl_checkpoints, set_rl_training
from marl_arena.systems.match_variant import sample_training_variant
from marl_arena.systems.simulation import ArenaSimulation


def test_domain_randomization_changes_variant() -> None:
    simulation = ArenaSimulation(seed=42, domain_randomization=True)
    first = simulation.match_variant.variant_id
    simulation.reset_match()
    second = simulation.match_variant.variant_id
    assert second == first + 1
    sampled = sample_training_variant(simulation.rng, CONFIG, 99)
    assert len(sampled.obstacles) >= CONFIG.dr_obstacle_count_min
    assert len(sampled.team_spawns) == 3


def test_rl_controllers_collect_rollouts_and_update() -> None:
    simulation = ArenaSimulation(seed=99, domain_randomization=True)
    set_rl_training(simulation.controllers, True)
    target = 30
    while simulation.total_env_steps < target:
        if simulation.step(0.1):
            simulation.finish_match()
            simulation.reset_match()

    for controller in simulation.controllers.values():
        assert isinstance(controller, RLTeamController)

    save_rl_checkpoints(simulation.controllers)
    for paradigm in ("cte", "dte", "ctde"):
        matches = list(CONFIG.rl_checkpoint_dir.glob(f"*_{paradigm}.pt"))
        assert matches, f"Checkpoint {paradigm} nao foi salvo."
