from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

os.environ["RL_TRAIN_TOTAL_STEPS"] = "50"
os.environ["MATCH_DURATION_SECONDS"] = "5"
os.environ["DOMAIN_RANDOMIZATION"] = "true"

from marl_arena.config import CONFIG
from marl_arena.controllers.rl_controller import RLTeamController, set_rl_training
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


def _flat_params(controller: RLTeamController) -> dict[str, object]:
    """Actor and critic share key names, so they have to be prefixed or they overwrite each other."""
    weights: dict[str, object] = {}
    for role, module in (("actor", controller.actor), ("critic", controller.critic)):
        if module is not None:
            weights.update({f"{role}.{k}": v.detach().clone() for k, v in module.state_dict().items()})
    return weights


def test_match_termination_triggers_a_real_ppo_update(tmp_path: Path) -> None:
    """Regression for the case where the loop was too short to ever end a match, so
    finish_match() - and therefore the only PPO update - never ran."""
    simulation = ArenaSimulation(seed=99, domain_randomization=True)
    set_rl_training(simulation.controllers, True)

    updates = 0
    steps = 0
    while updates < 2 and steps < 2000:
        steps += 1
        if simulation.step(0.1):
            before = {name: _flat_params(c) for name, c in simulation.controllers.items()}
            simulation.finish_match()
            updates += 1
            for name, controller in simulation.controllers.items():
                assert len(controller.buffer) == 0, "rollout buffer was not cleared after the update"
                after = _flat_params(controller)
                changed = any(
                    not torch.equal(before[name][key], value)
                    for key, value in after.items()
                    if isinstance(value, torch.Tensor)
                )
                assert changed, f"{name} ({controller.paradigm}) parameters did not move after an update"
            simulation.reset_match()

    assert updates == 2, f"no match terminated within {steps} steps"


def test_rl_controllers_collect_rollouts_and_save(tmp_path: Path) -> None:
    simulation = ArenaSimulation(seed=99, domain_randomization=True)
    set_rl_training(simulation.controllers, True)
    target = 30
    while simulation.total_env_steps < target:
        if simulation.step(0.1):
            simulation.finish_match()
            simulation.reset_match()

    for controller in simulation.controllers.values():
        assert isinstance(controller, RLTeamController)
        assert len(controller.buffer) > 0, "no rollout rows were collected"

    # Save into tmp_path: writing to CONFIG.rl_checkpoint_dir would overwrite the versioned policies.
    for controller in simulation.controllers.values():
        controller.save(tmp_path)
    for paradigm in ("cte", "dte", "ctde"):
        matches = list(tmp_path.glob(f"*_{paradigm}.pt"))
        assert matches, f"Checkpoint {paradigm} was not saved."
