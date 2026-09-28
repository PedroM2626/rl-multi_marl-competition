"""Training-path tests for the three CTDE variants.

Experiment 2 previously had only two shape assertions on the new network classes, leaving the
simulator, controllers and every PPO update path at 0 % coverage. These exercise the real update
routines for CTDE-VD, CTDE-CAC and CTDE-Comm, which are the code that actually decides the result.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ["MATCH_DURATION_SECONDS"] = "3"
os.environ["DOMAIN_RANDOMIZATION"] = "true"

from marl_arena.controllers.rl_controller import RLTeamController, set_rl_training  # noqa: E402
from marl_arena.systems.simulation import ArenaSimulation  # noqa: E402

EXPECTED_PARADIGMS = {"Team 1": "CTDE-VD", "Team 2": "CTDE-CAC", "Team 3": "CTDE-Comm"}


def _flat_params(controller: RLTeamController) -> dict[str, object]:
    weights: dict[str, object] = {}
    for role, module in (("actor", controller.actor), ("critic", controller.critic)):
        if module is not None:
            weights.update({f"{role}.{k}": v.detach().clone() for k, v in module.state_dict().items()})
    return weights


def test_each_ctde_variant_has_the_expected_networks() -> None:
    simulation = ArenaSimulation(seed=5, domain_randomization=False)
    for name, controller in simulation.controllers.items():
        assert controller.paradigm == EXPECTED_PARADIGMS[name]
        assert controller.actor is not None, f"{name} built no actor"
        assert controller.critic is not None, f"{name} built no critic"
    assert simulation.controllers["Team 1"].critic.agent_indices == [0, 1, 2]
    assert simulation.controllers["Team 3"].actor.msg_dim == 4


def test_ppo_update_moves_every_variant(tmp_path: Path) -> None:
    simulation = ArenaSimulation(seed=99, domain_randomization=True)
    set_rl_training(simulation.controllers, True)

    updates = 0
    steps = 0
    while updates < 2 and steps < 3000:
        steps += 1
        if simulation.step(0.1):
            before = {name: _flat_params(c) for name, c in simulation.controllers.items()}
            simulation.finish_match()
            updates += 1
            for name, controller in simulation.controllers.items():
                assert len(controller.buffer) == 0, f"{name} buffer was not cleared"
                after = _flat_params(controller)
                changed = any(
                    not torch.equal(before[name][key], value)
                    for key, value in after.items()
                    if isinstance(value, torch.Tensor)
                )
                assert changed, f"{name} ({controller.paradigm}) did not take a gradient step"
            simulation.reset_match()

    assert updates == 2, f"no match terminated within {steps} steps"


def test_comm_rollout_stores_team_observations() -> None:
    """The Comm arm concatenates all three allies' observations, so its buffer rows are 3x wider."""
    simulation = ArenaSimulation(seed=7, domain_randomization=True)
    set_rl_training(simulation.controllers, True)
    for _ in range(40):
        if simulation.step(0.1):
            break
    comm = simulation.controllers["Team 3"]
    assert comm.buffer.steps, "Comm collected no rollouts"
    assert comm.buffer.steps[0].local_obs.shape == (24,)
    assert comm.buffer.steps[0].agent_slot is not None
    assert comm.buffer.steps[0].global_obs is not None


def test_greedy_evaluation_is_deterministic_and_collects_nothing(tmp_path: Path) -> None:
    """The held-out evaluation in scripts/run_experiment.py runs entirely through the greedy branch of
    decide(), including the Comm path that assembles a team observation - otherwise never exercised."""
    from marl_arena.controllers.base import ControllerContext

    simulation = ArenaSimulation(seed=3, domain_randomization=False)
    set_rl_training(simulation.controllers, False)
    snapshots = simulation.build_snapshots()
    context = ControllerContext(step_index=1, arena_size=simulation.match_variant.arena_size,
                                time_delta=0.1, shoot_range=simulation.match_variant.shoot_range)

    for controller in simulation.controllers.values():
        agent = next(a for a in simulation.agents if a.team_name == controller.team_name)
        first = controller.decide(agent.snapshot(), snapshots, context)
        second = controller.decide(agent.snapshot(), snapshots, context)
        assert first.move == second.move and first.turn == second.turn
        assert first.shoot == second.shoot, "greedy actions must not vary between identical calls"
        assert not controller.pending_steps, "evaluation must not queue rollout steps"

    for _ in range(20):
        if simulation.step(0.1):
            break
    for controller in simulation.controllers.values():
        assert len(controller.buffer) == 0, "evaluation must not accumulate a rollout buffer"
