from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.config import seed_all
from marl_arena.controllers.base import ControllerContext
from marl_arena.controllers.rl_controller import set_rl_training
from marl_arena.systems.simulation import ArenaSimulation


def test_simulation_dispatches_live_teams_in_batches() -> None:
    seed_all(29)
    simulation = ArenaSimulation(seed=29, domain_randomization=False, load_checkpoints=False)
    calls: list[tuple[str, int]] = []
    for team_name, controller in simulation.controllers.items():
        original = controller.decide_many

        def record(agents, all_agents, context, *, _team=team_name, _original=original):
            calls.append((_team, len(agents)))
            return _original(agents, all_agents, context)

        controller.decide_many = record

    simulation.step(0.1)

    assert calls == [("Team 1", 3), ("Team 2", 3), ("Team 3", 3)]


def test_batched_decisions_and_rollouts_match_scalar_path() -> None:
    """Batching must preserve actions, seeded samples, and PPO transition inputs."""
    for rotation in range(3):
        seed_all(31 + rotation)
        simulation = ArenaSimulation(
            seed=rotation,
            domain_randomization=False,
            load_checkpoints=False,
            paradigm_rotation=rotation,
        )
        set_rl_training(simulation.controllers, False)
        snapshots = simulation.build_snapshots()
        context = ControllerContext(
            step_index=1,
            arena_size=simulation.match_variant.arena_size,
            time_delta=0.1,
            shoot_range=simulation.match_variant.shoot_range,
        )

        for controller in simulation.controllers.values():
            agents = [agent for agent in snapshots if agent.team_name == controller.team_name]
            controller.rng.seed(81)
            controller.action_rng = np.random.default_rng(81)
            scalar = {
                agent.agent_id: controller.decide(agent, snapshots, context)
                for agent in agents
            }
            controller.rng.seed(81)
            controller.action_rng = np.random.default_rng(81)
            batched = controller.decide_many(agents, snapshots, context)
            assert scalar.keys() == batched.keys()
            for agent_id in scalar:
                assert scalar[agent_id].move == batched[agent_id].move
                assert scalar[agent_id].turn == batched[agent_id].turn
                assert scalar[agent_id].jump == batched[agent_id].jump
                assert scalar[agent_id].shoot == batched[agent_id].shoot
                assert scalar[agent_id].debug_role == batched[agent_id].debug_role
                np.testing.assert_allclose(
                    scalar[agent_id].aim_direction, batched[agent_id].aim_direction, atol=1e-6
                )
                np.testing.assert_allclose(
                    scalar[agent_id].desired_target, batched[agent_id].desired_target, atol=1e-6
                )

            controller.set_training(True)
            controller.action_rng = np.random.default_rng(902 + rotation)
            controller.rng.seed(93)
            for agent in agents:
                controller.decide(agent, snapshots, context)
            scalar_rows = dict(controller.pending_steps)
            controller.pending_steps.clear()

            controller.action_rng = np.random.default_rng(902 + rotation)
            controller.rng.seed(93)
            controller.decide_many(agents, snapshots, context)
            batch_rows = dict(controller.pending_steps)
            controller.pending_steps.clear()
            assert scalar_rows.keys() == batch_rows.keys()
            for agent_id, scalar_row in scalar_rows.items():
                batch_row = batch_rows[agent_id]
                assert scalar_row.action == batch_row.action
                assert scalar_row.done == batch_row.done
                assert scalar_row.log_prob == pytest.approx(batch_row.log_prob, abs=1e-6)
                assert scalar_row.value == pytest.approx(batch_row.value, abs=1e-6)
                np.testing.assert_array_equal(scalar_row.local_obs, batch_row.local_obs)
                if scalar_row.global_obs is not None:
                    np.testing.assert_array_equal(scalar_row.global_obs, batch_row.global_obs)
                if scalar_row.agent_slot is not None:
                    np.testing.assert_array_equal(scalar_row.agent_slot, batch_row.agent_slot)
            controller.set_training(False)


def test_partial_team_decisions_match_full_team_rows() -> None:
    """Live subsets must read out exactly as their rows do inside a full-team batch.

    The inference path pads a team's live rows to a fixed batch of three and slices the
    logits, values and critic broadcast back on the host. A row's logits depend only on
    that row, so a call with a subset of the team must reproduce, bit for bit, the same
    agents' decisions and rollout rows as one call with every teammate live -- including
    when the extra rows are dead agents rather than padding.
    """
    for rotation in range(3):
        seed_all(61 + rotation)
        simulation = ArenaSimulation(
            seed=rotation,
            domain_randomization=False,
            load_checkpoints=False,
            paradigm_rotation=rotation,
        )
        snapshots = simulation.build_snapshots()
        context = ControllerContext(
            step_index=1,
            arena_size=simulation.match_variant.arena_size,
            time_delta=0.1,
            shoot_range=simulation.match_variant.shoot_range,
        )

        for controller in simulation.controllers.values():
            # CTDE-Comm is excluded by construction: its rows carry mean-pooled messages from
            # live teammates, so no per-row independence invariant exists to assert -- the
            # zero-filled dead slots in its team observation are the architecture, not padding.
            if controller.paradigm == "CTDE-Comm":
                continue
            roster = [agent for agent in snapshots if agent.team_name == controller.team_name]
            for live_count in (1, 2):
                dead = [
                    agent if index < live_count else replace(agent, alive=False)
                    for index, agent in enumerate(roster)
                ]
                world = [
                    next((row for row in dead if row.agent_id == agent.agent_id), agent)
                    for agent in snapshots
                ]
                for mode in (False, True):
                    controller.set_training(mode)
                    controller.rng.seed(71)
                    controller.action_rng = np.random.default_rng(71)
                    partial = controller.decide_many(roster[:live_count], world, context)
                    partial_rows = dict(controller.pending_steps)
                    controller.pending_steps.clear()

                    controller.rng.seed(71)
                    controller.action_rng = np.random.default_rng(71)
                    full = controller.decide_many(roster, world, context)
                    full_rows = dict(controller.pending_steps)
                    controller.pending_steps.clear()

                    assert set(partial) == {agent.agent_id for agent in roster[:live_count]}
                    for agent in roster[:live_count]:
                        assert partial[agent.agent_id].move == full[agent.agent_id].move
                        assert partial[agent.agent_id].turn == full[agent.agent_id].turn
                        assert partial[agent.agent_id].jump == full[agent.agent_id].jump
                        assert partial[agent.agent_id].shoot == full[agent.agent_id].shoot
                        np.testing.assert_array_equal(
                            partial[agent.agent_id].aim_direction,
                            full[agent.agent_id].aim_direction,
                        )
                        if mode:
                            row, reference = partial_rows[agent.agent_id], full_rows[agent.agent_id]
                            assert row.action == reference.action
                            assert row.log_prob == reference.log_prob
                            assert row.value == reference.value
                            np.testing.assert_array_equal(row.local_obs, reference.local_obs)
                            if reference.global_obs is not None:
                                np.testing.assert_array_equal(row.global_obs, reference.global_obs)
                            if reference.agent_slot is not None:
                                np.testing.assert_array_equal(row.agent_slot, reference.agent_slot)
            controller.set_training(False)
