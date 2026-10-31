"""The docs/10 remedies: each one does what its threat entry says, and its default changes nothing.

Every remedy is a configuration key whose default is the measured instrument (A-34's rule). The default
half is pinned by the byte-identity probe recorded in docs/11 and by the rest of this suite running
unchanged; these tests pin the other half -- that switching a remedy on changes exactly the thing it names.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.config import CONFIG  # noqa: E402
from marl_arena.controllers import rl_controller as controller_module  # noqa: E402
from marl_arena.controllers.base import obstacle_rays  # noqa: E402
from marl_arena.rl.buffer import RolloutBuffer, RolloutStep  # noqa: E402
from marl_arena.rl.networks import (  # noqa: E402
    CommActorNetwork,
    LocalValueDecompositionCriticNetwork,
    _mlp,
)
from marl_arena.systems import match_variant as variant_module  # noqa: E402
from marl_arena.systems import simulation as simulation_module  # noqa: E402
from marl_arena.systems.simulation import ArenaSimulation  # noqa: E402


def _configure(monkeypatch, **overrides):
    config = dataclasses.replace(CONFIG, **overrides)
    for module in (simulation_module, controller_module, variant_module):
        monkeypatch.setattr(module, "CONFIG", config)
    controller_module.matched_hidden_dim.cache_clear()
    return config


def _step(agent: str, reward: float, value: float, done: bool = False) -> RolloutStep:
    return RolloutStep(np.zeros(9), None, None, 0, 0.0, value, reward, done, agent_id=agent)


# --- GAE per agent ---------------------------------------------------------------------------------------

def test_per_agent_gae_follows_each_agent_not_the_interleaved_rows() -> None:
    """Two agents interleaved A, B, A, B: each A row must chain to the next A row, never to a B row."""
    gamma, lam = 0.9, 0.8
    buffer = RolloutBuffer()
    for step in (_step("a", 1.0, 0.5), _step("b", -1.0, 2.0), _step("a", 0.0, 0.25, done=True),
                 _step("b", 3.0, 1.0)):
        buffer.append(step)
    advantages, returns = buffer.compute_returns_per_agent(gamma, lam)

    def solo(rewards, values, dones, bootstrap):
        out, last = [0.0] * len(rewards), 0.0
        nexts = values[1:] + [bootstrap]
        for i in reversed(range(len(rewards))):
            mask = 1.0 - dones[i]
            delta = rewards[i] + gamma * nexts[i] * mask - values[i]
            last = delta + gamma * lam * mask * last
            out[i] = last
        return out

    a = solo([1.0, 0.0], [0.5, 0.25], [0, 1], bootstrap=0.25)
    b = solo([-1.0, 3.0], [2.0, 1.0], [0, 0], bootstrap=1.0)   # cut off by the match end: own last value
    np.testing.assert_allclose(advantages, [a[0], b[0], a[1], b[1]], rtol=1e-6)
    np.testing.assert_allclose(returns, advantages + np.array([0.5, 2.0, 0.25, 1.0]), rtol=1e-6)
    interleaved, _ = buffer.compute_returns(gamma, lam, last_value=1.0)
    assert not np.allclose(interleaved, advantages), "the remedy must change the interleaved recursion"


def test_rollout_rows_record_which_agent_produced_them() -> None:
    sim = ArenaSimulation(seed=3, domain_randomization=False, load_checkpoints=False)
    for _ in range(3):
        sim.step(CONFIG.sim_step_dt)
    for controller in sim.controllers.values():
        team = {agent.agent_id for agent in sim.agents if agent.team_name == controller.team_name}
        assert {step.agent_id for step in controller.buffer.steps} == team


# --- update volume and the KL guard ----------------------------------------------------------------------

def _trained_minibatches(monkeypatch, **overrides) -> dict[str, int]:
    _configure(monkeypatch, **overrides)
    sim = ArenaSimulation(seed=11, domain_randomization=False, load_checkpoints=False)
    while not sim.step(CONFIG.sim_step_dt):
        pass
    sim.finish_match()
    return {team: stats.minibatches for team, stats in sim.last_ppo_stats.items()}


def test_fixed_update_volume_is_the_same_for_every_team(monkeypatch) -> None:
    counts = _trained_minibatches(monkeypatch, rl_update_minibatches=5)
    assert set(counts.values()) == {5}, counts


def test_default_update_volume_grows_with_the_buffer(monkeypatch) -> None:
    """The confound being removed: under the default, gradient steps follow how many rows a team made."""
    sim_counts = _trained_minibatches(monkeypatch)
    assert all(count >= CONFIG.rl_ppo_epochs for count in sim_counts.values())


def test_the_kl_guard_stops_an_update_early(monkeypatch) -> None:
    counts = _trained_minibatches(monkeypatch, rl_update_minibatches=40, rl_target_kl=1e-9)
    assert all(count < 40 for count in counts.values()), counts


# --- matched capacity ------------------------------------------------------------------------------------

def test_matched_widths_bring_every_arm_to_the_same_parameter_count(monkeypatch) -> None:
    _configure(monkeypatch)
    counts = {p: controller_module.parameter_count(p, CONFIG.rl_hidden_dim) for p in controller_module.ALL_PARADIGMS}
    assert max(counts.values()) / min(counts.values()) > 2.0, "the unmatched spread docs/10 describes"
    target = max(counts.values())
    matched = {
        p: controller_module.parameter_count(p, controller_module.matched_hidden_dim(p, -1))
        for p in controller_module.ALL_PARADIGMS
    }
    assert max(matched.values()) / min(matched.values()) < 1.03, matched
    assert all(abs(count - target) / target < 0.03 for count in matched.values()), matched


# --- communication ---------------------------------------------------------------------------------------

def _team_obs(dead: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(0)
    rows = []
    for index in range(3):
        row = rng.normal(size=9)
        row[2], row[3] = math.sin(index), math.cos(index)
        rows.append(np.zeros(9) if index == dead else row)
    return np.concatenate(rows)[None, :].astype(np.float32)


def test_one_round_without_mask_is_the_measured_comm_network() -> None:
    legacy = CommActorNetwork(9, 8, hidden_dim=16, seed=4)
    explicit = CommActorNetwork(9, 8, hidden_dim=16, seed=4, rounds=1, mask_dead=False)
    assert "round_nets" not in legacy.params
    slot = np.eye(3, dtype=np.float32)[:1]
    np.testing.assert_array_equal(np.asarray(legacy(_team_obs(), slot)), np.asarray(explicit(_team_obs(), slot)))


def test_masking_removes_a_dead_allys_message() -> None:
    network = CommActorNetwork(9, 8, hidden_dim=16, seed=4, mask_dead=True)
    team = _team_obs(dead=2)
    observations = np.split(team[0], 3)
    params = network.params
    # Agent 0's context is ally 1's message alone, not the mean with msg_net(0).
    context = np.asarray(_mlp(params["msg_net"], jnp.asarray(observations[1][None, :])))
    expected = _mlp(params["policy_head"], jnp.asarray(np.concatenate((observations[0][None, :], context), axis=1)))
    slot = np.eye(3, dtype=np.float32)[:1]
    np.testing.assert_allclose(np.asarray(network(team, slot)), np.asarray(expected), rtol=1e-5, atol=1e-6)
    unmasked = CommActorNetwork(9, 8, hidden_dim=16, seed=4)
    assert not np.allclose(np.asarray(unmasked(team, slot)), np.asarray(expected), atol=1e-6)


def test_extra_rounds_add_parameters_and_change_the_policy_and_round_trip() -> None:
    one = CommActorNetwork(9, 8, hidden_dim=16, seed=4)
    two = CommActorNetwork(9, 8, hidden_dim=16, seed=4, rounds=2)
    slot = np.eye(3, dtype=np.float32)[:1]
    assert len(two.params["round_nets"]) == 1
    # The rounds share the first round's weights exactly, so only the extra round explains a difference.
    for name in ("msg_net", "policy_head"):
        for left, right in zip(one.params[name], two.params[name]):
            np.testing.assert_array_equal(np.asarray(left["kernel"]), np.asarray(right["kernel"]))
    assert not np.allclose(np.asarray(one(_team_obs(), slot)), np.asarray(two(_team_obs(), slot)))
    restored = CommActorNetwork(9, 8, hidden_dim=16, seed=99, rounds=2)
    restored.load_state_dict(two.state_dict())
    np.testing.assert_array_equal(np.asarray(restored(_team_obs(), slot)), np.asarray(two(_team_obs(), slot)))


# --- value decomposition over local observations -----------------------------------------------------------

def test_local_vd_critic_is_the_sum_of_per_agent_values_over_live_agents() -> None:
    critic = LocalValueDecompositionCriticNetwork(9, hidden_dim=16, seed=2)
    team = _team_obs(dead=1)
    rows = np.split(team[0], 3)
    expected = sum(float(np.asarray(_mlp(critic.params["agent_critic"], jnp.asarray(rows[i][None, :])))[0, 0])
                   for i in (0, 2))
    assert float(np.asarray(critic(team))[0]) == pytest.approx(expected, rel=1e-5)


def test_the_vd_arm_records_team_local_observations_for_its_critic(monkeypatch) -> None:
    _configure(monkeypatch, vd_local_critic=True, paradigm_cycle=("CTDE-VD", "CTDE-CAC", "CTDE-Comm"))
    monkeypatch.setattr(variant_module, "PARADIGM_CYCLE", ("CTDE-VD", "CTDE-CAC", "CTDE-Comm"))
    sim = ArenaSimulation(seed=3, domain_randomization=False, load_checkpoints=False, paradigm_rotation=0)
    sim.step(CONFIG.sim_step_dt)
    vd = sim.controllers["Team 1"]
    assert vd.critic.kind == "vd_local_critic"
    row = vd.buffer.steps[0]
    assert row.global_obs.shape == (27,)
    np.testing.assert_array_equal(row.global_obs[:9], vd.buffer.steps[0].local_obs)


# --- obstacle rays ---------------------------------------------------------------------------------------

def test_rays_measure_the_distance_to_a_box_and_to_the_walls() -> None:
    # Agent at the origin facing +z; a box 2 wide centred 5 ahead -> its near face is 4 away.
    rays = obstacle_rays(np.array([[0.0, 0.0]]), np.array([0.0]), np.array([[0.0, 5.0, 1.0, 1.0]]), 16.0, 4)
    np.testing.assert_allclose(rays[0], [4 / 16, 1.0, 1.0, 1.0])     # +z box, +x wall, -z wall, -x wall
    rays = obstacle_rays(np.array([[8.0, 0.0]]), np.array([90.0]), np.zeros((0, 4)), 16.0, 2)
    np.testing.assert_allclose(rays[0], [8 / 16, 1.0])               # facing +x: wall 8 away; behind: 24 -> 1


def test_rays_widen_both_observations_and_reach_the_policy(monkeypatch) -> None:
    _configure(monkeypatch, obs_obstacle_rays=6)
    sim = ArenaSimulation(seed=3, domain_randomization=False, load_checkpoints=False)
    sim.step(CONFIG.sim_step_dt)
    for controller in sim.controllers.values():
        row = controller.buffer.steps[0]
        if controller.paradigm == "CTE":
            assert row.global_obs.shape == (45 + 9 * 6,)
        else:
            assert row.local_obs.shape == (9 + 6,)
            assert np.all((row.local_obs[9:] >= 0) & (row.local_obs[9:] <= 1))


# --- arena geometry ---------------------------------------------------------------------------------------

def test_the_symmetric_layout_is_three_fold(monkeypatch) -> None:
    config = _configure(monkeypatch, arena_layout="symmetric")
    variant = variant_module.create_default_variant(config)
    radii = [float(np.hypot(s.spawn_center[0], s.spawn_center[2])) for s in variant.team_spawns]
    assert max(radii) - min(radii) < 1e-9
    centres = {(round(o.base_position[0], 6), round(o.base_position[2], 6)) for o in variant.obstacles}
    for obstacle in variant.obstacles:
        x, z = obstacle.base_position[0], obstacle.base_position[2]
        for turn in (120.0, 240.0):
            c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
            assert (round(c * x - s * z, 6), round(s * x + c * z, 6)) in centres
        assert obstacle.size[0] == obstacle.size[2], "every pillar is square"


def test_unknown_layouts_are_refused(monkeypatch) -> None:
    config = _configure(monkeypatch, arena_layout="hexagonal")
    with pytest.raises(ValueError):
        variant_module.create_default_variant(config)


def test_spawn_cycling_puts_every_team_at_every_position(monkeypatch) -> None:
    _configure(monkeypatch, spawn_cycle_per_match=True)
    sim = ArenaSimulation(seed=3, domain_randomization=False, load_checkpoints=False)
    seen: dict[str, set] = {}
    for _ in range(3):
        for spawn in sim.match_variant.team_spawns:
            seen.setdefault(spawn.team_name, set()).add(tuple(np.round(spawn.spawn_center, 6)))
        sim.reset_match()
    assert all(len(positions) == 3 for positions in seen.values()), seen


# --- opponent pool -----------------------------------------------------------------------------------------

def test_a_frozen_team_plays_its_snapshot_records_nothing_and_is_restored(monkeypatch) -> None:
    _configure(monkeypatch, rl_opponent_pool_prob=1.0, rl_opponent_pool_every=1)
    sim = ArenaSimulation(seed=5, domain_randomization=False, load_checkpoints=False)
    assert sim.frozen_team is None, "an empty pool cannot freeze anyone"
    while not sim.step(CONFIG.sim_step_dt):
        pass
    sim.finish_match()                     # snapshot taken here
    current = {team: c.actor.params for team, c in sim.controllers.items()}
    sim.reset_match()
    frozen = sim.frozen_team
    assert frozen is not None and sim.controllers[frozen].frozen
    for _ in range(5):
        sim.step(CONFIG.sim_step_dt)
    assert not sim.controllers[frozen].buffer.steps
    assert all(sim.controllers[t].buffer.steps for t in sim.controllers if t != frozen)
    while not sim.step(CONFIG.sim_step_dt):
        pass
    sim.finish_match()
    assert frozen not in sim.last_ppo_stats
    assert not sim.controllers[frozen].frozen
    assert sim.controllers[frozen].actor.params is current[frozen]


def test_the_pool_is_off_by_default() -> None:
    assert CONFIG.rl_opponent_pool_prob == 0.0
    assert ArenaSimulation(seed=5, domain_randomization=False, load_checkpoints=False).opponent_pool is None


# --- scripted opponent ---------------------------------------------------------------------------------------

def test_the_scripted_controller_engages_and_fires_in_range() -> None:
    from marl_arena.controllers.base import ControllerContext
    from marl_arena.controllers.scripted import ScriptedTeamController
    from marl_arena.models import AgentSnapshot

    def agent(agent_id, team, x):
        return AgentSnapshot(agent_id, team, "X", np.array([x, 1.0, 0.0]), 0.0, True, 0, 0, 0, 0.0)

    me, near, far = agent("1-1", "Team 1", 0.0), agent("2-1", "Team 2", 5.0), agent("3-1", "Team 3", 30.0)
    context = ControllerContext(1, 32.0, 0.1, 20.0)
    decision = ScriptedTeamController("Team 1", 0).decide(me, [me, near, far], context)
    assert decision.shoot and np.allclose(decision.aim_direction, [1.0, 0.0, 0.0])
    alone = ScriptedTeamController("Team 1", 0).decide(me, [me, far], context)
    assert not alone.shoot


# --- the coupled-arm statistics -----------------------------------------------------------------------------

def _report():
    import importlib.util

    spec = importlib.util.spec_from_file_location("report_study", PROJECT_ROOT / "scripts" / "report_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_friedman_matches_its_textbook_value() -> None:
    report = _report()
    # Three arms, four blocks, arm 0 always best and arm 2 always worst: rank sums 4, 8, 12.
    blocks = [[0.9, 0.5, 0.1], [0.8, 0.4, 0.2], [0.7, 0.6, 0.3], [0.6, 0.5, 0.4]]
    statistic, p, ranks = report.friedman(blocks)
    assert statistic == pytest.approx(8.0)
    assert p == pytest.approx(math.exp(-4.0), rel=1e-9)   # chi-square sf with 2 df is exp(-x / 2)
    assert ranks == [1.0, 2.0, 3.0]


def test_the_sign_flip_test_enumerates_every_assignment() -> None:
    report = _report()
    assert report.sign_flip_p([1.0, 1.0, 1.0]) == pytest.approx(2 / 8)
    differences = [0.3, -0.1, 0.2, 0.05]
    brute = sum(1 for signs in itertools.product((1, -1), repeat=4)
                if abs(sum(s * d for s, d in zip(signs, differences))) >= abs(sum(differences)) - 1e-12) / 16
    assert report.sign_flip_p(differences) == pytest.approx(brute)
