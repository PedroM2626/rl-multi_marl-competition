"""The reward and termination knobs must do what they say, and their defaults must be the old numbers.

This repository already has a documented defect of exactly this shape — a configuration key that was read
but never used (A-16, RESPAWN_ENABLED) — so a new set of knobs ships with tests that (a) pin the defaults
to the constants they replaced, and (b) prove each override actually changes behaviour.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.config import CONFIG
from marl_arena.systems import simulation as simulation_module
from marl_arena.systems.simulation import ArenaSimulation


def _sim_with(monkeypatch, **overrides) -> ArenaSimulation:
    config = dataclasses.replace(CONFIG, **overrides)
    monkeypatch.setattr(simulation_module, "CONFIG", config)
    return ArenaSimulation(seed=5, domain_randomization=False, load_checkpoints=False)


def test_defaults_reproduce_the_constants_the_knobs_replaced() -> None:
    """If one of these moves, every committed number is describing a different arena."""
    assert CONFIG.reward_alive_bonus == pytest.approx(0.015)
    assert CONFIG.reward_alive_decay_half_life == 0.0
    assert CONFIG.reward_kill == pytest.approx(1.2)
    assert CONFIG.reward_death == pytest.approx(-0.5)
    assert CONFIG.reward_hit == pytest.approx(-1.0)
    assert CONFIG.termination_cap_awards_win is True


def test_default_reward_matches_the_previous_hard_coded_expression(monkeypatch) -> None:
    sim = _sim_with(monkeypatch)
    agent = sim.agents[0]
    assert sim._reward_for_agent(agent, was_hit=False, scored_hit=False) == pytest.approx(0.015)
    agent.alive = False
    assert sim._reward_for_agent(agent, was_hit=False, scored_hit=False) == pytest.approx(-0.5)
    agent.alive = True
    assert sim._reward_for_agent(agent, was_hit=True, scored_hit=True) == pytest.approx(0.015 + 1.2 - 1.0)


def test_alive_bonus_can_be_switched_off(monkeypatch) -> None:
    sim = _sim_with(monkeypatch, reward_alive_bonus=0.0)
    assert sim._reward_for_agent(sim.agents[0], was_hit=False, scored_hit=False) == pytest.approx(0.0)


def test_alive_bonus_decays_over_the_match(monkeypatch) -> None:
    sim = _sim_with(monkeypatch, reward_alive_decay_half_life=30.0)
    agent = sim.agents[0]
    sim.match_time = 0.0
    start = sim._reward_for_agent(agent, was_hit=False, scored_hit=False)
    sim.match_time = 30.0
    at_one_half_life = sim._reward_for_agent(agent, was_hit=False, scored_hit=False)
    sim.match_time = 60.0
    at_two_half_lives = sim._reward_for_agent(agent, was_hit=False, scored_hit=False)
    assert at_one_half_life == pytest.approx(start / 2.0)
    assert at_two_half_lives == pytest.approx(start / 4.0)


def _unbalance(sim: ArenaSimulation) -> None:
    """Leave Team 1 whole and bleed the others, so a capped match has an unambiguous survivor ranking."""
    for agent in sim.agents:
        if agent.team_name == "Team 2" and agent.agent_id.endswith("-1"):
            agent.alive = False
        if agent.team_name == "Team 3" and agent.agent_id.endswith(("-1", "-2")):
            agent.alive = False


def _run_to_cap(sim: ArenaSimulation) -> None:
    dt = sim.config.sim_step_dt
    for _ in range(int(sim.match_variant.match_duration_seconds / dt) + 1):
        _unbalance(sim)
        if sim.step(dt):
            break


def test_a_capped_match_awards_the_most_alive_team_by_default(monkeypatch) -> None:
    sim = _sim_with(monkeypatch, match_duration_seconds=0.5)
    _run_to_cap(sim)
    result = sim.finish_match()
    assert result.winner_team == "Team 1", "default behaviour must not change"
    assert sim.cumulative_metrics["Team 1"].wins == 1


def test_a_capped_match_is_a_draw_when_the_knob_is_off(monkeypatch) -> None:
    sim = _sim_with(monkeypatch, match_duration_seconds=0.5, termination_cap_awards_win=False)
    _run_to_cap(sim)
    result = sim.finish_match()
    assert result.winner_team == simulation_module.DRAW_TEAM
    assert sum(sim.live_team_counts().values()) > 3, "a draw here must be a still-contested match"
    assert all(metrics.wins == 0 for metrics in sim.cumulative_metrics.values())


def test_a_wipeout_still_awards_a_winner_with_the_knob_off(monkeypatch) -> None:
    """The knob must remove the buzzer-beater win, not wins altogether."""
    sim = _sim_with(monkeypatch, termination_cap_awards_win=False)
    for agent in sim.agents:
        if agent.team_name != "Team 1":
            agent.alive = False
    result = sim.finish_match()
    assert result.winner_team == "Team 1"
