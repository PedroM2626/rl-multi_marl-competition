"""The reward and termination knobs must do what they say, and their defaults must be the old numbers.

This repository already has a documented defect of exactly this shape — a configuration key that was read
but never used (A-16, RESPAWN_ENABLED) — so a new set of knobs ships with tests that (a) pin the defaults
to the constants they replaced, and (b) prove each override actually changes behaviour. The last three
tests extend that to the configuration *files*: every variable the code reads has to be written down in
`.env`, at the value the measured arena used, or the file is not a description of the instrument.
"""

from __future__ import annotations

import dataclasses
import re
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


def test_simulation_defers_to_config_when_the_argument_is_omitted(monkeypatch) -> None:
    """The no-DR ablation reaches training only because the runner omits the argument; pin that fallback."""
    monkeypatch.setattr(simulation_module, "CONFIG",
                        dataclasses.replace(CONFIG, domain_randomization=False))
    assert ArenaSimulation(seed=5, load_checkpoints=False).domain_randomization is False
    # ... and an explicit argument still wins, which is what the held-out evaluation relies on.
    assert ArenaSimulation(seed=5, domain_randomization=True, load_checkpoints=False).domain_randomization is True


def test_the_training_runners_do_not_hard_code_domain_randomization() -> None:
    """A knob the runner overrides is not a knob (A-16's shape, one layer up).

    Both training entry points used to pass `domain_randomization=True` literally, so an ablation arm that
    set DOMAIN_RANDOMIZATION=false trained the control twice and reported the difference as an effect.
    """
    for relative in ("scripts/run_experiment.py", "scripts/train_rl.py"):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        hits = [line.strip() for line in source.splitlines()
                if "ArenaSimulation(" in line and "load_checkpoints" in line]
        assert hits, f"{relative} no longer constructs the training simulation where this test can see it"
        for line in hits:
            assert "domain_randomization=True" not in line, f"{relative}: {line}"


def test_the_study_runner_sets_domain_randomization_only_if_absent() -> None:
    """`os.environ[...] = ...` would clobber whatever the campaign script exported."""
    source = (PROJECT_ROOT / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
    assert 'os.environ.setdefault("DOMAIN_RANDOMIZATION"' in source
    assert 'os.environ["DOMAIN_RANDOMIZATION"] =' not in source


def _pinned_env(path: Path) -> dict[str, str]:
    pinned = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            pinned[name.strip()] = value.strip()
    return pinned


def _variables_config_reads() -> tuple[set[str], dict[str, str]]:
    """(every name read, name -> literal default) straight out of this tree's config.py."""
    source = (PROJECT_ROOT / "src" / "marl_arena" / "config.py").read_text(encoding="utf-8")
    names = set(re.findall(r'os\.getenv\("([A-Z_0-9]+)"', source)) | \
        set(re.findall(r'_read_bool\("([A-Z_0-9]+)"', source))
    defaults = dict(re.findall(r'os\.getenv\("([A-Z_0-9]+)", "([^"]*)"\)', source))
    defaults.update({m.group(1): m.group(2).lower()
                     for m in re.finditer(r'_read_bool\("([A-Z_0-9]+)", (True|False)\)', source)})
    return names, defaults


# A variable deliberately absent from the committed files: it re-roots the whole data tree, the study
# harness sets it to a fresh directory per replicate to guarantee random initialisation (A-4), and pinning
# it in a file that every run shares would put all of them in one place.
NOT_PINNED = {"ARENA_DATA_DIR"}

# The only two committed values that differ from the code default, both documented in README.md and
# docs/06_optimisation_procedure.md#67-hyperparameters-as-trained.
DOCUMENTED_OVERRIDES = {"RANDOM_SEED", "RL_TRAIN_TOTAL_STEPS"}


def test_every_variable_the_config_reads_is_written_down() -> None:
    """The `.env` files are the instrument's spec, so the two lists must agree in both directions.

    A key the code reads but the file omits can only be audited by opening the source, which is how the
    six reward and termination knobs ended up undocumented while the study was re-measuring them. A key the
    file sets and nothing reads is A-16 (RESPAWN_ENABLED) coming back.
    """
    read, _ = _variables_config_reads()
    for name in (".env", ".env.example"):
        pinned = set(_pinned_env(PROJECT_ROOT / name))
        assert read - pinned == NOT_PINNED, f"{name} omits {sorted(read - pinned - NOT_PINNED)}"
        assert not pinned - read, f"{name} sets variables no code reads: {sorted(pinned - read)}"


def test_the_pinned_values_are_the_arena_that_was_measured() -> None:
    """Copying `.env.example` must reproduce the committed arena rather than the code's fallback budget."""
    _, defaults = _variables_config_reads()
    pinned = _pinned_env(PROJECT_ROOT / ".env")
    for name, value in pinned.items():
        if name in DOCUMENTED_OVERRIDES:
            continue
        assert name in defaults, name
        if value.lower() == defaults[name].lower():
            continue
        try:
            equal = float(value) == float(defaults[name])
        except ValueError:
            equal = False
        assert equal, f"{name}: .env says {value!r}, the code default is {defaults[name]!r}"


def test_dotenv_and_dotenv_example_are_the_same_file() -> None:
    """docs/09 tells the reader to `cp .env.example .env`; both files are tracked, so they cannot differ."""
    assert (PROJECT_ROOT / ".env").read_bytes() == (PROJECT_ROOT / ".env.example").read_bytes()
