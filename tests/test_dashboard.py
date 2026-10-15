"""The in-game overlay is the only part of the UI a reader can verify without a window.

ui/dashboard.py was 0 % covered for the whole life of this repository, on the argument that it only
formats text. That is exactly why it is worth a test: the overlay is where the engine's per-agent state
becomes a claim a human reads, and it is the one surface with no assertion behind it.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.config import CONFIG
from marl_arena.systems.simulation import ArenaSimulation
from marl_arena.ui.dashboard import build_overlay_text


def _sim() -> ArenaSimulation:
    return ArenaSimulation(seed=3, domain_randomization=False, load_checkpoints=False)


def test_overlay_lists_every_agent_with_their_state() -> None:
    sim = _sim()
    text = build_overlay_text(sim)
    lines = text.splitlines()
    agent_lines = lines[lines.index("Agents:") + 1:]
    assert len(agent_lines) == len(sim.agents) == 9
    for agent, line in zip(sim.agents, agent_lines):
        fields = [part.strip() for part in line.split("|")]
        assert fields[0] == agent.agent_id
        assert fields[1] == agent.team_name
        assert fields[2] == agent.paradigm
        assert fields[3] == ("ALIVE" if agent.alive else "ELIMINATED")
        assert fields[4] == f"K:{agent.kills} H:{agent.hits} M:{agent.misses}"


def test_overlay_header_carries_the_match_clock_paradigms_and_alive_counts() -> None:
    sim = _sim()
    header = build_overlay_text(sim).split("\n\n", 1)[0]
    lines = header.splitlines()
    assert lines[0].startswith("Match 1")
    assert "Elapsed:" in lines[0]
    # One line per team naming the paradigm in its slot, which is what makes a rotated match readable.
    for spawn in sim.match_variant.team_spawns:
        row = next(line for line in lines if line.startswith(spawn.team_name))
        assert spawn.paradigm in row, "the overlay must show which paradigm is in which slot"
        assert "3 alive" in row
    assert sum("alive" in line for line in lines) == 3
    assert lines[-1].startswith("Obstacles:")
    assert f"{sum(sim.live_team_counts().values())}" in header


def test_overlay_shows_elimination_after_a_kill() -> None:
    sim = _sim()
    victim = sim.agents[0]
    victim.alive = False
    victim.kills = 2
    victim.hits = 3
    victim.misses = 4
    line = [ln for ln in build_overlay_text(sim).splitlines() if ln.startswith(victim.agent_id)][0]
    assert "ELIMINATED" in line
    assert "K:2 H:3 M:4" in line
    # The other eight are untouched, so a mutation here must not leak into their rows.
    assert sum("ALIVE" in ln for ln in build_overlay_text(sim).splitlines()) == 8


def test_overlay_tracks_the_simulation_as_it_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    sim = _sim()
    before = build_overlay_text(sim)
    for _ in range(int(6.0 / CONFIG.sim_step_dt)):
        sim.step(CONFIG.sim_step_dt)
    after = build_overlay_text(sim)
    assert "Agents:" in after
    assert f"{sim.match_time:.1f}" in after.splitlines()[0]
    assert before != after, "the overlay is static, so it cannot be showing simulation state"
