from __future__ import annotations

from marl_arena.systems.simulation import ArenaSimulation


def build_overlay_text(simulation: ArenaSimulation) -> str:
    lines = [simulation.match_status_text(), "", "Agents:"]
    for agent in simulation.agents:
        status = "ALIVE" if agent.alive else "ELIMINATED"
        lines.append(
            f"{agent.agent_id} | {agent.team_name} | {agent.paradigm} | {status} | K:{agent.kills} H:{agent.hits} M:{agent.misses}"
        )
    return "\n".join(lines)
