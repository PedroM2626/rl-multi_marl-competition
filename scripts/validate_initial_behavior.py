from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.systems.simulation import AGENT_RADIUS, ArenaSimulation


def _inside_any_obstacle(simulation: ArenaSimulation, position: np.ndarray) -> bool:
    for obstacle in simulation.obstacles:
        half_size = obstacle.size * 0.5
        if (
            abs(position[0] - obstacle.position[0]) <= half_size[0] + AGENT_RADIUS
            and abs(position[2] - obstacle.position[2]) <= half_size[2] + AGENT_RADIUS
        ):
            return True
    return False


def main() -> None:
    first_step_simulation = ArenaSimulation(seed=19)
    for controller in first_step_simulation.controllers.values():
        controller.update = lambda transitions: None
    first_positions = {agent.agent_id: agent.position.copy() for agent in first_step_simulation.agents}
    first_step_simulation.step(0.1)
    first_step_displacement = {
        agent.agent_id: float(np.linalg.norm(agent.position - first_positions[agent.agent_id]))
        for agent in first_step_simulation.agents
    }

    simulation = ArenaSimulation(seed=19)
    for controller in simulation.controllers.values():
        controller.update = lambda transitions: None

    initial_positions = {agent.agent_id: agent.position.copy() for agent in simulation.agents}
    initial_obstacles = {obstacle.obstacle_id: obstacle.position.copy() for obstacle in simulation.obstacles}

    collision_failures = 0
    for _ in range(30):
        simulation.step(0.1)
        for agent in simulation.agents:
            if agent.alive and _inside_any_obstacle(simulation, agent.position):
                collision_failures += 1

    displacement_by_agent = {
        agent.agent_id: float(np.linalg.norm(agent.position - initial_positions[agent.agent_id]))
        for agent in simulation.agents
    }
    moved_agents = {agent_id: distance for agent_id, distance in displacement_by_agent.items() if distance > 0.25}
    moved_obstacles = {
        obstacle.obstacle_id: float(np.linalg.norm(obstacle.position - initial_obstacles[obstacle.obstacle_id]))
        for obstacle in simulation.obstacles
        if float(np.linalg.norm(obstacle.position - initial_obstacles[obstacle.obstacle_id])) > 0.25
    }

    report = {
        "first_step_without_training": {
            "moved_agent_count": sum(1 for distance in first_step_displacement.values() if distance > 0.01),
            "total_agents": len(first_step_displacement),
            "displacement_by_agent": first_step_displacement,
        },
        "movement_without_training": {
            "moved_agent_count": len(moved_agents),
            "total_agents": len(simulation.agents),
            "displacement_by_agent": displacement_by_agent,
        },
        "obstacle_motion": {
            "moved_obstacle_count": len(moved_obstacles),
            "total_obstacles": len(simulation.obstacles),
            "displacement_by_obstacle": moved_obstacles,
        },
        "collision_validation": {
            "invalid_agent_obstacle_overlaps": collision_failures,
        },
        "interpretation": {
            "agents_should_be_idle_without_training": False,
            "reason": "As politicas atuais possuem heuristicas iniciais de perseguicao, flanqueamento, agrupamento e evasao; portanto os agentes ja demonstram atividade antes de qualquer ajuste online.",
        },
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
