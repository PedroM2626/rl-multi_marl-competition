from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.systems.simulation import ArenaSimulation, SimAgent


def _disable_learning(simulation: ArenaSimulation) -> None:
    for controller in simulation.controllers.values():
        controller.update = lambda transitions: None


def _agent(simulation: ArenaSimulation, agent_id: str) -> SimAgent:
    for candidate in simulation.agents:
        if candidate.agent_id == agent_id:
            return candidate
    raise RuntimeError(f"Agente {agent_id} nao encontrado.")


def _reset_agents(simulation: ArenaSimulation, alive_ids: set[str]) -> None:
    for agent in simulation.agents:
        agent.alive = agent.agent_id in alive_ids
        agent.vertical_velocity = 0.0
        agent.kills = 0
        agent.hits = 0
        agent.misses = 0
        agent.last_shot_at = -999.0
        agent.survival_time = 0.0


def _fire_once(
    simulation: ArenaSimulation,
    shooter_id: str,
    shooter_position: tuple[float, float, float],
    aim_direction: tuple[float, float, float],
    target_id: str | None = None,
    target_position: tuple[float, float, float] | None = None,
    use_default_obstacles: bool = True,
) -> dict[str, object]:
    _disable_learning(simulation)
    alive_ids = {shooter_id}
    if target_id is not None:
        alive_ids.add(target_id)
    _reset_agents(simulation, alive_ids)
    if not use_default_obstacles:
        simulation.obstacles = []
    simulation.projectiles.clear()
    simulation.pending_obstacle_hits.clear()
    simulation._projectile_id_counter = 0

    shooter = _agent(simulation, shooter_id)
    shooter.position = np.array(shooter_position, dtype=float)
    shooter.heading_deg = 0.0

    if target_id is not None and target_position is not None:
        target = _agent(simulation, target_id)
        target.position = np.array(target_position, dtype=float)
        target.heading_deg = 180.0

    projectile = simulation._spawn_projectile(shooter, np.array(aim_direction, dtype=float))
    if projectile is None:
        raise RuntimeError("Falha ao criar projetil no teste.")

    hit_status = {agent.agent_id: False for agent in simulation.agents}
    scored_status = {agent.agent_id: False for agent in simulation.agents}
    simulation._advance_projectiles(0.25, hit_status, scored_status)

    return {
        "simulation": simulation,
        "shooter": shooter,
        "target": _agent(simulation, target_id) if target_id is not None else None,
        "hit_status": hit_status,
        "scored_status": scored_status,
    }


def test_projectile_hits_central_wall_before_target() -> dict[str, object]:
    simulation = ArenaSimulation(seed=42)
    result = _fire_once(
        simulation=simulation,
        shooter_id="1-1",
        shooter_position=(0.0, 1.0, -12.0),
        aim_direction=(0.0, 0.0, 1.0),
        target_id="2-1",
        target_position=(0.0, 1.0, 2.0),
        use_default_obstacles=True,
    )
    shooter = result["shooter"]
    target = result["target"]
    collision_pos = simulation.pending_obstacle_hits[0]["position"] if simulation.pending_obstacle_hits else np.zeros(3, dtype=float)
    passed = (
        len(simulation.pending_obstacle_hits) == 1
        and len(simulation.projectiles) == 0
        and bool(target is not None and target.alive)
        and shooter.misses == 1
        and shooter.hits == 0
        and abs(float(collision_pos[2]) - (-6.12)) < 0.3
    )
    return {
        "test_scenario": "projectile_hits_central_wall_before_target",
        "collision_position": [round(float(value), 3) for value in collision_pos],
        "target_alive_after_wall_block": bool(target is not None and target.alive),
        "shooter_hits": shooter.hits,
        "shooter_misses": shooter.misses,
        "pass": passed,
    }


def test_projectile_hits_enemy_without_false_block() -> dict[str, object]:
    simulation = ArenaSimulation(seed=99)
    result = _fire_once(
        simulation=simulation,
        shooter_id="1-1",
        shooter_position=(-12.0, 1.0, -12.0),
        aim_direction=(0.0, 0.0, 1.0),
        target_id="2-1",
        target_position=(-12.0, 1.0, -4.0),
        use_default_obstacles=False,
    )
    shooter = result["shooter"]
    target = result["target"]
    hit_status = result["hit_status"]
    scored_status = result["scored_status"]
    passed = (
        len(simulation.pending_obstacle_hits) == 0
        and len(simulation.projectiles) == 0
        and bool(target is not None and not target.alive)
        and shooter.kills == 1
        and shooter.hits == 1
        and shooter.misses == 0
        and hit_status.get("2-1") is True
        and scored_status.get("1-1") is True
    )
    return {
        "test_scenario": "projectile_hits_enemy_without_false_block",
        "target_alive": bool(target is not None and target.alive),
        "shooter_kills": shooter.kills,
        "shooter_hits": shooter.hits,
        "shooter_misses": shooter.misses,
        "pass": passed,
    }


def test_projectile_stops_at_arena_boundary() -> dict[str, object]:
    simulation = ArenaSimulation(seed=7)
    result = _fire_once(
        simulation=simulation,
        shooter_id="1-1",
        shooter_position=(15.0, 1.0, 0.0),
        aim_direction=(1.0, 0.0, 0.0),
        target_id=None,
        target_position=None,
        use_default_obstacles=False,
    )
    shooter = result["shooter"]
    arena_half = simulation.config.arena_size * 0.5
    collision_pos = simulation.pending_obstacle_hits[0]["position"] if simulation.pending_obstacle_hits else np.zeros(3, dtype=float)
    passed = (
        len(simulation.pending_obstacle_hits) == 1
        and len(simulation.projectiles) == 0
        and shooter.misses == 1
        and abs(float(collision_pos[0]) - arena_half) < 0.05
        and abs(float(collision_pos[0])) <= arena_half + 1e-6
    )
    return {
        "test_scenario": "projectile_stops_at_arena_boundary",
        "arena_half": arena_half,
        "collision_position": [round(float(value), 3) for value in collision_pos],
        "shooter_misses": shooter.misses,
        "pass": passed,
    }


def test_pending_collision_events_are_step_local() -> dict[str, object]:
    simulation = ArenaSimulation(seed=123)
    _disable_learning(simulation)
    _reset_agents(simulation, set())
    simulation.pending_obstacle_hits.append(
        {
            "position": np.array([0.0, 1.0, 0.0], dtype=float),
            "team_name": "Equipe 1",
            "team_color": (1.0, 0.0, 0.0),
        }
    )
    simulation.step(0.0)
    passed = len(simulation.pending_obstacle_hits) == 0
    return {
        "test_scenario": "pending_collision_events_are_step_local",
        "pending_after_step": len(simulation.pending_obstacle_hits),
        "pass": passed,
    }


def test_headless_match_keeps_projectiles_inside_arena() -> dict[str, object]:
    simulation = ArenaSimulation(seed=202)
    _disable_learning(simulation)
    arena_half = simulation.config.arena_size * 0.5
    max_violation = 0.0
    total_obstacle_events = 0
    total_eliminations = 0

    for _ in range(180):
        simulation.step(0.1)
        total_obstacle_events += len(simulation.pending_obstacle_hits)
        total_eliminations = sum(agent.kills for agent in simulation.agents)
        for projectile in simulation.projectiles:
            max_violation = max(
                max_violation,
                max(0.0, abs(float(projectile.position[0])) - arena_half),
                max(0.0, abs(float(projectile.position[2])) - arena_half),
            )

    passed = max_violation <= 1e-6 and (total_obstacle_events > 0 or total_eliminations > 0)
    return {
        "test_scenario": "headless_match_keeps_projectiles_inside_arena",
        "max_bound_violation": round(float(max_violation), 6),
        "obstacle_events": total_obstacle_events,
        "eliminations": total_eliminations,
        "pass": passed,
    }


def main() -> None:
    results = [
        test_projectile_hits_central_wall_before_target(),
        test_projectile_hits_enemy_without_false_block(),
        test_projectile_stops_at_arena_boundary(),
        test_pending_collision_events_are_step_local(),
        test_headless_match_keeps_projectiles_inside_arena(),
    ]
    print(json.dumps(results, indent=2))
    all_pass = all(result["pass"] for result in results)
    print(f"\nTodos os testes passaram: {all_pass}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
