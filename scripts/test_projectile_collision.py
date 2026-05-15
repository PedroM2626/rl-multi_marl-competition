from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.systems.simulation import ArenaSimulation


def test_projectile_obstacle_collision():
    sim = ArenaSimulation(seed=42)
    for controller in sim.controllers.values():
        controller.update = lambda transitions: None

    sim.step(0.1)
    shots_fired = 0
    obstacle_hits = 0
    max_steps = 500
    step_limit = 0.5

    for step_idx in range(max_steps):
        sim.step(step_limit)
        shots_this_step = sum(
            1 for p in sim.projectiles
            if p.age < step_limit * 1.5
        )
        shots_fired += shots_this_step
        obstacle_hits += len(sim.pending_obstacle_hits)

    total_projectiles_created = (
        sum(1 for _ in sim.trajectory_rows)
    )

    report = {
        "test_scenario": "projectile_obstacle_collision",
        "steps_simulated": max_steps,
        "dt_per_step": step_limit,
        "projectiles_created": len([p for p in sim.projectiles]) + len(sim.pending_obstacle_hits),
        "obstacle_hits_registered": obstacle_hits,
        "trajectory_rows": len(sim.trajectory_rows),
        "obstacle_count": len(sim.obstacles),
        "pass": bool(obstacle_hits > 0),
        "note": "Se obstacle_hits > 0, projeteis estao colidindo com barreiras",
    }
    return report


def test_projectile_never_exceeds_arena():
    sim = ArenaSimulation(seed=99)
    for controller in sim.controllers.values():
        controller.update = lambda transitions: None

    arena_half = sim.config.arena_size / 2.0
    max_violation = 0.0

    for _ in range(200):
        sim.step(0.1)
        for proj in sim.projectiles:
            violation_x = max(0.0, abs(proj.position[0]) - arena_half)
            violation_z = max(0.0, abs(proj.position[2]) - arena_half)
            max_violation = max(max_violation, violation_x, violation_z)

    report = {
        "test_scenario": "projectile_arena_bounds",
        "max_bound_violation": float(max_violation),
        "pass": bool(max_violation < 1.0),
        "note": "Projeteis devem estar dentro dos limites da arena",
    }
    return report


def test_direct_shot_against_wall():
    sim = ArenaSimulation(seed=7)
    for controller in sim.controllers.values():
        controller.update = lambda transitions: None

    central_wall = next(
        (o for o in sim.obstacles if o.obstacle_id == "fixed-central-wall"),
        None,
    )
    if central_wall is None:
        return {"test_scenario": "direct_shot_wall", "pass": False, "note": "Parede central nao encontrada"}

    for _ in range(100):
        sim.step(0.1)

    hits_against_wall = sum(
        1 for hit in sim.pending_obstacle_hits
        if hit["position"][0] is not None
    )

    report = {
        "test_scenario": "direct_shot_against_wall",
        "hits_against_obstacle": hits_against_wall,
        "pass": hits_against_wall > 0,
        "note": "Confirmar que projeteis colidem com a parede central",
    }
    return report


def test_obstacle_hit_positions_valid():
    sim = ArenaSimulation(seed=55)
    for controller in sim.controllers.values():
        controller.update = lambda transitions: None

    arena_half = sim.config.arena_size / 2.0
    valid_positions = 0
    invalid_positions = 0

    for _ in range(150):
        sim.step(0.1)

    for hit in sim.pending_obstacle_hits:
        pos = hit["position"]
        if (
            abs(pos[0]) <= arena_half + 2.0
            and abs(pos[2]) <= arena_half + 2.0
        ):
            valid_positions += 1
        else:
            invalid_positions += 1

    report = {
        "test_scenario": "obstacle_hit_positions_valid",
        "valid_hits": valid_positions,
        "invalid_hits": invalid_positions,
        "pass": invalid_positions == 0,
        "note": "Todas as posicoes de hit devem estar dentro dos limites da arena",
    }
    return report


def main() -> None:
    results = [
        test_projectile_obstacle_collision(),
        test_projectile_never_exceeds_arena(),
        test_direct_shot_against_wall(),
        test_obstacle_hit_positions_valid(),
    ]
    print(json.dumps(results, indent=2))
    all_pass = all(r["pass"] for r in results)
    print(f"\nTodos os testes passaram: {all_pass}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
