"""Hand-checked geometry for the swept collision solver.

The engine's central correctness claim is that a projectile cannot pass through a target between two
simulation steps (docs/03_arena_system_model.md). Nothing else in the suite tests that, and every metric in
the study is downstream of it, so these cases assert exact entry parameters and hit positions computed on
paper rather than "it collides somehow".
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.systems.simulation import (
    AGENT_HALF_HEIGHT,
    AGENT_RADIUS,
    PROJECTILE_RADIUS,
    ProjectileState,
    SimObstacle,
    ArenaSimulation,
)


@pytest.fixture()
def sim() -> ArenaSimulation:
    return ArenaSimulation(seed=1, domain_randomization=False, load_checkpoints=False)


def box(center: tuple[float, float, float], size: tuple[float, float, float]) -> SimObstacle:
    return SimObstacle(
        obstacle_id="t",
        obstacle_type="fixed_barrier",
        base_position=np.array(center, dtype=float),
        size=np.array(size, dtype=float),
        color_rgb=(0.5, 0.5, 0.5),
    )


def point_in_box(point: np.ndarray, obstacle: SimObstacle, pad: float = 0.0) -> bool:
    """The naive per-step test the swept solver is supposed to replace."""
    half = obstacle.size * 0.5 + pad
    return bool(np.all(np.abs(point - obstacle.position) <= half))


# --- segment vs AABB, exact values ---------------------------------------------------------------


def test_head_on_shot_enters_at_the_facing_face(sim: ArenaSimulation) -> None:
    obstacle = box((0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
    start, end = np.array([-5.0, 1.0, 0.0]), np.array([5.0, 1.0, 0.0])
    hit, t, position = sim._segment_intersects_aabb(start, end, *sim._obstacle_bounds(obstacle))
    assert hit is True
    assert t == pytest.approx(0.4, abs=1e-12)          # travels 4 of the 10 units to reach x = -1
    np.testing.assert_allclose(position, [-1.0, 1.0, 0.0], atol=1e-12)


def test_shot_along_a_diagonal_hits_the_nearer_corner_plane(sim: ArenaSimulation) -> None:
    obstacle = box((0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
    start, end = np.array([-4.0, 1.0, -4.0]), np.array([4.0, 1.0, 4.0])
    hit, t, position = sim._segment_intersects_aabb(start, end, *sim._obstacle_bounds(obstacle))
    assert hit is True
    # x and z both go from -4 to 4, so the x = -1 plane and z = -1 plane are reached simultaneously.
    assert t == pytest.approx(3.0 / 8.0, abs=1e-12)
    np.testing.assert_allclose(position, [-1.0, 1.0, -1.0], atol=1e-12)


def test_shot_at_the_wrong_height_misses_even_through_the_footprint(sim: ArenaSimulation) -> None:
    """A flat shot passing over or under the box must not hit: the y axis is part of the slab test."""
    obstacle = box((0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
    for y in (3.0, -1.0):
        start, end = np.array([-5.0, y, 0.0]), np.array([5.0, y, 0.0])
        hit, _, _ = sim._segment_intersects_aabb(start, end, *sim._obstacle_bounds(obstacle))
        assert hit is False, f"y={y} is outside the box's [0, 2] span"


def test_receding_segment_never_hits(sim: ArenaSimulation) -> None:
    obstacle = box((0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
    start, end = np.array([5.0, 1.0, 0.0]), np.array([15.0, 1.0, 0.0])
    hit, _, _ = sim._segment_intersects_aabb(start, end, *sim._obstacle_bounds(obstacle))
    assert hit is False


def test_axis_aligned_segment_inside_the_box_reports_zero_travel(sim: ArenaSimulation) -> None:
    """Start already inside with no movement on any axis: entry is t = 0, not a miss."""
    obstacle = box((0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
    start = np.array([0.0, 1.0, 0.0])
    hit, t, position = sim._segment_intersects_aabb(start, start.copy(), *sim._obstacle_bounds(obstacle))
    assert hit is True
    assert t == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(position, start, atol=1e-12)


# --- the tunnelling claim ------------------------------------------------------------------------


def test_swept_solver_catches_what_per_endpoint_tests_miss(sim: ArenaSimulation) -> None:
    """One step lands on either side of the barrier: both endpoints are clear, the segment is not."""
    obstacle = box((0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
    sim.obstacles = [obstacle]
    start, end = np.array([-8.0, 1.0, 0.0]), np.array([10.0, 1.0, 0.0])
    pad = PROJECTILE_RADIUS
    assert not point_in_box(start, obstacle, pad)
    assert not point_in_box(end, obstacle, pad)

    hit, position, t = sim._first_obstacle_collision(start, end)
    assert hit is obstacle
    assert t == pytest.approx((8.0 - 1.0 - pad) / 18.0, abs=1e-12)
    # Entry is on the box surface expanded by the projectile radius, on the face the ray reaches first.
    np.testing.assert_allclose(position, [-(1.0 + pad), 1.0, 0.0], atol=1e-12)


def test_the_helpers_do_not_share_a_return_order(sim: ArenaSimulation) -> None:
    """_segment_intersects_aabb yields (flag, t, position) while the three searchers yield
    (target, position, t). The tests unpack them differently on purpose; this pins both orders so a
    future unification cannot silently swap them."""
    obstacle = box((0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
    sim.obstacles = [obstacle]
    start, end = np.array([-8.0, 1.0, 0.0]), np.array([10.0, 1.0, 0.0])

    raw = sim._segment_intersects_aabb(start, end, *sim._obstacle_bounds(obstacle))
    assert isinstance(raw[0], bool) and isinstance(raw[1], float) and raw[2].shape == (3,)

    found = sim._first_obstacle_collision(start, end)
    assert found[1].shape == (3,) and isinstance(found[2], float)

    half = sim.match_variant.arena_size * 0.5
    found = sim._arena_boundary_collision(start, np.array([half + 4.0, 1.0, 0.0]))
    assert found[1].shape == (3,) and isinstance(found[2], float)


def test_full_projectile_step_over_a_barrier_still_registers_a_hit(sim: ArenaSimulation) -> None:
    """Drive _advance_projectiles with a dt large enough to jump the barrier in one step."""
    assert sim.match_variant.arena_size * 0.5 > 15.0, "test geometry assumes a wide enough arena"
    obstacle = box((0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
    sim.obstacles = [obstacle]
    for agent in sim.agents:
        agent.alive = False
    speed = 60.0
    projectile = ProjectileState(
        projectile_id=991,
        shooter_agent_id="1-1",
        position=np.array([-8.0, 1.0, 0.0]),
        velocity=np.array([speed, 0.0, 0.0]),
        team_name="Team 1",
        team_color=(1.0, 0.0, 0.0),
        age=0.0,
        hit=False,
        obstacle_hit=False,
        hit_position=np.zeros(3),
        distance_travelled=0.0,
        max_distance=1_000.0,
    )
    sim.projectiles = [projectile]

    dt = 0.3  # an 18-unit step against a 2-unit barrier: the endpoints straddle it entirely
    before_misses = sim.cumulative_metrics["Team 1"].shots_missed
    sim._advance_projectiles(dt, {}, {})

    assert projectile.obstacle_hit is True
    assert projectile.hit is False
    assert projectile.position[0] == pytest.approx(-1.0 - PROJECTILE_RADIUS, abs=1e-9)
    assert sim.cumulative_metrics["Team 1"].shots_missed == before_misses + 1
    assert sim.pending_obstacle_hits, "the impact marker was not queued for rendering"


def test_the_nearest_obstacle_along_the_ray_wins(sim: ArenaSimulation) -> None:
    near = box((6.0, 1.0, 0.0), (2.0, 2.0, 2.0))
    far = box((14.0, 1.0, 0.0), (2.0, 2.0, 2.0))
    sim.obstacles = [far, near]  # deliberately out of order along the ray
    hit, position, t = sim._first_obstacle_collision(
        np.array([0.0, 1.0, 0.0]), np.array([20.0, 1.0, 0.0])
    )
    assert hit is near
    assert t == pytest.approx((6.0 - 1.0 - PROJECTILE_RADIUS) / 20.0, abs=1e-12)
    np.testing.assert_allclose(position, [5.0 - PROJECTILE_RADIUS, 1.0, 0.0], atol=1e-9)


# --- agent targets -------------------------------------------------------------------------------


def test_agent_bounds_are_the_body_box_plus_padding(sim: ArenaSimulation) -> None:
    agent = sim.agents[0]
    agent.position = np.array([0.0, 1.0, 0.0])
    minimum, maximum = sim._agent_bounds(agent, PROJECTILE_RADIUS)
    np.testing.assert_allclose(
        minimum,
        [-(AGENT_RADIUS + PROJECTILE_RADIUS), 1.0 - (AGENT_HALF_HEIGHT + PROJECTILE_RADIUS),
         -(AGENT_RADIUS + PROJECTILE_RADIUS)],
        atol=1e-12,
    )


def test_projectile_passes_through_team_mates_and_corpses_only(sim: ArenaSimulation) -> None:
    """Enemies on the ray are hit; own-team agents and the dead are skipped."""
    shooter_team = sim.agents[0].team_name
    enemy = next(a for a in sim.agents if a.team_name != shooter_team)
    mate = next(a for a in sim.agents if a.team_name == shooter_team and a is not sim.agents[0])

    # Line all three agents up on y = their own centre height, spaced 6 apart along x.
    sim.agents[0].position = np.array([0.0, 1.0, 0.0])
    mate.position = np.array([6.0, 1.0, 0.0])
    enemy.position = np.array([12.0, 1.0, 0.0])
    keep = {id(sim.agents[0]), id(mate), id(enemy)}
    for agent in sim.agents:
        agent.alive = id(agent) in keep

    projectile = ProjectileState(
        projectile_id=992,
        shooter_agent_id=sim.agents[0].agent_id,
        position=np.array([-6.0, 1.0, 0.0]),
        velocity=np.array([1.0, 0.0, 0.0]),
        team_name=shooter_team,
        team_color=(1.0, 0.0, 0.0),
        age=0.0,
        hit=False,
        obstacle_hit=False,
        hit_position=np.zeros(3),
        distance_travelled=0.0,
        max_distance=1_000.0,
    )
    target, position, t = sim._first_agent_collision(
        np.array([-6.0, 1.0, 0.0]), np.array([18.0, 1.0, 0.0]), projectile
    )
    assert target is enemy, "the only reachable body on the ray is the enemy"
    assert position is not None
    assert position[0] == pytest.approx(12.0 - (AGENT_RADIUS + PROJECTILE_RADIUS), abs=1e-9)
    assert t == pytest.approx((18.0 - (AGENT_RADIUS + PROJECTILE_RADIUS)) / 24.0, abs=1e-9)

    enemy.alive = False
    target, _, _ = sim._first_agent_collision(
        np.array([-6.0, 1.0, 0.0]), np.array([18.0, 1.0, 0.0]), projectile
    )
    assert target is None, "a dead agent must not absorb a shot"


# --- arena boundary ------------------------------------------------------------------------------


def test_outward_shot_hits_the_wall_at_the_arena_half_size(sim: ArenaSimulation) -> None:
    half = sim.match_variant.arena_size * 0.5
    start, end = np.array([0.0, 1.0, 0.0]), np.array([half + 5.0, 1.0, 0.0])
    hit, position, t = sim._arena_boundary_collision(start, end)
    assert hit is True
    assert position is not None
    np.testing.assert_allclose(position, [half, 1.0, 0.0], atol=1e-9)
    assert t == pytest.approx(half / (half + 5.0), abs=1e-9)


def test_shot_that_stays_inside_never_reports_a_boundary(sim: ArenaSimulation) -> None:
    half = sim.match_variant.arena_size * 0.5
    hit, _, _ = sim._arena_boundary_collision(
        np.array([-half * 0.5, 1.0, 0.0]), np.array([half * 0.5, 1.0, 0.0])
    )
    assert hit is False


def test_corner_shot_leaves_through_one_face_only(sim: ArenaSimulation) -> None:
    """A ray aimed at the corner still reports a single earliest face, not two."""
    half = sim.match_variant.arena_size * 0.5
    start = np.array([0.0, 1.0, 0.0])
    end = np.array([half * 2.0, 1.0, half * 2.0])
    hit, position, t = sim._arena_boundary_collision(start, end)
    assert hit is True
    assert position is not None
    assert max(abs(float(position[0])), abs(float(position[2]))) == pytest.approx(half, abs=1e-9)
    assert math.isclose(t, 0.5, rel_tol=1e-9)
