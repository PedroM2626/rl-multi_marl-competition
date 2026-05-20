from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from marl_arena.config import ArenaConfig


TEAM_META: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("Equipe 1", "CTE", (0.92, 0.25, 0.25)),
    ("Equipe 2", "DTE", (0.25, 0.55, 0.95)),
    ("Equipe 3", "CTDE", (0.25, 0.88, 0.45)),
)

AGENT_FORMATION_OFFSETS: tuple[np.ndarray, ...] = (
    np.array([-1.7, 0.0, 0.0], dtype=float),
    np.array([1.7, 0.0, 0.0], dtype=float),
    np.array([0.0, 0.0, 1.7], dtype=float),
)


@dataclass(frozen=True)
class TeamSpawnSpec:
    team_name: str
    paradigm: str
    spawn_center: np.ndarray
    color_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class ObstacleSpec:
    obstacle_id: str
    obstacle_type: str
    base_position: np.ndarray
    size: np.ndarray
    color_rgb: tuple[float, float, float]
    movement_axis: np.ndarray
    movement_amplitude: float
    movement_speed: float
    phase_offset: float


@dataclass(frozen=True)
class MatchVariant:
    variant_id: int
    arena_size: float
    match_duration_seconds: float
    agent_move_speed: float
    agent_turn_speed: float
    shoot_range: float
    shoot_cooldown: float
    team_spawns: tuple[TeamSpawnSpec, ...]
    obstacles: tuple[ObstacleSpec, ...]

    def summary(self) -> dict[str, float | int]:
        return {
            "variant_id": self.variant_id,
            "arena_size": self.arena_size,
            "obstacle_count": len(self.obstacles),
            "match_duration_seconds": self.match_duration_seconds,
            "agent_move_speed": self.agent_move_speed,
            "shoot_range": self.shoot_range,
            "shoot_cooldown": self.shoot_cooldown,
        }


def _default_obstacle_specs() -> tuple[ObstacleSpec, ...]:
    return (
        ObstacleSpec(
            "fixed-central-wall",
            "barreira_fixa",
            np.array([0.0, 1.5, -1.0], dtype=float),
            np.array([2.0, 3.0, 10.0], dtype=float),
            (0.60, 0.58, 0.56),
            np.zeros(3, dtype=float),
            0.0,
            0.0,
            0.0,
        ),
        ObstacleSpec(
            "fixed-east-block",
            "barreira_fixa",
            np.array([8.0, 1.5, 5.0], dtype=float),
            np.array([7.0, 3.0, 2.4], dtype=float),
            (0.54, 0.52, 0.50),
            np.zeros(3, dtype=float),
            0.0,
            0.0,
            0.0,
        ),
        ObstacleSpec(
            "fixed-west-pillar",
            "barreira_fixa",
            np.array([-8.0, 1.5, 6.0], dtype=float),
            np.array([2.8, 3.0, 2.8], dtype=float),
            (0.52, 0.50, 0.48),
            np.zeros(3, dtype=float),
            0.0,
            0.0,
            0.0,
        ),
        ObstacleSpec(
            "moving-north-sweeper",
            "obstaculo_movel",
            np.array([0.0, 1.0, 13.5], dtype=float),
            np.array([4.0, 2.0, 1.6], dtype=float),
            (0.93, 0.74, 0.25),
            np.array([1.0, 0.0, 0.0], dtype=float),
            7.0,
            0.8,
            0.0,
        ),
        ObstacleSpec(
            "moving-south-sweeper",
            "obstaculo_movel",
            np.array([0.0, 1.0, -12.0], dtype=float),
            np.array([4.6, 2.0, 1.6], dtype=float),
            (0.95, 0.66, 0.24),
            np.array([1.0, 0.0, 0.0], dtype=float),
            6.0,
            0.65,
            1.4,
        ),
        ObstacleSpec(
            "passage-left",
            "passagem_restrita",
            np.array([-3.8, 1.5, 9.0], dtype=float),
            np.array([2.6, 3.0, 4.5], dtype=float),
            (0.38, 0.42, 0.50),
            np.zeros(3, dtype=float),
            0.0,
            0.0,
            0.0,
        ),
        ObstacleSpec(
            "passage-right",
            "passagem_restrita",
            np.array([3.8, 1.5, 9.0], dtype=float),
            np.array([2.6, 3.0, 4.5], dtype=float),
            (0.38, 0.42, 0.50),
            np.zeros(3, dtype=float),
            0.0,
            0.0,
            0.0,
        ),
        ObstacleSpec(
            "passage-lower-left",
            "passagem_restrita",
            np.array([-3.8, 1.5, -9.0], dtype=float),
            np.array([2.6, 3.0, 4.0], dtype=float),
            (0.36, 0.40, 0.48),
            np.zeros(3, dtype=float),
            0.0,
            0.0,
            0.0,
        ),
        ObstacleSpec(
            "passage-lower-right",
            "passagem_restrita",
            np.array([3.8, 1.5, -9.0], dtype=float),
            np.array([2.6, 3.0, 4.0], dtype=float),
            (0.36, 0.40, 0.48),
            np.zeros(3, dtype=float),
            0.0,
            0.0,
            0.0,
        ),
    )


def _default_team_spawns(config: ArenaConfig) -> tuple[TeamSpawnSpec, ...]:
    margin = config.arena_size * 0.32
    centers = (
        np.array([-margin, 1.0, -margin], dtype=float),
        np.array([margin, 1.0, -margin], dtype=float),
        np.array([0.0, 1.0, margin], dtype=float),
    )
    spawns: list[TeamSpawnSpec] = []
    for (team_name, paradigm, color), center in zip(TEAM_META, centers, strict=True):
        spawns.append(TeamSpawnSpec(team_name, paradigm, center, color))
    return tuple(spawns)


def create_default_variant(config: ArenaConfig, variant_id: int = 0) -> MatchVariant:
    return MatchVariant(
        variant_id=variant_id,
        arena_size=config.arena_size,
        match_duration_seconds=config.match_duration_seconds,
        agent_move_speed=config.agent_move_speed,
        agent_turn_speed=config.agent_turn_speed,
        shoot_range=config.shoot_range,
        shoot_cooldown=config.shoot_cooldown,
        team_spawns=_default_team_spawns(config),
        obstacles=_default_obstacle_specs(),
    )


def _aabb_overlap(
    pos_a: np.ndarray,
    size_a: np.ndarray,
    pos_b: np.ndarray,
    size_b: np.ndarray,
    padding: float,
) -> bool:
    half_a = size_a * 0.5 + padding
    half_b = size_b * 0.5 + padding
    return (
        abs(pos_a[0] - pos_b[0]) < half_a[0] + half_b[0]
        and abs(pos_a[2] - pos_b[2]) < half_a[2] + half_b[2]
    )


def _point_blocked(
    point: np.ndarray,
    obstacles: Sequence[ObstacleSpec],
    agent_clear_radius: float = 2.2,
) -> bool:
    probe = np.array([point[0], 1.0, point[2]], dtype=float)
    probe_size = np.array([agent_clear_radius * 2, 2.0, agent_clear_radius * 2], dtype=float)
    for obstacle in obstacles:
        if _aabb_overlap(probe, probe_size, obstacle.base_position, obstacle.size, 0.35):
            return True
    return False


def _spawn_far_enough(centers: Sequence[np.ndarray], candidate: np.ndarray, min_distance: float) -> bool:
    for center in centers:
        if float(np.linalg.norm(center[[0, 2]] - candidate[[0, 2]])) < min_distance:
            return False
    return True


def _sample_team_spawns(rng: random.Random, arena_size: float, obstacles: Sequence[ObstacleSpec]) -> tuple[TeamSpawnSpec, ...]:
    margin = arena_size * rng.uniform(0.26, 0.36)
    min_sep = arena_size * 0.38
    candidates = [
        np.array([-margin, 1.0, -margin], dtype=float),
        np.array([margin, 1.0, -margin], dtype=float),
        np.array([0.0, 1.0, margin], dtype=float),
        np.array([-margin, 1.0, margin * 0.35], dtype=float),
        np.array([margin, 1.0, margin * 0.35], dtype=float),
        np.array([-margin, 1.0, margin], dtype=float),
        np.array([margin, 1.0, margin], dtype=float),
    ]
    rng.shuffle(candidates)
    chosen: list[np.ndarray] = []
    for _ in range(60):
        if len(chosen) >= 3:
            break
        candidate = candidates[rng.randrange(len(candidates))].copy()
        candidate[0] += rng.uniform(-2.0, 2.0)
        candidate[2] += rng.uniform(-2.0, 2.0)
        if not _spawn_far_enough(chosen, candidate, min_sep):
            continue
        if _point_blocked(candidate, obstacles):
            continue
        chosen.append(candidate)
    while len(chosen) < 3:
        chosen.append(np.array([rng.uniform(-6, 6), 1.0, rng.uniform(-6, 6)], dtype=float))
    spawns: list[TeamSpawnSpec] = []
    for (team_name, paradigm, color), center in zip(TEAM_META, chosen[:3], strict=True):
        spawns.append(TeamSpawnSpec(team_name, paradigm, center, color))
    return tuple(spawns)


def _sample_obstacle_specs(rng: random.Random, arena_size: float, config: ArenaConfig) -> tuple[ObstacleSpec, ...]:
    count = rng.randint(config.dr_obstacle_count_min, config.dr_obstacle_count_max)
    half = arena_size * 0.42
    specs: list[ObstacleSpec] = []
    type_pool = ("barreira_fixa", "barreira_fixa", "passagem_restrita", "obstaculo_movel")
    for index in range(count):
        for _ in range(40):
            obstacle_type = rng.choice(type_pool)
            x = rng.uniform(-half, half)
            z = rng.uniform(-half, half)
            if obstacle_type == "barreira_fixa":
                size = np.array(
                    [
                        rng.uniform(2.0, 8.0),
                        rng.uniform(2.5, 3.5),
                        rng.uniform(2.0, 8.0),
                    ],
                    dtype=float,
                )
                color = (0.55, 0.53, 0.51)
                movement_axis = np.zeros(3, dtype=float)
                amplitude = 0.0
                speed = 0.0
                phase = 0.0
            elif obstacle_type == "passagem_restrita":
                size = np.array(
                    [
                        rng.uniform(2.0, 3.5),
                        rng.uniform(2.5, 3.5),
                        rng.uniform(3.0, 5.5),
                    ],
                    dtype=float,
                )
                color = (0.38, 0.42, 0.50)
                movement_axis = np.zeros(3, dtype=float)
                amplitude = 0.0
                speed = 0.0
                phase = 0.0
            else:
                size = np.array(
                    [
                        rng.uniform(3.0, 5.5),
                        rng.uniform(1.6, 2.4),
                        rng.uniform(1.2, 2.0),
                    ],
                    dtype=float,
                )
                color = (0.93, 0.70, 0.24)
                axis = np.array([1.0, 0.0, 0.0], dtype=float)
                if rng.random() < 0.5:
                    axis = np.array([0.0, 0.0, 1.0], dtype=float)
                movement_axis = axis
                amplitude = rng.uniform(4.0, arena_size * 0.22)
                speed = rng.uniform(0.45, 1.0)
                phase = rng.uniform(0.0, 3.14)
            position = np.array([x, 1.0 if obstacle_type == "obstaculo_movel" else 1.5, z], dtype=float)
            candidate = ObstacleSpec(
                f"rnd-{index}",
                obstacle_type,
                position,
                size,
                color,
                movement_axis,
                amplitude,
                speed,
                phase,
            )
            if any(_aabb_overlap(candidate.base_position, candidate.size, s.base_position, s.size, 0.8) for s in specs):
                continue
            if abs(position[0]) < 2.0 and abs(position[2]) < 2.0:
                continue
            specs.append(candidate)
            break
    if not specs:
        return _default_obstacle_specs()
    return tuple(specs)


def sample_training_variant(rng: random.Random, config: ArenaConfig, variant_id: int) -> MatchVariant:
    arena_size = rng.uniform(config.dr_arena_size_min, config.dr_arena_size_max)
    obstacles = _sample_obstacle_specs(rng, arena_size, config)
    team_spawns = _sample_team_spawns(rng, arena_size, obstacles)
    return MatchVariant(
        variant_id=variant_id,
        arena_size=arena_size,
        match_duration_seconds=rng.uniform(config.dr_match_duration_min, config.dr_match_duration_max),
        agent_move_speed=rng.uniform(config.dr_move_speed_min, config.dr_move_speed_max),
        agent_turn_speed=rng.uniform(config.dr_turn_speed_min, config.dr_turn_speed_max),
        shoot_range=rng.uniform(config.dr_shoot_range_min, config.dr_shoot_range_max),
        shoot_cooldown=rng.uniform(config.dr_shoot_cooldown_min, config.dr_shoot_cooldown_max),
        team_spawns=team_spawns,
        obstacles=obstacles,
    )
