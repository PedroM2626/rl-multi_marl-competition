from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from marl_arena.models import AgentSnapshot, StepDecision, TransitionRecord


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.zeros_like(vector)
    return vector / norm


def angle_to_target(origin: np.ndarray, heading_deg: float, target: np.ndarray) -> float:
    offset = target - origin
    target_angle = math.degrees(math.atan2(offset[0], offset[2]))
    delta = (target_angle - heading_deg + 180.0) % 360.0 - 180.0
    return delta


@dataclass
class ControllerContext:
    step_index: int
    arena_size: float
    time_delta: float
    shoot_range: float
    # (M, 4) rows of (centre x, centre z, half-width x, half-width z), filled only when observations
    # carry obstacle rays; None keeps cover out of the observation, as every measured policy had it.
    obstacle_boxes: np.ndarray | None = None


def obstacle_rays(
    positions: np.ndarray, headings_deg: np.ndarray, boxes: np.ndarray, half_size: float, count: int,
) -> np.ndarray:
    """Distance from each agent to the first obstacle face or wall along `count` evenly spaced rays.

    Rays start at the agent's heading and go round the full circle in the ground plane; each distance is
    divided by `half_size` and capped at 1, so the feature is on the scale of the positions beside it.
    Returns an (agents, count) array. This is what makes cover observable (A-17).
    """
    angles = np.radians(headings_deg)[:, None] + 2.0 * np.pi * np.arange(count)[None, :] / count
    dx, dz = np.sin(angles), np.cos(angles)                                   # (N, K)
    origin_x, origin_z = positions[:, 0:1], positions[:, 1:2]                  # (N, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Walls: the ray leaves the square [-half, half]^2 through the face it is heading for.
        wall_x = np.where(dx > 0, (half_size - origin_x) / dx, np.where(dx < 0, (-half_size - origin_x) / dx, np.inf))
        wall_z = np.where(dz > 0, (half_size - origin_z) / dz, np.where(dz < 0, (-half_size - origin_z) / dz, np.inf))
        distance = np.minimum(wall_x, wall_z)
        if boxes is not None and len(boxes):
            # Slab test against every box at once: (N, K, M).
            cx, cz, hx, hz = (boxes[:, i][None, None, :] for i in range(4))
            ox, oz = origin_x[..., None], origin_z[..., None]
            rx, rz = dx[..., None], dz[..., None]
            tx1, tx2 = (cx - hx - ox) / rx, (cx + hx - ox) / rx
            tz1, tz2 = (cz - hz - oz) / rz, (cz + hz - oz) / rz
            inside_x = np.abs(ox - cx) <= hx
            inside_z = np.abs(oz - cz) <= hz
            t_near = np.maximum(
                np.where(rx == 0, np.where(inside_x, -np.inf, np.inf), np.minimum(tx1, tx2)),
                np.where(rz == 0, np.where(inside_z, -np.inf, np.inf), np.minimum(tz1, tz2)),
            )
            t_far = np.minimum(
                np.where(rx == 0, np.where(inside_x, np.inf, -np.inf), np.maximum(tx1, tx2)),
                np.where(rz == 0, np.where(inside_z, np.inf, -np.inf), np.maximum(tz1, tz2)),
            )
            hit = (t_far >= np.maximum(t_near, 0.0))
            entry = np.where(hit, np.maximum(t_near, 0.0), np.inf)
            distance = np.minimum(distance, entry.min(axis=2))
    return np.clip(np.nan_to_num(distance / half_size, nan=1.0, posinf=1.0), 0.0, 1.0)


class BaseTeamController(ABC):
    paradigm_name: str = "BASE"

    def __init__(self, team_name: str, rng_seed: int) -> None:
        self.team_name = team_name
        self.rng = random.Random(rng_seed)

    def get_team_agents(self, all_agents: Iterable[AgentSnapshot]) -> List[AgentSnapshot]:
        return [agent for agent in all_agents if agent.team_name == self.team_name]

    def get_enemy_agents(self, all_agents: Iterable[AgentSnapshot]) -> List[AgentSnapshot]:
        return [agent for agent in all_agents if agent.team_name != self.team_name and agent.alive]

    def nearest_enemy(self, agent: AgentSnapshot, all_agents: Iterable[AgentSnapshot]) -> AgentSnapshot | None:
        enemies = self.get_enemy_agents(all_agents)
        if not enemies:
            return None
        return min(enemies, key=lambda enemy: float(np.linalg.norm(enemy.position - agent.position)))

    def ally_centroid(self, all_agents: Iterable[AgentSnapshot]) -> np.ndarray:
        allies = self.get_team_agents(all_agents)
        if not allies:
            return np.zeros(3, dtype=float)
        positions = np.array([agent.position for agent in allies], dtype=float)
        return positions.mean(axis=0)

    def build_local_features(
        self,
        agent: AgentSnapshot,
        all_agents: Iterable[AgentSnapshot],
        scale: float = 1.0,
    ) -> np.ndarray:
        enemy = self.nearest_enemy(agent, all_agents)
        ally_centroid = self.ally_centroid(all_agents)
        if enemy is None:
            relative_enemy = np.zeros(3, dtype=float)
            enemy_distance = 0.0
        else:
            relative_enemy = enemy.position - agent.position
            enemy_distance = float(np.linalg.norm(relative_enemy))
        radians = math.radians(agent.heading_deg)
        return np.array(
            [
                agent.position[0] / scale,
                agent.position[2] / scale,
                math.sin(radians),
                math.cos(radians),
                relative_enemy[0] / scale,
                relative_enemy[2] / scale,
                enemy_distance / scale,
                (ally_centroid[0] - agent.position[0]) / scale,
                (ally_centroid[2] - agent.position[2]) / scale,
            ],
            dtype=float,
        )

    def build_global_features(
        self, all_agents: Iterable[AgentSnapshot], scale: float = 1.0, rays: dict[str, np.ndarray] | None = None,
        ray_count: int = 0,
    ) -> np.ndarray:
        """The 45 measured features, then -- only when rays are on -- each agent's rays in agent order.

        Appending rather than interleaving keeps the five-per-agent prefix the global-slice VD critic
        indexes into unchanged.
        """
        features: List[float] = []
        ordered = sorted(all_agents, key=lambda item: item.agent_id)
        for agent in ordered:
            radians = math.radians(agent.heading_deg)
            features.extend(
                [
                    agent.position[0] / scale,
                    agent.position[2] / scale,
                    math.sin(radians),
                    math.cos(radians),
                    1.0 if agent.alive else 0.0,
                ]
            )
        if ray_count:
            for agent in ordered:
                values = (rays or {}).get(agent.agent_id)
                features.extend(values.tolist() if values is not None and agent.alive else [0.0] * ray_count)
        return np.array(features, dtype=float)

    def candidate_targets(self, agent: AgentSnapshot, all_agents: Iterable[AgentSnapshot]) -> List[np.ndarray]:
        enemy = self.nearest_enemy(agent, all_agents)
        if enemy is None:
            # The action space always indexes four waypoints, so the list must stay that long even
            # when there is nothing to target; a short list makes action_to_decision raise.
            hold = agent.position.copy()
            return [hold, hold.copy(), hold.copy(), hold.copy()]

        chase = enemy.position.copy()
        flank = enemy.position + np.array([3.5, 0.0, -3.5], dtype=float)
        support = self.ally_centroid(all_agents)
        retreat = agent.position - normalize(enemy.position - agent.position) * 4.0
        return [chase, flank, support, retreat]

    def make_decision_from_target(
        self,
        agent: AgentSnapshot,
        target: np.ndarray,
        shoot: bool,
        role: str,
    ) -> StepDecision:
        direction = normalize(target - agent.position)
        turn_delta = angle_to_target(agent.position, agent.heading_deg, target)
        turn = max(-1.0, min(1.0, turn_delta / 35.0))
        move = 1.0 if float(np.linalg.norm(target - agent.position)) > 1.4 else 0.0
        jump = abs(turn_delta) < 20.0 and self.rng.random() < 0.015
        return StepDecision(
            move=move,
            turn=turn,
            jump=jump,
            shoot=shoot,
            aim_direction=direction,
            desired_target=target,
            debug_role=role,
        )

    @abstractmethod
    def decide(
        self,
        agent: AgentSnapshot,
        all_agents: Iterable[AgentSnapshot],
        context: ControllerContext,
    ) -> StepDecision:
        raise NotImplementedError

    @abstractmethod
    def update(self, transitions: List[TransitionRecord]) -> None:
        raise NotImplementedError
