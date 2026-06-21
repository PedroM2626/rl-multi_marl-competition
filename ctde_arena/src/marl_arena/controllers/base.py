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
    target_angle = math.degrees(math.atan2(offset[0], offset[1]))
    delta = (target_angle - heading_deg + 180.0) % 360.0 - 180.0
    return delta


@dataclass
class ControllerContext:
    step_index: int
    arena_size: float
    time_delta: float
    shoot_range: float


class BaseTeamController(ABC):
    paradigm_name: str = "BASE"

    def __init__(self, team_name: str, rng_seed: int) -> None:
        self.team_name = team_name
        self.rng = random.Random(rng_seed)
        self.transitions: List[TransitionRecord] = []

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

    def build_local_features(self, agent: AgentSnapshot, all_agents: Iterable[AgentSnapshot]) -> np.ndarray:
        enemy = self.nearest_enemy(agent, all_agents)
        ally_centroid = self.ally_centroid(all_agents)
        if enemy is None:
            relative_enemy = np.zeros(3, dtype=float)
            enemy_distance = 0.0
        else:
            relative_enemy = enemy.position - agent.position
            enemy_distance = float(np.linalg.norm(relative_enemy))
        return np.array(
            [
                agent.position[0],
                agent.position[2],
                agent.heading_deg / 180.0,
                relative_enemy[0],
                relative_enemy[2],
                enemy_distance,
                ally_centroid[0] - agent.position[0],
                ally_centroid[2] - agent.position[2],
            ],
            dtype=float,
        )

    def build_global_features(self, all_agents: Iterable[AgentSnapshot]) -> np.ndarray:
        features: List[float] = []
        for agent in sorted(all_agents, key=lambda item: item.agent_id):
            features.extend(
                [
                    agent.position[0],
                    agent.position[2],
                    agent.heading_deg / 180.0,
                    1.0 if agent.alive else 0.0,
                ]
            )
        return np.array(features, dtype=float)

    def candidate_targets(self, agent: AgentSnapshot, all_agents: Iterable[AgentSnapshot]) -> List[np.ndarray]:
        enemy = self.nearest_enemy(agent, all_agents)
        if enemy is None:
            return [agent.position.copy()]

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

    def update(
        self,
        transitions: List[TransitionRecord],
    ) -> None:
        self.transitions.extend(transitions)
