from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


Vec2 = Tuple[float, float]


@dataclass
class AgentSnapshot:
    agent_id: str
    team_name: str
    paradigm: str
    position: np.ndarray
    heading_deg: float
    alive: bool
    kills: int = 0
    hits: int = 0
    misses: int = 0
    survival_time: float = 0.0


@dataclass
class TeamMetrics:
    team_name: str
    paradigm: str
    wins: int = 0
    eliminations: int = 0
    shots_hit: int = 0
    shots_missed: int = 0
    survival_time_sum: float = 0.0
    matches_played: int = 0

    def as_summary(self) -> Dict[str, float]:
        matches = max(self.matches_played, 1)
        total_shots = self.shots_hit + self.shots_missed
        return {
            "team_name": self.team_name,
            "paradigm": self.paradigm,
            "win_rate": self.wins / matches,
            "eliminations_per_match": self.eliminations / matches,
            "mean_survival_time": self.survival_time_sum / (matches * 3),
            "shots_hit": self.shots_hit,
            "shots_missed": self.shots_missed,
            "shot_accuracy": self.shots_hit / max(total_shots, 1),
        }


@dataclass
class StepDecision:
    move: float
    turn: float
    jump: bool
    shoot: bool
    aim_direction: np.ndarray
    desired_target: np.ndarray
    debug_role: str = "engage"


@dataclass
class ObstacleSnapshot:
    obstacle_id: str
    obstacle_type: str
    position: np.ndarray
    size: np.ndarray
    movement_axis: np.ndarray
    movement_amplitude: float


@dataclass
class TransitionRecord:
    agent_id: str
    team_name: str
    state_features: np.ndarray
    action_features: np.ndarray
    reward: float
    next_state_features: np.ndarray
    done: bool


@dataclass
class MatchResult:
    winner_team: str
    duration_seconds: float
    team_rows: List[Dict[str, float]]
    agent_rows: List[Dict[str, float]]
    trajectory_rows: List[Dict[str, float]] = field(default_factory=list)
