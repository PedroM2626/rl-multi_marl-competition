from __future__ import annotations

from typing import Iterable, List

import numpy as np

from marl_arena.controllers.base import BaseTeamController, ControllerContext
from marl_arena.models import AgentSnapshot, StepDecision, TransitionRecord


class CTEController(BaseTeamController):
    paradigm_name = "CTE"

    def __init__(self, team_name: str, rng_seed: int) -> None:
        super().__init__(team_name, rng_seed)
        self.coordinator_weights = np.array([1.5, 1.2, 0.9, 0.7], dtype=float)

    def _score_targets(
        self,
        agent: AgentSnapshot,
        all_agents: Iterable[AgentSnapshot],
    ) -> List[float]:
        global_features = self.build_global_features(all_agents)
        ally_centroid = self.ally_centroid(all_agents)
        scores: List[float] = []
        for index, target in enumerate(self.candidate_targets(agent, all_agents)):
            distance = float(np.linalg.norm(target - agent.position))
            spread = float(np.linalg.norm(target - ally_centroid))
            crowding_penalty = abs(global_features.mean()) * 0.02 * (index + 1)
            score = (
                self.coordinator_weights[index]
                - 0.12 * distance
                - 0.06 * spread
                - crowding_penalty
            )
            scores.append(score)
        return scores

    def decide(
        self,
        agent: AgentSnapshot,
        all_agents: Iterable[AgentSnapshot],
        context: ControllerContext,
    ) -> StepDecision:
        scores = self._score_targets(agent, all_agents)
        best_index = int(np.argmax(scores))
        roles = ["pressure", "flank", "support", "retreat"]
        target = self.candidate_targets(agent, all_agents)[best_index]
        enemy = self.nearest_enemy(agent, all_agents)
        can_shoot = enemy is not None and float(np.linalg.norm(enemy.position - agent.position)) <= 18.0
        return self.make_decision_from_target(agent, target, can_shoot, roles[best_index])

    def update(self, transitions: List[TransitionRecord]) -> None:
        super().update(transitions)
        if not transitions:
            return
        reward_signal = float(np.mean([transition.reward for transition in transitions]))
        self.coordinator_weights += np.array(
            [0.02 * reward_signal, 0.015 * reward_signal, 0.008 * reward_signal, -0.01 * reward_signal],
            dtype=float,
        )
