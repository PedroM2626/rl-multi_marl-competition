from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from marl_arena.controllers.base import BaseTeamController, ControllerContext
from marl_arena.models import AgentSnapshot, StepDecision, TransitionRecord


class DTEController(BaseTeamController):
    paradigm_name = "DTE"

    def __init__(self, team_name: str, rng_seed: int) -> None:
        super().__init__(team_name, rng_seed)
        self.agent_weights: Dict[str, np.ndarray] = {}

    def _weights_for(self, agent_id: str) -> np.ndarray:
        if agent_id not in self.agent_weights:
            self.agent_weights[agent_id] = np.array([0.8, 0.55, 0.35, 0.15], dtype=float)
        return self.agent_weights[agent_id]

    def decide(
        self,
        agent: AgentSnapshot,
        all_agents: Iterable[AgentSnapshot],
        context: ControllerContext,
    ) -> StepDecision:
        features = self.build_local_features(agent, all_agents)
        weights = self._weights_for(agent.agent_id)
        targets = self.candidate_targets(agent, all_agents)
        scores: List[float] = []
        for index, target in enumerate(targets):
            distance = float(np.linalg.norm(target - agent.position))
            local_alignment = features[3] * 0.04 - features[4] * 0.03
            jitter = self.rng.uniform(-0.06, 0.06)
            score = weights[index] + local_alignment - 0.08 * distance + jitter
            scores.append(score)
        best_index = int(np.argmax(scores))
        enemy = self.nearest_enemy(agent, all_agents)
        can_shoot = enemy is not None and float(np.linalg.norm(enemy.position - agent.position)) <= 16.0
        roles = ["hunt", "skirmish", "group", "evade"]
        return self.make_decision_from_target(agent, targets[best_index], can_shoot, roles[best_index])

    def update(self, transitions: List[TransitionRecord]) -> None:
        super().update(transitions)
        for transition in transitions:
            if transition.agent_id not in self.agent_weights:
                continue
            weights = self.agent_weights[transition.agent_id]
            weights += 0.01 * transition.reward * transition.action_features[:4]
