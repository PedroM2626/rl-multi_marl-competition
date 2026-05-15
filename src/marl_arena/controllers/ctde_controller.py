from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from marl_arena.controllers.base import BaseTeamController, ControllerContext
from marl_arena.models import AgentSnapshot, StepDecision, TransitionRecord


class CTDEController(BaseTeamController):
    paradigm_name = "CTDE"

    def __init__(self, team_name: str, rng_seed: int) -> None:
        super().__init__(team_name, rng_seed)
        self.actor_weights: Dict[str, np.ndarray] = {}
        self.critic_weights = np.linspace(0.05, 0.15, 36, dtype=float)

    def _actor_for(self, agent_id: str) -> np.ndarray:
        if agent_id not in self.actor_weights:
            self.actor_weights[agent_id] = np.array([0.9, 0.8, 0.4, 0.2], dtype=float)
        return self.actor_weights[agent_id]

    def _critic_score(
        self,
        local_features: np.ndarray,
        global_features: np.ndarray,
        action_index: int,
    ) -> float:
        critic_input = np.concatenate(
            [
                local_features,
                global_features[: min(24, len(global_features))],
                np.array([action_index, 1.0], dtype=float),
            ]
        )
        padded = np.zeros_like(self.critic_weights)
        padded[: min(len(padded), len(critic_input))] = critic_input[: len(padded)]
        return float(np.dot(self.critic_weights, padded))

    def decide(
        self,
        agent: AgentSnapshot,
        all_agents: Iterable[AgentSnapshot],
        context: ControllerContext,
    ) -> StepDecision:
        local_features = self.build_local_features(agent, all_agents)
        global_features = self.build_global_features(all_agents)
        actor = self._actor_for(agent.agent_id)
        targets = self.candidate_targets(agent, all_agents)
        scores: List[float] = []
        for index, target in enumerate(targets):
            distance = float(np.linalg.norm(target - agent.position))
            actor_bias = actor[index]
            critic_bonus = 0.03 * self._critic_score(local_features, global_features, index)
            score = actor_bias + critic_bonus - 0.1 * distance
            scores.append(score)
        best_index = int(np.argmax(scores))
        enemy = self.nearest_enemy(agent, all_agents)
        can_shoot = enemy is not None and float(np.linalg.norm(enemy.position - agent.position)) <= 19.5
        roles = ["assault", "pinch", "anchor", "kite"]
        return self.make_decision_from_target(agent, targets[best_index], can_shoot, roles[best_index])

    def update(self, transitions: List[TransitionRecord]) -> None:
        super().update(transitions)
        if not transitions:
            return
        reward_mean = float(np.mean([transition.reward for transition in transitions]))
        self.critic_weights += reward_mean * 0.001
        for transition in transitions:
            agent_id = getattr(transition, "agent_id", "")
            if not agent_id or agent_id not in self.actor_weights:
                continue
            self.actor_weights[agent_id] += 0.01 * reward_mean * transition.action_features[:4]
