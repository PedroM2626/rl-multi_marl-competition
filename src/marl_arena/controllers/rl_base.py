from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from marl_arena.config import CONFIG
from marl_arena.controllers.base import BaseTeamController
from marl_arena.models import TransitionRecord
from marl_arena.rl.buffer import RolloutBuffer, RolloutStep
from marl_arena.rl.ppo import PPOStats, PPOTrainer


class RLTeamController(BaseTeamController):
    training_enabled: bool = True

    def __init__(self, team_name: str, rng_seed: int, device: torch.device) -> None:
        super().__init__(team_name, rng_seed)
        self.device = device
        self.buffer = RolloutBuffer()
        self.pending_steps: Dict[str, RolloutStep] = {}
        self.trainer = PPOTrainer(
            learning_rate=CONFIG.rl_learning_rate,
            clip_eps=CONFIG.rl_clip_eps,
            value_coef=CONFIG.rl_value_coef,
            entropy_coef=CONFIG.rl_entropy_coef,
            max_grad_norm=CONFIG.rl_max_grad_norm,
            ppo_epochs=CONFIG.rl_ppo_epochs,
            batch_size=CONFIG.rl_batch_size,
            device=device,
        )
        self._agent_slot_map: Dict[str, int] = {}

    def _agent_slot_vector(self, agent_id: str) -> np.ndarray:
        slot_index = self._agent_slot_map.get(agent_id, 0)
        vector = np.zeros(3, dtype=float)
        vector[slot_index] = 1.0
        return vector

    def _register_agent_slots(self, all_agents: Iterable) -> None:
        from marl_arena.models import AgentSnapshot

        allies = sorted(
            [agent for agent in all_agents if agent.team_name == self.team_name],
            key=lambda item: item.agent_id,
        )
        for index, agent in enumerate(allies):
            self._agent_slot_map[agent.agent_id] = index

    @abstractmethod
    def finish_episode(self) -> PPOStats | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, directory: Path) -> None:
        raise NotImplementedError

    def set_training(self, enabled: bool) -> None:
        self.training_enabled = enabled

    def update(self, transitions: List[TransitionRecord]) -> None:
        super().update(transitions)
        for transition in transitions:
            if transition.team_name != self.team_name:
                continue
            pending = self.pending_steps.pop(transition.agent_id, None)
            if pending is None:
                continue
            pending.reward = transition.reward
            pending.done = transition.done
            self.buffer.append(pending)

    def _greedy_action(self, logits: torch.Tensor) -> int:
        return int(torch.argmax(logits, dim=-1).item())
