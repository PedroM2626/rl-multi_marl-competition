from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class RolloutStep:
    local_obs: np.ndarray
    global_obs: np.ndarray | None
    agent_slot: np.ndarray | None
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


@dataclass
class RolloutBuffer:
    steps: List[RolloutStep] = field(default_factory=list)

    def clear(self) -> None:
        self.steps.clear()

    def append(self, step: RolloutStep) -> None:
        self.steps.append(step)

    def __len__(self) -> int:
        return len(self.steps)

    def compute_returns(self, gamma: float, gae_lambda: float, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.array([step.reward for step in self.steps], dtype=np.float32)
        values = np.array([step.value for step in self.steps] + [last_value], dtype=np.float32)
        dones = np.array([float(step.done) for step in self.steps], dtype=np.float32)
        advantages = np.zeros(len(self.steps), dtype=np.float32)
        last_advantage = 0.0
        for index in reversed(range(len(self.steps))):
            mask = 1.0 - dones[index]
            delta = rewards[index] + gamma * values[index + 1] * mask - values[index]
            last_advantage = delta + gamma * gae_lambda * mask * last_advantage
            advantages[index] = last_advantage
        returns = advantages + values[:-1]
        return advantages, returns

    def to_tensors(
        self,
        device: torch.device,
        use_global: bool,
        use_agent_slot: bool,
    ) -> dict[str, torch.Tensor]:
        local_obs = torch.tensor(
            np.stack([step.local_obs for step in self.steps]),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.tensor([step.action for step in self.steps], dtype=torch.long, device=device)
        old_log_probs = torch.tensor([step.log_prob for step in self.steps], dtype=torch.float32, device=device)
        payload: dict[str, torch.Tensor] = {
            "local_obs": local_obs,
            "actions": actions,
            "old_log_probs": old_log_probs,
        }
        if use_global:
            global_obs = torch.tensor(
                np.stack([step.global_obs for step in self.steps if step.global_obs is not None]),
                dtype=torch.float32,
                device=device,
            )
            payload["global_obs"] = global_obs
        if use_agent_slot:
            agent_slots = torch.tensor(
                np.stack([step.agent_slot for step in self.steps if step.agent_slot is not None]),
                dtype=torch.float32,
                device=device,
            )
            payload["agent_slots"] = agent_slots
        return payload
