from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


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
    # Which agent produced the row. Only the per-agent GAE path reads it; the team buffer interleaves
    # every live teammate's rows in step order.
    agent_id: str = ""


@dataclass
class RolloutBuffer:
    steps: List[RolloutStep] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

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

    def compute_returns_per_agent(self, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        """GAE over each agent's own trajectory rather than over the interleaved team buffer.

        The team buffer appends every live teammate's row each step, so `compute_returns` chains agent A's
        step t to agent B's step t as if one followed the other (docs/06 § 6.2). Here each row's successor
        is the same agent's next row. An agent whose last row is not terminal was cut off by the end of the
        match; it bootstraps from its own last value estimate, which is what the team path's `last_value`
        is as well (the critic's parameters do not change within a match).
        """
        rewards = np.array([step.reward for step in self.steps], dtype=np.float32)
        values = np.array([step.value for step in self.steps], dtype=np.float32)
        dones = np.array([float(step.done) for step in self.steps], dtype=np.float32)
        advantages = np.zeros(len(self.steps), dtype=np.float32)
        rows_by_agent: dict[str, list[int]] = {}
        for index, step in enumerate(self.steps):
            rows_by_agent.setdefault(step.agent_id, []).append(index)
        for rows in rows_by_agent.values():
            next_value = values[rows[-1]]
            last_advantage = 0.0
            for index in reversed(rows):
                mask = 1.0 - dones[index]
                delta = rewards[index] + gamma * next_value * mask - values[index]
                last_advantage = delta + gamma * gae_lambda * mask * last_advantage
                advantages[index] = last_advantage
                next_value = values[index]
        returns = advantages + values
        return advantages, returns

    def to_arrays(self, use_global: bool, use_agent_slot: bool) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "local_obs": np.stack([step.local_obs for step in self.steps]).astype(np.float32, copy=False),
            "actions": np.asarray([step.action for step in self.steps], dtype=np.int32),
            "old_log_probs": np.asarray([step.log_prob for step in self.steps], dtype=np.float32),
        }
        if use_global:
            payload["global_obs"] = np.stack([step.global_obs for step in self.steps]).astype(np.float32, copy=False)
        if use_agent_slot:
            payload["agent_slots"] = np.stack([step.agent_slot for step in self.steps]).astype(np.float32, copy=False)
        return payload
