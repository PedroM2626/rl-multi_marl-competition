from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from marl_arena.rl.buffer import RolloutBuffer
from marl_arena.rl.networks import ActorNetwork, CentralizedActorNetwork, CentralizedCriticNetwork


@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float


class PPOTrainer:
    def __init__(
        self,
        learning_rate: float,
        clip_eps: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        ppo_epochs: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer: torch.optim.Optimizer | None = None
        self.learning_rate = learning_rate

    def bind_optimizer(self, parameters: list[torch.nn.Parameter]) -> None:
        self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)

    def _optimize(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy: torch.Tensor,
    ) -> PPOStats:
        if self.optimizer is None:
            raise RuntimeError("Optimizer nao configurado.")
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [parameter for group in self.optimizer.param_groups for parameter in group["params"]],
            self.max_grad_norm,
        )
        self.optimizer.step()
        return PPOStats(
            policy_loss=float(policy_loss.item()),
            value_loss=float(value_loss.item()),
            entropy=float(entropy.item()),
            approx_kl=0.0,
        )

    def update_actor_critic(
        self,
        actor: ActorNetwork,
        buffer: RolloutBuffer,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> PPOStats | None:
        if len(buffer) == 0:
            return None
        advantages, returns = buffer.compute_returns(gamma, gae_lambda, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        tensors = buffer.to_tensors(self.device, use_global=False, use_agent_slot=False)
        last_stats: PPOStats | None = None
        sample_count = len(buffer)
        for _ in range(self.ppo_epochs):
            indices = np.arange(sample_count)
            np.random.shuffle(indices)
            for start in range(0, sample_count, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                obs_batch = tensors["local_obs"][batch_indices]
                actions_batch = tensors["actions"][batch_indices]
                old_log_probs_batch = tensors["old_log_probs"][batch_indices]
                returns_batch = returns_tensor[batch_indices]
                advantages_batch = advantages_tensor[batch_indices]
                log_probs, entropy, values = actor.evaluate(obs_batch, actions_batch)
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surrogate_1 = ratio * advantages_batch
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_batch
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = nn.functional.mse_loss(values, returns_batch)
                last_stats = self._optimize(policy_loss, value_loss, entropy.mean())
        return last_stats

    def update_ctde(
        self,
        actor: ActorNetwork,
        critic: CentralizedCriticNetwork,
        buffer: RolloutBuffer,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> PPOStats | None:
        if len(buffer) == 0:
            return None
        advantages, returns = buffer.compute_returns(gamma, gae_lambda, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        tensors = buffer.to_tensors(self.device, use_global=True, use_agent_slot=False)
        last_stats: PPOStats | None = None
        sample_count = len(buffer)
        for _ in range(self.ppo_epochs):
            indices = np.arange(sample_count)
            np.random.shuffle(indices)
            for start in range(0, sample_count, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                obs_batch = tensors["local_obs"][batch_indices]
                global_batch = tensors["global_obs"][batch_indices]
                actions_batch = tensors["actions"][batch_indices]
                old_log_probs_batch = tensors["old_log_probs"][batch_indices]
                returns_batch = returns_tensor[batch_indices]
                advantages_batch = advantages_tensor[batch_indices]
                log_probs, entropy, _ = actor.evaluate(obs_batch, actions_batch)
                values = critic(global_batch)
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surrogate_1 = ratio * advantages_batch
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_batch
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = nn.functional.mse_loss(values, returns_batch)
                last_stats = self._optimize(policy_loss, value_loss, entropy.mean())
        return last_stats

    def update_cte(
        self,
        actor: CentralizedActorNetwork,
        critic: CentralizedCriticNetwork,
        buffer: RolloutBuffer,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> PPOStats | None:
        if len(buffer) == 0:
            return None
        advantages, returns = buffer.compute_returns(gamma, gae_lambda, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        tensors = buffer.to_tensors(self.device, use_global=True, use_agent_slot=True)
        last_stats: PPOStats | None = None
        sample_count = len(buffer)
        for _ in range(self.ppo_epochs):
            indices = np.arange(sample_count)
            np.random.shuffle(indices)
            for start in range(0, sample_count, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                global_batch = tensors["global_obs"][batch_indices]
                slot_batch = tensors["agent_slots"][batch_indices]
                actions_batch = tensors["actions"][batch_indices]
                old_log_probs_batch = tensors["old_log_probs"][batch_indices]
                returns_batch = returns_tensor[batch_indices]
                advantages_batch = advantages_tensor[batch_indices]
                log_probs, entropy = actor.evaluate(global_batch, slot_batch, actions_batch)
                values = critic(global_batch)
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surrogate_1 = ratio * advantages_batch
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_batch
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = nn.functional.mse_loss(values, returns_batch)
                last_stats = self._optimize(policy_loss, value_loss, entropy.mean())
        return last_stats

    @staticmethod
    def bootstrap_value_critic(critic: CentralizedCriticNetwork, global_obs: np.ndarray, device: torch.device) -> float:
        obs_tensor = torch.tensor(global_obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return float(critic(obs_tensor).item())

    @staticmethod
    def bootstrap_value_actor(actor: ActorNetwork, local_obs: np.ndarray, device: torch.device) -> float:
        obs_tensor = torch.tensor(local_obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, value = actor(obs_tensor)
            return float(value.item())
