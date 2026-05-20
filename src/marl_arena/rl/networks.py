from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = _mlp(obs_dim, hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob, entropy, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        distribution = Categorical(logits=logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy, value


class CentralizedActorNetwork(nn.Module):
    def __init__(
        self,
        global_obs_dim: int,
        agent_slot_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = global_obs_dim + agent_slot_dim
        self.backbone = _mlp(input_dim, hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, global_obs: torch.Tensor, agent_slot: torch.Tensor) -> torch.Tensor:
        features = self.backbone(torch.cat([global_obs, agent_slot], dim=-1))
        return self.policy_head(features)

    def act(
        self,
        global_obs: torch.Tensor,
        agent_slot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(global_obs, agent_slot)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob, entropy

    def evaluate(
        self,
        global_obs: torch.Tensor,
        agent_slot: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(global_obs, agent_slot)
        distribution = Categorical(logits=logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy


class CentralizedCriticNetwork(nn.Module):
    def __init__(self, global_obs_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = _mlp(global_obs_dim, hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(global_obs)
        return self.value_head(features).squeeze(-1)


class CTDEActorNetwork(ActorNetwork):
    pass


class CTDECriticNetwork(CentralizedCriticNetwork):
    pass
