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


class ValueDecompositionCriticNetwork(nn.Module):
    def __init__(self, agent_indices: list[int], hidden_dim: int = 128) -> None:
        super().__init__()
        self.agent_indices = agent_indices
        # Each agent's state features: [pos_x, pos_z, heading, alive] (4 features)
        self.agent_critic = _mlp(4, 1, hidden_dim)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        v_tot = torch.zeros(global_obs.shape[0], device=global_obs.device)
        for idx in self.agent_indices:
            agent_features = global_obs[:, 4 * idx : 4 * (idx + 1)]
            v_i = self.agent_critic(agent_features).squeeze(-1)
            v_tot = v_tot + v_i
        return v_tot


class CommActorNetwork(nn.Module):
    def __init__(
        self,
        local_obs_dim: int,
        action_dim: int,
        msg_dim: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.local_obs_dim = local_obs_dim
        self.msg_dim = msg_dim
        self.msg_net = _mlp(local_obs_dim, msg_dim, hidden_dim)
        self.policy_head = _mlp(local_obs_dim + msg_dim, action_dim, hidden_dim)

    def forward(
        self,
        team_obs: torch.Tensor,
        agent_slot: torch.Tensor,
    ) -> torch.Tensor:
        o1 = team_obs[:, : self.local_obs_dim]
        o2 = team_obs[:, self.local_obs_dim : self.local_obs_dim * 2]
        o3 = team_obs[:, self.local_obs_dim * 2 :]

        m1 = self.msg_net(o1)
        m2 = self.msg_net(o2)
        m3 = self.msg_net(o3)

        c1 = 0.5 * (m2 + m3)
        c2 = 0.5 * (m1 + m3)
        c3 = 0.5 * (m1 + m2)

        s1 = agent_slot[:, 0:1]
        s2 = agent_slot[:, 1:2]
        s3 = agent_slot[:, 2:3]

        obs_selected = s1 * o1 + s2 * o2 + s3 * o3
        comm_selected = s1 * c1 + s2 * c2 + s3 * c3

        policy_input = torch.cat([obs_selected, comm_selected], dim=-1)
        logits = self.policy_head(policy_input)
        return logits

    def act(
        self,
        team_obs: torch.Tensor,
        agent_slot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(team_obs, agent_slot)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob, entropy

    def evaluate(
        self,
        team_obs: torch.Tensor,
        agent_slot: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(team_obs, agent_slot)
        distribution = Categorical(logits=logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy

