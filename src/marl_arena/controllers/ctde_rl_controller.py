from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from marl_arena.config import CONFIG
from marl_arena.controllers.rl_base import RLTeamController
from marl_arena.controllers.base import ControllerContext
from marl_arena.models import AgentSnapshot, StepDecision
from marl_arena.rl.buffer import RolloutStep
from marl_arena.rl.checkpoints import load_checkpoint, save_checkpoint
from marl_arena.rl.networks import ActorNetwork, CentralizedCriticNetwork
from marl_arena.rl.ppo import PPOStats, PPOTrainer
from marl_arena.rl.constants import GLOBAL_OBS_DIM, LOCAL_OBS_DIM, NUM_ACTIONS
from marl_arena.rl.spaces import action_to_decision


class CTDERLController(RLTeamController):
    paradigm_name = "CTDE"

    def __init__(self, team_name: str, rng_seed: int, device: torch.device) -> None:
        super().__init__(team_name, rng_seed, device)
        self.actor = ActorNetwork(LOCAL_OBS_DIM, NUM_ACTIONS, CONFIG.rl_hidden_dim).to(device)
        self.critic = CentralizedCriticNetwork(GLOBAL_OBS_DIM, CONFIG.rl_hidden_dim).to(device)
        self.trainer.bind_optimizer(list(self.actor.parameters()) + list(self.critic.parameters()))
        self._load_if_exists()

    def _checkpoint_path(self) -> Path:
        return CONFIG.rl_checkpoint_dir / f"{self.team_name.lower().replace(' ', '_')}_ctde.pt"

    def _load_if_exists(self) -> None:
        path = self._checkpoint_path()
        if not path.exists():
            return
        payload = load_checkpoint(path, self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])

    def load(self, directory: Path) -> None:
        path = directory / f"{self.team_name.lower().replace(' ', '_')}_ctde.pt"
        payload = load_checkpoint(path, self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])

    def save(self, directory: Path) -> None:
        save_checkpoint(
            directory / f"{self.team_name.lower().replace(' ', '_')}_ctde.pt",
            {
                "paradigm": self.paradigm_name,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
        )

    def decide(
        self,
        agent: AgentSnapshot,
        all_agents: Iterable[AgentSnapshot],
        context: ControllerContext,
    ) -> StepDecision:
        self._register_agent_slots(all_agents)
        local_obs = self.build_local_features(agent, all_agents)
        global_obs = self.build_global_features(all_agents)
        local_tensor = torch.tensor(local_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        global_tensor = torch.tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.training_enabled:
                action_tensor, log_prob, _, _ = self.actor.act(local_tensor)
                action_index = int(action_tensor.item())
                log_prob_value = float(log_prob.item())
                value_estimate = float(self.critic(global_tensor).item())
            else:
                logits, _ = self.actor(local_tensor)
                action_index = self._greedy_action(logits)
                log_prob_value = 0.0
                value_estimate = float(self.critic(global_tensor).item())
        if self.training_enabled:
            self.pending_steps[agent.agent_id] = RolloutStep(
                local_obs=local_obs,
                global_obs=global_obs,
                agent_slot=None,
                action=action_index,
                log_prob=log_prob_value,
                value=value_estimate,
                reward=0.0,
                done=False,
            )
        agent_list = list(all_agents)
        return action_to_decision(self, agent, agent_list, action_index)

    def finish_episode(self) -> PPOStats | None:
        last_value = 0.0
        if self.buffer.steps:
            last_step = self.buffer.steps[-1]
            if last_step.global_obs is not None:
                last_value = PPOTrainer.bootstrap_value_critic(self.critic, last_step.global_obs, self.device)
        stats = self.trainer.update_ctde(
            self.actor,
            self.critic,
            self.buffer,
            last_value,
            CONFIG.rl_gamma,
            CONFIG.rl_gae_lambda,
        )
        self.buffer.clear()
        self.pending_steps.clear()
        return stats
