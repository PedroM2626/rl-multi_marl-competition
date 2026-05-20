from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from marl_arena.config import CONFIG
from marl_arena.controllers.base import BaseTeamController, ControllerContext
from marl_arena.models import AgentSnapshot, StepDecision, TransitionRecord
from marl_arena.rl.actions import (
    GLOBAL_OBS_DIM,
    LOCAL_OBS_DIM,
    NUM_ACTIONS,
    action_to_decision,
    load_checkpoint,
    save_checkpoint,
)
from marl_arena.rl.buffer import RolloutBuffer, RolloutStep
from marl_arena.rl.networks import ActorNetwork, CentralizedActorNetwork, CentralizedCriticNetwork
from marl_arena.rl.ppo import PPOStats, PPOTrainer

TEAM_PARADIGMS = {
    "Equipe 1": "CTE",
    "Equipe 2": "DTE",
    "Equipe 3": "CTDE",
}


def resolve_device() -> torch.device:
    requested = CONFIG.rl_device.strip().lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RLTeamController(BaseTeamController):
    def __init__(self, team_name: str, paradigm: str, rng_seed: int, device: torch.device) -> None:
        super().__init__(team_name, rng_seed)
        self.paradigm = paradigm
        self.paradigm_name = paradigm
        self.device = device
        self.buffer = RolloutBuffer()
        self.pending_steps: Dict[str, RolloutStep] = {}
        self.training_enabled = True
        self._agent_slot_map: Dict[str, int] = {}
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
        self.actor: ActorNetwork | CentralizedActorNetwork | None = None
        self.critic: CentralizedCriticNetwork | None = None
        self._build_networks()
        self.trainer.bind_optimizer(self._trainable_parameters())
        self._load_if_exists()

    def _build_networks(self) -> None:
        hidden = CONFIG.rl_hidden_dim
        if self.paradigm == "DTE":
            self.actor = ActorNetwork(LOCAL_OBS_DIM, NUM_ACTIONS, hidden).to(self.device)
            return
        if self.paradigm == "CTDE":
            self.actor = ActorNetwork(LOCAL_OBS_DIM, NUM_ACTIONS, hidden).to(self.device)
            self.critic = CentralizedCriticNetwork(GLOBAL_OBS_DIM, hidden).to(self.device)
            return
        self.actor = CentralizedActorNetwork(
            GLOBAL_OBS_DIM,
            agent_slot_dim=3,
            action_dim=NUM_ACTIONS,
            hidden_dim=hidden,
        ).to(self.device)
        self.critic = CentralizedCriticNetwork(GLOBAL_OBS_DIM, hidden).to(self.device)

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        if self.actor is not None:
            params.extend(self.actor.parameters())
        if self.critic is not None:
            params.extend(self.critic.parameters())
        return params

    def _checkpoint_path(self, directory: Path | None = None) -> Path:
        root = CONFIG.rl_checkpoint_dir if directory is None else directory
        suffix = self.paradigm.lower()
        slug = self.team_name.lower().replace(" ", "_")
        return root / f"{slug}_{suffix}.pt"

    def _load_if_exists(self) -> None:
        path = self._checkpoint_path()
        if not path.exists():
            return
        payload = load_checkpoint(path, self.device)
        if self.actor is not None:
            self.actor.load_state_dict(payload["actor"])
        if self.critic is not None and "critic" in payload:
            self.critic.load_state_dict(payload["critic"])

    def load(self, directory: Path) -> None:
        payload = load_checkpoint(self._checkpoint_path(directory), self.device)
        if self.actor is not None:
            self.actor.load_state_dict(payload["actor"])
        if self.critic is not None and "critic" in payload:
            self.critic.load_state_dict(payload["critic"])

    def save(self, directory: Path) -> None:
        payload: dict[str, object] = {"paradigm": self.paradigm, "actor": self.actor.state_dict()}
        if self.critic is not None:
            payload["critic"] = self.critic.state_dict()
        save_checkpoint(self._checkpoint_path(directory), payload)

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

    def _register_agent_slots(self, all_agents: Iterable[AgentSnapshot]) -> None:
        allies = sorted(
            [agent for agent in all_agents if agent.team_name == self.team_name],
            key=lambda item: item.agent_id,
        )
        for index, agent in enumerate(allies):
            self._agent_slot_map[agent.agent_id] = index

    def _agent_slot_vector(self, agent_id: str) -> np.ndarray:
        slot_index = self._agent_slot_map.get(agent_id, 0)
        vector = np.zeros(3, dtype=float)
        vector[slot_index] = 1.0
        return vector

    def _record_step(
        self,
        agent_id: str,
        action_index: int,
        log_prob: float,
        value: float,
        local_obs: np.ndarray,
        global_obs: np.ndarray | None = None,
        agent_slot: np.ndarray | None = None,
    ) -> None:
        if not self.training_enabled:
            return
        self.pending_steps[agent_id] = RolloutStep(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_slot=agent_slot,
            action=action_index,
            log_prob=log_prob,
            value=value,
            reward=0.0,
            done=False,
        )

    def decide(
        self,
        agent: AgentSnapshot,
        all_agents: Iterable[AgentSnapshot],
        context: ControllerContext,
    ) -> StepDecision:
        self._register_agent_slots(all_agents)
        agent_list = list(all_agents)
        local_obs = self.build_local_features(agent, all_agents)

        if self.paradigm == "CTE":
            assert isinstance(self.actor, CentralizedActorNetwork)
            assert self.critic is not None
            global_obs = self.build_global_features(all_agents)
            agent_slot = self._agent_slot_vector(agent.agent_id)
            global_tensor = torch.tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            slot_tensor = torch.tensor(agent_slot, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                if self.training_enabled:
                    action_tensor, log_prob, _ = self.actor.act(global_tensor, slot_tensor)
                    action_index = int(action_tensor.item())
                    log_prob_value = float(log_prob.item())
                    value_estimate = float(self.critic(global_tensor).item())
                else:
                    logits = self.actor(global_tensor, slot_tensor)
                    action_index = self._greedy_action(logits)
                    log_prob_value = 0.0
                    value_estimate = float(self.critic(global_tensor).item())
            self._record_step(
                agent.agent_id,
                action_index,
                log_prob_value,
                value_estimate,
                local_obs,
                global_obs,
                agent_slot,
            )
            return action_to_decision(self, agent, agent_list, action_index)

        if self.paradigm == "CTDE":
            assert isinstance(self.actor, ActorNetwork)
            assert self.critic is not None
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
            self._record_step(
                agent.agent_id,
                action_index,
                log_prob_value,
                value_estimate,
                local_obs,
                global_obs,
            )
            return action_to_decision(self, agent, agent_list, action_index)

        assert isinstance(self.actor, ActorNetwork)
        obs_tensor = torch.tensor(local_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.training_enabled:
                action_tensor, log_prob, _, value = self.actor.act(obs_tensor)
                action_index = int(action_tensor.item())
                log_prob_value = float(log_prob.item())
                value_estimate = float(value.item())
            else:
                logits, value = self.actor(obs_tensor)
                action_index = self._greedy_action(logits)
                log_prob_value = 0.0
                value_estimate = float(value.item())
        self._record_step(agent.agent_id, action_index, log_prob_value, value_estimate, local_obs)
        return action_to_decision(self, agent, agent_list, action_index)

    def finish_episode(self) -> PPOStats | None:
        last_value = 0.0
        if self.buffer.steps:
            last_step = self.buffer.steps[-1]
            if self.paradigm == "DTE":
                assert isinstance(self.actor, ActorNetwork)
                last_value = PPOTrainer.bootstrap_value_actor(self.actor, last_step.local_obs, self.device)
            elif last_step.global_obs is not None and self.critic is not None:
                last_value = PPOTrainer.bootstrap_value_critic(self.critic, last_step.global_obs, self.device)

        stats: PPOStats | None = None
        if self.paradigm == "DTE" and isinstance(self.actor, ActorNetwork):
            stats = self.trainer.update_actor_critic(
                self.actor,
                self.buffer,
                last_value,
                CONFIG.rl_gamma,
                CONFIG.rl_gae_lambda,
            )
        elif self.paradigm == "CTDE" and isinstance(self.actor, ActorNetwork) and self.critic is not None:
            stats = self.trainer.update_ctde(
                self.actor,
                self.critic,
                self.buffer,
                last_value,
                CONFIG.rl_gamma,
                CONFIG.rl_gae_lambda,
            )
        elif isinstance(self.actor, CentralizedActorNetwork) and self.critic is not None:
            stats = self.trainer.update_cte(
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


def build_controllers(seed: int) -> dict[str, RLTeamController]:
    device = resolve_device()
    return {
        team_name: RLTeamController(team_name, TEAM_PARADIGMS[team_name], seed + offset, device)
        for team_name, offset in (
            ("Equipe 1", 11),
            ("Equipe 2", 23),
            ("Equipe 3", 37),
        )
    }


def finish_rl_episode(controllers: dict[str, RLTeamController]) -> dict[str, object]:
    stats: dict[str, object] = {}
    for team_name, controller in controllers.items():
        result = controller.finish_episode()
        if result is not None:
            stats[team_name] = result
    return stats


def save_rl_checkpoints(controllers: dict[str, RLTeamController]) -> None:
    for controller in controllers.values():
        controller.save(CONFIG.rl_checkpoint_dir)


def set_rl_training(controllers: dict[str, RLTeamController], enabled: bool) -> None:
    for controller in controllers.values():
        controller.set_training(enabled)
