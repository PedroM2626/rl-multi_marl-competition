from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List

from marl_arena.config import CONFIG

import jax
import numpy as np

from marl_arena.controllers.base import BaseTeamController, ControllerContext
from marl_arena.controllers.batching import decide_many as _decide_many
from marl_arena.models import AgentSnapshot, StepDecision, TransitionRecord
from marl_arena.rl.actions import (
    GLOBAL_OBS_DIM,
    LOCAL_OBS_DIM,
    NUM_ACTIONS,
    load_checkpoint,
    save_checkpoint,
)
from marl_arena.rl.buffer import RolloutBuffer, RolloutStep
from marl_arena.rl.networks import (
    ActorNetwork,
    CentralizedActorNetwork,
    CentralizedCriticNetwork,
    CommActorNetwork,
    JAXNetwork,
    ValueDecompositionCriticNetwork,
)
from marl_arena.rl.ppo import PPOStats, PPOTrainer
from marl_arena.systems.match_variant import paradigm_assignment


def resolve_device():
    """Resolve the requested JAX device, with CPU as the portable fallback."""
    requested = CONFIG.rl_device.strip().lower()
    if requested in {"cuda", "gpu"}:
        try:
            devices = jax.devices("gpu")
            if devices:
                return devices[0]
        except RuntimeError:
            pass
        warnings.warn("JAX GPU backend is unavailable; using CPU.", RuntimeWarning)
    elif requested == "tpu":
        try:
            devices = jax.devices("tpu")
            if devices:
                return devices[0]
        except RuntimeError:
            pass
        warnings.warn("JAX TPU backend is unavailable; using CPU.", RuntimeWarning)
    return jax.devices("cpu")[0]


class RLTeamController(BaseTeamController):
    def __init__(
        self,
        team_name: str,
        paradigm: str,
        rng_seed: int,
        device,
        load_checkpoints: bool = True,
    ) -> None:
        super().__init__(team_name, rng_seed)
        self.paradigm = paradigm
        self.paradigm_name = paradigm
        self.device = device
        self.action_rng = np.random.default_rng(rng_seed)
        self.buffer = RolloutBuffer(rng=np.random.default_rng(rng_seed))
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
        self.actor: JAXNetwork | None = None
        self.critic: JAXNetwork | None = None
        self._build_networks(rng_seed)
        self.trainer.bind_optimizer(self.actor, self.critic)
        if load_checkpoints:
            self._load_if_exists()

    def _build_networks(self, seed: int) -> None:
        hidden = CONFIG.rl_hidden_dim
        if self.paradigm == "CTDE-VD":
            team_indices = {"Team 1": [0, 1, 2], "Team 2": [3, 4, 5], "Team 3": [6, 7, 8]}
            self.actor = ActorNetwork(LOCAL_OBS_DIM, NUM_ACTIONS, hidden, seed=seed).to(self.device)
            self.critic = ValueDecompositionCriticNetwork(
                team_indices.get(self.team_name, [0, 1, 2]), hidden, seed=seed + 1
            ).to(self.device)
        elif self.paradigm == "CTDE-Comm":
            self.actor = CommActorNetwork(
                LOCAL_OBS_DIM, NUM_ACTIONS, msg_dim=4, hidden_dim=hidden, seed=seed
            ).to(self.device)
            self.critic = CentralizedCriticNetwork(GLOBAL_OBS_DIM, hidden, seed=seed + 1).to(self.device)
        elif self.paradigm in {"CTDE", "CTDE-CAC"}:
            self.actor = ActorNetwork(LOCAL_OBS_DIM, NUM_ACTIONS, hidden, seed=seed).to(self.device)
            self.critic = CentralizedCriticNetwork(GLOBAL_OBS_DIM, hidden, seed=seed + 1).to(self.device)
        elif self.paradigm == "CTE":
            self.actor = CentralizedActorNetwork(
                GLOBAL_OBS_DIM, agent_slot_dim=3, action_dim=NUM_ACTIONS,
                hidden_dim=hidden, seed=seed,
            ).to(self.device)
            self.critic = CentralizedCriticNetwork(GLOBAL_OBS_DIM, hidden, seed=seed + 1).to(self.device)
        else:  # DTE and defensive fallback
            self.actor = ActorNetwork(LOCAL_OBS_DIM, NUM_ACTIONS, hidden, seed=seed).to(self.device)

    def _checkpoint_path(self, directory: Path | None = None) -> Path:
        root = CONFIG.rl_checkpoint_dir if directory is None else directory
        suffix = self.paradigm.lower()
        slug = self.team_name.lower().replace(" ", "_")
        return root / f"{slug}_{suffix}.npz"

    def _load_if_exists(self) -> None:
        path = self._checkpoint_path()
        if not path.exists():
            legacy = path.with_suffix(".pt")
            if legacy.exists():
                warnings.warn(
                    f"Found legacy PyTorch checkpoint {legacy.name}; convert it with "
                    "scripts/convert_torch_checkpoints.py to use this JAX runtime."
                )
                return
            siblings = sorted(p.name for p in path.parent.glob("*.npz")) if path.parent.is_dir() else []
            warnings.warn(
                f"No checkpoint at {path.name}; {self.team_name} ({self.paradigm}) starts from random "
                f"initialisation. {path.parent} holds {siblings or 'no policies'}."
            )
            return
        payload = load_checkpoint(path)
        if self.actor is not None:
            try:
                self.actor.load_state_dict(payload["actor"])
                if self.critic is not None and payload.get("critic") is not None:
                    self.critic.load_state_dict(payload["critic"])
            except ValueError as exc:
                warnings.warn(f"Ignoring incompatible checkpoint {path.name}: {exc}")

    def load(self, directory: Path) -> None:
        payload = load_checkpoint(self._checkpoint_path(directory))
        if self.actor is not None:
            self.actor.load_state_dict(payload["actor"])
        if self.critic is not None and payload.get("critic") is not None:
            self.critic.load_state_dict(payload["critic"])

    def save(self, directory: Path) -> None:
        assert self.actor is not None
        payload: dict[str, object] = {"paradigm": self.paradigm, "actor": self.actor.state_dict()}
        if self.critic is not None:
            payload["critic"] = self.critic.state_dict()
        save_checkpoint(self._checkpoint_path(directory), payload)

    def set_training(self, enabled: bool) -> None:
        self.training_enabled = enabled

    def update(self, transitions: List[TransitionRecord]) -> None:
        for transition in transitions:
            if transition.team_name != self.team_name:
                continue
            pending = self.pending_steps.pop(transition.agent_id, None)
            if pending is None:
                continue
            pending.reward = transition.reward
            pending.done = transition.done
            self.buffer.append(pending)

    def _greedy_action(self, logits) -> int:
        return int(np.asarray(logits).argmax(axis=-1).reshape(-1)[0])

    def _eval_action(self, logits) -> int:
        """Use argmax for the published zero-temperature readout, or a seeded JAX sample."""
        temperature = CONFIG.eval_temperature
        values = np.asarray(logits, dtype=np.float64)
        if temperature <= 0.0:
            return self._greedy_action(values)
        shifted = values / temperature
        probabilities = np.exp(shifted - shifted.max())
        probabilities /= probabilities.sum()
        return int(self.action_rng.choice(len(probabilities), p=probabilities))

    def _register_agent_slots(self, all_agents: Iterable[AgentSnapshot]) -> None:
        allies = sorted(
            [agent for agent in all_agents if agent.team_name == self.team_name],
            key=lambda item: item.agent_id,
        )
        for index, agent in enumerate(allies):
            self._agent_slot_map[agent.agent_id] = index

    def get_team_agents(self, agents: Iterable[AgentSnapshot]) -> list[AgentSnapshot]:
        return [agent for agent in agents if agent.team_name == self.team_name]

    def _agent_slot_vector(self, agent_id: str) -> np.ndarray:
        slot_index = self._agent_slot_map.get(agent_id, 0)
        vector = np.zeros(3, dtype=np.float32)
        if slot_index < len(vector):
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
        return self.decide_many([agent], all_agents, context)[agent.agent_id]

    def decide_many(
        self,
        agents: list[AgentSnapshot],
        all_agents: Iterable[AgentSnapshot],
        context: ControllerContext,
    ) -> dict[str, StepDecision]:
        return _decide_many(self, agents, all_agents, context)

    def finish_episode(self) -> PPOStats | None:
        assert self.actor is not None
        last_value = 0.0
        if self.buffer.steps:
            last_step = self.buffer.steps[-1]
            if self.critic is not None and last_step.global_obs is not None:
                last_value = PPOTrainer.bootstrap_value_critic(self.critic, last_step.global_obs, self.device)
            elif self.critic is None:
                last_value = PPOTrainer.bootstrap_value_actor(self.actor, last_step.local_obs, self.device)

        stats = None
        if self.critic is None:
            stats = self.trainer.update_actor_critic(
                self.actor, self.buffer, last_value, CONFIG.rl_gamma, CONFIG.rl_gae_lambda
            )
        elif self.critic.kind == "vd_critic":
            stats = self.trainer.update_ctde_vd(
                self.actor, self.critic, self.buffer, last_value, CONFIG.rl_gamma, CONFIG.rl_gae_lambda
            )
        elif self.actor.kind == "comm_actor":
            stats = self.trainer.update_ctde_comm(
                self.actor, self.critic, self.buffer, last_value, CONFIG.rl_gamma, CONFIG.rl_gae_lambda
            )
        elif self.actor.kind == "central_actor":
            stats = self.trainer.update_cte(
                self.actor, self.critic, self.buffer, last_value, CONFIG.rl_gamma, CONFIG.rl_gae_lambda
            )
        else:
            stats = self.trainer.update_ctde(
                self.actor, self.critic, self.buffer, last_value, CONFIG.rl_gamma, CONFIG.rl_gae_lambda
            )
        self.buffer.clear()
        self.pending_steps.clear()
        return stats


def build_controllers(
    seed: int, load_checkpoints: bool = True, paradigm_rotation: int = 0
) -> dict[str, RLTeamController]:
    device = resolve_device()
    assignment = paradigm_assignment(paradigm_rotation)
    return {
        team_name: RLTeamController(
            team_name, assignment[team_name], seed + offset, device, load_checkpoints
        )
        for team_name, offset in (("Team 1", 11), ("Team 2", 23), ("Team 3", 37))
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
