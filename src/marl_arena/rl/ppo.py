from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from marl_arena.rl.buffer import RolloutBuffer
from marl_arena.rl.networks import (
    JAXNetwork,
    _actor_forward,
    _central_actor_forward,
    _comm_actor_forward,
    _critic_forward,
    _vd_critic_forward,
)


@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float


@partial(jax.jit, static_argnames=(
    "actor_kind", "critic_kind", "actor_agent_indices", "critic_agent_indices", "use_actor_value",
))
def _ppo_step(
    params: dict[str, Any], opt_state: dict[str, Any], observations: jax.Array, global_obs: jax.Array,
    slots: jax.Array, actions: jax.Array, old_log_probs: jax.Array, returns: jax.Array,
    valid_mask: jax.Array,
    advantages: jax.Array, learning_rate: float, clip_eps: float, value_coef: float,
    entropy_coef: float, max_grad_norm: float, actor_kind: str, critic_kind: str | None,
    actor_agent_indices: tuple[int, ...], critic_agent_indices: tuple[int, ...], use_actor_value: bool,
) -> tuple[dict[str, Any], dict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array]:
    def loss_fn(current: dict[str, Any]) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        if actor_kind == "actor":
            logits, actor_values = _actor_forward(current["actor"], observations)
        elif actor_kind == "central_actor":
            logits = _central_actor_forward(current["actor"], global_obs, slots)
            actor_values = jnp.zeros_like(returns)
        else:
            logits = _comm_actor_forward(current["actor"], observations, slots)
            actor_values = jnp.zeros_like(returns)

        log_distribution = jax.nn.log_softmax(logits)
        probabilities = jax.nn.softmax(logits)
        log_probs = jnp.take_along_axis(log_distribution, actions[:, None], axis=-1)[:, 0]
        entropy = -jnp.sum(probabilities * log_distribution, axis=-1)
        ratio = jnp.exp(log_probs - old_log_probs)
        surrogate_1 = ratio * advantages
        surrogate_2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        mask_total = jnp.maximum(valid_mask.sum(), 1.0)
        policy_loss = -jnp.sum(jnp.minimum(surrogate_1, surrogate_2) * valid_mask) / mask_total

        if use_actor_value:
            values = actor_values
        elif critic_kind == "critic":
            values = _critic_forward(current["critic"], global_obs)
        else:
            values = _vd_critic_forward(current["critic"], global_obs, critic_agent_indices)
        value_loss = jnp.sum((values - returns) ** 2 * valid_mask) / mask_total
        entropy_mean = jnp.sum(entropy * valid_mask) / mask_total
        approx_kl = jnp.sum((old_log_probs - log_probs) * valid_mask) / mask_total
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
        return total_loss, (policy_loss, value_loss, entropy_mean, approx_kl)

    (_, stats), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params)
    squared_norm = sum(jnp.sum(leaf * leaf) for leaf in jax.tree_util.tree_leaves(gradients))
    gradient_norm = jnp.sqrt(squared_norm)
    scale = jnp.minimum(1.0, max_grad_norm / (gradient_norm + 1e-6))
    gradients = jax.tree_util.tree_map(lambda grad: grad * scale, gradients)

    count = opt_state["count"] + 1
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    first = jax.tree_util.tree_map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, opt_state["first"], gradients)
    second = jax.tree_util.tree_map(lambda old, grad: beta2 * old + (1.0 - beta2) * (grad * grad), opt_state["second"], gradients)
    first_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1 ** count), first)
    second_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2 ** count), second)
    params = jax.tree_util.tree_map(
        lambda value, m, v: value - learning_rate * m / (jnp.sqrt(v) + epsilon),
        params, first_hat, second_hat,
    )
    opt_state = {"count": count, "first": first, "second": second}
    return params, opt_state, *stats


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
        device: Any,
    ) -> None:
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.actor: JAXNetwork | None = None
        self.critic: JAXNetwork | None = None
        self.opt_state: dict[str, Any] | None = None

    def bind_optimizer(self, actor: JAXNetwork, critic: JAXNetwork | None) -> None:
        self.actor, self.critic = actor, critic
        params = {"actor": actor.params, "critic": critic.params if critic is not None else {}}
        self.opt_state = {
            "count": jnp.asarray(0, dtype=jnp.int32),
            "first": jax.tree_util.tree_map(jnp.zeros_like, params),
            "second": jax.tree_util.tree_map(jnp.zeros_like, params),
        }

    def _update(
        self, actor: JAXNetwork, critic: JAXNetwork | None, buffer: RolloutBuffer,
        last_value: float, gamma: float, gae_lambda: float,
    ) -> PPOStats | None:
        if len(buffer) == 0:
            return None
        if self.actor is None or self.opt_state is None:
            self.bind_optimizer(actor, critic)
        assert self.actor is actor and self.opt_state is not None

        advantages, returns = buffer.compute_returns(gamma, gae_lambda, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        use_global = critic is not None or actor.kind == "central_actor"
        use_slots = actor.kind in {"central_actor", "comm_actor"}
        tensors = buffer.to_arrays(use_global=use_global, use_agent_slot=use_slots)
        sample_count = len(buffer)
        stats = None

        for _ in range(self.ppo_epochs):
            indices = np.arange(sample_count)
            buffer.rng.shuffle(indices)
            for start in range(0, sample_count, self.batch_size):
                valid_indices = indices[start : start + self.batch_size]
                valid_count = len(valid_indices)
                # Bucket and pad the final minibatch so JAX compiles only a small set of shapes.
                # Masked duplicate rows contribute zero to every loss term and gradient.
                bucket_size = min(self.batch_size, 1 << max(0, (valid_count - 1).bit_length()))
                if bucket_size < valid_count:
                    bucket_size = self.batch_size
                batch_indices = np.pad(
                    valid_indices,
                    (0, bucket_size - valid_count),
                    mode="edge",
                )
                valid_mask = jnp.asarray(
                    np.concatenate((np.ones(valid_count, dtype=np.float32), np.zeros(bucket_size - valid_count, dtype=np.float32))),
                    dtype=jnp.float32,
                )
                obs = jnp.asarray(tensors["local_obs"][batch_indices], dtype=jnp.float32)
                if "global_obs" in tensors:
                    global_batch = tensors["global_obs"][batch_indices]
                else:
                    global_batch = np.zeros((len(batch_indices), 0), dtype=np.float32)
                if "agent_slots" in tensors:
                    slot_batch = tensors["agent_slots"][batch_indices]
                else:
                    slot_batch = np.zeros((len(batch_indices), 0), dtype=np.float32)
                global_obs = jnp.asarray(global_batch, dtype=jnp.float32)
                slots = jnp.asarray(slot_batch, dtype=jnp.float32)
                actions = jnp.asarray(tensors["actions"][batch_indices], dtype=jnp.int32)
                old_log_probs = jnp.asarray(tensors["old_log_probs"][batch_indices], dtype=jnp.float32)
                returns_batch = jnp.asarray(returns[batch_indices], dtype=jnp.float32)
                advantages_batch = jnp.asarray(advantages[batch_indices], dtype=jnp.float32)

                params = {"actor": actor.params, "critic": critic.params if critic is not None else {}}
                params, self.opt_state, policy_loss, value_loss, entropy, approx_kl = _ppo_step(
                    params, self.opt_state, obs, global_obs, slots, actions, old_log_probs,
                    returns_batch, valid_mask, advantages_batch, self.learning_rate, self.clip_eps,
                    self.value_coef, self.entropy_coef, self.max_grad_norm, actor.kind,
                    critic.kind if critic is not None else None, actor.agent_indices,
                    critic.agent_indices if critic is not None else (), critic is None,
                )
                actor.params = params["actor"]
                if critic is not None:
                    critic.params = params["critic"]
                values = jax.device_get((policy_loss, value_loss, entropy, approx_kl))
                stats = PPOStats(*(float(value) for value in values))
        return stats

    def update_actor_critic(self, actor: JAXNetwork, buffer: RolloutBuffer, last_value: float, gamma: float, gae_lambda: float) -> PPOStats | None:
        return self._update(actor, None, buffer, last_value, gamma, gae_lambda)

    def update_ctde(self, actor: JAXNetwork, critic: JAXNetwork, buffer: RolloutBuffer, last_value: float, gamma: float, gae_lambda: float) -> PPOStats | None:
        return self._update(actor, critic, buffer, last_value, gamma, gae_lambda)

    def update_cte(self, actor: JAXNetwork, critic: JAXNetwork, buffer: RolloutBuffer, last_value: float, gamma: float, gae_lambda: float) -> PPOStats | None:
        return self._update(actor, critic, buffer, last_value, gamma, gae_lambda)

    def update_ctde_vd(self, actor: JAXNetwork, critic: JAXNetwork, buffer: RolloutBuffer, last_value: float, gamma: float, gae_lambda: float) -> PPOStats | None:
        return self._update(actor, critic, buffer, last_value, gamma, gae_lambda)

    def update_ctde_comm(self, actor: JAXNetwork, critic: JAXNetwork, buffer: RolloutBuffer, last_value: float, gamma: float, gae_lambda: float) -> PPOStats | None:
        return self._update(actor, critic, buffer, last_value, gamma, gae_lambda)

    @staticmethod
    def bootstrap_value_critic(critic: JAXNetwork, global_obs: np.ndarray, device: Any) -> float:
        batch = jnp.asarray(global_obs, dtype=jnp.float32).reshape(1, -1)
        return float(jax.device_get(critic.forward(batch)[0]))

    @staticmethod
    def bootstrap_value_actor(actor: JAXNetwork, local_obs: np.ndarray, device: Any) -> float:
        batch = jnp.asarray(local_obs, dtype=jnp.float32).reshape(1, -1)
        return float(jax.device_get(actor.forward(batch)[1][0]))
