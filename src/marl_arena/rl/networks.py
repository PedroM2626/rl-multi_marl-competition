"""JAX policies and critics used by the arena's PPO trainer."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _dense_init(key: jax.Array, input_dim: int, output_dim: int) -> dict[str, jax.Array]:
    # Match torch.nn.Linear's fan-in uniform range while using explicit JAX keys.
    bound = 1.0 / np.sqrt(input_dim)
    weight_key, bias_key = jax.random.split(key)
    return {
        "kernel": jax.random.uniform(weight_key, (input_dim, output_dim), minval=-bound, maxval=bound),
        "bias": jax.random.uniform(bias_key, (output_dim,), minval=-bound, maxval=bound),
    }


def _mlp_init(key: jax.Array, input_dim: int, output_dim: int, hidden_dim: int) -> tuple[dict[str, jax.Array], ...]:
    dimensions = (input_dim, hidden_dim, hidden_dim, output_dim)
    keys = jax.random.split(key, len(dimensions) - 1)
    return tuple(_dense_init(k, before, after) for k, before, after in zip(keys, dimensions, dimensions[1:]))


def _mlp(layers: tuple[dict[str, jax.Array], ...], inputs: jax.Array) -> jax.Array:
    value = inputs
    for index, layer in enumerate(layers):
        value = value @ layer["kernel"] + layer["bias"]
        if index + 1 < len(layers):
            value = jnp.tanh(value)
    return value


@jax.jit
def _actor_forward(params: dict[str, Any], observations: jax.Array) -> tuple[jax.Array, jax.Array]:
    features = _mlp(params["backbone"], observations)
    logits = features @ params["policy_head"]["kernel"] + params["policy_head"]["bias"]
    value = (features @ params["value_head"]["kernel"] + params["value_head"]["bias"])[..., 0]
    return logits, value


@jax.jit
def _central_actor_forward(params: dict[str, Any], global_obs: jax.Array, slots: jax.Array) -> jax.Array:
    features = _mlp(params["backbone"], jnp.concatenate((global_obs, slots), axis=-1))
    return features @ params["policy_head"]["kernel"] + params["policy_head"]["bias"]


@jax.jit
def _critic_forward(params: dict[str, Any], global_obs: jax.Array) -> jax.Array:
    features = _mlp(params["backbone"], global_obs)
    return (features @ params["value_head"]["kernel"] + params["value_head"]["bias"])[..., 0]


@partial(jax.jit, static_argnums=(2,))
def _vd_critic_forward(params: dict[str, Any], global_obs: jax.Array, agent_indices: tuple[int, ...]) -> jax.Array:
    total = jnp.zeros((global_obs.shape[0],), dtype=global_obs.dtype)
    for index in agent_indices:
        features = global_obs[:, 5 * index : 5 * (index + 1)]
        total = total + _mlp(params["agent_critic"], features)[..., 0]
    return total


@jax.jit
def _comm_actor_forward(params: dict[str, Any], team_obs: jax.Array, slots: jax.Array) -> jax.Array:
    obs_dim = 9
    observations = (team_obs[:, :obs_dim], team_obs[:, obs_dim : 2 * obs_dim], team_obs[:, 2 * obs_dim :])
    messages = tuple(_mlp(params["msg_net"], item) for item in observations)
    contexts = (
        0.5 * (messages[1] + messages[2]),
        0.5 * (messages[0] + messages[2]),
        0.5 * (messages[0] + messages[1]),
    )
    selected_obs = sum(slots[:, i : i + 1] * observations[i] for i in range(3))
    selected_context = sum(slots[:, i : i + 1] * contexts[i] for i in range(3))
    return _mlp(params["policy_head"], jnp.concatenate((selected_obs, selected_context), axis=-1))


def _export_layers(prefix: str, layers: tuple[dict[str, jax.Array], ...]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for index, layer in enumerate(layers):
        module_index = 2 * index
        result[f"{prefix}.{module_index}.weight"] = np.asarray(jax.device_get(layer["kernel"])).T.copy()
        result[f"{prefix}.{module_index}.bias"] = np.asarray(jax.device_get(layer["bias"])).copy()
    return result


def _export_dense(prefix: str, layer: dict[str, jax.Array]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.weight": np.asarray(jax.device_get(layer["kernel"])).T.copy(),
        f"{prefix}.bias": np.asarray(jax.device_get(layer["bias"])).copy(),
    }


class JAXNetwork:
    def __init__(self, kind: str, seed: int = 0, agent_indices: list[int] | None = None) -> None:
        self.kind = kind
        self.agent_indices = tuple(agent_indices or ())
        keys = iter(jax.random.split(jax.random.PRNGKey(seed), 8))
        if kind == "actor":
            self.params = {
                "backbone": _mlp_init(next(keys), self.input_dim, self.hidden_dim, self.hidden_dim),
                "policy_head": _dense_init(next(keys), self.hidden_dim, self.action_dim),
                "value_head": _dense_init(next(keys), self.hidden_dim, 1),
            }
        elif kind == "central_actor":
            self.params = {
                "backbone": _mlp_init(next(keys), self.input_dim + self.slot_dim, self.hidden_dim, self.hidden_dim),
                "policy_head": _dense_init(next(keys), self.hidden_dim, self.action_dim),
            }
        elif kind == "critic":
            self.params = {
                "backbone": _mlp_init(next(keys), self.input_dim, self.hidden_dim, self.hidden_dim),
                "value_head": _dense_init(next(keys), self.hidden_dim, 1),
            }
        elif kind == "vd_critic":
            self.params = {"agent_critic": _mlp_init(next(keys), 5, 1, self.hidden_dim)}
        elif kind == "comm_actor":
            self.params = {
                "msg_net": _mlp_init(next(keys), self.input_dim, 4, self.hidden_dim),
                "policy_head": _mlp_init(next(keys), self.input_dim + 4, self.action_dim, self.hidden_dim),
            }
        else:
            raise ValueError(f"unknown JAX network kind: {kind}")

    def to(self, device: Any) -> "JAXNetwork":
        if device is not None:
            self.params = jax.device_put(self.params, device)
        return self

    def forward(self, *inputs: Any) -> Any:
        arrays = tuple(jnp.asarray(item, dtype=jnp.float32) for item in inputs)
        if self.kind == "actor":
            return _actor_forward(self.params, arrays[0])
        if self.kind == "central_actor":
            return _central_actor_forward(self.params, arrays[0], arrays[1])
        if self.kind == "critic":
            return _critic_forward(self.params, arrays[0])
        if self.kind == "vd_critic":
            return _vd_critic_forward(self.params, arrays[0], self.agent_indices)
        if self.kind == "comm_actor":
            return _comm_actor_forward(self.params, arrays[0], arrays[1])
        raise ValueError(self.kind)

    def __call__(self, *inputs: Any) -> Any:
        return self.forward(*inputs)

    def state_dict(self) -> dict[str, np.ndarray]:
        if self.kind in {"actor", "central_actor", "critic"}:
            result = _export_layers("backbone", self.params["backbone"])
            if "policy_head" in self.params:
                result.update(_export_dense("policy_head", self.params["policy_head"]))
            if "value_head" in self.params:
                result.update(_export_dense("value_head", self.params["value_head"]))
            return result
        if self.kind == "vd_critic":
            return _export_layers("agent_critic", self.params["agent_critic"])
        result = _export_layers("msg_net", self.params["msg_net"])
        result.update(_export_layers("policy_head", self.params["policy_head"]))
        return result

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()
        if set(state) != set(expected):
            missing = sorted(set(expected) - set(state))
            unexpected = sorted(set(state) - set(expected))
            raise ValueError(f"checkpoint parameter names mismatch; missing={missing}, unexpected={unexpected}")
        for name, expected_value in expected.items():
            if np.asarray(state[name]).shape != expected_value.shape:
                raise ValueError(f"checkpoint shape mismatch at {name}: {np.asarray(state[name]).shape} vs {expected_value.shape}")

        params = dict(self.params)

        def import_layers(prefix: str, current: tuple[dict[str, jax.Array], ...]) -> tuple[dict[str, jax.Array], ...]:
            return tuple({
                "kernel": jnp.asarray(state[f"{prefix}.{2 * index}.weight"], dtype=jnp.float32).T,
                "bias": jnp.asarray(state[f"{prefix}.{2 * index}.bias"], dtype=jnp.float32),
            } for index, _ in enumerate(current))

        def import_dense(prefix: str) -> dict[str, jax.Array]:
            return {
                "kernel": jnp.asarray(state[f"{prefix}.weight"], dtype=jnp.float32).T,
                "bias": jnp.asarray(state[f"{prefix}.bias"], dtype=jnp.float32),
            }

        if self.kind in {"actor", "central_actor", "critic"}:
            params["backbone"] = import_layers("backbone", params["backbone"])
            for name in ("policy_head", "value_head"):
                if name in params:
                    params[name] = import_dense(name)
        elif self.kind == "vd_critic":
            params["agent_critic"] = import_layers("agent_critic", params["agent_critic"])
        else:
            params["msg_net"] = import_layers("msg_net", params["msg_net"])
            params["policy_head"] = import_layers("policy_head", params["policy_head"])
        self.params = params

    def parameters(self) -> list[jax.Array]:
        return list(jax.tree_util.tree_leaves(self.params))


class ActorNetwork(JAXNetwork):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, seed: int = 0) -> None:
        self.input_dim, self.action_dim, self.hidden_dim, self.slot_dim = obs_dim, action_dim, hidden_dim, 0
        super().__init__("actor", seed)

    def act(self, obs: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        logits, value = self.forward(obs)
        log_probs = jax.nn.log_softmax(logits)
        action = jax.random.categorical(key, logits, axis=-1)
        selected = jnp.take_along_axis(log_probs, action[..., None], axis=-1)[..., 0]
        entropy = -jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1)
        return action, selected, entropy, value

    def evaluate(self, obs: jax.Array, actions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        logits, value = self.forward(obs)
        log_probs = jax.nn.log_softmax(logits)
        selected = jnp.take_along_axis(log_probs, jnp.asarray(actions)[..., None], axis=-1)[..., 0]
        entropy = -jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1)
        return selected, entropy, value


class CentralizedActorNetwork(JAXNetwork):
    def __init__(self, global_obs_dim: int, agent_slot_dim: int, action_dim: int, hidden_dim: int = 128, seed: int = 0) -> None:
        self.input_dim, self.slot_dim, self.action_dim, self.hidden_dim = global_obs_dim, agent_slot_dim, action_dim, hidden_dim
        super().__init__("central_actor", seed)

    def evaluate(self, global_obs: jax.Array, agent_slot: jax.Array, actions: jax.Array) -> tuple[jax.Array, jax.Array]:
        logits = self.forward(global_obs, agent_slot)
        log_probs = jax.nn.log_softmax(logits)
        selected = jnp.take_along_axis(log_probs, jnp.asarray(actions)[..., None], axis=-1)[..., 0]
        entropy = -jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1)
        return selected, entropy


class CentralizedCriticNetwork(JAXNetwork):
    def __init__(self, global_obs_dim: int, hidden_dim: int = 128, seed: int = 0) -> None:
        self.input_dim, self.action_dim, self.hidden_dim, self.slot_dim = global_obs_dim, 0, hidden_dim, 0
        super().__init__("critic", seed)


class ValueDecompositionCriticNetwork(JAXNetwork):
    def __init__(self, agent_indices: list[int], hidden_dim: int = 128, seed: int = 0) -> None:
        self.input_dim, self.action_dim, self.hidden_dim, self.slot_dim = 5, 0, hidden_dim, 0
        super().__init__("vd_critic", seed, agent_indices)


class CommActorNetwork(JAXNetwork):
    def __init__(self, local_obs_dim: int, action_dim: int, msg_dim: int = 4, hidden_dim: int = 128, seed: int = 0) -> None:
        if local_obs_dim != 9 or msg_dim != 4:
            raise ValueError("the arena communication policy expects 9 local features and 4 message features")
        self.input_dim, self.action_dim, self.hidden_dim, self.slot_dim = local_obs_dim, action_dim, hidden_dim, 3
        self.msg_dim = msg_dim
        super().__init__("comm_actor", seed)

    def act(
        self, team_obs: jax.Array, agent_slot: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        logits = self.forward(team_obs, agent_slot)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        action = jax.random.categorical(key, logits, axis=-1)
        selected = jnp.take_along_axis(log_probs, action[..., None], axis=-1)[..., 0]
        entropy = -jnp.sum(jax.nn.softmax(logits, axis=-1) * log_probs, axis=-1)
        return action, selected, entropy

    def evaluate(self, team_obs: jax.Array, agent_slot: jax.Array, actions: jax.Array) -> tuple[jax.Array, jax.Array]:
        logits = self.forward(team_obs, agent_slot)
        log_probs = jax.nn.log_softmax(logits)
        selected = jnp.take_along_axis(log_probs, jnp.asarray(actions)[..., None], axis=-1)[..., 0]
        entropy = -jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1)
        return selected, entropy
