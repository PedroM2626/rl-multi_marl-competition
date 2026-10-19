from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.rl.actions import GLOBAL_OBS_DIM, LOCAL_OBS_DIM, NUM_ACTIONS, parse_action
from marl_arena.rl.networks import ActorNetwork, CentralizedActorNetwork, CentralizedCriticNetwork


def test_actor_forward_shape_and_explicit_key_sampling() -> None:
    actor = ActorNetwork(LOCAL_OBS_DIM, NUM_ACTIONS, hidden_dim=32, seed=17)
    obs = jnp.ones((5, LOCAL_OBS_DIM), dtype=jnp.float32)
    logits, values = actor.forward(obs)
    assert logits.shape == (5, NUM_ACTIONS)
    assert values.shape == (5,)
    np.testing.assert_array_equal(actor.act(obs, jax.random.PRNGKey(4))[0], actor.act(obs, jax.random.PRNGKey(4))[0])


def test_centralized_network_shapes() -> None:
    actor = CentralizedActorNetwork(GLOBAL_OBS_DIM, agent_slot_dim=3, action_dim=NUM_ACTIONS, hidden_dim=32)
    critic = CentralizedCriticNetwork(GLOBAL_OBS_DIM, hidden_dim=32)
    global_obs = jnp.ones((4, GLOBAL_OBS_DIM), dtype=jnp.float32)
    slots = jnp.eye(3, dtype=jnp.float32)[jnp.asarray([0, 1, 2, 0])]
    assert actor.forward(global_obs, slots).shape == (4, NUM_ACTIONS)
    assert critic.forward(global_obs).shape == (4,)


def test_numpy_checkpoint_round_trip(tmp_path: Path) -> None:
    from marl_arena.rl.actions import load_checkpoint, save_checkpoint

    actor = ActorNetwork(LOCAL_OBS_DIM, NUM_ACTIONS, hidden_dim=32, seed=31)
    path = tmp_path / "policy.npz"
    save_checkpoint(path, {"paradigm": "DTE", "actor": actor.state_dict()})
    payload = load_checkpoint(path)
    restored = ActorNetwork(LOCAL_OBS_DIM, NUM_ACTIONS, hidden_dim=32, seed=99)
    restored.load_state_dict(payload["actor"])
    for name, value in actor.state_dict().items():
        np.testing.assert_array_equal(value, restored.state_dict()[name])


def test_parse_action_roundtrip() -> None:
    for action_index in range(NUM_ACTIONS):
        parsed = parse_action(action_index)
        assert 0 <= parsed.target_index < 4
        assert parsed.shoot == bool(action_index % 2)
