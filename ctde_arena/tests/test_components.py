import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marl_arena.rl.actions import GLOBAL_OBS_DIM, LOCAL_OBS_DIM, NUM_ACTIONS
from marl_arena.rl.networks import ValueDecompositionCriticNetwork, CommActorNetwork


def test_value_decomposition_critic():
    critic = ValueDecompositionCriticNetwork(agent_indices=[0, 1, 2], hidden_dim=64)
    assert GLOBAL_OBS_DIM == 45, "VD critic slices five components per agent"
    global_obs = jnp.ones((4, GLOBAL_OBS_DIM), dtype=jnp.float32)
    val = critic.forward(global_obs)
    assert val.shape == (4,)
    assert np.isfinite(np.asarray(val)).all()


def test_comm_actor():
    actor = CommActorNetwork(local_obs_dim=LOCAL_OBS_DIM, action_dim=NUM_ACTIONS, msg_dim=4, hidden_dim=64)
    team_obs = jnp.ones((4, 3 * LOCAL_OBS_DIM), dtype=jnp.float32)
    agent_slot = jnp.tile(jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32), (4, 1))

    logits = actor.forward(team_obs, agent_slot)
    assert logits.shape == (4, NUM_ACTIONS)

    action, log_prob, entropy = actor.act(team_obs, agent_slot, jax.random.PRNGKey(1))
    assert action.shape == (4,)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)

    actions = jnp.arange(4, dtype=jnp.int32) % NUM_ACTIONS
    eval_log_prob, eval_entropy = actor.evaluate(team_obs, agent_slot, actions)
    assert eval_log_prob.shape == (4,)
    assert eval_entropy.shape == (4,)
