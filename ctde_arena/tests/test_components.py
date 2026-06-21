import sys
from pathlib import Path
import torch

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marl_arena.rl.networks import ValueDecompositionCriticNetwork, CommActorNetwork


def test_value_decomposition_critic():
    device = torch.device("cpu")
    critic = ValueDecompositionCriticNetwork(agent_indices=[0, 1, 2], hidden_dim=64).to(device)
    global_obs = torch.randn(4, 36, device=device)
    val = critic(global_obs)
    assert val.shape == (4,)
    assert not torch.isnan(val).any()


def test_comm_actor():
    device = torch.device("cpu")
    actor = CommActorNetwork(local_obs_dim=8, action_dim=8, msg_dim=4, hidden_dim=64).to(device)
    team_obs = torch.randn(4, 24, device=device)
    agent_slot = torch.zeros(4, 3, device=device)
    agent_slot[:, 0] = 1.0  # agent 1

    logits = actor(team_obs, agent_slot)
    assert logits.shape == (4, 8)

    action, log_prob, entropy = actor.act(team_obs, agent_slot)
    assert action.shape == (4,)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)

    actions = torch.randint(0, 8, (4,), device=device)
    eval_log_prob, eval_entropy = actor.evaluate(team_obs, agent_slot, actions)
    assert eval_log_prob.shape == (4,)
    assert eval_entropy.shape == (4,)
