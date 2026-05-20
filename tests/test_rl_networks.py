from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.rl.actions import GLOBAL_OBS_DIM, LOCAL_OBS_DIM, NUM_ACTIONS, parse_action
from marl_arena.rl.networks import ActorNetwork, CentralizedActorNetwork, CentralizedCriticNetwork


def test_actor_forward_shape() -> None:
    actor = ActorNetwork(LOCAL_OBS_DIM, NUM_ACTIONS, hidden_dim=32)
    obs = torch.randn(5, LOCAL_OBS_DIM)
    logits, values = actor(obs)
    assert logits.shape == (5, NUM_ACTIONS)
    assert values.shape == (5,)


def test_centralized_actor_forward_shape() -> None:
    actor = CentralizedActorNetwork(GLOBAL_OBS_DIM, agent_slot_dim=3, action_dim=NUM_ACTIONS, hidden_dim=32)
    global_obs = torch.randn(4, GLOBAL_OBS_DIM)
    slots = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    logits = actor(global_obs, slots)
    assert logits.shape == (4, NUM_ACTIONS)


def test_parse_action_roundtrip() -> None:
    for action_index in range(NUM_ACTIONS):
        parsed = parse_action(action_index)
        assert 0 <= parsed.target_index < 4
