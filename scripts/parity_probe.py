"""Parity probe: does a converted .npz load as the same policy as the .pt it came from?

Run once with --kind torch against the reference worktree, once with --kind jax against the ported tree;
both print the logits of one fixed observation, and the diff between the two outputs is the number the
documentation has to state. This is the instrument behind the max-|Δ logit| table in A-45
(docs/11_code_audit.md#the-runtime-swap-orphaned-the-committed-checkpoint-archive) and the 4e-07 figure
§ 9.6 quotes; it was renamed from a root-level `parity_probe_tmp.py` that nothing referenced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=Path, required=True)
parser.add_argument("--kind", choices=["torch", "jax"], required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--head", choices=["dte", "cte", "ctde"], required=True)
parser.add_argument("--obs-dim", type=int, default=None)
args = parser.parse_args()

sys.path.insert(0, str(args.src))

from marl_arena.rl.actions import LOCAL_OBS_DIM, GLOBAL_OBS_DIM, NUM_ACTIONS  # noqa: E402

obs_dim = args.obs_dim or (GLOBAL_OBS_DIM if args.head == "cte" else LOCAL_OBS_DIM)
obs = (np.linspace(-1.0, 1.0, obs_dim, dtype=np.float32)).reshape(1, -1)

if args.kind == "torch":
    import torch

    from marl_arena.rl.networks import ActorNetwork, CentralizedActorNetwork, CentralizedCriticNetwork

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.head == "cte":
        actor = CentralizedActorNetwork(GLOBAL_OBS_DIM, agent_slot_dim=3, action_dim=NUM_ACTIONS, hidden_dim=128)
        actor.load_state_dict(payload["actor"])
        slot = np.zeros((1, 3), dtype=np.float32)
        slot[0, 0] = 1.0
        with torch.no_grad():
            logits = actor(torch.from_numpy(obs), torch.from_numpy(slot))
        logits = logits.detach().numpy()
    else:
        actor = ActorNetwork(obs_dim, NUM_ACTIONS, 128)
        actor.load_state_dict(payload["actor"])
        with torch.no_grad():
            out = actor(torch.from_numpy(obs))
        logits = (out[0] if isinstance(out, tuple) else out).detach().numpy()
    print("torch logits:", np.array2string(logits, precision=9, separator=","))
else:
    from marl_arena.rl.actions import load_checkpoint
    from marl_arena.rl.networks import ActorNetwork, CentralizedActorNetwork

    payload = load_checkpoint(args.checkpoint)
    if args.head == "cte":
        actor = CentralizedActorNetwork(GLOBAL_OBS_DIM, agent_slot_dim=3, action_dim=NUM_ACTIONS,
                                        hidden_dim=128, seed=0)
        actor.load_state_dict(payload["actor"])
        slot = np.zeros((1, 3), dtype=np.float32)
        slot[0, 0] = 1.0
        logits = actor.forward(obs, slot)
    else:
        actor = ActorNetwork(obs_dim, NUM_ACTIONS, 128, seed=0)
        actor.load_state_dict(payload["actor"])
        logits = actor.forward(obs)[0]
    logits = np.asarray(logits)
    print("jax logits:", np.array2string(logits.reshape(1, -1), precision=9, separator=","))
