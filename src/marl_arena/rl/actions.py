from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marl_arena.models import AgentSnapshot, StepDecision


LOCAL_OBS_DIM = 9
GLOBAL_OBS_DIM = 45
NUM_TARGETS = 4
NUM_ACTIONS = NUM_TARGETS * 2


@dataclass(frozen=True)
class ParsedAction:
    target_index: int
    shoot: bool


def parse_action(action_index: int) -> ParsedAction:
    clamped = int(action_index) % NUM_ACTIONS
    return ParsedAction(target_index=clamped // 2, shoot=bool(clamped % 2))


def action_to_decision(
    controller,
    agent: AgentSnapshot,
    all_agents: list[AgentSnapshot],
    action_index: int,
    shoot_range: float,
) -> StepDecision:
    parsed = parse_action(action_index)
    targets = controller.candidate_targets(agent, all_agents)
    target = targets[parsed.target_index]
    enemy = controller.nearest_enemy(agent, all_agents)
    can_shoot = False
    if enemy is not None and parsed.shoot:
        distance = float(np.linalg.norm(enemy.position - agent.position))
        can_shoot = distance <= shoot_range
    roles = ["engage", "flank", "support", "evade"]
    return controller.make_decision_from_target(agent, target, can_shoot, roles[parsed.target_index])


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write a non-pickle JAX checkpoint; parameters remain in Torch-compatible layer layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key in {"actor", "critic"} and value is not None:
            for parameter_name, parameter in value.items():
                arrays[f"{key}::{parameter_name}"] = np.asarray(parameter)
        else:
            metadata[key] = value
    arrays["__metadata__"] = np.asarray(json.dumps(metadata))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_checkpoint(path: Path, device: Any = None) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    result: dict[str, Any] = {"actor": {}, "critic": {}}
    with np.load(path, allow_pickle=False) as saved:
        result.update(json.loads(str(saved["__metadata__"])))
        for key in saved.files:
            if key == "__metadata__":
                continue
            group, parameter_name = key.split("::", maxsplit=1)
            result[group][parameter_name] = saved[key]
    if not result["critic"]:
        result["critic"] = None
    return result
