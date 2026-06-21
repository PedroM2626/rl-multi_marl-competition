from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marl_arena.models import AgentSnapshot, StepDecision


LOCAL_OBS_DIM = 8
GLOBAL_OBS_DIM = 36
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
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {path}")
    return torch.load(path, map_location=device, weights_only=False)
