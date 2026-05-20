from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl_arena.config import CONFIG
from marl_arena.models import AgentSnapshot, StepDecision
from marl_arena.rl.constants import GLOBAL_OBS_DIM, LOCAL_OBS_DIM, NUM_ACTIONS, NUM_TARGETS


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
) -> StepDecision:
    parsed = parse_action(action_index)
    targets = controller.candidate_targets(agent, all_agents)
    target = targets[parsed.target_index]
    enemy = controller.nearest_enemy(agent, all_agents)
    can_shoot = False
    if enemy is not None and parsed.shoot:
        distance = float(np.linalg.norm(enemy.position - agent.position))
        can_shoot = distance <= CONFIG.shoot_range
    roles = ["engage", "flank", "support", "evade"]
    return controller.make_decision_from_target(agent, target, can_shoot, roles[parsed.target_index])
