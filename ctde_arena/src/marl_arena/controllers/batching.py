"""Batched policy inference for the agents that share one team controller."""

from __future__ import annotations

import math
from typing import Iterable

import jax
import numpy as np

from marl_arena.controllers.base import ControllerContext, normalize
from marl_arena.models import AgentSnapshot, StepDecision
from marl_arena.rl.actions import parse_action


def _observation_cache(
    agents: list[AgentSnapshot],
    all_agents: list[AgentSnapshot],
    scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, AgentSnapshot | None], dict[str, float], np.ndarray]:
    team_name = agents[0].team_name
    team_members = [agent for agent in all_agents if agent.team_name == team_name]
    centroid = (
        np.array([agent.position for agent in team_members], dtype=float).mean(axis=0)
        if team_members else np.zeros(3, dtype=float)
    )
    enemies = [agent for agent in all_agents if agent.team_name != team_name and agent.alive]
    enemy_positions = (
        np.array([agent.position for agent in enemies], dtype=float)
        if enemies else np.empty((0, 3), dtype=float)
    )

    observations: dict[str, np.ndarray] = {}
    nearest_enemies: dict[str, AgentSnapshot | None] = {}
    enemy_distances: dict[str, float] = {}
    for agent in agents:
        if not agent.alive:
            continue
        if not enemies:
            enemy = None
            relative_enemy = np.zeros(3, dtype=float)
            enemy_distance = 0.0
        else:
            enemy_offsets = enemy_positions - agent.position
            distances = np.linalg.norm(enemy_offsets, axis=1)
            nearest_index = int(np.argmin(distances))
            enemy = enemies[nearest_index]
            relative_enemy = enemy_offsets[nearest_index]
            enemy_distance = float(distances[nearest_index])
        radians = math.radians(agent.heading_deg)
        ally_delta = centroid - agent.position
        observations[agent.agent_id] = np.array(
            [
                agent.position[0] / scale,
                agent.position[2] / scale,
                math.sin(radians),
                math.cos(radians),
                relative_enemy[0] / scale,
                relative_enemy[2] / scale,
                enemy_distance / scale,
                ally_delta[0] / scale,
                ally_delta[2] / scale,
            ],
            dtype=float,
        )
        nearest_enemies[agent.agent_id] = enemy
        enemy_distances[agent.agent_id] = enemy_distance
    return observations, nearest_enemies, enemy_distances, centroid


def _cached_decision(
    controller,
    agent: AgentSnapshot,
    action_index: int,
    shoot_range: float,
    enemy: AgentSnapshot | None,
    enemy_distance: float,
    ally_centroid: np.ndarray,
) -> StepDecision:
    parsed = parse_action(action_index)
    roles = ("engage", "flank", "support", "evade")
    if enemy is None:
        target = agent.position.copy()
    elif parsed.target_index == 0:
        target = enemy.position.copy()
    elif parsed.target_index == 1:
        target = enemy.position + np.array([3.5, 0.0, -3.5], dtype=float)
    elif parsed.target_index == 2:
        target = ally_centroid
    else:
        target = agent.position - normalize(enemy.position - agent.position) * 4.0
    shoot = enemy is not None and parsed.shoot and enemy_distance <= shoot_range
    return controller.make_decision_from_target(
        agent, target, shoot, roles[parsed.target_index]
    )


def decide_many(
    controller,
    agents: list[AgentSnapshot],
    all_agents: Iterable[AgentSnapshot],
    context: ControllerContext,
) -> dict[str, StepDecision]:
    """Decide for multiple live teammates with one actor forward pass.

    Each rollout row keeps its per-agent observations and slot exactly as in ``decide``. Sampling and
    Python-side decision construction remain in agent order, so each controller consumes its random
    streams in the same order as the scalar path.
    """
    agent_list = list(all_agents)
    controller._register_agent_slots(agent_list)
    scale = context.arena_size * 0.5
    paradigm = controller.paradigm
    observation_agents = agents
    if paradigm == "CTDE-Comm":
        observation_agents = sorted(
            (agent for agent in agent_list if agent.team_name == controller.team_name and agent.alive),
            key=lambda item: item.agent_id,
        )
    local_obs_by_id, nearest_by_id, enemy_distance_by_id, centroid = _observation_cache(
        observation_agents, agent_list, scale
    )

    if paradigm == "CTDE-Comm":
        local_obs_by_id = {
            agent.agent_id: local_obs_by_id[agent.agent_id]
            if agent.alive else np.zeros(9, dtype=float)
            for agent in sorted(controller.get_team_agents(agent_list), key=lambda item: item.agent_id)
        }
        team_obs = np.concatenate(list(local_obs_by_id.values()))
        policy_obs = [team_obs] * len(agents)
        saved_local_obs = [team_obs] * len(agents)
    else:
        local_obs = [local_obs_by_id[agent.agent_id] for agent in agents]
        policy_obs = local_obs
        saved_local_obs = local_obs

    global_obs = None if paradigm == "DTE" else controller.build_global_features(agent_list, scale)
    needs_slots = paradigm in {"CTE", "CTDE-Comm"}
    slots = (
        [controller._agent_slot_vector(agent.agent_id) for agent in agents]
        if needs_slots else [None] * len(agents)
    )
    # A team has three fixed roster slots. Padding keeps JAX inference on one batch shape
    # when agents are eliminated, avoiding recompilation for batch sizes one and two.
    inference_batch = 3

    def padded_rows(rows: np.ndarray) -> np.ndarray:
        if len(rows) >= inference_batch:
            return rows
        padding = np.zeros((inference_batch - len(rows), rows.shape[1]), dtype=rows.dtype)
        return np.concatenate((rows, padding), axis=0)

    # Everything below feeds JAX with host arrays and pulls results back with a single sync. The eager
    # device-side slicing, expanding and broadcasting this path used to do were extra dispatches per
    # step with byte-identical results to the host equivalents, so the host versions win on throughput
    # without changing a single sampled value: the padded batch shapes the jitted programs see are
    # unchanged, and slicing/broadcasting float32 rows before or after the transfer is exact.
    global_rows = np.asarray(global_obs, dtype=np.float32)[None, :] if global_obs is not None else None
    slot_tensor = padded_rows(np.stack(slots)) if needs_slots else None

    if paradigm == "CTE":
        assert global_rows is not None and slot_tensor is not None
        policy_rows = np.broadcast_to(global_rows, (inference_batch, global_rows.shape[-1]))
        logits = controller.actor.forward(policy_rows, slot_tensor)
        values = controller.critic.forward(global_rows)
    elif paradigm == "CTDE-Comm":
        assert global_rows is not None and slot_tensor is not None
        team_tensor = padded_rows(np.stack(policy_obs))
        logits = controller.actor.forward(team_tensor, slot_tensor)
        values = controller.critic.forward(global_rows)
    else:
        policy_tensor = padded_rows(np.stack(policy_obs))
        logits, values = controller.actor.forward(policy_tensor)
        if paradigm != "DTE":
            assert global_rows is not None
            values = controller.critic.forward(global_rows)

    logits_result, values_result = jax.device_get((logits, values))
    logits_host = np.asarray(logits_result, dtype=np.float64)[:len(agents)]
    shifted_logits = logits_host - logits_host.max(axis=-1, keepdims=True)
    log_distribution_host = shifted_logits - np.log(
        np.exp(shifted_logits).sum(axis=-1, keepdims=True)
    )
    if controller.training_enabled:
        action_rows = [
            int(controller.action_rng.choice(logits_host.shape[-1], p=np.exp(log_distribution_host[i])))
            for i in range(len(agents))
        ]
        log_prob_rows = [
            float(log_distribution_host[i, action_rows[i]]) for i in range(len(agents))
        ]
    else:
        action_rows = [controller._eval_action(logits_host[i]) for i in range(len(agents))]
        log_prob_rows = [0.0] * len(agents)

    if paradigm == "DTE":
        value_rows = np.asarray(values_result)[:len(agents)].tolist()
    else:
        value_rows = np.broadcast_to(np.asarray(values_result), (len(agents),)).tolist()

    decisions: dict[str, StepDecision] = {}
    for index, agent in enumerate(agents):
        action_index = int(action_rows[index])
        controller._record_step(
            agent.agent_id,
            action_index,
            float(log_prob_rows[index]),
            float(value_rows[index]),
            saved_local_obs[index],
            global_obs,
            slots[index] if paradigm in {"CTE", "CTDE-Comm"} else None,
        )
        decisions[agent.agent_id] = _cached_decision(
            controller,
            agent,
            action_index,
            context.shoot_range,
            nearest_by_id[agent.agent_id],
            enemy_distance_by_id[agent.agent_id],
            centroid,
        )
    return decisions
