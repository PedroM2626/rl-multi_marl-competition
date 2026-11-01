"""Batched policy inference for the agents that share one team controller."""

from __future__ import annotations

import math
from typing import Iterable

import jax
import numpy as np

from marl_arena.controllers.base import ControllerContext, normalize, obstacle_rays
from marl_arena.models import AgentSnapshot, StepDecision
from marl_arena.rl.actions import parse_action


def _observation_cache(
    agents: list[AgentSnapshot],
    all_agents: list[AgentSnapshot],
    scale: float,
    rays: dict[str, np.ndarray] | None = None,
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
        if rays is not None:
            observations[agent.agent_id] = np.concatenate((observations[agent.agent_id], rays[agent.agent_id]))
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
    prepared = _prepare(controller, agents, all_agents, context)
    logits, values = _forward(controller, [prepared])[0]
    return _finish(controller, prepared, logits, values)


def decide_many_across(
    controller,
    calls: list[tuple[list[AgentSnapshot], list[AgentSnapshot], ControllerContext]],
) -> list[dict[str, StepDecision]]:
    """`decide_many` for one team in several arenas at once, with one actor and one critic pass.

    This is what lets a batch of arenas share a policy without paying a device dispatch per arena per team
    per step -- the cost A-46 found dominating a step. One call is exactly `decide_many`; with several, the
    batch is three rows per arena, so float32 rows can round differently from a three-row pass by the ~6e-8
    A-44 measured, which is why the published instruments keep the one-arena path.
    """
    prepared = [_prepare(controller, *call) for call in calls]
    outputs = _forward(controller, prepared)
    return [
        _finish(controller, prep, logits, values)
        for prep, (logits, values) in zip(prepared, outputs, strict=True)
    ]


# A team has three fixed roster slots. Padding keeps JAX inference on one batch shape when agents are
# eliminated, avoiding recompilation for batch sizes one and two.
INFERENCE_BATCH = 3


def _padded_rows(rows: np.ndarray) -> np.ndarray:
    if len(rows) >= INFERENCE_BATCH:
        return rows
    padding = np.zeros((INFERENCE_BATCH - len(rows), rows.shape[1]), dtype=rows.dtype)
    return np.concatenate((rows, padding), axis=0)


def _prepare(
    controller,
    agents: list[AgentSnapshot],
    all_agents: Iterable[AgentSnapshot],
    context: ControllerContext,
) -> dict:
    """Every host-side input of one team's decision in one arena: observations, slots and padded rows."""
    agent_list = list(all_agents)
    controller._register_agent_slots(agent_list)
    scale = context.arena_size * 0.5
    paradigm = controller.paradigm
    ray_count = getattr(controller, "ray_count", 0)
    # The local-observation VD critic sums over teammates, so like the Comm actor it needs every live
    # teammate's observation, not only those deciding this call.
    needs_team_obs = paradigm == "CTDE-Comm" or getattr(controller, "vd_local", False)
    rays_by_id = None
    if ray_count:
        live = [agent for agent in agent_list if agent.alive]
        ray_rows = obstacle_rays(
            np.array([[agent.position[0], agent.position[2]] for agent in live], dtype=float),
            np.array([agent.heading_deg for agent in live], dtype=float),
            context.obstacle_boxes, scale, ray_count,
        )
        rays_by_id = {agent.agent_id: row for agent, row in zip(live, ray_rows, strict=True)}
    observation_agents = agents
    if needs_team_obs:
        observation_agents = sorted(
            (agent for agent in agent_list if agent.team_name == controller.team_name and agent.alive),
            key=lambda item: item.agent_id,
        )
    local_obs_by_id, nearest_by_id, enemy_distance_by_id, centroid = _observation_cache(
        observation_agents, agent_list, scale, rays_by_id
    )
    team_obs = None
    if needs_team_obs:
        obs_dim = 9 + ray_count
        team_obs = np.concatenate([
            local_obs_by_id[agent.agent_id] if agent.alive else np.zeros(obs_dim, dtype=float)
            for agent in sorted(controller.get_team_agents(agent_list), key=lambda item: item.agent_id)
        ])

    if paradigm == "CTDE-Comm":
        policy_obs = [team_obs] * len(agents)
        saved_local_obs = [team_obs] * len(agents)
    else:
        local_obs = [local_obs_by_id[agent.agent_id] for agent in agents]
        policy_obs = local_obs
        saved_local_obs = local_obs

    # Whether a value head beyond the actor exists is what decides the value path, not the arm name: DTE has
    # none, and neither does an untrained controller under a label outside the six arms (the RANDOM opponent).
    if controller.critic is None:
        global_obs = None
    elif getattr(controller, "vd_local", False):
        # What the VDN critic reads is the team's local observations, so that is what the rollout row keeps.
        global_obs = team_obs
    else:
        global_obs = controller.build_global_features(agent_list, scale, rays_by_id, ray_count)
    needs_slots = paradigm in {"CTE", "CTDE-Comm"}
    slots = (
        [controller._agent_slot_vector(agent.agent_id) for agent in agents]
        if needs_slots else [None] * len(agents)
    )

    # Everything below feeds JAX with host arrays and pulls results back with a single sync. The eager
    # device-side slicing, expanding and broadcasting this path used to do were extra dispatches per
    # step with byte-identical results to the host equivalents, so the host versions win on throughput
    # without changing a single sampled value: the padded batch shapes the jitted programs see are
    # unchanged, and slicing/broadcasting float32 rows before or after the transfer is exact.
    global_rows = np.asarray(global_obs, dtype=np.float32)[None, :] if global_obs is not None else None
    slot_tensor = _padded_rows(np.stack(slots)) if needs_slots else None
    if paradigm == "CTE":
        assert global_rows is not None and slot_tensor is not None
        policy_rows = np.broadcast_to(global_rows, (INFERENCE_BATCH, global_rows.shape[-1]))
    else:
        policy_rows = _padded_rows(np.stack(policy_obs))
    return {
        "agents": agents, "context": context, "global_obs": global_obs, "global_rows": global_rows,
        "slots": slots, "slot_tensor": slot_tensor, "policy_rows": policy_rows,
        "saved_local_obs": saved_local_obs, "nearest_by_id": nearest_by_id,
        "enemy_distance_by_id": enemy_distance_by_id, "centroid": centroid,
    }


def _forward(controller, prepared: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    """One actor pass (and one critic pass) over every prepared call, split back per call after one sync."""
    paradigm = controller.paradigm
    count = len(prepared)

    def stacked(key: str) -> np.ndarray:
        if count == 1:
            return prepared[0][key]
        return np.concatenate([prep[key] for prep in prepared], axis=0)

    if paradigm in {"CTE", "CTDE-Comm"}:
        logits = controller.actor.forward(stacked("policy_rows"), stacked("slot_tensor"))
        values = controller.critic.forward(stacked("global_rows"))
    else:
        logits, values = controller.actor.forward(stacked("policy_rows"))
        if controller.critic is not None:
            values = controller.critic.forward(stacked("global_rows"))

    logits_result, values_result = jax.device_get((logits, values))
    if count == 1:
        return [(logits_result, values_result)]
    rows = INFERENCE_BATCH
    value_rows = rows if controller.critic is None else 1
    return [
        (logits_result[index * rows:(index + 1) * rows],
         values_result[index * value_rows:(index + 1) * value_rows])
        for index in range(count)
    ]


def _finish(controller, prepared: dict, logits_result, values_result) -> dict[str, StepDecision]:
    """Sample or read out the actions, record the rollout rows and build the movement decisions."""
    agents, context = prepared["agents"], prepared["context"]
    paradigm = controller.paradigm
    logits_host = np.asarray(logits_result, dtype=np.float64)[:len(agents)]
    shifted_logits = logits_host - logits_host.max(axis=-1, keepdims=True)
    log_distribution_host = shifted_logits - np.log(
        np.exp(shifted_logits).sum(axis=-1, keepdims=True)
    )
    if controller.training_enabled or getattr(controller, "frozen", False):
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

    if controller.critic is None:
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
            prepared["saved_local_obs"][index],
            prepared["global_obs"],
            prepared["slots"][index] if paradigm in {"CTE", "CTDE-Comm"} else None,
        )
        decisions[agent.agent_id] = _cached_decision(
            controller,
            agent,
            action_index,
            context.shoot_range,
            prepared["nearest_by_id"][agent.agent_id],
            prepared["enemy_distance_by_id"][agent.agent_id],
            prepared["centroid"],
        )
    return decisions
