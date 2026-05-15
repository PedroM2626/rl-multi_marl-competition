from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from marl_arena.config import CONFIG
from marl_arena.controllers import CTDEController, CTEController, DTEController
from marl_arena.controllers.base import ControllerContext, normalize
from marl_arena.models import AgentSnapshot, MatchResult, ObstacleSnapshot, TeamMetrics, TransitionRecord


TEAM_DEFINITIONS = (
    ("Equipe 1", "CTE", np.array([-10.0, 1.0, -10.0], dtype=float), (0.92, 0.25, 0.25)),
    ("Equipe 2", "DTE", np.array([10.0, 1.0, -10.0], dtype=float), (0.25, 0.55, 0.95)),
    ("Equipe 3", "CTDE", np.array([0.0, 1.0, 10.0], dtype=float), (0.25, 0.88, 0.45)),
)

AGENT_RADIUS = 0.58


@dataclass
class SimAgent:
    agent_id: str
    team_name: str
    paradigm: str
    color_rgb: tuple[float, float, float]
    position: np.ndarray
    heading_deg: float
    vertical_velocity: float = 0.0
    alive: bool = True
    kills: int = 0
    hits: int = 0
    misses: int = 0
    survival_time: float = 0.0
    respawn_timer: float = 0.0
    last_shot_at: float = -999.0

    def snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(
            agent_id=self.agent_id,
            team_name=self.team_name,
            paradigm=self.paradigm,
            position=self.position.copy(),
            heading_deg=self.heading_deg,
            alive=self.alive,
            kills=self.kills,
            hits=self.hits,
            misses=self.misses,
            survival_time=self.survival_time,
        )


@dataclass
class SimObstacle:
    obstacle_id: str
    obstacle_type: str
    base_position: np.ndarray
    size: np.ndarray
    color_rgb: tuple[float, float, float]
    movement_axis: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    movement_amplitude: float = 0.0
    movement_speed: float = 0.0
    phase_offset: float = 0.0
    position: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = self.base_position.copy()

    def update(self, match_time: float) -> None:
        if self.movement_amplitude <= 0.0 or self.movement_speed <= 0.0:
            self.position = self.base_position.copy()
            return
        self.position = (
            self.base_position
            + normalize(self.movement_axis)
            * self.movement_amplitude
            * math.sin(self.phase_offset + match_time * self.movement_speed)
        )

    def snapshot(self) -> ObstacleSnapshot:
        return ObstacleSnapshot(
            obstacle_id=self.obstacle_id,
            obstacle_type=self.obstacle_type,
            position=self.position.copy(),
            size=self.size.copy(),
            movement_axis=self.movement_axis.copy(),
            movement_amplitude=self.movement_amplitude,
        )


class ArenaSimulation:
    def __init__(self, seed: int | None = None) -> None:
        self.config = CONFIG
        self.seed = self.config.random_seed if seed is None else seed
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.controllers = {
            "Equipe 1": CTEController("Equipe 1", self.seed + 11),
            "Equipe 2": DTEController("Equipe 2", self.seed + 23),
            "Equipe 3": CTDEController("Equipe 3", self.seed + 37),
        }
        self.cumulative_metrics = {
            team_name: TeamMetrics(team_name=team_name, paradigm=paradigm)
            for team_name, paradigm, _, _ in TEAM_DEFINITIONS
        }
        self.match_index = 0
        self.agents: List[SimAgent] = []
        self.obstacles: List[SimObstacle] = []
        self.match_time = 0.0
        self.step_index = 0
        self.trajectory_rows: List[Dict[str, float]] = []
        self.last_match_result: MatchResult | None = None
        self.reset_match()

    def _create_obstacles(self) -> List[SimObstacle]:
        return [
            SimObstacle(
                obstacle_id="fixed-central-wall",
                obstacle_type="barreira_fixa",
                base_position=np.array([0.0, 1.5, -1.0], dtype=float),
                size=np.array([2.0, 3.0, 10.0], dtype=float),
                color_rgb=(0.60, 0.58, 0.56),
            ),
            SimObstacle(
                obstacle_id="fixed-east-block",
                obstacle_type="barreira_fixa",
                base_position=np.array([8.0, 1.5, 5.0], dtype=float),
                size=np.array([7.0, 3.0, 2.4], dtype=float),
                color_rgb=(0.54, 0.52, 0.50),
            ),
            SimObstacle(
                obstacle_id="fixed-west-pillar",
                obstacle_type="barreira_fixa",
                base_position=np.array([-8.0, 1.5, 6.0], dtype=float),
                size=np.array([2.8, 3.0, 2.8], dtype=float),
                color_rgb=(0.52, 0.50, 0.48),
            ),
            SimObstacle(
                obstacle_id="moving-north-sweeper",
                obstacle_type="obstaculo_movel",
                base_position=np.array([0.0, 1.0, 13.5], dtype=float),
                size=np.array([4.0, 2.0, 1.6], dtype=float),
                color_rgb=(0.93, 0.74, 0.25),
                movement_axis=np.array([1.0, 0.0, 0.0], dtype=float),
                movement_amplitude=7.0,
                movement_speed=0.8,
                phase_offset=0.0,
            ),
            SimObstacle(
                obstacle_id="moving-south-sweeper",
                obstacle_type="obstaculo_movel",
                base_position=np.array([0.0, 1.0, -12.0], dtype=float),
                size=np.array([4.6, 2.0, 1.6], dtype=float),
                color_rgb=(0.95, 0.66, 0.24),
                movement_axis=np.array([1.0, 0.0, 0.0], dtype=float),
                movement_amplitude=6.0,
                movement_speed=0.65,
                phase_offset=1.4,
            ),
            SimObstacle(
                obstacle_id="passage-left",
                obstacle_type="passagem_restrita",
                base_position=np.array([-3.8, 1.5, 9.0], dtype=float),
                size=np.array([2.6, 3.0, 4.5], dtype=float),
                color_rgb=(0.38, 0.42, 0.50),
            ),
            SimObstacle(
                obstacle_id="passage-right",
                obstacle_type="passagem_restrita",
                base_position=np.array([3.8, 1.5, 9.0], dtype=float),
                size=np.array([2.6, 3.0, 4.5], dtype=float),
                color_rgb=(0.38, 0.42, 0.50),
            ),
            SimObstacle(
                obstacle_id="passage-lower-left",
                obstacle_type="passagem_restrita",
                base_position=np.array([-3.8, 1.5, -9.0], dtype=float),
                size=np.array([2.6, 3.0, 4.0], dtype=float),
                color_rgb=(0.36, 0.40, 0.48),
            ),
            SimObstacle(
                obstacle_id="passage-lower-right",
                obstacle_type="passagem_restrita",
                base_position=np.array([3.8, 1.5, -9.0], dtype=float),
                size=np.array([2.6, 3.0, 4.0], dtype=float),
                color_rgb=(0.36, 0.40, 0.48),
            ),
        ]

    def reset_match(self) -> None:
        self.match_index += 1
        self.match_time = 0.0
        self.step_index = 0
        self.last_match_result = None
        self.trajectory_rows = []
        self.agents = []
        self.obstacles = self._create_obstacles()
        for obstacle in self.obstacles:
            obstacle.update(0.0)
        for team_idx, (team_name, paradigm, spawn_center, color_rgb) in enumerate(TEAM_DEFINITIONS):
            offsets = [
                np.array([-1.7, 0.0, 0.0], dtype=float),
                np.array([1.7, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.7], dtype=float),
            ]
            for agent_idx, offset in enumerate(offsets):
                agent = SimAgent(
                    agent_id=f"{team_idx + 1}-{agent_idx + 1}",
                    team_name=team_name,
                    paradigm=paradigm,
                    color_rgb=color_rgb,
                    position=spawn_center + offset,
                    heading_deg=self.rng.uniform(0.0, 360.0),
                )
                self.agents.append(agent)

    def live_team_counts(self) -> Dict[str, int]:
        counts = {team_name: 0 for team_name, _, _, _ in TEAM_DEFINITIONS}
        for agent in self.agents:
            if agent.alive:
                counts[agent.team_name] += 1
        return counts

    def build_snapshots(self) -> List[AgentSnapshot]:
        return [agent.snapshot() for agent in self.agents]

    def build_obstacle_snapshots(self) -> List[ObstacleSnapshot]:
        return [obstacle.snapshot() for obstacle in self.obstacles]

    def _apply_jump_and_gravity(self, agent: SimAgent, dt: float) -> None:
        ground_y = 1.0
        if agent.position[1] <= ground_y + 1e-4:
            agent.position[1] = ground_y
            if agent.vertical_velocity < 0.0:
                agent.vertical_velocity = 0.0
        agent.vertical_velocity -= self.config.gravity * dt
        agent.position[1] += agent.vertical_velocity * dt
        if agent.position[1] <= ground_y:
            agent.position[1] = ground_y
            agent.vertical_velocity = 0.0

    def _forward_from_heading(self, heading_deg: float) -> np.ndarray:
        radians = math.radians(heading_deg)
        return np.array([math.sin(radians), 0.0, math.cos(radians)], dtype=float)

    def _clamp_position_to_arena(self, position: np.ndarray) -> np.ndarray:
        half = self.config.arena_size * 0.5 - AGENT_RADIUS
        clamped = position.copy()
        clamped[0] = float(np.clip(clamped[0], -half, half))
        clamped[2] = float(np.clip(clamped[2], -half, half))
        return clamped

    def _position_blocked(self, position: np.ndarray) -> bool:
        for obstacle in self.obstacles:
            half_size = obstacle.size * 0.5
            dx = abs(position[0] - obstacle.position[0])
            dz = abs(position[2] - obstacle.position[2])
            if dx <= half_size[0] + AGENT_RADIUS and dz <= half_size[2] + AGENT_RADIUS:
                return True
        return False

    def _resolve_movement(self, current_position: np.ndarray, desired_position: np.ndarray) -> np.ndarray:
        desired_position = self._clamp_position_to_arena(desired_position)
        if not self._position_blocked(desired_position):
            return desired_position

        slide_x = desired_position.copy()
        slide_x[2] = current_position[2]
        slide_x = self._clamp_position_to_arena(slide_x)
        if not self._position_blocked(slide_x):
            return slide_x

        slide_z = desired_position.copy()
        slide_z[0] = current_position[0]
        slide_z = self._clamp_position_to_arena(slide_z)
        if not self._position_blocked(slide_z):
            return slide_z

        for obstacle in self.obstacles:
            half_size = obstacle.size * 0.5
            dx = desired_position[0] - obstacle.position[0]
            dz = desired_position[2] - obstacle.position[2]
            overlap_x = (half_size[0] + AGENT_RADIUS) - abs(dx)
            overlap_z = (half_size[2] + AGENT_RADIUS) - abs(dz)
            if overlap_x > 0.0 and overlap_z > 0.0:
                corrected = desired_position.copy()
                if overlap_x < overlap_z:
                    corrected[0] += overlap_x if dx >= 0.0 else -overlap_x
                else:
                    corrected[2] += overlap_z if dz >= 0.0 else -overlap_z
                corrected = self._clamp_position_to_arena(corrected)
                if not self._position_blocked(corrected):
                    return corrected
        return current_position.copy()

    def _push_agents_out_of_obstacles(self) -> None:
        for agent in self.agents:
            if not agent.alive:
                continue
            corrected = self._resolve_movement(agent.position, agent.position)
            if np.allclose(corrected, agent.position):
                for obstacle in self.obstacles:
                    half_size = obstacle.size * 0.5
                    dx = agent.position[0] - obstacle.position[0]
                    dz = agent.position[2] - obstacle.position[2]
                    overlap_x = (half_size[0] + AGENT_RADIUS) - abs(dx)
                    overlap_z = (half_size[2] + AGENT_RADIUS) - abs(dz)
                    if overlap_x > 0.0 and overlap_z > 0.0:
                        if overlap_x < overlap_z:
                            agent.position[0] += overlap_x + 0.05 if dx >= 0.0 else -(overlap_x + 0.05)
                        else:
                            agent.position[2] += overlap_z + 0.05 if dz >= 0.0 else -(overlap_z + 0.05)
                        agent.position = self._clamp_position_to_arena(agent.position)
            else:
                agent.position = corrected

    def _update_obstacles(self) -> None:
        for obstacle in self.obstacles:
            obstacle.update(self.match_time)
        self._push_agents_out_of_obstacles()

    def _segment_hits_obstacle(self, start: np.ndarray, end: np.ndarray, obstacle: SimObstacle) -> bool:
        half = obstacle.size * 0.5
        min_x = obstacle.position[0] - half[0]
        max_x = obstacle.position[0] + half[0]
        min_z = obstacle.position[2] - half[2]
        max_z = obstacle.position[2] + half[2]
        samples = 20
        for t in np.linspace(0.0, 1.0, samples):
            x = start[0] + (end[0] - start[0]) * t
            z = start[2] + (end[2] - start[2]) * t
            if min_x <= x <= max_x and min_z <= z <= max_z:
                return True
        return False

    def _record_trajectory(self) -> None:
        team_counts = self.live_team_counts()
        for team_name, team_metrics in self.cumulative_metrics.items():
            shots = team_metrics.shots_hit + team_metrics.shots_missed
            self.trajectory_rows.append(
                {
                    "match_index": self.match_index,
                    "time_seconds": round(self.match_time, 3),
                    "team_name": team_name,
                    "paradigm": team_metrics.paradigm,
                    "alive_agents": team_counts[team_name],
                    "cumulative_eliminations": team_metrics.eliminations,
                    "cumulative_accuracy": team_metrics.shots_hit / max(shots, 1),
                    "cumulative_win_rate": team_metrics.wins / max(team_metrics.matches_played, 1),
                    "obstacle_count": len(self.obstacles),
                }
            )

    def _pick_hit_target(self, shooter: SimAgent, aim_direction: np.ndarray) -> SimAgent | None:
        best_target: SimAgent | None = None
        best_distance = float("inf")
        for candidate in self.agents:
            if not candidate.alive or candidate.team_name == shooter.team_name:
                continue
            delta = candidate.position - shooter.position
            horizontal = np.array([delta[0], 0.0, delta[2]], dtype=float)
            distance = float(np.linalg.norm(horizontal))
            if distance > self.config.shoot_range or distance <= 1e-6:
                continue
            direction = normalize(horizontal)
            alignment = float(np.dot(direction, normalize(np.array([aim_direction[0], 0.0, aim_direction[2]], dtype=float))))
            if alignment < 0.93:
                continue
            if any(self._segment_hits_obstacle(shooter.position, candidate.position, obstacle) for obstacle in self.obstacles):
                continue
            if distance < best_distance:
                best_target = candidate
                best_distance = distance
        return best_target

    def _reward_for_agent(self, agent: SimAgent, was_hit: bool, scored_hit: bool) -> float:
        reward = 0.015 if agent.alive else -0.5
        if scored_hit:
            reward += 1.2
        if was_hit:
            reward -= 1.0
        return reward

    def step(self, dt: float) -> bool:
        self.match_time += dt
        self.step_index += 1
        self._update_obstacles()
        snapshots = self.build_snapshots()
        context = ControllerContext(
            step_index=self.step_index,
            arena_size=self.config.arena_size,
            time_delta=dt,
        )
        decisions: Dict[str, object] = {}
        hit_status = {agent.agent_id: False for agent in self.agents}
        scored_status = {agent.agent_id: False for agent in self.agents}

        for agent in self.agents:
            if not agent.alive:
                continue
            controller = self.controllers[agent.team_name]
            decisions[agent.agent_id] = controller.decide(agent.snapshot(), snapshots, context)

        for agent in self.agents:
            decision = decisions.get(agent.agent_id)
            if not agent.alive or decision is None:
                self._apply_jump_and_gravity(agent, dt)
                continue

            agent.heading_deg = (agent.heading_deg + decision.turn * self.config.agent_turn_speed * dt) % 360.0
            if decision.jump and agent.position[1] <= 1.02:
                agent.vertical_velocity = self.config.jump_speed
            forward = self._forward_from_heading(agent.heading_deg)
            desired_position = agent.position + forward * decision.move * self.config.agent_move_speed * dt
            agent.position = self._resolve_movement(agent.position, desired_position)
            self._apply_jump_and_gravity(agent, dt)
            if agent.alive:
                agent.survival_time += dt

        for agent in self.agents:
            decision = decisions.get(agent.agent_id)
            if not agent.alive or decision is None or not decision.shoot:
                continue
            if self.match_time - agent.last_shot_at < self.config.shoot_cooldown:
                continue
            agent.last_shot_at = self.match_time
            target = self._pick_hit_target(agent, decision.aim_direction)
            team_metrics = self.cumulative_metrics[agent.team_name]
            if target is None:
                agent.misses += 1
                team_metrics.shots_missed += 1
                continue

            target.alive = False
            target.vertical_velocity = 0.0
            agent.kills += 1
            agent.hits += 1
            team_metrics.eliminations += 1
            team_metrics.shots_hit += 1
            hit_status[target.agent_id] = True
            scored_status[agent.agent_id] = True

        post_snapshots = self.build_snapshots()
        for team_name, controller in self.controllers.items():
            transitions: List[TransitionRecord] = []
            for agent in self.agents:
                decision = decisions.get(agent.agent_id)
                if agent.team_name != team_name or decision is None:
                    continue
                transitions.append(
                    TransitionRecord(
                        agent_id=agent.agent_id,
                        team_name=agent.team_name,
                        state_features=controller.build_local_features(agent.snapshot(), post_snapshots),
                        action_features=np.array(
                            [decision.move, decision.turn, 1.0 if decision.shoot else 0.0, 1.0 if decision.jump else 0.0],
                            dtype=float,
                        ),
                        reward=self._reward_for_agent(agent, hit_status[agent.agent_id], scored_status[agent.agent_id]),
                        next_state_features=controller.build_local_features(agent.snapshot(), post_snapshots),
                        done=not agent.alive,
                    )
                )
            controller.update(transitions)

        self._record_trajectory()
        return self._is_match_finished()

    def _is_match_finished(self) -> bool:
        alive_by_team = self.live_team_counts()
        alive_teams = [team_name for team_name, count in alive_by_team.items() if count > 0]
        if len(alive_teams) <= 1:
            return True
        return self.match_time >= self.config.match_duration_seconds

    def finish_match(self) -> MatchResult:
        team_alive = self.live_team_counts()
        sorted_alive = sorted(team_alive.items(), key=lambda item: (item[1], self.cumulative_metrics[item[0]].eliminations), reverse=True)
        winner_team = sorted_alive[0][0]
        self.cumulative_metrics[winner_team].wins += 1

        team_rows: List[Dict[str, float]] = []
        agent_rows: List[Dict[str, float]] = []

        for team_name, paradigm, _, _ in TEAM_DEFINITIONS:
            team_agents = [agent for agent in self.agents if agent.team_name == team_name]
            team_metrics = self.cumulative_metrics[team_name]
            team_metrics.matches_played += 1
            team_metrics.survival_time_sum += float(sum(agent.survival_time for agent in team_agents))
            team_rows.append(
                {
                    "match_index": self.match_index,
                    "team_name": team_name,
                    "paradigm": paradigm,
                    "winner": 1 if team_name == winner_team else 0,
                    "eliminations": sum(agent.kills for agent in team_agents),
                    "mean_survival_time": float(np.mean([agent.survival_time for agent in team_agents])),
                    "shots_hit": sum(agent.hits for agent in team_agents),
                    "shots_missed": sum(agent.misses for agent in team_agents),
                    "shot_accuracy": sum(agent.hits for agent in team_agents)
                    / max(sum(agent.hits + agent.misses for agent in team_agents), 1),
                    "remaining_agents": team_alive[team_name],
                    "match_duration_seconds": self.match_time,
                }
            )
            for agent in team_agents:
                agent_rows.append(
                    {
                        "match_index": self.match_index,
                        "agent_id": agent.agent_id,
                        "team_name": team_name,
                        "paradigm": paradigm,
                        "kills": agent.kills,
                        "shots_hit": agent.hits,
                        "shots_missed": agent.misses,
                        "survival_time": agent.survival_time,
                        "alive_at_end": 1 if agent.alive else 0,
                    }
                )

        self.last_match_result = MatchResult(
            winner_team=winner_team,
            duration_seconds=self.match_time,
            team_rows=team_rows,
            agent_rows=agent_rows,
            trajectory_rows=self.trajectory_rows.copy(),
        )
        return self.last_match_result

    def match_status_text(self) -> str:
        counts = self.live_team_counts()
        obstacle_breakdown: Dict[str, int] = {}
        for obstacle in self.obstacles:
            obstacle_breakdown[obstacle.obstacle_type] = obstacle_breakdown.get(obstacle.obstacle_type, 0) + 1
        lines = [
            f"Partida {self.match_index}  Tempo: {self.match_time:05.1f}s",
            f"Equipe 1 / CTE: {counts['Equipe 1']} vivos",
            f"Equipe 2 / DTE: {counts['Equipe 2']} vivos",
            f"Equipe 3 / CTDE: {counts['Equipe 3']} vivos",
            f"Obstaculos: {len(self.obstacles)} | Fixos {obstacle_breakdown.get('barreira_fixa', 0)} | Moveis {obstacle_breakdown.get('obstaculo_movel', 0)} | Restritos {obstacle_breakdown.get('passagem_restrita', 0)}",
        ]
        return "\n".join(lines)
