from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from marl_arena.config import CONFIG
from marl_arena.controllers.base import ControllerContext, normalize
from marl_arena.controllers.rl_controller import build_controllers, finish_rl_episode
from marl_arena.models import AgentSnapshot, MatchResult, ObstacleSnapshot, ProjectileSnapshot, TeamMetrics, TransitionRecord
from marl_arena.systems.match_variant import (
    AGENT_FORMATION_OFFSETS,
    TEAM_META,
    MatchVariant,
    create_default_variant,
    sample_training_variant,
)

AGENT_RADIUS = 0.58
AGENT_HALF_HEIGHT = 1.1
PROJECTILE_RADIUS = 0.12
PROJECTILE_SPEED_MULTIPLIER = 2.5
FLOAT_EPSILON = 1e-6


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


@dataclass
class ProjectileState:
    projectile_id: int
    shooter_agent_id: str
    position: np.ndarray
    velocity: np.ndarray
    team_name: str
    team_color: tuple[float, float, float]
    age: float
    hit: bool
    obstacle_hit: bool
    hit_position: np.ndarray
    distance_travelled: float
    max_distance: float

    def snapshot(self) -> ProjectileSnapshot:
        return ProjectileSnapshot(
            projectile_id=self.projectile_id,
            position=self.position.copy(),
            direction=self.velocity.copy(),
            team_name=self.team_name,
            team_color=self.team_color,
            age=self.age,
            hit=self.hit,
            obstacle_hit=self.obstacle_hit,
            hit_position=tuple(self.hit_position.tolist()) if self.hit_position is not None else (0.0, 0.0, 0.0),
        )


class ArenaSimulation:
    def __init__(self, seed: int | None = None, domain_randomization: bool | None = None) -> None:
        self.config = CONFIG
        self.seed = self.config.random_seed if seed is None else seed
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.domain_randomization = (
            self.config.domain_randomization if domain_randomization is None else domain_randomization
        )
        self.controllers = build_controllers(self.seed)
        self.cumulative_metrics = {
            team_name: TeamMetrics(team_name=team_name, paradigm=paradigm)
            for team_name, paradigm, _ in TEAM_META
        }
        self.match_index = 0
        self.variant_counter = 0
        self.total_env_steps = 0
        self.match_variant: MatchVariant = create_default_variant(self.config)
        self.agents: List[SimAgent] = []
        self.obstacles: List[SimObstacle] = []
        self.projectiles: List[ProjectileState] = []
        self._projectile_id_counter: int = 0
        self.match_time = 0.0
        self.pending_obstacle_hits: List[Dict[str, object]] = []
        self.step_index = 0
        self.trajectory_rows: List[Dict[str, float]] = []
        self.last_match_result: MatchResult | None = None
        self.reset_match()

    def _sample_next_variant(self) -> MatchVariant:
        self.variant_counter += 1
        if self.domain_randomization:
            return sample_training_variant(self.rng, self.config, self.variant_counter)
        return create_default_variant(self.config, self.variant_counter)

    def _create_obstacles(self) -> List[SimObstacle]:
        obstacles: List[SimObstacle] = []
        for spec in self.match_variant.obstacles:
            obstacles.append(
                SimObstacle(
                    obstacle_id=spec.obstacle_id,
                    obstacle_type=spec.obstacle_type,
                    base_position=spec.base_position.copy(),
                    size=spec.size.copy(),
                    color_rgb=spec.color_rgb,
                    movement_axis=spec.movement_axis.copy(),
                    movement_amplitude=spec.movement_amplitude,
                    movement_speed=spec.movement_speed,
                    phase_offset=spec.phase_offset,
                )
            )
        return obstacles

    def reset_match(self) -> None:
        self.match_index += 1
        self.match_time = 0.0
        self.step_index = 0
        self.last_match_result = None
        self.trajectory_rows = []
        self.match_variant = self._sample_next_variant()
        self.agents = []
        self.obstacles = self._create_obstacles()
        for obstacle in self.obstacles:
            obstacle.update(0.0)
        for team_idx, spawn in enumerate(self.match_variant.team_spawns):
            for agent_idx, offset in enumerate(AGENT_FORMATION_OFFSETS):
                agent = SimAgent(
                    agent_id=f"{team_idx + 1}-{agent_idx + 1}",
                    team_name=spawn.team_name,
                    paradigm=spawn.paradigm,
                    color_rgb=spawn.color_rgb,
                    position=spawn.spawn_center + offset,
                    heading_deg=self.rng.uniform(0.0, 360.0),
                )
                self.agents.append(agent)
        self.projectiles.clear()
        self._projectile_id_counter = 0
        self.pending_obstacle_hits.clear()

    def live_team_counts(self) -> Dict[str, int]:
        counts = {team_name: 0 for team_name, _, _ in TEAM_META}
        for agent in self.agents:
            if agent.alive:
                counts[agent.team_name] += 1
        return counts

    def build_snapshots(self) -> List[AgentSnapshot]:
        return [agent.snapshot() for agent in self.agents]

    def build_obstacle_snapshots(self) -> List[ObstacleSnapshot]:
        return [obstacle.snapshot() for obstacle in self.obstacles]

    def build_projectile_snapshots(self) -> List[ProjectileSnapshot]:
        return [proj.snapshot() for proj in self.projectiles]

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
        half = self.match_variant.arena_size * 0.5 - AGENT_RADIUS
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

    def _coerce_vec3(self, value: np.ndarray, context: str) -> np.ndarray | None:
        try:
            vector = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            warnings.warn(f"{context}: nao foi possivel converter o vetor para um formato numerico valido.")
            return None
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            warnings.warn(f"{context}: esperado vetor 3D finito, recebido shape={vector.shape}.")
            return None
        return vector

    def _segment_intersects_aabb(
        self,
        start: np.ndarray,
        end: np.ndarray,
        min_corner: np.ndarray,
        max_corner: np.ndarray,
    ) -> tuple[bool, float, np.ndarray]:
        direction = end - start
        t_enter = 0.0
        t_exit = 1.0
        for axis in range(3):
            delta = direction[axis]
            if abs(delta) <= FLOAT_EPSILON:
                if start[axis] < min_corner[axis] or start[axis] > max_corner[axis]:
                    return False, 1.0, end.copy()
                continue
            inv_delta = 1.0 / delta
            t0 = (min_corner[axis] - start[axis]) * inv_delta
            t1 = (max_corner[axis] - start[axis]) * inv_delta
            if t0 > t1:
                t0, t1 = t1, t0
            t_enter = max(t_enter, t0)
            t_exit = min(t_exit, t1)
            if t_enter - t_exit > FLOAT_EPSILON:
                return False, 1.0, end.copy()
        hit_t = float(np.clip(t_enter, 0.0, 1.0))
        hit_position = start + direction * hit_t
        return True, hit_t, hit_position

    def _obstacle_bounds(self, obstacle: SimObstacle, padding: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        half = obstacle.size * 0.5 + padding
        return obstacle.position - half, obstacle.position + half

    def _agent_bounds(self, agent: SimAgent, padding: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        half = np.array(
            [AGENT_RADIUS + padding, AGENT_HALF_HEIGHT + padding, AGENT_RADIUS + padding],
            dtype=float,
        )
        return agent.position - half, agent.position + half

    def _first_obstacle_collision(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> tuple[SimObstacle | None, np.ndarray | None, float]:
        best_obstacle: SimObstacle | None = None
        best_position: np.ndarray | None = None
        best_t = float("inf")
        for obstacle in self.obstacles:
            min_corner, max_corner = self._obstacle_bounds(obstacle, PROJECTILE_RADIUS)
            intersects, hit_t, hit_position = self._segment_intersects_aabb(start, end, min_corner, max_corner)
            if intersects and hit_t < best_t:
                best_obstacle = obstacle
                best_position = hit_position
                best_t = hit_t
        return best_obstacle, best_position, best_t

    def _first_agent_collision(
        self,
        start: np.ndarray,
        end: np.ndarray,
        projectile: ProjectileState,
    ) -> tuple[SimAgent | None, np.ndarray | None, float]:
        best_agent: SimAgent | None = None
        best_position: np.ndarray | None = None
        best_t = float("inf")
        for candidate in self.agents:
            if (
                not candidate.alive
                or candidate.team_name == projectile.team_name
                or candidate.agent_id == projectile.shooter_agent_id
            ):
                continue
            min_corner, max_corner = self._agent_bounds(candidate, PROJECTILE_RADIUS)
            intersects, hit_t, hit_position = self._segment_intersects_aabb(start, end, min_corner, max_corner)
            if intersects and hit_t < best_t:
                best_agent = candidate
                best_position = hit_position
                best_t = hit_t
        return best_agent, best_position, best_t

    def _arena_boundary_collision(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> tuple[bool, np.ndarray | None, float]:
        half = self.match_variant.arena_size * 0.5
        direction = end - start
        best_t = float("inf")
        best_position: np.ndarray | None = None
        for axis in (0, 2):
            delta = direction[axis]
            if abs(delta) <= FLOAT_EPSILON:
                continue
            for boundary in (-half, half):
                hit_t = (boundary - start[axis]) / delta
                if hit_t <= FLOAT_EPSILON or hit_t > 1.0 + FLOAT_EPSILON:
                    continue
                hit_position = start + direction * hit_t
                if abs(hit_position[0]) <= half + FLOAT_EPSILON and abs(hit_position[2]) <= half + FLOAT_EPSILON:
                    if hit_t < best_t:
                        best_t = float(hit_t)
                        best_position = hit_position
        return best_position is not None, best_position, best_t

    def _find_agent(self, agent_id: str) -> SimAgent | None:
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def _register_projectile_miss(self, projectile: ProjectileState) -> None:
        shooter = self._find_agent(projectile.shooter_agent_id)
        if shooter is None:
            warnings.warn(
                f"Projetil {projectile.projectile_id} descartado sem autor valido para registrar erro de disparo."
            )
            return
        shooter.misses += 1
        team_metrics = self.cumulative_metrics.get(projectile.team_name)
        if team_metrics is None:
            warnings.warn(f"Equipe '{projectile.team_name}' nao encontrada ao registrar miss do projetil.")
            return
        team_metrics.shots_missed += 1

    def _register_projectile_hit(
        self,
        projectile: ProjectileState,
        target: SimAgent,
        hit_status: Dict[str, bool],
        scored_status: Dict[str, bool],
    ) -> None:
        shooter = self._find_agent(projectile.shooter_agent_id)
        target.alive = False
        target.vertical_velocity = 0.0
        hit_status[target.agent_id] = True
        if shooter is None:
            warnings.warn(
                f"Projetil {projectile.projectile_id} acertou {target.agent_id}, mas o autor nao foi encontrado."
            )
            return
        shooter.kills += 1
        shooter.hits += 1
        scored_status[shooter.agent_id] = True
        team_metrics = self.cumulative_metrics.get(projectile.team_name)
        if team_metrics is None:
            warnings.warn(f"Equipe '{projectile.team_name}' nao encontrada ao registrar hit do projetil.")
            return
        team_metrics.eliminations += 1
        team_metrics.shots_hit += 1

    def _spawn_projectile(self, shooter: SimAgent, aim_direction: np.ndarray) -> ProjectileState | None:
        direction_input = self._coerce_vec3(aim_direction, f"Disparo do agente {shooter.agent_id}")
        if direction_input is None:
            return None
        direction = normalize(direction_input)
        speed = max(self.match_variant.shoot_range * PROJECTILE_SPEED_MULTIPLIER, FLOAT_EPSILON)
        if float(np.linalg.norm(direction)) <= FLOAT_EPSILON:
            warnings.warn(f"Disparo do agente {shooter.agent_id} ignorado por direcao nula.")
            return None
        projectile_origin = shooter.position.copy() + direction * (AGENT_RADIUS + PROJECTILE_RADIUS + 0.05)
        half = self.match_variant.arena_size * 0.5 - PROJECTILE_RADIUS
        projectile_origin[0] = float(np.clip(projectile_origin[0], -half, half))
        projectile_origin[2] = float(np.clip(projectile_origin[2], -half, half))
        projectile = ProjectileState(
            projectile_id=self._projectile_id_counter,
            shooter_agent_id=shooter.agent_id,
            position=projectile_origin,
            velocity=direction * speed,
            team_name=shooter.team_name,
            team_color=shooter.color_rgb,
            age=0.0,
            hit=False,
            obstacle_hit=False,
            hit_position=np.zeros(3, dtype=float),
            distance_travelled=0.0,
            max_distance=self.match_variant.shoot_range,
        )
        self._projectile_id_counter += 1
        self.projectiles.append(projectile)
        return projectile

    def _advance_projectiles(
        self,
        dt: float,
        hit_status: Dict[str, bool],
        scored_status: Dict[str, bool],
    ) -> None:
        if dt <= 0.0:
            return
        remaining_projectiles: List[ProjectileState] = []
        for projectile in self.projectiles:
            try:
                start = self._coerce_vec3(projectile.position, f"Projetil {projectile.projectile_id} posicao")
                velocity = self._coerce_vec3(projectile.velocity, f"Projetil {projectile.projectile_id} velocidade")
                if start is None or velocity is None:
                    raise ValueError("estado do projetil invalido")
                speed = float(np.linalg.norm(velocity))
                if speed <= FLOAT_EPSILON:
                    raise ValueError("velocidade nula")
                remaining_distance = max(0.0, projectile.max_distance - projectile.distance_travelled)
                if remaining_distance <= FLOAT_EPSILON:
                    self._register_projectile_miss(projectile)
                    continue
                travel_distance = min(speed * dt, remaining_distance)
                direction = velocity / speed
                end = start + direction * travel_distance
                projectile.age += dt

                best_kind: str | None = None
                best_t = float("inf")
                best_position: np.ndarray | None = None
                best_target: SimAgent | None = None

                obstacle, obstacle_position, obstacle_t = self._first_obstacle_collision(start, end)
                if obstacle is not None and obstacle_position is not None:
                    best_kind = "obstacle"
                    best_t = obstacle_t
                    best_position = obstacle_position

                target, target_position, target_t = self._first_agent_collision(start, end, projectile)
                if target is not None and target_position is not None and target_t < best_t - FLOAT_EPSILON:
                    best_kind = "agent"
                    best_t = target_t
                    best_position = target_position
                    best_target = target

                boundary_hit, boundary_position, boundary_t = self._arena_boundary_collision(start, end)
                if boundary_hit and boundary_position is not None and boundary_t < best_t - FLOAT_EPSILON:
                    best_kind = "boundary"
                    best_t = boundary_t
                    best_position = boundary_position

                if best_kind == "agent" and best_position is not None and best_target is not None:
                    projectile.hit = True
                    projectile.hit_position = best_position.copy()
                    projectile.position = best_position.copy()
                    projectile.distance_travelled += travel_distance * best_t
                    self._register_projectile_hit(projectile, best_target, hit_status, scored_status)
                    continue

                if best_kind in {"obstacle", "boundary"} and best_position is not None:
                    projectile.obstacle_hit = True
                    projectile.hit_position = best_position.copy()
                    projectile.position = best_position.copy()
                    projectile.distance_travelled += travel_distance * best_t
                    self.pending_obstacle_hits.append(
                        {
                            "position": best_position.copy(),
                            "team_name": projectile.team_name,
                            "team_color": projectile.team_color,
                        }
                    )
                    self._register_projectile_miss(projectile)
                    continue

                projectile.position = end
                projectile.distance_travelled += travel_distance
                if projectile.distance_travelled >= projectile.max_distance - FLOAT_EPSILON:
                    projectile.hit_position = end.copy()
                    self._register_projectile_miss(projectile)
                    continue
                remaining_projectiles.append(projectile)
            except ValueError as exc:
                warnings.warn(f"Projetil {projectile.projectile_id} removido: {exc}.")
                self._register_projectile_miss(projectile)
        self.projectiles = remaining_projectiles

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

    def _reward_for_agent(self, agent: SimAgent, was_hit: bool, scored_hit: bool) -> float:
        reward = 0.015 if agent.alive else -0.5
        if scored_hit:
            reward += 1.2
        if was_hit:
            reward -= 1.0
        return reward

    def step(self, dt: float) -> bool:
        self.pending_obstacle_hits.clear()
        self.match_time += dt
        self.step_index += 1
        self.total_env_steps += 1
        self._update_obstacles()
        snapshots = self.build_snapshots()
        context = ControllerContext(
            step_index=self.step_index,
            arena_size=self.match_variant.arena_size,
            time_delta=dt,
            shoot_range=self.match_variant.shoot_range,
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

            agent.heading_deg = (
                agent.heading_deg + decision.turn * self.match_variant.agent_turn_speed * dt
            ) % 360.0
            if decision.jump and agent.position[1] <= 1.02:
                agent.vertical_velocity = self.config.jump_speed
            forward = self._forward_from_heading(agent.heading_deg)
            desired_position = (
                agent.position + forward * decision.move * self.match_variant.agent_move_speed * dt
            )
            agent.position = self._resolve_movement(agent.position, desired_position)
            self._apply_jump_and_gravity(agent, dt)
            if agent.alive:
                agent.survival_time += dt

        for agent in self.agents:
            decision = decisions.get(agent.agent_id)
            if not agent.alive or decision is None or not decision.shoot:
                continue
            if self.match_time - agent.last_shot_at < self.match_variant.shoot_cooldown:
                continue
            agent.last_shot_at = self.match_time
            self._spawn_projectile(agent, decision.aim_direction)

        self._advance_projectiles(dt, hit_status, scored_status)

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
        return self.match_time >= self.match_variant.match_duration_seconds

    def finish_match(self) -> MatchResult:
        team_alive = self.live_team_counts()
        sorted_alive = sorted(team_alive.items(), key=lambda item: (item[1], self.cumulative_metrics[item[0]].eliminations), reverse=True)
        winner_team = sorted_alive[0][0]
        self.cumulative_metrics[winner_team].wins += 1

        team_rows: List[Dict[str, float]] = []
        agent_rows: List[Dict[str, float]] = []

        for spawn in self.match_variant.team_spawns:
            team_name = spawn.team_name
            paradigm = spawn.paradigm
            team_agents = [agent for agent in self.agents if agent.team_name == team_name]
            team_metrics = self.cumulative_metrics[team_name]
            team_metrics.matches_played += 1
            team_metrics.survival_time_sum += float(sum(agent.survival_time for agent in team_agents))
            team_rows.append(
                {
                    "match_index": self.match_index,
                    "variant_id": self.match_variant.variant_id,
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
        finish_rl_episode(self.controllers)
        return self.last_match_result

    def match_status_text(self) -> str:
        counts = self.live_team_counts()
        obstacle_breakdown: Dict[str, int] = {}
        for obstacle in self.obstacles:
            obstacle_breakdown[obstacle.obstacle_type] = obstacle_breakdown.get(obstacle.obstacle_type, 0) + 1
        lines = [
            f"Partida {self.match_index}  Tempo: {self.match_time:05.1f}s",
            f"Equipe 1 / CTDE-VD: {counts['Equipe 1']} vivos",
            f"Equipe 2 / CTDE-CAC: {counts['Equipe 2']} vivos",
            f"Equipe 3 / CTDE-Comm: {counts['Equipe 3']} vivos",
            f"Obstaculos: {len(self.obstacles)} | Fixos {obstacle_breakdown.get('barreira_fixa', 0)} | Moveis {obstacle_breakdown.get('obstaculo_movel', 0)} | Restritos {obstacle_breakdown.get('passagem_restrita', 0)}",
        ]
        return "\n".join(lines)
