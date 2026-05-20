from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ursina import (
    AmbientLight,
    Button,
    DirectionalLight,
    EditorCamera,
    Entity,
    Text,
    Ursina,
    Vec3,
    WindowPanel,
    color,
    destroy,
    invoke,
    time,
    window,
)

from marl_arena.controllers.rl_controller import set_rl_training
from marl_arena.systems.metrics import MetricsStore
from marl_arena.systems.simulation import ArenaSimulation
from marl_arena.ui.dashboard import build_overlay_text


class CapsuleVisual(Entity):
    _team_hsv = {
        "Equipe 1": (0.0, 0.73, 0.92),
        "Equipe 2": (220.0, 0.74, 0.95),
        "Equipe 3": (145.0, 0.72, 0.92),
    }

    def __init__(self, agent, **kwargs) -> None:
        super().__init__(position=Vec3(*agent.position), **kwargs)
        self.agent = agent
        hue, base_sat, base_val = self._team_hsv[agent.team_name]
        agent_idx = int(agent.agent_id.split("-")[1]) - 1
        sat_scale = 0.60 + agent_idx * 0.20
        val_scale = 0.60 + agent_idx * 0.18
        alive_color = color.color(hue, sat_scale, val_scale)
        dead_color = color.color(hue, sat_scale * 0.25, val_scale * 0.30)
        self._alive_color = alive_color
        self._dead_color = dead_color
        Entity(parent=self, model="cube", scale=(0.7, 1.2, 0.7), color=alive_color, y=0.0)
        Entity(parent=self, model="sphere", scale=(0.72, 0.72, 0.72), color=alive_color, y=0.74)
        Entity(parent=self, model="sphere", scale=(0.72, 0.72, 0.72), color=alive_color, y=-0.74)
        self.label = Text(
            text=f"{agent.team_name[-1]}-{agent.agent_id}",
            world_parent=self,
            position=(0, 1.75, 0),
            scale=10,
            origin=(0, 0),
            billboard=True,
        )

    def sync(self) -> None:
        self.position = Vec3(*self.agent.position)
        self.rotation_y = self.agent.heading_deg
        current_color = self._alive_color if self.agent.alive else self._dead_color
        for child in self.children:
            if hasattr(child, "color"):
                child.color = current_color


class ObstacleVisual(Entity):
    def __init__(self, obstacle, **kwargs) -> None:
        super().__init__(position=Vec3(*obstacle.position), **kwargs)
        self.obstacle = obstacle
        tint = color.color(0.0, 0.0, 0.55)
        Entity(parent=self, model="cube", scale=tuple(obstacle.size), color=tint)
        label_text = obstacle.obstacle_type.replace("_", " ").title()
        self.label = Text(
            text=label_text,
            world_parent=self,
            position=(0, obstacle.size[1] * 0.6 + 0.7, 0),
            scale=7,
            origin=(0, 0),
            billboard=True,
        )

    def sync(self) -> None:
        self.position = Vec3(*self.obstacle.position)


class ProjectileExplosion(Entity):
    _team_hues = {
        "Equipe 1": 0.0,
        "Equipe 2": 220.0,
        "Equipe 3": 145.0,
    }

    def __init__(self, hit_data, **kwargs) -> None:
        pos = hit_data["position"]
        team_name = hit_data.get("team_name", "")
        hue = self._team_hues.get(team_name, 180.0)
        self._elapsed = 0.0
        self._duration = 0.5
        super().__init__(position=Vec3(pos[0], pos[1], pos[2]), **kwargs)
        self._ring = Entity(
            parent=self,
            model="sphere",
            scale=(0.0, 0.0, 0.0),
            color=color.color(hue, 0.9, 1.0),
        )
        self._core = Entity(
            parent=self,
            model="sphere",
            scale=(0.3, 0.3, 0.3),
            color=color.color(hue, 0.6, 1.0),
        )

    def update(self) -> None:
        self._elapsed += time.dt
        progress = min(self._elapsed / self._duration, 1.0)
        ring_scale = progress * 2.5
        self._ring.scale = (ring_scale, ring_scale * 0.3, ring_scale)
        self._core.scale = (0.3 * (1.0 - progress * 0.9), 0.3 * (1.0 - progress * 0.9), 0.3 * (1.0 - progress * 0.9))
        if self._elapsed >= self._duration:
            destroy(self)


class ProjectileVisual(Entity):
    _team_hues = {
        "Equipe 1": 0.0,
        "Equipe 2": 220.0,
        "Equipe 3": 145.0,
    }

    def __init__(self, snapshot, **kwargs) -> None:
        self.snapshot = snapshot
        hue = self._team_hues.get(getattr(snapshot, "team_name", ""), 180.0)
        proj_color = color.color(hue, 0.85, 1.0)
        trail_color = color.color(hue, 0.60, 1.0)
        super().__init__(position=Vec3(*snapshot.position), color=proj_color, **kwargs)
        Entity(parent=self, model="sphere", scale=(0.25, 0.25, 0.7), color=proj_color)
        Entity(parent=self, model="sphere", scale=(0.17, 0.17, 0.45), color=trail_color)

    def sync(self, snapshot) -> None:
        self.snapshot = snapshot
        self.position = Vec3(*snapshot.position)


class ArenaApp:
    def __init__(self) -> None:
        self.metrics = MetricsStore()
        self.simulation = ArenaSimulation(domain_randomization=False)
        set_rl_training(self.simulation.controllers, False)
        self.agent_visuals: dict[str, CapsuleVisual] = {}
        self.obstacle_visuals: dict[str, ObstacleVisual] = {}
        self.projectile_visuals: dict[int, ProjectileVisual] = {}
        self.explosions: list[ProjectileExplosion] = []
        self.match_banner: Text | None = None
        self.finished = False
        self._build_scene()
        self._spawn_obstacle_visuals()
        self._spawn_agent_visuals()

    def _build_scene(self) -> None:
        arena_size = self.simulation.match_variant.arena_size
        arena_half = arena_size * 0.5
        window.title = "RL Multi MARL Competition"
        window.borderless = False
        window.color = color.rgb(22, 24, 31)
        Entity(
            model="plane",
            scale=(arena_size, 1, arena_size),
            texture="white_cube",
            texture_scale=(max(arena_size / 2.0, 1.0), max(arena_size / 2.0, 1.0)),
            color=color.rgb(70, 75, 82),
        )
        wall_color = color.rgb(45, 48, 55)
        Entity(model="cube", scale=(arena_size, 3, 1), position=(0, 1.5, arena_half), color=wall_color)
        Entity(model="cube", scale=(arena_size, 3, 1), position=(0, 1.5, -arena_half), color=wall_color)
        Entity(model="cube", scale=(1, 3, arena_size), position=(arena_half, 1.5, 0), color=wall_color)
        Entity(model="cube", scale=(1, 3, arena_size), position=(-arena_half, 1.5, 0), color=wall_color)
        AmbientLight(color=color.rgba(180, 180, 180, 0.7))
        sun = DirectionalLight()
        sun.look_at(Vec3(1, -1, -1))
        EditorCamera(position=(0, 32, -28), rotation=(37, 0, 0))
        self.overlay = Text(text="", x=-0.86, y=0.47, scale=0.78, background=True)
        self.export_label = Text(text="", x=-0.86, y=-0.43, scale=0.72, background=True)
        WindowPanel(
            title="Legenda",
            content=(
                Text("Equipe 1 = vermelho / CTE"),
                Text("Equipe 2 = azul / DTE"),
                Text("Equipe 3 = verde / CTDE"),
                Text("Cinza = barreira fixa"),
                Text("Amarelo = obstaculo movel"),
                Text("Aco = passagem restrita"),
            ),
            x=0.53,
            y=0.32,
        )
        Button(text="Reiniciar Partida", x=0.63, y=-0.43, scale=(0.2, 0.06), on_click=self.restart_match)

    def _spawn_obstacle_visuals(self) -> None:
        for visual in self.obstacle_visuals.values():
            destroy(visual)
        self.obstacle_visuals = {}
        for obstacle in self.simulation.obstacles:
            self.obstacle_visuals[obstacle.obstacle_id] = ObstacleVisual(obstacle)

    def _spawn_agent_visuals(self) -> None:
        for visual in self.agent_visuals.values():
            destroy(visual)
        self.agent_visuals = {}
        for agent in self.simulation.agents:
            self.agent_visuals[agent.agent_id] = CapsuleVisual(agent)

    def restart_match(self) -> None:
        self.finished = False
        if self.match_banner is not None:
            destroy(self.match_banner)
            self.match_banner = None
        for visual in self.projectile_visuals.values():
            destroy(visual)
        self.projectile_visuals.clear()
        for exp in self.explosions:
            destroy(exp)
        self.explosions.clear()
        self.simulation.reset_match()
        self._spawn_obstacle_visuals()
        self._spawn_agent_visuals()

    def _handle_match_end(self) -> None:
        if self.finished:
            return
        self.finished = True
        result = self.simulation.finish_match()
        exported = self.metrics.record_match(result, self.simulation.cumulative_metrics)
        export_text = "Arquivos: " + ", ".join(path.name for path in exported) if exported else "Arquivos: nenhum"
        self.export_label.text = export_text
        self.match_banner = Text(
            text=f"Vencedora: {result.winner_team} | Reinicio em 4s",
            scale=1.8,
            y=0.34,
            background=True,
        )
        invoke(self.restart_match, delay=4.0)

    def update(self) -> None:
        if not self.finished:
            finished = self.simulation.step(time.dt)
            if finished:
                self._handle_match_end()
        for visual in self.obstacle_visuals.values():
            visual.sync()
        for visual in self.agent_visuals.values():
            visual.sync()
        active_ids = set()
        to_create = []
        for snapshot in self.simulation.build_projectile_snapshots():
            active_ids.add(snapshot.projectile_id)
            if snapshot.projectile_id in self.projectile_visuals:
                self.projectile_visuals[snapshot.projectile_id].sync(snapshot)
            else:
                to_create.append(snapshot)
        for snapshot in to_create:
            self.projectile_visuals[snapshot.projectile_id] = ProjectileVisual(snapshot)
        stale_ids = set(self.projectile_visuals.keys()) - active_ids
        for pid in stale_ids:
            destroy(self.projectile_visuals.pop(pid))
        for hit_data in self.simulation.pending_obstacle_hits:
            self.explosions.append(ProjectileExplosion(hit_data))
        self.simulation.pending_obstacle_hits.clear()
        self.explosions = [e for e in self.explosions if e is not None]
        self.overlay.text = build_overlay_text(self.simulation)


class ArenaLoopDriver(Entity):
    def __init__(self, arena: ArenaApp) -> None:
        super().__init__()
        self.arena = arena

    def update(self) -> None:
        self.arena.update()


def main() -> None:
    app = Ursina()
    arena = ArenaApp()
    ArenaLoopDriver(arena)
    app.run()


if __name__ == "__main__":
    main()
