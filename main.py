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

from marl_arena.systems.metrics import MetricsStore
from marl_arena.systems.simulation import ArenaSimulation
from marl_arena.ui.dashboard import build_overlay_text


class CapsuleVisual(Entity):
    def __init__(self, agent) -> None:
        super().__init__(position=Vec3(*agent.position))
        self.agent = agent
        tint = color.rgba(
            int(agent.color_rgb[0] * 255),
            int(agent.color_rgb[1] * 255),
            int(agent.color_rgb[2] * 255),
            255,
        )
        Entity(parent=self, model="cube", scale=(0.7, 1.2, 0.7), color=tint, y=0.0)
        Entity(parent=self, model="sphere", scale=(0.72, 0.72, 0.72), color=tint, y=0.74)
        Entity(parent=self, model="sphere", scale=(0.72, 0.72, 0.72), color=tint, y=-0.74)
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
        alpha = 1.0 if self.agent.alive else 0.18
        for child in self.children:
            if hasattr(child, "color"):
                child.color = color.rgba(child.color.r, child.color.g, child.color.b, int(alpha * 255))


class ObstacleVisual(Entity):
    def __init__(self, obstacle) -> None:
        super().__init__(position=Vec3(*obstacle.position))
        self.obstacle = obstacle
        tint = color.rgba(
            int(obstacle.color_rgb[0] * 255),
            int(obstacle.color_rgb[1] * 255),
            int(obstacle.color_rgb[2] * 255),
            255,
        )
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


class ArenaApp:
    def __init__(self) -> None:
        self.metrics = MetricsStore()
        self.simulation = ArenaSimulation()
        self.agent_visuals: dict[str, CapsuleVisual] = {}
        self.obstacle_visuals: dict[str, ObstacleVisual] = {}
        self.match_banner: Text | None = None
        self.finished = False
        self._build_scene()
        self._spawn_obstacle_visuals()
        self._spawn_agent_visuals()

    def _build_scene(self) -> None:
        window.title = "RL Multi MARL Competition"
        window.borderless = False
        window.color = color.rgb(22, 24, 31)
        Entity(model="plane", scale=(40, 1, 40), texture="white_cube", texture_scale=(20, 20), color=color.rgb(70, 75, 82))
        wall_color = color.rgb(45, 48, 55)
        Entity(model="cube", scale=(40, 3, 1), position=(0, 1.5, 20), color=wall_color)
        Entity(model="cube", scale=(40, 3, 1), position=(0, 1.5, -20), color=wall_color)
        Entity(model="cube", scale=(1, 3, 40), position=(20, 1.5, 0), color=wall_color)
        Entity(model="cube", scale=(1, 3, 40), position=(-20, 1.5, 0), color=wall_color)
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
