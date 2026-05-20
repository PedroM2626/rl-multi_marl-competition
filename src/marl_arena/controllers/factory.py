from __future__ import annotations

import torch

from marl_arena.config import CONFIG
from marl_arena.controllers.base import BaseTeamController
from marl_arena.controllers.ctde_controller import CTDEController
from marl_arena.controllers.ctde_rl_controller import CTDERLController
from marl_arena.controllers.cte_controller import CTEController
from marl_arena.controllers.cte_rl_controller import CTERLController
from marl_arena.controllers.dte_controller import DTEController
from marl_arena.controllers.dte_rl_controller import DTERLController
from marl_arena.controllers.rl_base import RLTeamController


def resolve_device() -> torch.device:
    requested = CONFIG.rl_device.strip().lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_controllers(seed: int) -> dict[str, BaseTeamController]:
    if CONFIG.controller_mode == "rl":
        device = resolve_device()
        return {
            "Equipe 1": CTERLController("Equipe 1", seed + 11, device),
            "Equipe 2": DTERLController("Equipe 2", seed + 23, device),
            "Equipe 3": CTDERLController("Equipe 3", seed + 37, device),
        }
    return {
        "Equipe 1": CTEController("Equipe 1", seed + 11),
        "Equipe 2": DTEController("Equipe 2", seed + 23),
        "Equipe 3": CTDEController("Equipe 3", seed + 37),
    }


def finish_rl_episode(controllers: dict[str, BaseTeamController]) -> dict[str, object]:
    stats: dict[str, object] = {}
    for team_name, controller in controllers.items():
        if isinstance(controller, RLTeamController):
            result = controller.finish_episode()
            if result is not None:
                stats[team_name] = result
    return stats


def save_rl_checkpoints(controllers: dict[str, BaseTeamController]) -> None:
    for controller in controllers.values():
        if isinstance(controller, RLTeamController):
            controller.save(CONFIG.rl_checkpoint_dir)


def set_rl_training(controllers: dict[str, BaseTeamController], enabled: bool) -> None:
    for controller in controllers.values():
        if isinstance(controller, RLTeamController):
            controller.set_training(enabled)
