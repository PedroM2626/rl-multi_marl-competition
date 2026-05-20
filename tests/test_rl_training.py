from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ["CONTROLLER_MODE"] = "rl"
os.environ["RL_TRAIN_MATCHES"] = "2"
os.environ["MATCH_DURATION_SECONDS"] = "5"

from marl_arena.config import CONFIG
from marl_arena.controllers.factory import save_rl_checkpoints, set_rl_training
from marl_arena.controllers.rl_base import RLTeamController
from marl_arena.systems.simulation import ArenaSimulation


def test_rl_controllers_collect_rollouts_and_update() -> None:
    simulation = ArenaSimulation(seed=99)
    set_rl_training(simulation.controllers, True)
    for _ in range(2):
        while True:
            if simulation.step(0.1):
                break
        simulation.finish_match()
        simulation.reset_match()

    for controller in simulation.controllers.values():
        assert isinstance(controller, RLTeamController)

    save_rl_checkpoints(simulation.controllers)
    for paradigm in ("cte", "dte", "ctde"):
        matches = list(CONFIG.rl_checkpoint_dir.glob(f"*_{paradigm}.pt"))
        assert matches, f"Checkpoint {paradigm} nao foi salvo."
