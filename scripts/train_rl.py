from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.config import CONFIG
from marl_arena.controllers.factory import save_rl_checkpoints, set_rl_training
from marl_arena.systems.metrics import MetricsStore
from marl_arena.systems.simulation import ArenaSimulation


def main() -> None:
    if CONFIG.controller_mode != "rl":
        raise RuntimeError("Defina CONTROLLER_MODE=rl no arquivo .env antes de treinar.")

    metrics = MetricsStore()
    simulation = ArenaSimulation()
    set_rl_training(simulation.controllers, True)
    total_matches = CONFIG.rl_train_matches
    training_log: list[dict[str, object]] = []

    print(f"Iniciando treino PPO por {total_matches} partidas | device={CONFIG.rl_device}")
    for match_number in range(1, total_matches + 1):
        while True:
            finished = simulation.step(0.1)
            if finished:
                break
        result = simulation.finish_match()
        metrics.record_match(result, simulation.cumulative_metrics)
        simulation.reset_match()

        if match_number % CONFIG.rl_save_every == 0 or match_number == total_matches:
            save_rl_checkpoints(simulation.controllers)
            print(f"[{match_number}/{total_matches}] checkpoints salvos em {CONFIG.rl_checkpoint_dir}")

        if match_number % 10 == 0 or match_number == 1:
            summary = {
                team_metrics.team_name: team_metrics.as_summary()
                for team_metrics in simulation.cumulative_metrics.values()
            }
            training_log.append({"match": match_number, "winner": result.winner_team, "summary": summary})
            print(
                f"Partida {match_number} | vencedora={result.winner_team} | "
                f"win_rates={{k: round(v['win_rate'], 3) for k, v in summary.items()}}"
            )

    log_path = CONFIG.rl_checkpoint_dir / "training_log.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(training_log, handle, indent=2)
    print(f"Treino finalizado. Log: {log_path}")


if __name__ == "__main__":
    main()
