from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.config import CONFIG
from marl_arena.controllers.rl_controller import save_rl_checkpoints, set_rl_training
from marl_arena.systems.metrics import MetricsStore
from marl_arena.systems.simulation import ArenaSimulation


def main() -> None:
    metrics = MetricsStore()
    simulation = ArenaSimulation(domain_randomization=True)
    set_rl_training(simulation.controllers, True)
    target_steps = CONFIG.rl_train_total_steps
    dt = CONFIG.sim_step_dt
    training_log: list[dict[str, object]] = []
    last_saved_steps = 0
    last_logged_steps = 0

    print(
        f"Treino PPO | alvo={target_steps:,} steps | dt={dt} | "
        f"domain_randomization={simulation.domain_randomization} | device={CONFIG.rl_device}"
    )

    while simulation.total_env_steps < target_steps:
        finished = simulation.step(dt)
        if not finished:
            continue

        result = simulation.finish_match()
        if simulation.match_index % CONFIG.rl_metrics_every_matches == 0:
            metrics.record_match(result, simulation.cumulative_metrics)

        if simulation.total_env_steps - last_logged_steps >= CONFIG.rl_log_every_steps:
            last_logged_steps = simulation.total_env_steps
            summary = {
                team_metrics.team_name: team_metrics.as_summary()
                for team_metrics in simulation.cumulative_metrics.values()
            }
            training_log.append(
                {
                    "env_steps": simulation.total_env_steps,
                    "matches": simulation.match_index,
                    "winner": result.winner_team,
                    "variant": simulation.match_variant.summary(),
                    "summary": summary,
                }
            )
            print(
                f"steps={simulation.total_env_steps:,}/{target_steps:,} | partidas={simulation.match_index} | "
                f"vencedora={result.winner_team} | variant={simulation.match_variant.variant_id} | "
                f"win_rates={{k: round(v['win_rate'], 3) for k, v in summary.items()}}"
            )

        if simulation.total_env_steps - last_saved_steps >= CONFIG.rl_save_every_steps:
            last_saved_steps = simulation.total_env_steps
            save_rl_checkpoints(simulation.controllers)
            print(f"[save] checkpoints em {CONFIG.rl_checkpoint_dir} (@ {simulation.total_env_steps:,} steps)")

        simulation.reset_match()

    save_rl_checkpoints(simulation.controllers)
    log_path = CONFIG.rl_checkpoint_dir / "training_log.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "target_steps": target_steps,
                "completed_steps": simulation.total_env_steps,
                "matches_played": simulation.match_index,
                "entries": training_log,
            },
            handle,
            indent=2,
        )
    print(f"Treino finalizado em {simulation.total_env_steps:,} steps | log: {log_path}")


if __name__ == "__main__":
    main()
