from __future__ import annotations

import json
import sys
from pathlib import Path
import mlflow
import mlflow.pytorch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.config import CONFIG
from marl_arena.controllers.rl_controller import save_rl_checkpoints, set_rl_training
from marl_arena.systems.metrics import MetricsStore
from marl_arena.systems.simulation import ArenaSimulation


def main() -> None:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CTDE_Comparison_Arena")

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

    with mlflow.start_run(run_name="ppo_training_run") as run:
        # Log config configurations
        for key, val in vars(CONFIG).items():
            if isinstance(val, (int, float, str, bool)):
                mlflow.log_param(key, val)

        # Log system information
        mlflow.set_tag("domain_randomization", str(simulation.domain_randomization))
        mlflow.set_tag("device", str(CONFIG.rl_device))

        while simulation.total_env_steps < target_steps:
            finished = simulation.step(dt)
            if not finished:
                continue

            result = simulation.finish_match()
            
            # Record match and log dashboard plots to mlflow
            if simulation.match_index % CONFIG.rl_metrics_every_matches == 0:
                exported_paths = metrics.record_match(result, simulation.cumulative_metrics)
                for plot_path in exported_paths:
                    mlflow.log_artifact(str(plot_path), artifact_path="plots")

            if simulation.total_env_steps - last_logged_steps >= CONFIG.rl_log_every_steps:
                last_logged_steps = simulation.total_env_steps
                summary = {
                    team_metrics.team_name: team_metrics.as_summary()
                    for team_metrics in simulation.cumulative_metrics.values()
                }
                
                # Log metrics for each team
                for team_name, stats in summary.items():
                    team_slug = team_name.lower().replace(" ", "_")
                    mlflow.log_metric(f"{team_slug}_win_rate", stats["win_rate"], step=simulation.total_env_steps)
                    mlflow.log_metric(f"{team_slug}_shot_accuracy", stats["shot_accuracy"], step=simulation.total_env_steps)
                    mlflow.log_metric(f"{team_slug}_mean_survival_time", stats["mean_survival_time"], step=simulation.total_env_steps)
                    mlflow.log_metric(f"{team_slug}_eliminations_per_match", stats["eliminations_per_match"], step=simulation.total_env_steps)

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
                mlflow.log_artifacts(str(CONFIG.rl_checkpoint_dir), artifact_path="checkpoints")

            simulation.reset_match()

        # Save final checkpoints and logs
        save_rl_checkpoints(simulation.controllers)
        mlflow.log_artifacts(str(CONFIG.rl_checkpoint_dir), artifact_path="checkpoints")

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
        mlflow.log_artifact(str(log_path), artifact_path="logs")

        # Log PyTorch models
        for team_name, controller in simulation.controllers.items():
            team_slug = team_name.lower().replace(" ", "_")
            if controller.actor is not None:
                mlflow.pytorch.log_model(
                    controller.actor,
                    artifact_path=f"model_{team_slug}_actor",
                    registered_model_name=f"CTDE_Arena_{team_slug.upper()}_Actor"
                )

        print(f"Treino finalizado em {simulation.total_env_steps:,} steps | log: {log_path}")


if __name__ == "__main__":
    main()

