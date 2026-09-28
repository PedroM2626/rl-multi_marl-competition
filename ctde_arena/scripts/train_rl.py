from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CTDE-VD / CAC / Comm policies with PPO and MLflow.")
    parser.add_argument("--seed", type=int, default=None, help="overrides RANDOM_SEED for this run")
    parser.add_argument("--steps", type=int, default=None, help="overrides RL_TRAIN_TOTAL_STEPS")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="do not load the checkpoints already in data/checkpoints; train from random weights",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="train without contacting the MLflow store")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from marl_arena.config import CONFIG, seed_all
    from marl_arena.controllers.rl_controller import save_rl_checkpoints, set_rl_training
    from marl_arena.systems.metrics import MetricsStore
    from marl_arena.systems.simulation import ArenaSimulation

    seed = CONFIG.random_seed if args.seed is None else args.seed
    seed_all(seed)

    mlflow = None
    mlflow_pytorch = None
    if not args.no_mlflow:
        import mlflow
        import mlflow.pytorch

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("CTDE_Comparison_Arena")

    metrics = MetricsStore()
    simulation = ArenaSimulation(seed=seed, domain_randomization=True, load_checkpoints=not args.from_scratch)
    set_rl_training(simulation.controllers, True)
    target_steps = CONFIG.rl_train_total_steps if args.steps is None else args.steps
    dt = CONFIG.sim_step_dt
    training_log: list[dict[str, object]] = []
    last_saved_steps = 0
    last_logged_steps = 0

    print(
        f"PPO training | target={target_steps:,} steps | dt={dt} | seed={seed} | "
        f"init={'checkpoint' if not args.from_scratch else 'random'} | "
        f"domain_randomization={simulation.domain_randomization} | device={CONFIG.rl_device}"
    )

    def log_entry(final: bool) -> None:
        summary = {
            team_metrics.team_name: team_metrics.as_summary()
            for team_metrics in simulation.cumulative_metrics.values()
        }
        ppo_stats = {
            team: {
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "entropy": stats.entropy,
                "approx_kl": stats.approx_kl,
            }
            for team, stats in simulation.last_ppo_stats.items()
        }
        training_log.append(
            {
                "env_steps": simulation.total_env_steps,
                "matches": simulation.match_index,
                "winner": simulation.last_match_result.winner_team if simulation.last_match_result else None,
                "variant": simulation.match_variant.summary(),
                "summary": summary,
                "ppo_stats": ppo_stats,
                "final": final,
            }
        )
        if mlflow is not None:
            for team_name, stats in summary.items():
                slug = team_name.lower().replace(" ", "_")
                step = simulation.total_env_steps
                mlflow.log_metric(f"{slug}_win_rate", stats["win_rate"], step=step)
                mlflow.log_metric(f"{slug}_shot_accuracy", stats["shot_accuracy"], step=step)
                mlflow.log_metric(f"{slug}_mean_survival_time", stats["mean_survival_time"], step=step)
                mlflow.log_metric(f"{slug}_eliminations_per_match", stats["eliminations_per_match"], step=step)
            for team_name, stats in ppo_stats.items():
                slug = team_name.lower().replace(" ", "_")
                for key, value in stats.items():
                    mlflow.log_metric(f"{slug}_ppo_{key}", value, step=simulation.total_env_steps)
        win_rates = ", ".join(f"{team}={stats['win_rate']:.3f}" for team, stats in summary.items())
        print(
            f"steps={simulation.total_env_steps:,}/{target_steps:,} | matches={simulation.match_index} | "
            f"{win_rates}"
        )

    def run_loop() -> None:
        nonlocal last_saved_steps, last_logged_steps
        while simulation.total_env_steps < target_steps:
            finished = simulation.step(dt)
            if not finished:
                continue

            result = simulation.finish_match()
            if simulation.match_index % CONFIG.rl_metrics_every_matches == 0:
                exported_paths = metrics.record_match(result, simulation.cumulative_metrics)
                if mlflow is not None:
                    for plot_path in exported_paths:
                        mlflow.log_artifact(str(plot_path), artifact_path="plots")

            if simulation.total_env_steps - last_logged_steps >= CONFIG.rl_log_every_steps:
                last_logged_steps = simulation.total_env_steps
                log_entry(final=False)

            if simulation.total_env_steps - last_saved_steps >= CONFIG.rl_save_every_steps:
                last_saved_steps = simulation.total_env_steps
                save_rl_checkpoints(simulation.controllers)
                if mlflow is not None:
                    mlflow.log_artifacts(str(CONFIG.rl_checkpoint_dir), artifact_path="checkpoints")

            simulation.reset_match()

    if mlflow is None:
        run_loop()
    else:
        with mlflow.start_run(run_name=f"ppo_seed{seed}") as run:
            for key, value in vars(CONFIG).items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)
            mlflow.log_param("seed", seed)
            mlflow.log_param("from_scratch", args.from_scratch)
            mlflow.set_tag("domain_randomization", str(simulation.domain_randomization))
            mlflow.set_tag("device", str(CONFIG.rl_device))

            run_loop()

            save_rl_checkpoints(simulation.controllers)
            # The in-loop logger only fires at a match boundary and only every RL_LOG_EVERY_STEPS, so
            # with a 50k cadence and a 100k budget it fired once and the final policy was never logged.
            log_entry(final=True)
            mlflow.log_artifacts(str(CONFIG.rl_checkpoint_dir), artifact_path="checkpoints")

            for team_name, controller in simulation.controllers.items():
                slug = team_name.lower().replace(" ", "_")
                if controller.actor is not None:
                    mlflow.pytorch.log_model(
                        controller.actor,
                        artifact_path=f"model_{slug}_actor",
                        registered_model_name=f"CTDE_Arena_{slug.upper()}_Actor",
                    )

    save_rl_checkpoints(simulation.controllers)
    log_path = CONFIG.rl_checkpoint_dir / "training_log.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": seed,
                "target_steps": target_steps,
                "completed_steps": simulation.total_env_steps,
                "matches_played": simulation.match_index,
                "entries": training_log,
            },
            handle,
            indent=2,
        )
    if mlflow is not None:
        mlflow.log_artifact(str(log_path), artifact_path="logs")
    print(f"Training finished at {simulation.total_env_steps:,} steps | log: {log_path}")


if __name__ == "__main__":
    main()
