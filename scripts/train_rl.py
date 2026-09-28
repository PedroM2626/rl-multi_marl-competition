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
    parser = argparse.ArgumentParser(description="Train the CTE / DTE / CTDE policies with PPO.")
    parser.add_argument("--seed", type=int, default=None, help="overrides RANDOM_SEED for this run")
    parser.add_argument("--steps", type=int, default=None, help="overrides RL_TRAIN_TOTAL_STEPS")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="do not load the checkpoints already in data/checkpoints; train from random weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from marl_arena.config import CONFIG, seed_all
    from marl_arena.controllers.rl_controller import save_rl_checkpoints, set_rl_training
    from marl_arena.systems.metrics import MetricsStore
    from marl_arena.systems.simulation import ArenaSimulation

    seed = CONFIG.random_seed if args.seed is None else args.seed
    seed_all(seed)

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
        entry = {
            "env_steps": simulation.total_env_steps,
            "matches": simulation.match_index,
            "winner": simulation.last_match_result.winner_team if simulation.last_match_result else None,
            "variant": simulation.match_variant.summary(),
            "summary": summary,
            "ppo_stats": {
                team: {
                    "policy_loss": stats.policy_loss,
                    "value_loss": stats.value_loss,
                    "entropy": stats.entropy,
                    "approx_kl": stats.approx_kl,
                }
                for team, stats in simulation.last_ppo_stats.items()
            },
            "final": final,
        }
        training_log.append(entry)
        win_rates = ", ".join(f"{team}={stats['win_rate']:.3f}" for team, stats in summary.items())
        print(
            f"steps={simulation.total_env_steps:,}/{target_steps:,} | matches={simulation.match_index} | "
            f"{win_rates}"
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
            log_entry(final=False)

        if simulation.total_env_steps - last_saved_steps >= CONFIG.rl_save_every_steps:
            last_saved_steps = simulation.total_env_steps
            save_rl_checkpoints(simulation.controllers)
            print(f"[save] checkpoints at {CONFIG.rl_checkpoint_dir} (@ {simulation.total_env_steps:,} steps)")

        simulation.reset_match()

    save_rl_checkpoints(simulation.controllers)
    # The in-loop logger only fires at a match boundary and only every RL_LOG_EVERY_STEPS, so a run
    # whose budget is not a large multiple of that cadence never records its end state at all.
    log_entry(final=True)

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
    print(f"Training finished at {simulation.total_env_steps:,} steps | log: {log_path}")


if __name__ == "__main__":
    main()
