"""One replicated run: seeded PPO training plus held-out greedy evaluation.

Identical in both experiment trees; which `marl_arena` package it binds to is decided by the
calling working directory, the same way `train_rl.py` already works.

Usage:
    python scripts/run_experiment.py --seed 1 --steps 1000000 --out data/runs/exp1_s1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path, required=True, help="run artefact directory")
    parser.add_argument("--eval-matches", type=int, default=200, help="greedy matches at the final checkpoint")
    parser.add_argument("--curve-matches", type=int, default=60, help="greedy matches at each curve point")
    parser.add_argument("--curve-at", type=str, default="0.25,0.5,0.75,1.0", help="fractions of the step budget")
    parser.add_argument("--metrics-every", type=int, default=100, help="record training metrics every N matches")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Delegates to the library helper so the study and scripts/train_rl.py seed identically."""
    from marl_arena.config import seed_all

    seed_all(seed)


def evaluate(matches: int, seed: int) -> dict[str, object]:
    """Greedy, fixed-variant, no-domain-randomisation evaluation of the checkpoints on disk."""
    from marl_arena.config import CONFIG
    from marl_arena.controllers.rl_controller import set_rl_training
    from marl_arena.systems.simulation import ArenaSimulation

    sim = ArenaSimulation(seed=seed, domain_randomization=False, load_checkpoints=True)
    set_rl_training(sim.controllers, False)

    per_team: dict[str, dict[str, float]] = {
        name: {"wins": 0, "eliminations": 0, "survival_sum": 0.0, "hits": 0, "misses": 0, "matches": 0}
        for name in sim.cumulative_metrics
    }
    durations: list[float] = []
    rows: list[dict[str, object]] = []
    dt = CONFIG.sim_step_dt

    for _ in range(matches):
        while not sim.step(dt):
            pass
        result = sim.finish_match()
        durations.append(result.duration_seconds)
        for row in result.team_rows:
            stats = per_team[row["team_name"]]
            stats["wins"] += row["winner"]
            stats["eliminations"] += row["eliminations"]
            stats["survival_sum"] += row["mean_survival_time"]
            stats["hits"] += row["shots_hit"]
            stats["misses"] += row["shots_missed"]
            stats["matches"] += 1
            rows.append({k: row[k] for k in ("match_index", "team_name", "paradigm", "winner",
                                             "eliminations", "mean_survival_time", "shot_accuracy")})
        sim.reset_match()

    summary = {}
    for name, stats in per_team.items():
        n = max(stats["matches"], 1)
        total_shots = stats["hits"] + stats["misses"]
        summary[name] = {
            "win_rate": stats["wins"] / n,
            "eliminations_per_match": stats["eliminations"] / n,
            "mean_survival_time": stats["survival_sum"] / n,
            "shots_per_match": total_shots / n,
            "shot_accuracy": stats["hits"] / max(total_shots, 1),
        }
    return {"matches": matches, "per_team": summary, "rows": rows,
            "median_duration_seconds": sorted(durations)[len(durations) // 2] if durations else None}


def main() -> None:
    args = parse_args()
    run_dir = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ARENA_DATA_DIR"] = str(run_dir)
    os.environ["RANDOM_SEED"] = str(args.seed)
    os.environ["RL_TRAIN_TOTAL_STEPS"] = str(args.steps)
    os.environ["RL_METRICS_EVERY_MATCHES"] = str(args.metrics_every)
    os.environ["DOMAIN_RANDOMIZATION"] = "true"

    seed_everything(args.seed)

    import torch

    from marl_arena.config import CONFIG
    from marl_arena.controllers.rl_controller import save_rl_checkpoints, set_rl_training
    from marl_arena.systems.metrics import MetricsStore
    from marl_arena.systems.simulation import ArenaSimulation

    torch.set_num_threads(1)

    fractions = sorted(float(f) for f in args.curve_at.split(","))
    next_curve = 0

    metrics = MetricsStore()
    sim = ArenaSimulation(seed=args.seed, domain_randomization=True, load_checkpoints=False)
    set_rl_training(sim.controllers, True)

    started = time.time()
    eval_seconds = 0.0
    curve: list[dict[str, object]] = []
    while sim.total_env_steps < args.steps:
        if sim.step(CONFIG.sim_step_dt):
            result = sim.finish_match()
            if sim.match_index % CONFIG.rl_metrics_every_matches == 0:
                metrics.record_match(result, sim.cumulative_metrics)
            sim.reset_match()

            while next_curve < len(fractions) and sim.total_env_steps >= fractions[next_curve] * args.steps:
                save_rl_checkpoints(sim.controllers)
                eval_started = time.time()
                point = evaluate(args.curve_matches, args.seed + 5000)
                eval_seconds += time.time() - eval_started
                curve.append({
                    "env_steps": sim.total_env_steps,
                    "fraction": fractions[next_curve],
                    "elapsed_seconds": round(time.time() - started, 1),
                    "per_team": point["per_team"],
                })
                print(f"[curve] steps={sim.total_env_steps:,} "
                      + " ".join(f"{t}={s['win_rate']:.3f}" for t, s in point["per_team"].items()), flush=True)
                next_curve += 1

    save_rl_checkpoints(sim.controllers)
    train_seconds = time.time() - started - eval_seconds

    eval_started = time.time()
    final = evaluate(args.eval_matches, args.seed + 9000)
    eval_seconds += time.time() - eval_started
    curve.append({"env_steps": sim.total_env_steps, "fraction": 1.0,
                  "elapsed_seconds": round(train_seconds, 1), "per_team": final["per_team"]})

    eval_path = run_dir / "evaluation"
    eval_path.mkdir(exist_ok=True)
    with (eval_path / "eval_match_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(final["rows"])

    payload = {
        "seed": args.seed,
        "target_steps": args.steps,
        "completed_steps": sim.total_env_steps,
        "matches_played": sim.match_index,
        "train_seconds": round(train_seconds, 1),
        "evaluation_seconds": round(eval_seconds, 1),
        "steps_per_second": round(sim.total_env_steps / max(train_seconds, 1e-9), 1),
        "device": CONFIG.rl_device,
        "paradigm_rotation": sim.paradigm_rotation,
        "paradigms": {name: controller.paradigm for name, controller in sim.controllers.items()},
        "training_cumulative": {t: m.as_summary() for t, m in sim.cumulative_metrics.items()},
        "held_out_greedy": {
            "matches": final["matches"],
            "median_duration_seconds": final["median_duration_seconds"],
            "per_team": final["per_team"],
        },
        "curve": curve,
    }
    (eval_path / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] seed={args.seed} steps={sim.total_env_steps:,} "
          f"{payload['steps_per_second']:.0f} steps/s -> {eval_path / 'run_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
