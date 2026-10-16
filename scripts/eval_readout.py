"""Re-evaluate the study's existing checkpoints under the other readout of the same policy.

[§ 8.1](../docs/08_results.md#readout-divergence) established that the held-out greedy win rate the study is
judged by is not the trained behaviour: the greedy readout has 2.5x the between-seed spread of the sampled
policy it summarises, and it preserved the within-replicate ranking in only 35.6 % of comparable pairs. That
was measured from artefacts the training run had already written. This script measures it directly, by playing
the same weights twice: once at `EVAL_TEMPERATURE=0` (argmax, the published readout) and once by sampling the
policy at a fixed temperature.

Nothing is trained. The protocol is the study's own held-out evaluation reproduced exactly -- the run
directory's three checkpoints, `ArenaSimulation(seed=replicate_seed + 9000, domain_randomization=False)`,
`set_rl_training(False)`, 150 matches, `finish_match` then `reset_match` -- so the only thing that differs
between the two arms is how an action is chosen from the logits.

The output is deliberately the same schema as `run_study.py`'s `per_seed_metrics.csv`, which means the
existing instrument reads it:

    python scripts/report_study.py --results-dir results/readout/T1 --compare-to results/study --label "readout T=1"

gives per-seed paired tables of the sampled readout against the published greedy one. Running `--temperature 0`
into its own directory is the control: that difference is pure re-measurement noise, and it is the yardstick
for reading the temperature arms.

    python scripts/eval_readout.py --temperature 1.0 --seeds 1,2,3,4,5,6,7,8,9,10
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, required=True,
                        help="softmax temperature for the sampled readout; 0 replays the published argmax")
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10",
                        help="training seeds of the replicates to re-evaluate")
    parser.add_argument("--runs-dir", type=Path, default=Path("data/runs"),
                        help="the study scratch holding <experiment>_seedN/checkpoints")
    parser.add_argument("--experiments", default="exp1", help="comma-separated: exp1, exp2")
    parser.add_argument("--matches", type=int, default=150, help="held-out matches per replicate")
    parser.add_argument("--seed-offset", type=int, default=9000,
                        help="the offset run_experiment.py adds to the replicate seed for its held-out stream")
    parser.add_argument("--out", type=Path, default=None,
                        help="default results/readout/T<temperature>, in the run_study.py CSV schema")
    parser.add_argument("--require-checkpoints", action="store_true",
                        help="fail if a replicate's checkpoints are absent instead of skipping it")
    return parser.parse_args()


def evaluate(matches: int, seed: int) -> dict:
    """The study's own held-out protocol, with whatever `CONFIG.eval_temperature` says.

    Copied from `run_experiment.py::evaluate` rather than imported, because that function is inside a training
    script that would also want to run training; keeping it here is what makes the two arms' match streams
    comparable, and `tests/test_readout_parity.py` pins the two bodies line-for-line except for the readout.
    """
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
    return {"per_team": summary, "median_duration_seconds": sorted(durations)[len(durations) // 2]}


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import csv

    from marl_arena.config import CONFIG, seed_all
    from marl_arena.controllers import rl_controller

    out_dir = args.out or Path("results") / "readout" / f"T{args.temperature:g}"
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else ROOT / args.runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    skipped: list[str] = []
    started = time.time()
    for experiment in args.experiments.split(","):
        for seed in (int(s) for s in args.seeds.split(",")):
            directory = runs_dir / f"{experiment}_seed{seed}"
            checkpoints = directory / "checkpoints"
            if not checkpoints.is_dir() or not list(checkpoints.glob("*.pt")):
                if args.require_checkpoints:
                    raise SystemExit(f"{experiment}_seed{seed}: no checkpoints under {checkpoints}")
                skipped.append(f"{experiment}_seed{seed}")
                print(f"  {experiment}_seed{seed}: no scratch checkpoints, skipped", flush=True)
                continue
            # The readout and the checkpoint directory are both swapped per replicate, in the module the
            # controller resolves them from -- CONFIG is frozen and read at call time inside the loop.
            rl_controller.CONFIG = dataclasses.replace(CONFIG, eval_temperature=args.temperature,
                                                      rl_checkpoint_dir=checkpoints)
            seed_all(seed)
            elapsed = time.time()
            result = evaluate(args.matches, seed + args.seed_offset)
            payload = json.loads((directory / "evaluation" / "run_summary.json").read_text(encoding="utf-8")) \
                if (directory / "evaluation" / "run_summary.json").is_file() else {}
            for team, stats in result["per_team"].items():
                records.append({
                    "experiment": experiment,
                    "seed": seed,
                    "team": team,
                    "paradigm": payload.get("paradigms", {}).get(team, ""),
                    "eval_win_rate": round(stats["win_rate"], 4),
                    "eval_eliminations_per_match": round(stats["eliminations_per_match"], 4),
                    "eval_mean_survival": round(stats["mean_survival_time"], 4),
                    "eval_shots_per_match": round(stats["shots_per_match"], 4),
                    "eval_shot_accuracy": round(stats["shot_accuracy"], 4),
                    "train_win_rate": round(payload.get("training_cumulative", {}).get(team, {})
                                            .get("win_rate", float("nan")), 4),
                    "steps_per_second": payload.get("steps_per_second"),
                    "train_seconds": payload.get("train_seconds"),
                    "matches_played": payload.get("matches_played"),
                    "eval_matches": args.matches,
                    "published_greedy_win_rate": round(payload.get("held_out_greedy", {}).get("per_team", {})
                                                       .get(team, {}).get("win_rate", float("nan")), 4),
                    "eval_seconds": round(time.time() - elapsed, 1),
                    "median_duration_seconds": round(result["median_duration_seconds"], 2),
                })
            print(f"  {experiment}_seed{seed}: win rates "
                  f"{ {r['team']: r['eval_win_rate'] for r in records if r['seed'] == seed and r['experiment'] == experiment} }"
                  f" in {time.time() - elapsed:.0f}s", flush=True)

    if not records:
        raise SystemExit("no replicate could be evaluated -- check --runs-dir and --experiments")

    fields = list(records[0])
    with (out_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    (out_dir / "summary.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "readout": "argmax" if args.temperature == 0 else f"sampled at temperature {args.temperature}",
        "eval_temperature": args.temperature,
        "protocol": {"matches_per_replicate": args.matches, "seed_offset": args.seed_offset,
                     "domain_randomization": False, "training_mode": False,
                     "hidden_dim": CONFIG.rl_hidden_dim,
                     "arena": {"alive_bonus": CONFIG.reward_alive_bonus,
                               "alive_decay_half_life_s": CONFIG.reward_alive_decay_half_life,
                               "cap_awards_win": CONFIG.termination_cap_awards_win,
                               "match_duration_seconds": CONFIG.match_duration_seconds}},
        "replicates": len({(r["experiment"], r["seed"]) for r in records}),
        "skipped_for_absent_checkpoints": skipped,
        "wall_seconds": round(time.time() - started, 1),
        "note": "The run directories are git-ignored scratch, so this re-evaluation only runs where the "
                "study was trained. The versioned blocks under data/training_replicates/ are the portable "
                "equivalent, and the round-robin plays those.",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'per_seed_metrics.csv'} ({len(records)} rows) and summary.json")


if __name__ == "__main__":
    sys.exit(main())
