"""Launch the replicated study across seeds and both experiment trees.

Runs `scripts/run_experiment.py` as one subprocess per (experiment, seed) with torch pinned to a
single thread, so N runs occupy N cores instead of contending. Collects every `run_summary.json`
into an aggregated CSV and JSON under `results/study/`.

Usage:
    python scripts/run_study.py --steps 1000000 --seeds 1,2,3,4,5 --jobs 10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREES = {"exp1": ROOT, "exp2": ROOT / "ctde_arena"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--experiments", type=str, default="exp1,exp2")
    parser.add_argument("--jobs", type=int, default=10, help="concurrent runs")
    parser.add_argument("--eval-matches", type=int, default=200)
    parser.add_argument("--curve-matches", type=int, default=60)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "runs")
    return parser.parse_args()


def launch(experiment: str, tree: Path, out_dir: Path, seed: int, args: argparse.Namespace) -> tuple[str, int, float, str]:
    label = f"{experiment}_seed{seed}"
    log_path = out_dir / f"{label}.log"
    command = [
        sys.executable, "scripts/run_experiment.py",
        "--seed", str(seed),
        "--steps", str(args.steps),
        "--out", str(out_dir / label),
        "--eval-matches", str(args.eval_matches),
        "--curve-matches", str(args.curve_matches),
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=tree, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return label, proc.returncode, time.time() - started, str(out_dir / label / "evaluation" / "run_summary.json")


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion estimated from `n` Bernoulli trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def fisher_exact_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Exact two-sided Fisher test for two binomials, via the hypergeometric tail."""
    from math import comb

    total = k1 + k2
    n = n1 + n2
    if total == 0 or total > 400:  # keep the summation cheap and bounded
        return float("nan")
    lo, hi = max(0, total - n2), min(total, n1)
    denom = comb(n, total)
    observed = comb(n1, k1) * comb(n2, total - k1) / denom
    p = 0.0
    for x in range(lo, hi + 1):
        px = comb(n1, x) * comb(n2, total - x) / denom
        if px <= observed * (1 + 1e-9):
            p += px
    return min(1.0, p)


def collect(out_dir: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for summary in sorted(out_dir.glob("*/evaluation/run_summary.json")):
        payload = json.loads(summary.read_text(encoding="utf-8"))
        experiment = summary.parents[1].name.split("_seed")[0]
        for team, stats in payload["held_out_greedy"]["per_team"].items():
            records.append({
                "experiment": experiment,
                "seed": payload["seed"],
                "team": team,
                "paradigm": payload["paradigms"][team],
                "eval_win_rate": round(stats["win_rate"], 4),
                "eval_eliminations_per_match": round(stats["eliminations_per_match"], 4),
                "eval_mean_survival": round(stats["mean_survival_time"], 4),
                "eval_shots_per_match": round(stats["shots_per_match"], 4),
                "eval_shot_accuracy": round(stats["shot_accuracy"], 4),
                "train_win_rate": round(payload["training_cumulative"][team]["win_rate"], 4),
                "steps_per_second": payload["steps_per_second"],
                "train_seconds": payload["train_seconds"],
                "matches_played": payload["matches_played"],
                "eval_matches": payload["held_out_greedy"]["matches"],
            })
    return records


def analyse(records: list[dict[str, object]]) -> dict[str, object]:
    analysis: dict[str, object] = {}
    experiments = sorted({str(r["experiment"]) for r in records})
    for experiment in experiments:
        rows = [r for r in records if r["experiment"] == experiment]
        paradigms = sorted({str(r["paradigm"]) for r in rows})
        per_paradigm: dict[str, object] = {}
        for paradigm in paradigms:
            subset = [r for r in rows if r["paradigm"] == paradigm]
            wins = sum(r["eval_win_rate"] * r["eval_matches"] for r in subset)
            trials = sum(r["eval_matches"] for r in subset)
            lo, hi = wilson(round(wins), int(trials))
            per_paradigm[paradigm] = {
                "seeds": len(subset),
                "pooled_win_rate": round(wins / trials, 4),
                "wilson_95_ci_pooled": [round(lo, 4), round(hi, 4)],
                "per_seed_win_rates": [r["eval_win_rate"] for r in sorted(subset, key=lambda x: x["seed"])],
                "mean_survival": round(sum(r["eval_mean_survival"] for r in subset) / len(subset), 3),
                "eliminations_per_match": round(sum(r["eval_eliminations_per_match"] for r in subset) / len(subset), 3),
                "shots_per_match": round(sum(r["eval_shots_per_match"] for r in subset) / len(subset), 3),
                "shot_accuracy": round(sum(r["eval_shot_accuracy"] for r in subset) / len(subset), 4),
            }
        pairs = {}
        for i, a in enumerate(paradigms):
            for b in paradigms[i + 1:]:
                sa = [r for r in rows if r["paradigm"] == a]
                sb = [r for r in rows if r["paradigm"] == b]
                ka = round(sum(r["eval_win_rate"] * r["eval_matches"] for r in sa))
                kb = round(sum(r["eval_win_rate"] * r["eval_matches"] for r in sb))
                na = int(sum(r["eval_matches"] for r in sa))
                nb = int(sum(r["eval_matches"] for r in sb))
                pairs[f"{a} vs {b}"] = {
                    "wins": [ka, na, kb, nb],
                    "fisher_two_sided_p": round(fisher_exact_two_sided(ka, na, kb, nb), 6),
                }
        analysis[experiment] = {"per_paradigm": per_paradigm, "pairwise": pairs}
    return analysis


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    experiments = args.experiments.split(",")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(exp, TREES[exp], out_dir, seed, args) for exp in experiments for seed in seeds]
    print(f"launching {len(jobs)} runs, {min(args.jobs, len(jobs))} concurrent, {args.steps:,} steps each", flush=True)

    started = time.time()
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(launch, *job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            label, code, seconds, path = future.result()
            status = "ok" if code == 0 else f"FAILED rc={code}"
            print(f"  [{status}] {label} in {seconds/60:.1f} min -> {path}", flush=True)
            if code != 0:
                failures.append(label)

    records = collect(out_dir, args)
    results_dir = ROOT / "results" / "study"
    results_dir.mkdir(parents=True, exist_ok=True)
    if records:
        with (results_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        (results_dir / "analysis.json").write_text(json.dumps(analyse(records), indent=2), encoding="utf-8")

    print(f"\nwall clock {(time.time() - started)/60:.1f} min | {len(records)} per-seed rows | failures: {failures or 'none'}")
    print(f"wrote {results_dir / 'per_seed_metrics.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
