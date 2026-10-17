"""Correlate the optimiser's own trace with what the study measured, and say what the correlation is worth.

Open work item 4 of the audit asked for exactly this: the study judges a replicate by its held-out win rate,
and nothing yet connects that number to what the update steps looked like from the inside. The re-training
campaign (`scripts/run_traces_campaign.sh`) replays the twenty replicates with the same seeds and the same
budget so that `run_experiment.py` now leaves a `training_log.json` behind, and this script joins those
traces to the published per-replicate metrics.

Two controls come out of the same pass and are written into the record, because the analysis is worthless
without them. The first is whether the re-training reproduced the committed weights at all: if the
checkpoints hash-match `data/training_replicates/`, the trace belongs to the networks the tables quote; if
not, it belongs to networks trained the same way, which is a weaker statement and has to be labelled. The
second is whether the re-trained held-out numbers still equal `results/study/`, which is the same question
asked of the outcome rather than the weights.

Usage:
    python scripts/analyse_traces.py --runs-dir data/runs_traces --out results/trace_analysis.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FEATURES = ("approx_kl_final", "approx_kl_mean", "approx_kl_max", "entropy_final", "entropy_mean",
            "value_loss_final", "value_loss_mean", "policy_loss_final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "data" / "runs_traces",
                        help="the re-trained replicates whose checkpoints/ hold a training_log.json")
    parser.add_argument("--study", type=Path, default=ROOT / "results" / "study" / "per_seed_metrics.csv",
                        help="the published per-replicate metrics the traces are joined to")
    parser.add_argument("--replicates", type=Path, default=ROOT / "data" / "training_replicates",
                        help="the versioned checkpoints the re-training is expected to reproduce")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "trace_analysis.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(values: list[float]) -> list[float]:
    """Average ranks, ties shared -- the input to a Spearman that needs no scipy."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = average
        i = j + 1
    return out


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or statistics.pstdev(a) == 0 or statistics.pstdev(b) == 0:
        return float("nan")
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return numerator / denominator if denominator else float("nan")


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(rank(a), rank(b))


def trace_features(log: Path) -> dict[str, dict[str, float]]:
    """Per-team summary statistics of one replicate's optimiser trace."""
    payload = json.loads(log.read_text(encoding="utf-8"))
    entries = payload["entries"]
    by_team: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        for team, stats in entry["ppo_stats"].items():
            for key, value in stats.items():
                by_team[team][key].append(float(value))
    out: dict[str, dict[str, float]] = {}
    for team, series in by_team.items():
        kl, entropy, value, policy = (series.get("approx_kl", []), series.get("entropy", []),
                                      series.get("value_loss", []), series.get("policy_loss", []))
        out[team] = {
            "approx_kl_final": kl[-1] if kl else float("nan"),
            "approx_kl_mean": statistics.fmean(kl) if kl else float("nan"),
            "approx_kl_max": max(kl) if kl else float("nan"),
            "entropy_final": entropy[-1] if entropy else float("nan"),
            "entropy_mean": statistics.fmean(entropy) if entropy else float("nan"),
            "value_loss_final": value[-1] if value else float("nan"),
            "value_loss_mean": statistics.fmean(value) if value else float("nan"),
            "policy_loss_final": policy[-1] if policy else float("nan"),
            "trace_points": len(entries),
        }
    return out


def collect(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    with (args.study if args.study.is_absolute() else ROOT / args.study).open(
            newline="", encoding="utf-8") as handle:
        study_rows = list(csv.DictReader(handle))

    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else ROOT / args.runs_dir
    replicates = args.replicates if args.replicates.is_absolute() else ROOT / args.replicates
    cells: list[dict[str, object]] = []
    weight_matches = weight_differs = 0
    held_out_deltas: list[float] = []
    retrained_rates: dict[tuple[str, str, str], float] = {}

    for run_dir in sorted(runs_dir.glob("exp*_seed*")):
        summary = run_dir / "evaluation" / "run_summary.json"
        if summary.is_file():
            payload = json.loads(summary.read_text(encoding="utf-8"))
            rotation = payload.get("paradigms", {})
            for team, stats in payload.get("held_out_greedy", {}).get("per_team", {}).items():
                experiment = "exp1" if run_dir.name.startswith("exp1") else "exp2"
                retrained_rates[(experiment, run_dir.name.rsplit("seed", 1)[1], rotation.get(team, ""))] = \
                    float(stats["win_rate"])
        for candidate in sorted((run_dir / "checkpoints").glob("*.pt")):
            twin = replicates / run_dir.name / candidate.name
            if twin.is_file() and sha256(twin) == sha256(candidate):
                weight_matches += 1
            else:
                weight_differs += 1

    for row in study_rows:
        experiment, seed, paradigm = row["experiment"], row["seed"], row["paradigm"]
        run_dir = runs_dir / f"{experiment}_seed{seed}"
        log = run_dir / "checkpoints" / "training_log.json"
        if not log.is_file():
            raise SystemExit(f"{log} missing: the re-training did not produce a trace for "
                             f"{experiment}_seed{seed}")
        features = trace_features(log).get(row["team"])
        if not features:
            raise SystemExit(f"{log} records no stats for {row['team']}")
        key = (experiment, seed, paradigm)
        if key in retrained_rates:
            held_out_deltas.append(retrained_rates[key] - float(row["eval_win_rate"]))
        cells.append({
            "experiment": experiment, "seed": int(seed), "paradigm": paradigm, "team": row["team"],
            "held_out_win_rate": float(row["eval_win_rate"]),
            "retrained_held_out_win_rate": retrained_rates.get(key),
            "train_win_rate": float(row["train_win_rate"]),
            "divergence": abs(float(row["eval_win_rate"]) - float(row["train_win_rate"])),
            **{feature: value for feature, value in features.items() if feature in FEATURES},
            "trace_points": features["trace_points"],
        })
    controls = {
        "checkpoint_comparisons": weight_matches + weight_differs,
        "checkpoints_hash_matching_the_versioned_set": weight_matches,
        "checkpoints_differing": weight_differs,
        "retrained_runs_with_a_held_out_summary": len(retrained_rates),
        "max_abs_held_out_delta_vs_study": max((abs(d) for d in held_out_deltas), default=float("nan")),
        "mean_abs_held_out_delta_vs_study": statistics.fmean(
            [abs(d) for d in held_out_deltas]) if held_out_deltas else float("nan"),
    }
    return cells, controls


def correlate(cells: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    targets = {
        "held_out_win_rate": [float(c["held_out_win_rate"]) for c in cells],
        "train_win_rate": [float(c["train_win_rate"]) for c in cells],
        "divergence": [float(c["divergence"]) for c in cells],
    }
    out: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        values = [float(c[feature]) for c in cells]
        if all(math.isnan(v) for v in values):
            continue
        out[feature] = {f"spearman_with_{target}": spearman(values, series)
                        for target, series in targets.items()}
        out[feature].update({f"pearson_with_{target}": pearson(values, series)
                             for target, series in targets.items()})
    return out


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cells, controls = collect(args)
    print(f"{len(cells)} arm-replicate cells; controls: {json.dumps(controls)}", flush=True)
    correlations = correlate(cells)
    print("\n| feature | Spearman with held-out | with in-training | with the divergence |")
    print("|---|---:|---:|---:|")
    for feature, values in sorted(correlations.items(),
                                  key=lambda kv: -abs(kv[1]["spearman_with_held_out_win_rate"])):
        print("| {feature} | {a:+.3f} | {b:+.3f} | {c:+.3f} |".format(
            feature=feature, a=values["spearman_with_held_out_win_rate"],
            b=values["spearman_with_train_win_rate"], c=values["spearman_with_divergence"]))

    extreme = sorted(cells, key=lambda c: float(c["held_out_win_rate"]))[:3] + \
        sorted(cells, key=lambda c: float(c["held_out_win_rate"]))[-2:]
    payload = {
        "cells": len(cells),
        "controls": controls,
        "correlations": correlations,
        "extreme_cells": [{k: c[k] for k in ("experiment", "seed", "paradigm", "held_out_win_rate",
                                             "train_win_rate", "divergence", "approx_kl_final",
                                             "entropy_final", "value_loss_final")} for c in extreme],
        "per_cell": cells,
    }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
