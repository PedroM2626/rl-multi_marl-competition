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


def run_roots(runs_dir: Path) -> list[Path]:
    """Both trees, because each child resolves a relative `--out` against its own working directory.

    Experiment 1's replicates land in `data/...` and experiment 2's in `ctde_arena/data/...`. Reading only
    the launcher's root is the defect the audit records as A-35, and it produced an aggregate with half the
    replicates and no failure to report; an analysis that silently sees only one experiment is the same bug
    one step later, so this raises instead.
    """
    roots = [runs_dir]
    sibling = ROOT / "ctde_arena" / runs_dir.relative_to(ROOT) if runs_dir.is_relative_to(ROOT) else None
    if sibling is not None and sibling.is_dir() and sibling not in roots:
        roots.append(sibling)
    return roots


def find_run(runs_dirs: list[Path], name: str) -> Path | None:
    for base in runs_dirs:
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return None


def collect(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    with (args.study if args.study.is_absolute() else ROOT / args.study).open(
            newline="", encoding="utf-8") as handle:
        study_rows = list(csv.DictReader(handle))

    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else ROOT / args.runs_dir
    runs_dirs = run_roots(runs_dir)
    replicates = args.replicates if args.replicates.is_absolute() else ROOT / args.replicates
    cells: list[dict[str, object]] = []
    weight_matches = weight_differs = 0
    held_out_deltas: list[float] = []
    retrained_rates: dict[tuple[str, str, str], float] = {}

    for base in runs_dirs:
        for run_dir in sorted(base.glob("exp*_seed*")):
            summary = run_dir / "evaluation" / "run_summary.json"
            if not summary.is_file():
                continue
            payload = json.loads(summary.read_text(encoding="utf-8"))
            rotation = payload.get("paradigms", {})
            experiment = "exp1" if run_dir.name.startswith("exp1") else "exp2"
            for team, stats in payload.get("held_out_greedy", {}).get("per_team", {}).items():
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
        run_dir = find_run(runs_dirs, f"{experiment}_seed{seed}")
        if run_dir is None:
            raise SystemExit(f"no re-trained run directory for {experiment}_seed{seed} under "
                             f"{[p.as_posix() for p in runs_dirs]}")
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
            # The signed version, because the absolute one conflates the two directions: a cell where the
            # frozen policy reads better than it trained and one where it reads worse move the same way.
            "gap": float(row["eval_win_rate"]) - float(row["train_win_rate"]),
            **{feature: value for feature, value in features.items() if feature in FEATURES},
            "trace_points": features["trace_points"],
        })
    controls = {
        # A-42: a committed record spells paths with forward slashes and relative to the repository,
        # so a reader on another host can still see where the traces were read from.
        "run_roots": [p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix()
                    for p in runs_dirs],
        "checkpoint_comparisons": weight_matches + weight_differs,
        "checkpoints_hash_matching_the_versioned_set": weight_matches,
        "checkpoints_differing": weight_differs,
        "retrained_runs_with_a_held_out_summary": len({k[:2] for k in retrained_rates}),
        "max_abs_held_out_delta_vs_study": max((abs(d) for d in held_out_deltas), default=float("nan")),
        "mean_abs_held_out_delta_vs_study": statistics.fmean(
            [abs(d) for d in held_out_deltas]) if held_out_deltas else float("nan"),
    }
    return cells, controls


def spearman_p(value: float, n: int) -> float:
    """Two-sided p for a Spearman rho by Fisher's z, which is the approximation the sample size supports.

    n = 60 with 24 correlations on the table is exactly the situation where a bare rho invites a reading it
    cannot support, so the p-values and the corrected threshold are computed here rather than left to whoever
    quotes them.
    """
    if math.isnan(value) or n < 5 or abs(value) >= 1.0:
        return float("nan")
    z = math.atanh(value) * math.sqrt(n - 3)
    return math.erfc(abs(z) / math.sqrt(2))


def correlate(cells: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    targets = {
        "held_out_win_rate": [float(c["held_out_win_rate"]) for c in cells],
        "train_win_rate": [float(c["train_win_rate"]) for c in cells],
        "divergence": [float(c["divergence"]) for c in cells],
        "gap": [float(c["gap"]) for c in cells],
    }
    out: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        values = [float(c[feature]) for c in cells]
        if all(math.isnan(v) for v in values):
            continue
        entry: dict[str, float] = {}
        for target, series in targets.items():
            pairs = [(v, s) for v, s in zip(values, series) if not math.isnan(v)]
            rho = spearman([v for v, _ in pairs], [s for _, s in pairs])
            entry[f"spearman_with_{target}"] = rho
            entry[f"p_with_{target}"] = spearman_p(rho, len(pairs))
        out[feature] = entry
    tests = sum(len(v) // 2 for v in out.values())
    for entry in out.values():
        entry["bonferroni_alpha"] = round(0.05 / tests, 6) if tests else float("nan")
    return out


def run_level(cells: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """The same correlations with the run as the unit, which is the only independent one.

    Three arms inside a run share its RNG stream, its opponents and its match history, so 60 cells are 20
    draws with three readings each. A signal that is real across runs survives the collapse; one that was
    carried by within-run structure does not, and the difference between those two is the whole content of
    this block.
    """
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        key = (str(cell["experiment"]), int(cell["seed"]))
        grouped[key]["held_out_win_rate"].append(float(cell["held_out_win_rate"]))
        for feature in FEATURES:
            value = float(cell[feature])
            if not math.isnan(value):
                grouped[key][feature].append(value)
    held = [statistics.fmean(values["held_out_win_rate"]) for values in grouped.values()]
    out: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        series = [statistics.fmean(values[feature]) for values in grouped.values() if values.get(feature)]
        if len(series) != len(held):
            continue
        rho = spearman(held, series)
        out[feature] = {"spearman_with_held_out_win_rate": rho,
                        "p_with_held_out_win_rate": spearman_p(rho, len(series)),
                        "runs": len(series)}
    return out


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cells, controls = collect(args)
    print(f"{len(cells)} arm-replicate cells; controls: {json.dumps(controls)}", flush=True)
    correlations = correlate(cells)
    print("\n| feature | ρ vs held-out | p | ρ vs in-training | ρ vs the signed gap | p | ρ vs abs(gap) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for feature, values in sorted(correlations.items(),
                                  key=lambda kv: -abs(kv[1]["spearman_with_held_out_win_rate"])):
        print("| {feature} | {a:+.3f} | {pa:.4f} | {b:+.3f} | {g:+.3f} | {pg:.4f} | {c:+.3f} |".format(
            feature=feature, a=values["spearman_with_held_out_win_rate"],
            pa=values["p_with_held_out_win_rate"], b=values["spearman_with_train_win_rate"],
            g=values["spearman_with_gap"], pg=values["p_with_gap"],
            c=values["spearman_with_divergence"]))

    alpha = next(iter(correlations.values()))["bonferroni_alpha"]
    print(f"\n{sum(len(v) - 1 for v in correlations.values()) // 2 * 3} correlations; "
          f"Bonferroni alpha = {alpha:.5f}")

    per_run = run_level(cells)
    runs = next(iter(per_run.values()))["runs"]
    print(f"\nSame statistics with the **run** as the unit (n = {runs} independent draws, not {len(cells)} "
          "cells):")
    print("| feature | rho vs held-out, run-level | p | cell-level rho for comparison |")
    print("|---|---:|---:|---:|")
    for feature, values in sorted(per_run.items(),
                                  key=lambda kv: -abs(kv[1]["spearman_with_held_out_win_rate"])):
        print("| {feature} | {r:+.3f} | {p:.4f} | {c:+.3f} |".format(
            feature=feature, r=values["spearman_with_held_out_win_rate"],
            p=values["p_with_held_out_win_rate"],
            c=correlations[feature]["spearman_with_held_out_win_rate"]))

    extreme = sorted(cells, key=lambda c: float(c["held_out_win_rate"]))[:3] + \
        sorted(cells, key=lambda c: float(c["held_out_win_rate"]))[-2:]
    payload = {
        "cells": len(cells),
        "runs": runs,
        "controls": controls,
        "correlations": correlations,
        "run_level_correlations": per_run,
        "bonferroni_alpha": alpha,
        "extreme_cells": [{k: c[k] for k in ("experiment", "seed", "paradigm", "held_out_win_rate",
                                             "train_win_rate", "divergence", "gap", "approx_kl_final",
                                             "entropy_final", "value_loss_final")} for c in extreme],
        "per_cell": cells,
    }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
