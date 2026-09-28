"""Render the replicated study into markdown tables for docs/08_results.md.

Reads results/study/per_seed_metrics.csv (written by run_study.py) and prints, per experiment:
a per-paradigm summary with between-seed spread and a pooled Wilson interval, a per-seed matrix so a
single-seed fluke is visible, and Fisher exact tests for every pairing.

Usage:  python scripts/report_study.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "study" / "per_seed_metrics.csv"


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def fisher_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    from math import comb

    total = k1 + k2
    n = n1 + n2
    lo, hi = max(0, total - n2), min(total, n1)
    denom = comb(n, total)
    observed = comb(n1, k1) * comb(n2, total - k1) / denom
    p = 0.0
    for x in range(lo, hi + 1):
        px = comb(n1, x) * comb(n2, total - x) / denom
        if px <= observed * (1 + 1e-9):
            p += px
    return min(1.0, p)


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    for row in rows:
        for key in row:
            if key not in {"experiment", "team", "paradigm"}:
                row[key] = float(row[key]) if key != "seed" else int(row[key])

    by_experiment: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_experiment[row["experiment"]].append(row)

    for experiment, exp_rows in sorted(by_experiment.items()):
        paradigms = sorted({r["paradigm"] for r in exp_rows})
        seeds = sorted({int(r["seed"]) for r in exp_rows})
        print(f"\n### {experiment} — {len(seeds)} seeds x {int(exp_rows[0]['eval_matches'])} held-out greedy matches\n")
        print("| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Accuracy |")
        print("|---|---:|---:|---:|---|---:|---:|---:|---:|")
        counts: dict[str, tuple[int, int]] = {}
        for paradigm in paradigms:
            sub = [r for r in exp_rows if r["paradigm"] == paradigm]
            rates = [r["eval_win_rate"] for r in sub]
            wins = round(sum(r["eval_win_rate"] * r["eval_matches"] for r in sub))
            trials = int(sum(r["eval_matches"] for r in sub))
            counts[paradigm] = (wins, trials)
            lo, hi = wilson(wins, trials)
            print(
                f"| {paradigm} | **{sum(rates)/len(rates):.3f}** | {sd(rates):.3f} | {wins}/{trials} "
                f"| [{lo:.3f}, {hi:.3f}] "
                f"| {sum(r['eval_eliminations_per_match'] for r in sub)/len(sub):.2f} "
                f"| {sum(r['eval_mean_survival'] for r in sub)/len(sub):.2f} "
                f"| {sum(r['eval_shots_per_match'] for r in sub)/len(sub):.2f} "
                f"| {sum(r['eval_shot_accuracy'] for r in sub)/len(sub):.4f} |"
            )

        print(f"\nPer-seed held-out win rate (each column is an independent replicate):\n")
        print("| Paradigm | " + " | ".join(f"seed {s}" for s in seeds) + " | best seed? |")
        print("|---|" + "---:|" * (len(seeds) + 1))
        for paradigm in paradigms:
            cells = []
            for s in seeds:
                match = [r for r in exp_rows if r["paradigm"] == paradigm and int(r["seed"]) == s]
                cells.append(f"{match[0]['eval_win_rate']:.3f}" if match else "—")
            best = sum(
                1
                for s in seeds
                if [r for r in exp_rows if int(r["seed"]) == s and r["paradigm"] == paradigm]
                and max(
                    (r for r in exp_rows if int(r["seed"]) == s), key=lambda r: r["eval_win_rate"]
                )["paradigm"]
                == paradigm
            )
            print(f"| {paradigm} | " + " | ".join(cells) + f" | {best}/{len(seeds)} |")

        print("\nPairwise Fisher exact tests on pooled held-out wins:\n")
        print("| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (α=0.0167) |")
        print("|---|---|---:|---:|---|")
        for i, a in enumerate(paradigms):
            for b in paradigms[i + 1:]:
                ka, na = counts[a]
                kb, nb = counts[b]
                p = fisher_two_sided(ka, na, kb, nb)
                odds = (ka / max(na - ka, 1)) / max(kb / max(nb - kb, 1), 1e-12)
                verdict = "**significant**" if p < 0.0167 else ("nominal only" if p < 0.05 else "not significant")
                print(f"| {a} vs {b} | {ka}/{na} vs {kb}/{nb} | {odds:.2f} | {p:.4f} | {verdict} |")


if __name__ == "__main__":
    main()
