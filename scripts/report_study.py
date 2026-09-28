"""Render the replicated study into markdown tables for docs/08_results.md.

Reads results/study/per_seed_metrics.csv (written by run_study.py) and prints, per experiment:
a per-paradigm summary with between-seed spread and a pooled Wilson interval, a per-seed matrix so a
single-seed fluke is visible, and Fisher exact tests for every pairing.

Usage:  python scripts/report_study.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "study" / "per_seed_metrics.csv"

# The tables contain em dashes; Windows consoles default to cp1252 and would raise on them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def fisher_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Exact two-sided Fisher test for two binomials via the hypergeometric tail.

    Computed in log space with lgamma so it stays usable at the sample sizes the study produces -
    a direct comb() form overflows or underflows well before 1500 trials.
    """
    from math import exp, inf, lgamma

    def log_comb(a: int, b: int) -> float:
        if b < 0 or b > a:
            return -inf
        return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)

    total = k1 + k2
    n = n1 + n2
    if total == 0 or n == 0:
        return float("nan")
    lo, hi = max(0, total - n2), min(total, n1)
    log_denom = log_comb(n, total)
    log_pmf = {x: log_comb(n1, x) + log_comb(n2, total - x) - log_denom for x in range(lo, hi + 1)}
    observed = log_pmf[k1]
    p = sum(exp(l) for l in log_pmf.values() if l <= observed + 1e-9)
    return min(1.0, p)

def sd(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _student_sf(x: float, df: int) -> float:
    """One-sided survival function of Student's t, via the regularised incomplete beta."""
    from math import lgamma

    def betacf(a: float, b: float, z: float) -> float:
        tiny = 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * z / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        result = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * z / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            result *= d * c
            aa = -(a + m) * (qab + m) * z / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            result *= delta
            if abs(delta - 1.0) < 1e-13:
                break
        return result

    def betai(a: float, b: float, z: float) -> float:
        if z <= 0.0:
            return 0.0
        if z >= 1.0:
            return 1.0
        lbeta = lgamma(a + b) - lgamma(a) - lgamma(b) + a * math.log(z) + b * math.log(1.0 - z)
        if z < (a + 1.0) / (a + b + 2.0):
            return math.exp(lbeta) / a * betacf(a, b, z)
        return 1.0 - math.exp(
            lgamma(a + b) - lgamma(a) - lgamma(b) + b * math.log(1.0 - z) + a * math.log(z)
        ) / b * betacf(b, a, 1.0 - z)

    return 0.5 * betai(df / 2.0, 0.5, df / (df + x * x))


def paired_t(differences: list[float]) -> tuple[float, float]:
    """Two-sided paired t-test over per-seed differences; the seed, not the match, is the unit."""
    n = len(differences)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    if variance <= 0.0:
        return (float("nan"), 1.0)
    t_stat = mean / math.sqrt(variance / n)
    return (t_stat, 2.0 * _student_sf(abs(t_stat), n - 1))


def student_ppf(p: float, df: int) -> float:
    """Upper-tail quantile of the t distribution, by bisection on _student_sf."""
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _student_sf(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def replicates_for_power(mean_diff: float, diff_sd: float, alpha: float, power: float) -> int:
    """Smallest n for a two-sided paired t-test to reach `power` at the observed effect size.

    Solves n = ((t_crit + t_beta) * sd / mean_diff)^2 self-consistently, because t_crit depends on n.
    """
    if diff_sd <= 0.0 or mean_diff == 0.0:
        return 10**9
    n = 4.0
    for _ in range(200):
        df = max(int(math.ceil(n)) - 1, 1)
        t_crit = student_ppf(alpha / 2.0, df)
        t_beta = student_ppf(1.0 - power, df)
        n = ((t_crit + t_beta) * diff_sd / abs(mean_diff)) ** 2
        if n > 10**8:
            return 10**9
    return min(int(math.ceil(n)), 10**9)


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

        print("\nPairwise tests at the **seed** level — n = %d replicates, the independent unit:\n" % len(seeds))
        print("| Comparison | Mean per-seed difference | SD | Paired t | p (df=%d) | seeds favouring A |" % (len(seeds) - 1))
        print("|---|---:|---:|---:|---:|---|")
        observed: list[tuple[str, str, float, float]] = []
        for i, a in enumerate(paradigms):
            for b in paradigms[i + 1:]:
                diffs = []
                for s in seeds:
                    ra = [r for r in exp_rows if r["paradigm"] == a and int(r["seed"]) == s]
                    rb = [r for r in exp_rows if r["paradigm"] == b and int(r["seed"]) == s]
                    if ra and rb:
                        diffs.append(ra[0]["eval_win_rate"] - rb[0]["eval_win_rate"])
                if len(diffs) < 2:
                    continue
                mean_diff = sum(diffs) / len(diffs)
                t_stat, p_value = paired_t(diffs)
                favour = sum(1 for d in diffs if d > 0)
                verdict = "**significant**" if p_value < 0.0167 else ("nominal only" if p_value < 0.05 else "not significant")
                observed.append((a, b, mean_diff, sd(diffs)))
                print(
                    f"| {a} vs {b} | {mean_diff:+.3f} | {sd(diffs):.3f} | {t_stat:+.2f} "
                    f"| {p_value:.3f} | {favour}/{len(diffs)} — {verdict} |"
                )

        print(
            "\nForward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power"
            " — replicates needed to detect the **observed** difference at its observed spread:\n"
        )
        print("| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |")
        print("|---|---:|---:|---:|---:|")
        for a, b, mean_diff, diff_sd in observed:
            needed = replicates_for_power(mean_diff, diff_sd, 0.0167, 0.80)
            cohen_d = mean_diff / diff_sd if diff_sd > 0 else float("nan")
            needed_text = f"~{needed}" if needed < 10**8 else "not reachable in this design"
            print(
                f"| {a} vs {b} | {mean_diff:+.3f} | {diff_sd:.3f} | {cohen_d:+.3f} | {needed_text} |"
            )

        print("\nPairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):\n")
        print("| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |")
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
