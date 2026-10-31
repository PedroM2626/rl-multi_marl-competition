"""Render the replicated study into markdown tables for docs/08_results.md.

Reads results/study/per_seed_metrics.csv (written by run_study.py) and prints, per experiment:
a per-paradigm summary with between-seed spread and a pooled Wilson interval, a per-seed matrix so a
single-seed fluke is visible, and Fisher exact tests for every pairing.

Usage:  python scripts/report_study.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results" / "study"

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


def shots_per_alive_second(shots_per_match: float, mean_survival_seconds: float) -> float:
    """Firing volume per second in which the team actually had agents alive.

    `mean_survival_seconds` is a per-agent mean — `simulation.py` accrues `agent.survival_time` only while the
    agent is alive and reports the mean over its three — so a team's alive life in one match is three times it.
    The unnormalised shot column cannot separate "this arm fires less" from "this arm is dead earlier", and a
    dead agent does not fire: the ratio is what makes the two statements different claims.
    """
    return shots_per_match / (3 * mean_survival_seconds) if mean_survival_seconds > 0 else 0.0


def average_ranks(values: list[float]) -> list[float]:
    """Ranks 1..k, highest value first; tied values share the mean of the ranks they span."""
    order = sorted(range(len(values)), key=lambda index: -values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        for index in order[position:end + 1]:
            ranks[index] = (position + end) / 2 + 1
        position = end + 1
    return ranks


def friedman(blocks: list[list[float]]) -> tuple[float, float, list[float]]:
    """Friedman's test of k coupled arms over n blocks (here: three paradigms over n seeds).

    This is the test the exactly-one-winner coupling asks for (docs/10 § 10.4, item 5): within a replicate
    the three win rates are not independent samples -- they come out of the same matches and nearly sum to
    one -- so the unit is the replicate and the observation is each arm's *rank* inside it. Returns the
    tie-corrected statistic, its chi-square p-value on k - 1 degrees of freedom, and the mean rank per arm.
    """
    n, k = len(blocks), len(blocks[0])
    ranks = [average_ranks(block) for block in blocks]
    rank_sums = [sum(row[j] for row in ranks) for j in range(k)]
    statistic = 12.0 / (n * k * (k + 1)) * sum(r * r for r in rank_sums) - 3.0 * n * (k + 1)
    ties = sum(count ** 3 - count for row in blocks for count in
               (sum(1 for value in row if value == level) for level in set(row)))
    correction = 1.0 - ties / (n * (k ** 3 - k))
    statistic = statistic / correction if correction > 0 else 0.0
    return statistic, chi2_sf(statistic, k - 1), [total / n for total in rank_sums]


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of a chi-square distribution, through the regularised incomplete gamma function."""
    if x <= 0:
        return 1.0
    a, z = df / 2.0, x / 2.0
    if z < a + 1:  # series for the lower tail
        term = total = 1.0 / a
        for n in range(1, 500):
            term *= z / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    # continued fraction for the upper tail
    b, c, d = z + 1 - a, 1 / 1e-300, 1 / (z + 1 - a)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-z + a * math.log(z) - math.lgamma(a)) * h


def sign_flip_p(differences: list[float]) -> float:
    """Exact two-sided permutation p for a paired mean difference, by enumerating every sign flip.

    Valid whatever the within-match coupling does to the arms, because under the null each replicate's
    difference is as likely to have either sign; with ten replicates that is all 1,024 assignments.
    """
    observed = abs(sum(differences))
    if len(differences) > 20:
        raise ValueError("exact enumeration is meant for the study's replicate counts")
    hits = sum(
        1 for signs in itertools.product((1, -1), repeat=len(differences))
        if abs(sum(sign * d for sign, d in zip(signs, differences))) >= observed - 1e-12
    )
    return hits / 2 ** len(differences)


def bootstrap_ci(values: list[float], resamples: int = 10_000, seed: int = 20261031) -> tuple[float, float]:
    """Percentile 95 % interval of the mean, resampling replicates (the independent unit)."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


SECONDARY = (
    ("eval_eliminations_per_match", "Elim./match", 2),
    ("eval_mean_survival", "Survival (s)", 1),
    ("eval_shots_per_match", "Shots/match", 1),
    ("eval_shot_accuracy", "Accuracy", 4),
)


def read_metrics(results_dir: Path) -> list[dict]:
    rows = list(csv.DictReader((results_dir / "per_seed_metrics.csv").open(newline="", encoding="utf-8")))
    for row in rows:
        for key in row:
            if key not in {"experiment", "team", "paradigm"}:
                row[key] = float(row[key]) if key != "seed" else int(row[key])
    return rows


def compare_arms(control: list[dict], arm: list[dict], arm_name: str) -> None:
    """Paired per-seed comparison of one campaign against the control, paradigm by paradigm.

    The seed is the unit, and the pairing is on the seed number, which is what makes the arms comparable:
    each arm reuses the control's seeds, so the paradigm-to-slot rotation is identical in both.
    """
    def keyed(rows: list[dict]) -> dict:
        return {(r["experiment"], r["paradigm"], int(r["seed"])): r for r in rows}

    left, right = keyed(control), keyed(arm)
    shared = sorted(set(left) & set(right))
    if not shared:
        print(f"\n### {arm_name}: no seeds in common with the control, nothing to pair\n")
        return
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key in shared:
        groups[(key[0], key[1])].append(key[2])

    print(f"\n### {arm_name} vs control, paired on seed ({len(shared) // len(groups)} shared seeds per arm)\n")
    print("| experiment | paradigm | control | arm | paired diff | t | p | seeds below control | "
          "survival (control -> arm) | elim./match | shots/match | shots per alive-second |")
    print("|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|")
    for (experiment, paradigm), seeds in sorted(groups.items()):
        c = [left[(experiment, paradigm, s)] for s in seeds]
        a = [right[(experiment, paradigm, s)] for s in seeds]
        differences = [x["eval_win_rate"] - y["eval_win_rate"] for x, y in zip(a, c)]
        mean = sum(differences) / len(differences)
        t_stat, p = paired_t(differences)
        below = sum(1 for d in differences if d < 0)
        surv_c = statistics.fmean([r["eval_mean_survival"] for r in c])
        surv_a = statistics.fmean([r["eval_mean_survival"] for r in a])
        shots_c = statistics.fmean([r["eval_shots_per_match"] for r in c])
        shots_a = statistics.fmean([r["eval_shots_per_match"] for r in a])
        print(f"| {experiment} | {paradigm} | {statistics.fmean([r['eval_win_rate'] for r in c]):.3f} "
              f"| {statistics.fmean([r['eval_win_rate'] for r in a]):.3f} | {mean:+.3f} "
              f"| {t_stat:+.2f} | {p:.3f} | {below}/{len(seeds)} "
              f"| {surv_c:.1f} -> {surv_a:.1f}s "
              f"| {statistics.fmean([r['eval_eliminations_per_match'] for r in c]):.2f} -> "
              f"{statistics.fmean([r['eval_eliminations_per_match'] for r in a]):.2f} "
              f"| {shots_c:.0f} -> {shots_a:.0f} "
              f"| {shots_per_alive_second(shots_c, surv_c):.2f} -> "
              f"{shots_per_alive_second(shots_a, surv_a):.2f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="a run_study.py output directory; the arena-v2 campaign and the ablations are "
                             "read with this flag rather than by editing the path")
    parser.add_argument("--label", default="",
                        help="a word for the banner, e.g. 'arena v2', so a re-measurement is not mistaken "
                             "for the committed study")
    parser.add_argument("--compare-to", type=Path, default=None, metavar="CONTROL_DIR",
                        help="pair this campaign against a control campaign, per paradigm and per seed")
    parser.add_argument("--sampled-readout", type=Path, default=None, metavar="READOUT_DIR",
                        help="a results/readout/T* directory; prints the headline table under that sampled "
                             "readout beside the greedy one (docs/10 § 10.4, item 11)")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="write the seed-level tests (Friedman, sign-flip, bootstrap intervals) here")
    args = parser.parse_args()
    coupled: dict[str, dict] = {}
    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    if args.label:
        print(f"=== {args.label}: {results_dir.name} ===")
    rows = read_metrics(results_dir)

    by_experiment: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_experiment[row["experiment"]].append(row)

    for experiment, exp_rows in sorted(by_experiment.items()):
        paradigms = sorted({r["paradigm"] for r in exp_rows})
        seeds = sorted({int(r["seed"]) for r in exp_rows})
        print(f"\n### {experiment} — {len(seeds)} seeds x {int(exp_rows[0]['eval_matches'])} held-out greedy matches\n")
        print("| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |")
        print("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
        counts: dict[str, tuple[int, int]] = {}
        for paradigm in paradigms:
            sub = [r for r in exp_rows if r["paradigm"] == paradigm]
            rates = [r["eval_win_rate"] for r in sub]
            wins = round(sum(r["eval_win_rate"] * r["eval_matches"] for r in sub))
            trials = int(sum(r["eval_matches"] for r in sub))
            counts[paradigm] = (wins, trials)
            lo, hi = wilson(wins, trials)
            survival = sum(r["eval_mean_survival"] for r in sub) / len(sub)
            shots = sum(r["eval_shots_per_match"] for r in sub) / len(sub)
            print(
                f"| {paradigm} | **{sum(rates)/len(rates):.3f}** | {sd(rates):.3f} | {wins}/{trials} "
                f"| [{lo:.3f}, {hi:.3f}] "
                f"| {sum(r['eval_eliminations_per_match'] for r in sub)/len(sub):.2f} "
                f"| {survival:.2f} "
                f"| {shots:.2f} "
                f"| {shots_per_alive_second(shots, survival):.3f} "
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

        # The coupled-arm treatment (docs/10 § 10.4, items 5 and 8): rank-based omnibus over the replicate,
        # an exact permutation test per pair, and intervals on the secondary metrics -- all with the
        # replicate as the unit, none assuming the arms of one match are independent.
        complete = [s for s in seeds
                    if all([r for r in exp_rows if r["paradigm"] == a and int(r["seed"]) == s] for a in paradigms)]
        blocks = [[next(r["eval_win_rate"] for r in exp_rows if r["paradigm"] == a and int(r["seed"]) == s)
                   for a in paradigms] for s in complete]
        statistic, p_friedman, mean_ranks = friedman(blocks)
        print(f"\nFriedman test over {len(complete)} replicates (the three arms are coupled inside a match, so the "
              f"replicate's ranking is the observation): chi2 = {statistic:.3f}, df = {len(paradigms) - 1}, "
              f"p = {p_friedman:.3f}. Mean rank (1 = best): "
              + ", ".join(f"{a} {rank:.2f}" for a, rank in zip(paradigms, mean_ranks)) + "\n")
        print("| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |")
        print("|---|---:|---:|---|")
        pairs = {}
        for i, a in enumerate(paradigms):
            for b in paradigms[i + 1:]:
                diffs = [block[i] - block[paradigms.index(b)] for block in blocks]
                p_flip = sign_flip_p(diffs)
                verdict = "**significant**" if p_flip < 0.0167 else ("nominal only" if p_flip < 0.05 else "not significant")
                pairs[f"{a} vs {b}"] = {"mean_difference": round(statistics.fmean(diffs), 4),
                                        "sign_flip_p": round(p_flip, 4)}
                print(f"| {a} vs {b} | {statistics.fmean(diffs):+.3f} | {p_flip:.3f} | {verdict} |")
        print("\nSecondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the "
              "replicates):\n")
        print("| Paradigm | " + " | ".join(label for _, label, _ in SECONDARY) + " |")
        print("|---|" + "---|" * len(SECONDARY))
        secondary = {}
        for paradigm in paradigms:
            sub_rows = [r for r in exp_rows if r["paradigm"] == paradigm]
            cells, secondary[paradigm] = [], {}
            for key, _, digits in SECONDARY:
                values = [r[key] for r in sub_rows]
                lo, hi = bootstrap_ci(values)
                mean = statistics.fmean(values)
                secondary[paradigm][key] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
                cells.append(f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]")
            print(f"| {paradigm} | " + " | ".join(cells) + " |")
        coupled[experiment] = {
            "replicates": len(complete),
            "friedman": {"chi2": round(statistic, 4), "df": len(paradigms) - 1, "p": round(p_friedman, 4),
                         "mean_rank": {a: round(rank, 3) for a, rank in zip(paradigms, mean_ranks)}},
            "pairwise_sign_flip": pairs,
            "secondary_bootstrap": secondary,
        }

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

    if args.sampled_readout is not None:
        readout_dir = args.sampled_readout if args.sampled_readout.is_absolute() else ROOT / args.sampled_readout
        dual_readout(rows, read_metrics(readout_dir), readout_dir.name)

    if args.compare_to is not None:
        control_dir = args.compare_to if args.compare_to.is_absolute() else ROOT / args.compare_to
        compare_arms(read_metrics(control_dir), rows, results_dir.name)

    if args.json_out is not None:
        path = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
        path.write_text(json.dumps(coupled, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"\nwrote {path}")


def dual_readout(greedy: list[dict], sampled: list[dict], label: str) -> None:
    """The headline table under both readouts, side by side, with seed-level intervals on each.

    docs/10 § 10.4 item 11: the greedy readout decides experiment 2's winner on its own, so a table that
    shows only argmax carries a qualification no reader can see. This prints the two together.
    """
    print(f"\n### Dual readout: greedy (T = 0) beside sampled ({label})\n")
    print("| experiment | paradigm | greedy mean [95 % CI] | rank | sampled mean [95 % CI] | rank |")
    print("|---|---|---|---:|---|---:|")
    for experiment in sorted({r["experiment"] for r in greedy}):
        table = {}
        for name, rows in (("greedy", greedy), ("sampled", sampled)):
            for paradigm in sorted({r["paradigm"] for r in rows if r["experiment"] == experiment}):
                values = [r["eval_win_rate"] for r in rows if r["experiment"] == experiment and r["paradigm"] == paradigm]
                table.setdefault(paradigm, {})[name] = (statistics.fmean(values), *bootstrap_ci(values))
        ranks = {name: dict(zip(table, average_ranks([table[p][name][0] for p in table]))) for name in ("greedy", "sampled")}
        for paradigm, cells in sorted(table.items(), key=lambda kv: -kv[1]["greedy"][0]):
            g, s_ = cells["greedy"], cells["sampled"]
            print(f"| {experiment} | {paradigm} | {g[0]:.3f} [{g[1]:.3f}, {g[2]:.3f}] | {ranks['greedy'][paradigm]:.0f} "
                  f"| {s_[0]:.3f} [{s_[1]:.3f}, {s_[2]:.3f}] | {ranks['sampled'][paradigm]:.0f} |")


if __name__ == "__main__":
    main()
