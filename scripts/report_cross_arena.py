"""Report the cross-experiment round-robin: six architectures on a common opponent pool.

The study's two experiments never fight each other, so their win rates are comparable only by assumption.
This analysis comes from `run_cross_arena.py`, which plays every three-architecture combination with the six
versioned policies, putting all six on one scale.

Each match contributes three rows whose outcomes are negatively coupled -- exactly one team wins or the match
is drawn -- so overlapping confidence intervals are the wrong test here. The pairwise comparison is
therefore McNemar's exact test on the discordant outcomes of the matches where the two architectures
actually met, which is paired by construction and does not assume independence between arms.

Usage:
    python scripts/report_cross_arena.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAW = "draw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", type=Path, default=Path("results/cross_arena/matches.csv"))
    parser.add_argument("--json-out", type=Path, default=Path("results/cross_arena/pairwise.json"))
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def wilson(wins: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 0.0)
    p = wins / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (centre - half, centre + half)


def log_comb(a: int, b: int) -> float:
    if b < 0 or b > a:
        return -math.inf
    return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for the discordant counts b and c.

    Under the null each discordant outcome is a fair coin, so the statistic is the binomial tail at 0.5
    computed in log space -- b + c reaches a few hundred here, where a direct comb() would underflow.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.exp(log_comb(n, i) - n * math.log(2.0)) for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    src = PROJECT_ROOT / args.src if not args.src.is_absolute() else args.src
    rows = list(csv.DictReader(src.open(newline="", encoding="utf-8")))
    for row in rows:
        for key in ("eliminations", "remaining_agents", "shots", "shots_hit", "permutation", "match"):
            row[key] = int(row[key])
        for key in ("survival_seconds", "duration_seconds"):
            row[key] = float(row[key])

    # One record per match, rebuilt from the three team rows it produced. The harness seed is part of the
    # key: without it, five replicated runs of the same triple all collapse onto match index 0..29 and the
    # paired tests silently see a fifth of the data.
    def match_key(row: dict[str, object]) -> tuple:
        seed = row.get("harness_seed", "")
        return (str(seed), row["triple"], row["match"])

    matches: dict[tuple, dict[str, str]] = defaultdict(dict)
    for row in rows:
        matches[match_key(row)][row["paradigm"]] = row["winner"]
    durations = {match_key(row): row["duration_seconds"] for row in rows}

    per_arch: dict[str, dict[str, float]] = {}
    for row in rows:
        stats = per_arch.setdefault(row["paradigm"], defaultdict(float))
        stats["matches"] += 1
        stats["wins"] += int(row["winner"] == row["paradigm"])
        stats["draws"] += int(row["winner"] == DRAW)
        stats["eliminations"] += row["eliminations"]
        stats["survival"] += row["survival_seconds"]
        stats["shots"] += row["shots"]
        stats["hits"] += row["shots_hit"]
        stats[f"slot_{row['slot'][-1]}"] += int(row["winner"] == row["paradigm"])

    print(f"\n{len(matches)} matches over {len({key[1] for key in matches})} triples, "
          f"{len(per_arch)} architectures, {len(rows)} team-matches\n")

    match_outcomes = {"wipeout": 0, "timeout": 0, "draw": 0}
    cap = max(durations.values())
    for key, winners in matches.items():
        if DRAW in winners.values():
            match_outcomes["draw"] += 1
        elif durations[key] >= cap - 0.5:
            match_outcomes["timeout"] += 1
        else:
            match_outcomes["wipeout"] += 1
    total_matches = len(matches)
    print("match outcomes: " + "  ".join(
        f"{name} {count / total_matches:.1%}" for name, count in match_outcomes.items()
    ))
    print("(a match counted as 'timeout' was decided with the clock running out, so it measures survival, "
          "not elimination)\n")

    print("| architecture | win rate | 95 % CI | vs 1/3 | elim./match | survival | shots | accuracy |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    table = {}
    for paradigm, stats in sorted(per_arch.items(), key=lambda kv: -kv[1]["wins"] / kv[1]["matches"]):
        n = int(stats["matches"])
        wins = int(stats["wins"])
        lo, hi = wilson(wins, n)
        shots = max(stats["shots"], 1)
        table[paradigm] = {
            "matches": n, "wins": wins, "win_rate": round(wins / n, 4),
            "wilson_95_ci": [round(lo, 4), round(hi, 4)],
            "eliminations_per_match": round(stats["eliminations"] / n, 3),
            "mean_survival_seconds": round(stats["survival"] / n, 2),
            "shots_per_match": round(stats["shots"] / n, 2),
            "shot_accuracy": round(stats["hits"] / shots, 4),
            "win_rate_by_slot": {f"Team {i}": round(stats[f"slot_{i}"] / max(
                sum(1 for r in rows if r["paradigm"] == paradigm and r["slot"] == f"Team {i}"), 1), 4)
                for i in (1, 2, 3)},
        }
        t = table[paradigm]
        print(f"| {paradigm} | **{t['win_rate']:.3f}** | [{t['wilson_95_ci'][0]:.3f}, "
              f"{t['wilson_95_ci'][1]:.3f}] | {t['win_rate'] - 1/3:+.3f} "
              f"| {t['eliminations_per_match']:.2f} | {t['mean_survival_seconds']:.1f}s "
              f"| {t['shots_per_match']:.1f} | {t['shot_accuracy']:.4f} |")

    names = sorted(per_arch)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    comparisons = len(pairs)
    bonferroni = args.alpha / comparisons

    print(f"\nMcNemar exact tests on the matches where the two actually met "
          f"(Bonferroni alpha = {bonferroni:.5f} over {comparisons} pairs)\n")
    print("| pair | met | A won | B won | discordant b/c | p | verdict |")
    print("|---|---:|---:|---:|---|---:|---|")
    pairwise = {}
    for a, b in pairs:
        common = [w for w in matches.values() if a in w and b in w]
        a_won = sum(1 for w in common if w[a] == a)
        b_won = sum(1 for w in common if w[b] == b)
        # Discordant for the pair: a won while b did not, and the reverse. Draws and third-party wins
        # count on both sides and cancel, so they do not enter b/c.
        b_disc = sum(1 for w in common if w[a] == a and w[b] != b)
        c_disc = sum(1 for w in common if w[b] == b and w[a] != a)
        p = mcnemar_exact(b_disc, c_disc)
        verdict = "**significant**" if p < bonferroni else ("nominal only" if p < args.alpha else "not significant")
        pairwise[f"{a} vs {b}"] = {
            "met": len(common), "a_wins": a_won, "b_wins": b_won,
            "discordant": [b_disc, c_disc], "p": round(p, 5), "verdict": verdict,
        }
        print(f"| {a} vs {b} | {len(common)} | {a_won} | {b_won} | {b_disc}/{c_disc} | {p:.4f} | {verdict} |")

    print("\nSlot control: win rate of each architecture by the team slot it occupied. A flat profile means "
          "the rotation worked; a sloped one is the confound the study introduced rotation to remove.\n")
    print("| architecture | Team 1 | Team 2 | Team 3 | spread |")
    print("|---|---:|---:|---:|---:|")
    for paradigm, stats in table.items():
        by_slot = stats["win_rate_by_slot"]
        values = [by_slot[f"Team {i}"] for i in (1, 2, 3)]
        print(f"| {paradigm} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} "
              f"| {max(values) - min(values):.3f} |")

    out = PROJECT_ROOT / args.json_out if not args.json_out.is_absolute() else args.json_out
    out.parent.mkdir(parents=True, exist_ok=True)

    stability: dict[str, object] = {}
    if rows and "harness_seed" in rows[0]:
        # Replication across harness seeds answers the question the pooled table cannot: whether the
        # ordering is a property of the architectures or of one match stream.
        per_seed: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for row in rows:
            cell = per_seed[str(row["harness_seed"])][row["paradigm"]]
            cell[1] += 1
            cell[0] += int(row["winner"] == row["paradigm"])
        seed_names = sorted(per_seed)
        print("\nReplication across harness seeds: win rate per architecture, and where it ranked that run\n")
        print("| architecture | " + " | ".join(f"seed {s}" for s in seed_names) + " | mean | SD | ranks seen |")
        print("|---|" + "---:|" * (len(seed_names) + 3))
        rates: dict[str, list[float]] = {}
        for paradigm in sorted(names):
            values = []
            for s in seed_names:
                wins, n = per_seed[s][paradigm]
                values.append(wins / n if n else float("nan"))
            rates[paradigm] = values
            ranks = sorted(
                1 + sum(1 for other in names if other != paradigm and per_seed[s][other][0] / max(per_seed[s][other][1], 1) > v)
                for s, v in zip(seed_names, values)
            )
            mean = statistics.fmean(values)
            sd = statistics.pstdev(values)
            stability[paradigm] = {
                "per_seed": [round(v, 4) for v in values], "mean": round(mean, 4),
                "sd": round(sd, 4), "ranks": ranks,
            }
            print(f"| {paradigm} | " + " | ".join(f"{v:.3f}" for v in values)
                  + f" | {mean:.3f} | {sd:.3f} | {ranks} |")
        top = max(names, key=lambda p: statistics.fmean(rates[p]))
        bottom = min(names, key=lambda p: statistics.fmean(rates[p]))
        print(f"\nmean-ranked leader across seeds: {top}; mean-ranked laggard: {bottom}")

    out.write_text(json.dumps({
        "matches": total_matches,
        "match_outcomes": match_outcomes,
        "bonferroni_alpha": round(bonferroni, 6),
        "per_architecture": table,
        "pairwise_mcnemar": pairwise,
        "harness_seed_stability": stability,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
