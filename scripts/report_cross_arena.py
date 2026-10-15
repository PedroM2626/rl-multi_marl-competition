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
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAW = "draw"


class MatchKey(NamedTuple):
    """Identity of one match inside a pooled run.

    A NamedTuple rather than a bare tuple because the first analysis silently dropped a replication column
    from it and quietly pooled five match streams onto thirty keys; naming the fields is what keeps a later
    column from being forgotten in the same way.
    """
    training_block: str
    harness_seed: str
    triple: str
    index: int

    @classmethod
    def of(cls, row: dict) -> "MatchKey":
        return cls(str(row.get("training_block", "")), str(row.get("harness_seed", "")),
                   row["triple"], int(row["match"]))


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


def fmt_p(p: float) -> str:
    """Report tables print at four decimals, which turns the significant results into a column of 0.0000s."""
    return f"{p:.2e}" if p < 1e-4 else f"{p:.5f}"


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

    # One record per match, rebuilt from the three team rows it produced. Every replication column has to
    # be part of the key: omitting the harness seed made five replicated runs collapse onto the same 30
    # keys and the paired tests silently saw a fifth of the data, and the training block is the same trap
    # one level up.
    matches: dict[MatchKey, dict[str, str]] = defaultdict(dict)
    for row in rows:
        matches[MatchKey.of(row)][row["paradigm"]] = row["winner"]
    durations = {MatchKey.of(row): row["duration_seconds"] for row in rows}

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

    print(f"\n{len(matches)} matches over {len({key.triple for key in matches})} triples, "
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
            "discordant": [b_disc, c_disc], "p": round(p, 12), "verdict": verdict,
        }
        print(f"| {a} vs {b} | {len(common)} | {a_won} | {b_won} | {b_disc}/{c_disc} | {fmt_p(p)} "
              f"| {verdict} |")

    # Sensitivity of each corrected-significant pair to the training block that produced it. Pooling blocks
    # raises power but a pooled p can still come from one set of checkpoints, which is the unit-of-analysis
    # mistake this chapter makes a point of elsewhere; leaving a block out and re-testing is what shows it.
    blocks = sorted({key.training_block for key in matches})
    sensitivity: dict[str, object] = {}
    if len(blocks) >= 2:
        def pair_p(arch_a: str, arch_b: str, keep) -> tuple[float, int, int]:
            common = [w for key, w in matches.items()
                      if arch_a in w and arch_b in w and keep(key)]
            b_disc = sum(1 for w in common if w[arch_a] == arch_a and w[arch_b] != arch_b)
            c_disc = sum(1 for w in common if w[arch_b] == arch_b and w[arch_a] != arch_a)
            return mcnemar_exact(b_disc, c_disc), b_disc, c_disc

        print(f"\nLeave-one-training-block-out for the pairs that pass correction, over {len(blocks)} blocks. "
              f"Robust means every subset of {len(blocks) - 1} blocks still reaches "
              f"alpha = {bonferroni:.5f}.\n")
        print("| pair | pooled p | " + " | ".join(f"without block {i + 1}" for i in range(len(blocks)))
              + " | per-block discordant | robust |")
        print("|---|---:|" + "---:|" * len(blocks) + "---|---:|")
        for name, record in pairwise.items():
            if record["p"] >= bonferroni:
                continue
            arch_a, arch_b = name.split(" vs ")
            # Re-derived rather than read back from the JSON record, whose p is rounded; the smallest pooled
            # p here is 1e-15, which a ten-decimal rounding would print as zero.
            pooled_p, _, _ = pair_p(arch_a, arch_b, lambda k: True)
            dropped, per_block = {}, {}
            for block in blocks:
                leave_p, _, _ = pair_p(arch_a, arch_b, lambda k, _b=block: k.training_block != _b)
                solo_p, solo_b, solo_c = pair_p(arch_a, arch_b, lambda k, _b=block: k.training_block == _b)
                dropped[block] = leave_p
                per_block[block] = {"discordant": [solo_b, solo_c], "solo_p": solo_p}
            # The same rule as the verdict column: a pair is robust only if it still clears the corrected
            # threshold with any one set of checkpoints removed.
            robust = all(value < bonferroni for value in dropped.values())
            sensitivity[name] = {
                "pooled_p": round(pooled_p, 12),
                "without_block_p": {k: round(v, 12) for k, v in dropped.items()},
                "per_block": {k: {"discordant": v["discordant"], "solo_p": round(v["solo_p"], 12)}
                              for k, v in per_block.items()},
                "robust_to_block": robust,
            }
            print(f"| {name} | {fmt_p(pooled_p)} | "
                  + " | ".join(fmt_p(dropped[blk]) for blk in blocks)
                  + " | " + " ".join(f"{per_block[blk]['discordant'][0]}/"
                                     f"{per_block[blk]['discordant'][1]}"
                                     f"({fmt_p(per_block[blk]['solo_p'])})" for blk in blocks)
                  + f" | {'**yes**' if robust else 'no'} |")

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

    def stability_for(key: str, title: str, dimension: str) -> dict[str, dict[str, object]]:
        """Win rate per architecture within each level of a replication dimension, plus its rank there."""
        if not rows or key not in rows[0]:
            return {}
        cells: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for row in rows:
            cell = cells[str(row[key])][row["paradigm"]]
            cell[1] += 1
            cell[0] += int(row["winner"] == row["paradigm"])
        levels = sorted(cells)
        print("")
        print(title)
        print("")
        print("| architecture | " + " | ".join(str(level) for level in levels) + " | mean | SD | ranks seen |")
        print("|---|" + "---:|" * (len(levels) + 3))
        result: dict[str, dict[str, object]] = {}
        for paradigm in sorted(names):
            values = []
            for level in levels:
                wins, count = cells[level][paradigm]
                values.append(wins / count if count else float("nan"))
            ranks = sorted(
                1 + sum(
                    1 for other in names
                    if other != paradigm
                    and cells[level][other][0] / max(cells[level][other][1], 1)
                    > cells[level][paradigm][0] / max(cells[level][paradigm][1], 1)
                )
                for level in levels
            )
            mean = statistics.fmean(values)
            result[paradigm] = {
                "levels": {level: round(value, 4) for level, value in zip(levels, values)},
                "mean": round(mean, 4), "sd": round(statistics.pstdev(values), 4), "ranks": ranks,
            }
            print(f"| {paradigm} | " + " | ".join(f"{v:.3f}" for v in values)
                  + f" | {mean:.3f} | {statistics.pstdev(values):.3f} | {ranks} |")
        leader = max(names, key=lambda par: statistics.fmean(
            [result[par]["levels"][level] for level in levels]))
        laggard = min(names, key=lambda par: statistics.fmean(
            [result[par]["levels"][level] for level in levels]))
        print("")
        print(f"leader across {dimension} ({len(levels)} levels): {leader}; laggard: {laggard}")
        return result

    seed_stability = stability_for(
        "harness_seed",
        "Replication across harness seeds (the match stream): win rate per architecture, and its rank",
        "harness seeds")
    block_stability = stability_for(
        "training_block",
        "Replication across TRAINING blocks (different checkpoints per architecture): win rate and rank",
        "training blocks")

    out.write_text(json.dumps({
        "matches": total_matches,
        "match_outcomes": match_outcomes,
        "bonferroni_alpha": round(bonferroni, 6),
        "per_architecture": table,
        "pairwise_mcnemar": pairwise,
        "training_blocks": blocks,
        "block_sensitivity": sensitivity,
        "harness_seed_stability": seed_stability,
        "training_block_stability": block_stability,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
