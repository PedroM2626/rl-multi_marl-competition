"""Absolute strength: each trained policy against a fixed pool of uniform-random opponents.

Every win rate in this repository is relative -- an arm beating two other arms that trained against it.
That is why no number here is comparable to the literature, and it is the gap chapter 10 lists as open.
This script closes the cheap half of it: put one trained policy in one team, put two uniform-random
controllers in the others, and measure. Random opponents do not adapt, co-evolve, or share a run's
initialisation, so the result is a floor rather than a ranking.

It is a *second scale*, not a better version of the first: [§ 8.11](docs/08_results.md#absolute-strength)
reports both, and the two orderings of the six arms disagree, which is the reason neither is quoted alone.

It reuses the round-robin harness rather than reimplementing the arena wiring, so the variant, the slot
rotation and the checkpoint loading are the same code paths that produced results/cross_arena.

Usage:
    cd ctde_arena
    python ../scripts/run_absolute_eval.py --matches 150 --seeds 1,2,3
    python ../scripts/run_absolute_eval.py --blocks 1:7,10:5,9:2 --matches 50 --seeds 1,2,3 \
        --out results/absolute_blocks
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_harness():
    """Import scripts/run_cross_arena.py, which owns the CrossArena wiring."""
    path = Path(__file__).resolve().parent / "run_cross_arena.py"
    spec = importlib.util.spec_from_file_location("cross_arena_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recorded_path(value: "Path | None") -> "str | None":
    """How a path the summary names should be spelled: repository-relative, forward slashes.

    `str(Path)` uses the host separator, so a summary written on Windows recorded
    `results\\baseline\\...`. `--report-only` inherits that string and joins it onto the repository root,
    which on a POSIX host is a single file whose name contains backslashes -- so the artefact gate in CI
    died reading its own provenance while passing on the machine that wrote it (A-42).
    """
    if value is None:
        return None
    resolved = value if value.is_absolute() else PROJECT_ROOT / value
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_blocks(spec: str, harness, campaign: str = "") -> list[tuple[str, dict]]:
    """(label, policies) per training block; an empty spec means the promoted checkpoint set.

    The labels are the strings the round-robin uses, so the two instruments can be joined on the checkpoint
    set they measured rather than on the architecture name alone. `campaign` switches both to an ablation
    set under `data/campaign_replicates/`, prefixed on the label so a pooled table cannot read an ablation
    block and a study block trained from the same two seeds as one checkpoint set.
    """
    if not spec.strip():
        if campaign:
            raise SystemExit("--campaign names a set of training replicates, so --blocks has to say which "
                             "(exp1:exp2 pairs); the promoted checkpoints are not that set")
        return [("promoted", dict(harness.POLICIES))]
    blocks = []
    for pair in spec.split(","):
        exp1, _, exp2 = pair.partition(":")
        if not exp2:
            raise SystemExit(f"--blocks wants exp1:exp2 pairs, got {pair!r}")
        label = f"exp1_seed{exp1}+exp2_seed{exp2}"
        policies = harness.replicate_policies(int(exp1), int(exp2), campaign)
        if campaign:
            harness.check_campaign_capacity(campaign)
            label = f"{campaign}:{label}"
        blocks.append((label, policies))
    return blocks


def rank_of(values: dict[str, float]) -> dict[str, float]:
    """Average-rank ordering, high value first; ties share the mean of the ranks they span."""
    order = sorted(values, key=lambda key: -values[key])
    ranks: dict[str, float] = {}
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        share = (position + end) / 2 + 1
        for key in order[position:end + 1]:
            ranks[key] = share
        position = end + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else float("nan")


def compare_scales(pooled: dict, pool_json: Path) -> dict:
    """Join the random-pool result with the trained-peer round-robin and say how much they agree.

    This is the measurement behind "strength is not one quantity": the two orders are nearly unrelated, and
    none of the pool run's own component metrics predicts the absolute result.
    """
    peer = json.loads(pool_json.read_text(encoding="utf-8"))["per_architecture"]
    arms = sorted(set(pooled) & set(peer))
    abs_rate = {arm: pooled[arm]["win_rate_vs_random"] for arm in arms}
    pool_rate = {arm: peer[arm]["win_rate"] for arm in arms}
    abs_rank, pool_rank = rank_of(abs_rate), rank_of(pool_rate)
    n = len(arms)
    d2 = sum((abs_rank[arm] - pool_rank[arm]) ** 2 for arm in arms)
    spearman = 1 - 6 * d2 / (n * (n * n - 1))
    components = ("eliminations_per_match", "shots_per_match", "mean_survival_seconds", "shot_accuracy")
    rows = {
        "arms": n,
        "spearman_rank_correlation": round(spearman, 4),
        "pearson_win_rate": round(pearson([abs_rate[a] for a in arms], [pool_rate[a] for a in arms]), 4),
        "pearson_vs_pool_components": {
            component: round(pearson([abs_rate[a] for a in arms], [peer[a][component] for a in arms]), 4)
            for component in components
        },
        "per_arm": {arm: {"vs_random": abs_rate[arm], "vs_peers": pool_rate[arm],
                          "rank_random": abs_rank[arm], "rank_peers": pool_rank[arm]} for arm in arms},
    }
    print("\n--- agreement between the two scales (random pool vs trained peers) ---")
    print("| architecture | vs random | rank | vs peers | rank |")
    print("|---|---:|---:|---:|---:|")
    for arm in sorted(arms, key=lambda a: -abs_rate[a]):
        print(f"| {arm} | {abs_rate[arm]:.3f} | {abs_rank[arm]:.1f} | {pool_rate[arm]:.3f} "
              f"| {pool_rank[arm]:.1f} |")
    print(f"Spearman {spearman:+.3f} · Pearson on win rates {rows['pearson_win_rate']:+.3f} · "
          + " ".join(f"{component.split('_')[0]} r={rows['pearson_vs_pool_components'][component]:+.3f}"
                     for component in components))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=150,
                        help="matches per architecture, per seed, per block")
    parser.add_argument("--seeds", type=str, default="1",
                        help="comma-separated harness seeds; each is an independent match stream over the "
                             "same checkpoints, so pooling them narrows the CI without touching training")
    parser.add_argument("--blocks", type=str, default="",
                        help="comma-separated exp1:exp2 training-replicate pairs, e.g. 1:7,10:5,9:2; empty "
                             "uses the promoted checkpoints, which is one set and cannot separate an "
                             "architecture from the seed that produced it")
    parser.add_argument("--out", type=Path, default=Path("results/absolute"))
    parser.add_argument("--campaign", default="",
                        help="name of an ablation set under data/campaign_replicates/ whose checkpoints "
                             "--blocks should load; the round-robin's capacity guard runs, so the running "
                             "RL_HIDDEN_DIM has to match the set")
    parser.add_argument("--compare-pool", type=Path, default=None,
                        help="a report_cross_arena.py JSON; joins its per-architecture win rates to these "
                             "and reports how much the two scales agree")
    parser.add_argument("--null-json", type=Path, default=None,
                        help="a random_baseline.py uniform-mode JSON; its per-slot win rates are the matched "
                             "null, because three mutually random teams do not split the arena evenly")
    parser.add_argument("--report-only", action="store_true",
                        help="replay the aggregation from the matches.csv already in --out, without playing "
                             "any matches; the summary is then verifiably derived from the raw record")
    return parser.parse_args()


def matched_null(null_json: Path) -> dict[str, float]:
    """Per-slot win rate of a team whose policy is uniform random, from the § 8.9 baseline run.

    This is the right null for the absolute evaluation because the focal team's opponents are the same two
    random teams in both measurements -- and it is not 1/3 per slot, because the arena's three spawn
    positions are not equivalent (A-32).
    """
    payload = json.loads(null_json.read_text(encoding="utf-8"))
    if payload.get("action_policy") != "uniform":
        raise SystemExit(f"{null_json} is not the uniform baseline; the greedy one is not a random null")
    return {team: float(stats["win_rate"]) for team, stats in payload["per_team"].items()}


def summarise(rows: list[dict], harness) -> dict[str, dict]:
    """Win/draw/loss against the random pool per architecture, with the split by trained team."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    slots: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in rows:
        paradigm = str(row["paradigm"])
        won = int(row["winner"] == row["trained_team"])
        counts[paradigm]["wins"] += won
        counts[paradigm]["matches"] += 1
        counts[paradigm]["draws"] += int(row["winner"] == harness.DRAW_TEAM)
        slot_cell = slots[paradigm][str(row["trained_team"])]
        slot_cell[0] += won
        slot_cell[1] += 1

    out: dict[str, dict] = {}
    for paradigm, cell in counts.items():
        n, wins, draws = cell["matches"], cell["wins"], cell["draws"]
        lo, hi = harness.wilson(wins, n)
        out[paradigm] = {
            "matches": n, "wins": wins, "draws": draws, "losses": n - wins - draws,
            "win_rate_vs_random": round(wins / n, 4),
            "wilson_95_ci": [round(lo, 4), round(hi, 4)],
            "draw_rate": round(draws / n, 4),
            "loss_rate": round((n - wins - draws) / n, 4),
            "median_duration_seconds": float(statistics.median(
                [row["duration_seconds"] for row in rows if row["paradigm"] == paradigm])),
            "win_rate_by_trained_team": {s: round(v[0] / v[1], 4) for s, v in sorted(slots[paradigm].items())},
        }
    return out


def by_level(rows: list[dict], key: str, harness) -> dict[str, dict]:
    levels = sorted({str(row[key]) for row in rows})
    if len(levels) < 2:
        return {}
    return {level: summarise([r for r in rows if str(r[key]) == level], harness) for level in levels}


def spread(per_level: dict) -> dict:
    levels = sorted(per_level)
    return {
        paradigm: {
            "sd": round(statistics.pstdev([per_level[level][paradigm]["win_rate_vs_random"]
                                           for level in levels]), 4),
            "levels": {level: per_level[level][paradigm]["win_rate_vs_random"] for level in levels},
        }
        for paradigm in sorted(per_level[levels[0]])
    }


def load_rows(path: Path) -> list[dict]:
    """Read a matches.csv back so the summary can be re-derived without replaying any match."""
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for row in rows:
        row["match"] = int(row["match"])
        row["duration_seconds"] = float(row["duration_seconds"])
        row["harness_seed"] = int(row["harness_seed"])
        row.setdefault("training_block", "promoted")
    return rows


def add_matched_null(pooled: dict, null: dict[str, float]) -> float:
    """Attach the slot-matched null to every architecture, and return the slot-balanced null."""
    balanced = statistics.fmean(null.values())
    for paradigm, stats in pooled.items():
        stats["excess_vs_matched_null"] = round(stats["win_rate_vs_random"] - balanced, 4)
        stats["excess_by_trained_team"] = {
            slot: round(rate - null[slot], 4) for slot, rate in stats["win_rate_by_trained_team"].items()
            if slot in null
        }
    return round(balanced, 4)


def arena_snapshot(args, previous: dict) -> dict | None:
    """The A-34 knobs these matches were played under, so a re-measurement cannot be mistaken for the study.

    `--report-only` carries the snapshot the run itself recorded rather than reading the environment: the
    shell that re-derives a summary is not evidence of the arena that produced the matches inside it.
    """
    if args.report_only:
        return previous.get("arena")
    from marl_arena.config import CONFIG

    return {
        # The harness supplies every match variant explicitly, so domain randomisation is off in play
        # whatever the environment says; recorded anyway because that is what makes the no_dr arm's
        # checkpoints comparable to the study's on this scale.
        "domain_randomization": False,
        "hidden_dim": CONFIG.rl_hidden_dim,
        "match_duration_seconds": CONFIG.match_duration_seconds,
        "reward": {
            "alive_bonus": CONFIG.reward_alive_bonus,
            "alive_decay_half_life_s": CONFIG.reward_alive_decay_half_life,
            "kill": CONFIG.reward_kill,
            "death": CONFIG.reward_death,
            "hit": CONFIG.reward_hit,
            "cap_awards_win": CONFIG.termination_cap_awards_win,
        },
    }


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    harness = load_harness()
    out_dir = PROJECT_ROOT / args.out if not args.out.is_absolute() else args.out
    previous: dict = {}
    if (out_dir / "summary.json").is_file():
        previous = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    arena_settings = arena_snapshot(args, previous)
    if args.report_only and (args.null_json is None or args.compare_pool is None):
        # Re-deriving must reproduce the committed summary, so inherit the inputs it was made with rather
        # than quietly emitting a summary with the null and agreement blocks missing.
        inherited = previous.get("inputs") or {}
        args.null_json = args.null_json or (Path(inherited["null_json"]) if inherited.get("null_json") else None)
        args.compare_pool = (args.compare_pool
                             or (Path(inherited["compare_pool"]) if inherited.get("compare_pool") else None))
    rows: list[dict[str, object]] = []

    if args.report_only:
        rows = load_rows(out_dir / "matches.csv")
        blocks = sorted({str(row["training_block"]) for row in rows})
        seeds = sorted({row["harness_seed"] for row in rows})
        print(f"re-deriving the summary of {len(rows)} recorded matches "
              f"({len(blocks)} block(s) x {len(seeds)} seed(s)); no matches played", flush=True)
    else:
        from marl_arena.config import CONFIG
        from marl_arena.rl.actions import NUM_ACTIONS
        from marl_arena.systems.match_variant import TeamSpawnSpec
        import numpy as np

        seeds = [int(value) for value in args.seeds.split(",")]
        parsed = parse_blocks(args.blocks, harness, args.campaign)
        blocks = [label for label, _ in parsed]
        for block_label, policies in parsed:
            for seed in seeds:
                arena = harness.CrossArena(seed, policies)
                rng = np.random.default_rng(seed)

                def make_random_controller(team_name: str, slot: int):
                    """A real controller -- so the simulation is untouched -- forced to sample uniformly.

                    The greedy argmax of an untrained network is a near-constant policy (A-30), which is why
                    the opponent here is drawn from the action space rather than read off an untrained actor.
                    """
                    controller = arena.ctde_controller(team_name, "RANDOM", 1000 + slot, arena.device, False)
                    controller._greedy_action = lambda logits, _r=rng: int(_r.integers(NUM_ACTIONS))
                    return controller

                print(f"\n--- {block_label}, harness seed {seed}, {args.matches} matches per architecture ---",
                      flush=True)
                for paradigm in sorted(policies):
                    for index in range(args.matches):
                        # Rotate which slot the trained policy occupies, so a corner advantage cannot land on it.
                        slot = index % 3
                        team_names = [spec.team_name for spec in arena.base_variant.team_spawns]
                        spawns = tuple(
                            TeamSpawnSpec(name, paradigm if position == slot else "RANDOM",
                                          spec.spawn_center, spec.color_rgb)
                            for position, (name, spec) in enumerate(
                                zip(team_names, arena.base_variant.team_spawns))
                        )
                        arena.sim.pending = harness.replace(arena.base_variant, team_spawns=spawns)
                        controllers = {}
                        for position, spec in enumerate(spawns):
                            if position == slot:
                                controllers[spec.team_name] = arena.policy_controller(
                                    spec.team_name, paradigm, position)
                            else:
                                controllers[spec.team_name] = make_random_controller(spec.team_name, position)
                        arena.sim.controllers = controllers
                        arena.set_training(arena.sim.controllers, False)
                        arena.sim.reset_match()

                        dt = CONFIG.sim_step_dt
                        while not arena.sim.step(dt):
                            pass
                        result = arena.sim.finish_match()
                        rows.append({"training_block": block_label, "harness_seed": seed,
                                     "paradigm": paradigm, "match": index,
                                     "trained_team": team_names[slot], "winner": result.winner_team,
                                     "duration_seconds": round(result.duration_seconds, 2)})

                stats = summarise([r for r in rows if r["training_block"] == block_label
                                   and r["harness_seed"] == seed and r["paradigm"] == paradigm], harness)[paradigm]
                print(f"{paradigm:11s} win {stats['win_rate_vs_random']:.3f} "
                      f"[{stats['wilson_95_ci'][0]:.3f}, {stats['wilson_95_ci'][1]:.3f}]  "
                      f"draw {stats['draw_rate']:.3f}  loss {stats['loss_rate']:.3f}  "
                      f"median {stats['median_duration_seconds']:5.1f}s", flush=True)

    pooled = summarise(rows, harness)
    per_block, per_seed = by_level(rows, "training_block", harness), by_level(rows, "harness_seed", harness)

    null_by_slot: dict[str, float] = {}
    balanced_null = None
    if args.null_json is not None:
        path = PROJECT_ROOT / args.null_json if not args.null_json.is_absolute() else args.null_json
        null_by_slot = matched_null(path)
        balanced_null = add_matched_null(pooled, null_by_slot)

    print(f"\n--- pooled: {len(blocks)} block(s) x {len(seeds)} seed(s) = {len(rows)} matches ---", flush=True)
    if balanced_null is not None:
        print(f"matched null: {', '.join(f'{k} {v:.4f}' for k, v in sorted(null_by_slot.items()))}"
              f" -> slot-balanced {balanced_null:.4f} (not 0.3333)", flush=True)
    print("| architecture | win rate vs random | 95 % CI | excess over null | draws | losses | by trained team |")
    print("|---|---:|---|---:|---:|---:|---|")
    for paradigm, p in sorted(pooled.items(), key=lambda kv: -kv[1]["win_rate_vs_random"]):
        slots = " ".join(
            f"{s.split()[-1]}:{v:.3f}"
            + (f"({p['excess_by_trained_team'][s]:+.3f})" if p.get("excess_by_trained_team") else "")
            for s, v in p["win_rate_by_trained_team"].items())
        excess = f"{p['excess_vs_matched_null']:+.3f}" if "excess_vs_matched_null" in p else "—"
        print(f"| {paradigm} | **{p['win_rate_vs_random']:.3f}** | [{p['wilson_95_ci'][0]:.3f}, "
              f"{p['wilson_95_ci'][1]:.3f}] | {excess} | {p['draw_rate']:.3f} | {p['loss_rate']:.3f} | {slots} |")
    for title, per_level in (("training block", per_block), ("harness seed", per_seed)):
        if not per_level:
            print(f"\n(no spread across {title}s: one level)")
            continue
        print(f"\nSpread across {title}s")
        print("| architecture | " + " | ".join(str(level) for level in sorted(per_level)) + " | SD |")
        print("|---|" + "---:|" * (len(per_level) + 1))
        for paradigm, s in sorted(spread(per_level).items(), key=lambda kv: -kv[1]["sd"]):
            print(f"| {paradigm} | " + " | ".join(
                f"{s['levels'][level]:.3f}" for level in sorted(per_level)) + f" | {s['sd']:.3f} |")

    comparison = None
    if args.compare_pool is not None:
        path = PROJECT_ROOT / args.compare_pool if not args.compare_pool.is_absolute() else args.compare_pool
        comparison = compare_scales(pooled, path)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "design": "one trained policy versus two uniform-random controllers, trained slot rotated",
        "blocks": blocks,
        "seeds": seeds,
        "campaign": args.campaign or "study",
        "arena": arena_settings,
        # Provenance of the two derived blocks below: which file supplied the slot-matched null, and which
        # peer pool the agreement statistic was computed against. Section 8.11's scope-mismatch lesson is the
        # reason this is recorded rather than left to the command line of whoever ran it, and the separators
        # are forward slashes because `--report-only` reads these strings back on hosts where a backslash is
        # a filename character, not a directory one (A-42).
        "inputs": {"null_json": recorded_path(args.null_json),
                   "compare_pool": recorded_path(args.compare_pool)},
        # In --report-only the cell size is a property of the record being re-derived, not of this
        # invocation: inherit what the run wrote instead of blanking a published figure.
        "matches_per_architecture_per_block_per_seed": (
            args.matches if not args.report_only
            else previous.get("matches_per_architecture_per_block_per_seed", args.matches)),
        "chance_level": round(1 / 3, 4),
        "matched_null_by_slot": null_by_slot,
        "matched_null_balanced": balanced_null,
        "pooled": pooled,
        "per_training_block": per_block,
        "per_harness_seed": per_seed,
        "spread_across_training_blocks": spread(per_block) if per_block else {},
        "spread_across_harness_seeds": spread(per_seed) if per_seed else {},
        "agreement_with_peer_pool": comparison,
    }, indent=2), encoding="utf-8")
    if not args.report_only:
        with (out_dir / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nwrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
