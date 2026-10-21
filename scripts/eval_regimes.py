"""Replay the study's held-out evaluation and record how every match was decided.

[§ 8.1](../docs/08_results.md#two-regimes) reports that five of twenty replicates play a different game
from the rest: their matches end on the 90 s clock rather than by elimination, so a pooled win rate mixes
"who wins a fight" with "who is most alive at the buzzer". The round-robin could split those regimes
([§ 8.10](../docs/08_results.md#cross-arena)) because its per-match rows carry `duration_seconds`; the
study could not, because `run_experiment.py`'s held-out recorder kept everything except the duration --
which is audit item 8's open half. This script closes the gap the only way it can be closed, given that the
durations were never written: by replaying the study's own held-out protocol over the versioned
checkpoints and recording the per-match rows the original runs should have kept.

Nothing is trained, and the readout is the published one (argmax). The protocol is `run_experiment.py`'s
held-out evaluation reproduced exactly -- the run's three checkpoints from
`data/training_replicates/<experiment>_seedN`, `ArenaSimulation(seed=replicate_seed + 9000,
domain_randomization=False)`, `set_rl_training(False)`, 150 matches, `finish_match` then `reset_match` --
so the replay is the published measurement with a recorder switched on, not a new one. The control that
makes that claim checkable is built into the campaign: the per-team win rates this replay produces must
equal the published `results/study/per_seed_metrics.csv` on every cell, and `report_regimes.py` refuses to
report if they do not.

The output is one row per team per match -- `run_experiment.py`'s `eval_match_metrics.csv` schema plus the
`duration_seconds` column that was missing -- so the regime classification can be derived rather than
guessed from medians:

    python scripts/eval_regimes.py --seeds 1,2,3,4,5,6,7,8,9,10   # from the tree that owns --experiments
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _readout_module():
    """`paradigms_from_files` lives in eval_readout.py; borrowing it is what keeps the two instruments
    from drifting apart on which arm is which -- the same resolution rule the round-robin relies on."""
    path = Path(__file__).with_name("eval_readout.py")
    spec = importlib.util.spec_from_file_location("_regimes_eval_readout", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10",
                        help="training seeds of the replicates to replay")
    parser.add_argument("--experiments", default="exp1", help="comma-separated: exp1, exp2")
    parser.add_argument("--matches", type=int, default=150, help="held-out matches per replicate")
    parser.add_argument("--seed-offset", type=int, default=9000,
                        help="the offset run_experiment.py adds to the replicate seed for its held-out stream")
    parser.add_argument("--out", type=Path, default=None,
                        help="default results/study/parts/<experiment>_regimes")
    parser.add_argument("--require-checkpoints", action="store_true",
                        help="fail if a replicate's checkpoints are absent instead of skipping it")
    return parser.parse_args()


def evaluate(matches: int, seed: int) -> dict:
    """The study's own held-out protocol, with the per-match recorder switched on.

    Copied from `run_experiment.py::evaluate` rather than imported, for the same reason
    `scripts/eval_readout.py` copies it: the body *is* the protocol, and
    `tests/test_readout_parity.py` pins the lines that must not drift between the three copies.
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
    rows: list[dict[str, object]] = []
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
            rows.append({**{k: row[k] for k in ("match_index", "team_name", "paradigm", "winner",
                                                 "eliminations", "mean_survival_time", "shot_accuracy")},
                         "duration_seconds": round(result.duration_seconds, 2)})
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
    return {"per_team": summary, "rows": rows,
            "median_duration_seconds": sorted(durations)[len(durations) // 2] if durations else None}


def versioned_replicates_root() -> Path:
    """The study's versioned blocks live once, at the repository root, whichever tree evaluates them.

    `data/training_replicates/` holds all twenty replicates -- both experiments -- so ctde_arena reaches
    it through its parent, the same resolution `run_cross_arena.py` uses for the blocks it plays. Resolving
    against the evaluating tree instead silently skips every replicate of the other experiment.
    """
    return (ROOT.parent if ROOT.name == "ctde_arena" else ROOT) / "data" / "training_replicates"


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import csv

    from marl_arena.config import CONFIG, seed_all
    from marl_arena.controllers import rl_controller

    readout = _readout_module()
    started = time.time()
    records: list[dict[str, object]] = []
    win_rates: dict[str, dict[str, float]] = {}
    skipped: list[str] = []
    for experiment in args.experiments.split(","):
        if (experiment == "exp2") != (ROOT.name == "ctde_arena"):
            # Each tree's controller is the code that trained its own arms; loading experiment 2's weights
            # through experiment 1's actuator would report a number that looks fine and is wrong -- the
            # same guard eval_readout.py carries, for the same reason.
            raise SystemExit(f"{experiment} checkpoints must be evaluated from "
                             f"{'ctde_arena/' if experiment == 'exp2' else 'the repository root'}, "
                             f"not from {ROOT.name}/")
        for seed in (int(s) for s in args.seeds.split(",")):
            checkpoints = versioned_replicates_root() / f"{experiment}_seed{seed}"
            # .npz is the format this runtime loads (A-44/A-45); .pt only ever exists next to it.
            files = sorted(checkpoints.glob("*.npz")) or sorted(checkpoints.glob("*.pt"))
            if not checkpoints.is_dir() or not files:
                if args.require_checkpoints:
                    raise SystemExit(f"{experiment}_seed{seed}: no checkpoints under {checkpoints}")
                skipped.append(f"{experiment}_seed{seed}")
                print(f"  {experiment}_seed{seed}: no versioned checkpoints, skipped", flush=True)
                continue
            # The checkpoint directory is swapped per replicate, in the module the controller resolves it
            # from -- CONFIG is frozen and read at call time inside the loop, exactly as eval_readout.py
            # swaps the readout.
            rl_controller.CONFIG = dataclasses.replace(CONFIG, rl_checkpoint_dir=checkpoints)
            seed_all(seed)
            elapsed = time.time()
            result = evaluate(args.matches, seed + args.seed_offset)
            teams = readout.paradigms_from_files(experiment, files)
            if set(result["per_team"]) != set(teams):
                raise SystemExit(f"{experiment}_seed{seed}: the simulation seated {sorted(result['per_team'])} "
                                 f"but the checkpoint names say {sorted(teams)}")
            win_rates[f"{experiment}_seed{seed}"] = {
                team: round(stats["win_rate"], 4) for team, stats in result["per_team"].items()}
            for row in result["rows"]:
                records.append({
                    "experiment": experiment,
                    "seed": seed,
                    **row,
                })
            print(f"  {experiment}_seed{seed}: win rates {win_rates[f'{experiment}_seed{seed}']} "
                  f"median {result['median_duration_seconds']:.1f}s "
                  f"in {time.time() - elapsed:.0f}s", flush=True)

    if not records:
        raise SystemExit("no replicate could be evaluated -- are the versioned replicates in this checkout?")

    out_dir = args.out or Path("results") / "study" / "parts" / f"{args.experiments.split(',')[0]}_regimes"
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with (out_dir / "match_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    (out_dir / "part_summary.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "instrument": "eval_regimes",
        "readout": "argmax",
        "checkpoint_set": "versioned",
        "protocol": {"matches_per_replicate": args.matches, "seed_offset": args.seed_offset,
                     "domain_randomization": False, "training_mode": False,
                     "hidden_dim": CONFIG.rl_hidden_dim,
                     "eval_temperature": CONFIG.eval_temperature,
                     "arena": {"alive_bonus": CONFIG.reward_alive_bonus,
                               "alive_decay_half_life_s": CONFIG.reward_alive_decay_half_life,
                               "cap_awards_win": CONFIG.termination_cap_awards_win,
                               "match_duration_seconds": CONFIG.match_duration_seconds}},
        "experiments": sorted({str(r["experiment"]) for r in records}),
        "replicates": len(win_rates),
        "match_rows": len(records),
        "replayed_win_rates": win_rates,
        "skipped_for_absent_checkpoints": skipped,
        "wall_seconds": round(time.time() - started, 1),
        "note": "One row per team per match: run_experiment.py's eval_match_metrics.csv schema plus "
                "duration_seconds, the column the study's own evaluation never wrote. The versioned "
                "checkpoints under data/training_replicates/ are what the round-robin plays, so this "
                "replay is runnable from a clone.",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'match_rows.csv'} ({len(records)} team-match rows) and part_summary.json")


if __name__ == "__main__":
    sys.exit(main())
