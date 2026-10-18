"""Item 7: what changes when the only difference between two studies is the step budget.

`scripts/run_three_million_campaign.sh` trains the same twenty replicates at 3,000,000 steps that the
committed study trained at 500,000. Two questions come out of that, and they need different instruments:

* **Does the ranking change?** Held-out greedy win rate is the study's own measure, so this compares
  `results/study_3m/per_seed_metrics.csv` against `results/study/per_seed_metrics.csv` paired on seed number,
  with the same `report_study.paired_t` the study tables use.
* **Does CTDE-Comm's late improvement continue?** That is a curve question, and the curve is the held-out
  snapshots each run writes at 25/50/75/100 % of *its own* budget. Fractions are not steps: 25 % of 3 M is
  750,000, already half again the whole committed budget, so the two curves are compared as drift within a
  budget, not as points on a shared x-axis.

The record it writes carries its own controls, because the comparison is worthless without them: the budget
each side actually trained, the seeds the two sides share, whether the 500 k side is the committed study's
own numbers rather than a re-measurement of them, and how many held-out matches sit behind each cell.

    python scripts/analyse_budget.py --three-m data/runs_3m --half-m data/runs \
        --three-m-results results/study_3m --out results/study_3m/budget_analysis.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def trees() -> tuple[Path, Path]:
    """Experiment 2 is trained inside its own tree, so a run directory can be under either (A-35)."""
    return ROOT, ROOT / "ctde_arena"


def load_report_study():
    """The real scripts directory, not ROOT: tests patch ROOT to point at synthetic trees."""
    here = Path(__file__).resolve().parent / "report_study.py"
    spec = importlib.util.spec_from_file_location("_budget_report", here)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_run(runs_dir: Path, experiment: str, seed: int) -> Path:
    for tree in trees():
        candidate = tree / runs_dir / f"{experiment}_seed{seed}"
        if (candidate / "evaluation" / "run_summary.json").is_file():
            return candidate
    raise SystemExit(f"no completed run for {experiment}_seed{seed} under {runs_dir} in either tree")


def curve_by_paradigm(runs_dir: Path, experiment: str, seed: int) -> dict[str, dict[str, float]]:
    """{paradigm: {fraction: held-out win rate}} for one replicate, from the run's own snapshot record."""
    summary = json.loads((find_run(runs_dir, experiment, seed) / "evaluation"
                          / "run_summary.json").read_text(encoding="utf-8"))
    paradigm_of = summary["paradigms"]
    out: dict[str, dict[str, float]] = {name: {} for name in paradigm_of.values()}
    for point in summary["curve"]:
        fraction = round(float(point["fraction"]), 2)
        for team, rates in point["per_team"].items():
            out[paradigm_of[team]][str(fraction)] = float(rates["win_rate"])
    return out


def budget_of(runs_dir: Path, experiment: str, seed: int) -> dict[str, object]:
    summary = json.loads((find_run(runs_dir, experiment, seed) / "evaluation"
                          / "run_summary.json").read_text(encoding="utf-8"))
    return {"target_steps": summary["target_steps"], "completed_steps": summary["completed_steps"],
            "held_out_matches": summary["held_out_greedy"]["matches"],
            "curve_matches": next((len(p["per_team"]) for p in summary["curve"]), 0)}


def means_by_paradigm(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        out.setdefault((row["experiment"], row["paradigm"]), []).append(float(row["eval_win_rate"]))
    return out


def drift(curve: dict[str, float]) -> float | None:
    """Held-out win rate at the end of the run's own budget minus its value at a quarter of it."""
    if "0.25" not in curve or "1.0" not in curve:
        return None
    return curve["1.0"] - curve["0.25"]


def analyse(three_m: Path, half_m: Path, three_m_results: Path) -> dict[str, object]:
    report = load_report_study()
    control_rows = report.read_metrics(ROOT / "results" / "study")
    arm_rows = report.read_metrics(ROOT / three_m_results)
    control = {(r["experiment"], r["paradigm"], int(r["seed"])): r for r in control_rows}
    arm = {(r["experiment"], r["paradigm"], int(r["seed"])): r for r in arm_rows}
    shared = sorted(set(control) & set(arm))
    cells = sorted({(e, p) for e, p, _ in shared})

    paired: dict[str, object] = {}
    for experiment, paradigm in cells:
        keys = [k for k in shared if k[0] == experiment and k[1] == paradigm]
        differences = [float(arm[k]["eval_win_rate"]) - float(control[k]["eval_win_rate"]) for k in keys]
        t_stat, p_value = report.paired_t(differences)
        paired[f"{experiment}|{paradigm}"] = {
            "seeds": len(keys),
            "held_out_500k": round(sum(float(control[k]["eval_win_rate"]) for k in keys) / len(keys), 4),
            "held_out_3m": round(sum(float(arm[k]["eval_win_rate"]) for k in keys) / len(keys), 4),
            "paired_diff": round(sum(differences) / len(differences), 4),
            "t": round(t_stat, 3),
            "p": p_value,
            "seeds_below_500k": sum(1 for d in differences if d < 0),
        }

    ranking = {}
    for label, table in (("500k", control_rows), ("3m", arm_rows)):
        means = means_by_paradigm(table)
        ranking[label] = {}
        for experiment in ("exp1", "exp2"):
            in_experiment = [(p, sum(v) / len(v)) for (e, p), v in means.items() if e == experiment]
            ranking[label][experiment] = [{"paradigm": p, "mean": round(value, 4)}
                                          for p, value in sorted(in_experiment, key=lambda item: -item[1])]

    curves: dict[str, object] = {}
    for experiment, paradigm in cells:
        seeds = sorted({k[2] for k in shared if k[0] == experiment and k[1] == paradigm})
        entry: dict[str, object] = {"seeds": seeds, "500k": {}, "3m": {}, "drift": {}}
        for label, runs_dir in (("500k", half_m), ("3m", three_m)):
            per_fraction: dict[str, list[float]] = {}
            drifts: list[float] = []
            for seed in seeds:
                curve = curve_by_paradigm(runs_dir, experiment, seed)[paradigm]
                for fraction, value in curve.items():
                    per_fraction.setdefault(fraction, []).append(value)
                step = drift(curve)
                if step is not None:
                    drifts.append(step)
            entry[label] = {fraction: round(sum(values) / len(values), 4)
                           for fraction, values in sorted(per_fraction.items())}
            entry["drift"][label] = round(sum(drifts) / len(drifts), 4) if drifts else None
        curves[f"{experiment}|{paradigm}"] = entry

    budgets = {}
    for label, runs_dir in (("500k", half_m), ("3m", three_m)):
        runs = {(experiment, seed): budget_of(runs_dir, experiment, seed)
                for experiment in ("exp1", "exp2")
                for seed in sorted({k[2] for k in shared if k[0] == experiment})}
        seen = list(runs.values())
        budgets[label] = {
            "runs": len(runs),
            "target_steps": sorted({s["target_steps"] for s in seen}),
            "completed_steps": sorted({s["completed_steps"] for s in seen}),
            "held_out_matches_per_arm": sorted({s["held_out_matches"] for s in seen}),
        }

    # The 500 k side has to be the committed study's own numbers, not a second measurement of the same runs:
    # the per-team held-out rate the run wrote, against the row the study's CSV carries for that cell.
    parity = 0.0
    for key in shared:
        run = find_run(half_m, key[0], key[2])
        summary = json.loads((run / "evaluation" / "run_summary.json").read_text(encoding="utf-8"))
        team = next(t for t, name in summary["paradigms"].items() if name == key[1])
        recorded = float(summary["held_out_greedy"]["per_team"][team]["win_rate"])
        parity = max(parity, abs(recorded - float(control[key]["eval_win_rate"])))

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "docs/11_code_audit.md#open-work-suggested-measurements item 7: the study re-measured at "
                   "the 3,000,000-step budget the code defaults to, against the committed 500,000-step study "
                   "paired on seed number.",
        "produced_by": "bash scripts/run_three_million_campaign.sh, then scripts/analyse_budget.py",
        # Paths in a committed artefact are a cross-host API: as_posix, never str(Path) (A-42).
        "inputs": {"three_m_runs": three_m.as_posix(), "half_m_runs": half_m.as_posix(),
                   "three_m_results": three_m_results.as_posix(),
                   "control_results": "results/study/per_seed_metrics.csv"},
        "controls": {
            "budgets": budgets,
            "shared_cells": len(shared),
            "half_m_parity_max_abs_delta_vs_committed_study": round(parity, 6),
            "note": "curve points are held-out snapshots at fractions of each run's OWN budget, so a "
                    "fraction is not a step count: 0.25 of 3,000,000 is 750,000, more than the whole "
                    "committed budget. Drift is end-minus-quarter within one budget. The parity figure is "
                    "expected to be at the rounding floor of results/study/per_seed_metrics.csv, which "
                    "stores four decimal places -- it says this analysis read the committed study's own "
                    "runs, not a second measurement of them.",
        },
        "paired_held_out": paired,
        "ranking": ranking,
        "held_out_curves": curves,
    }


def check(record: dict[str, object]) -> None:
    """Refuse to publish a budget comparison whose runs did not train the budget.

    A 3 M arm that stopped at 2.5 M would still produce a paired table and a drift figure, and both would read
    as a shorter budget rather than as a failed run -- so the guard is on the record, not on the campaign log.
    """
    budgets = record["controls"]["budgets"]
    for label, budget_record in budgets.items():
        if budget_record["target_steps"] != budget_record["completed_steps"]:
            raise SystemExit(f"the {label} side did not complete every run it claimed: {budget_record}")
        if len(budget_record["target_steps"]) != 1:
            raise SystemExit(f"the {label} side mixes budgets: {budget_record['target_steps']}")
    if len(record["paired_held_out"]) != 6:
        raise SystemExit(f"expected six (experiment, paradigm) cells, got {sorted(record['paired_held_out'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-m", type=Path, default=Path("data/runs_3m"))
    parser.add_argument("--half-m", type=Path, default=Path("data/runs"))
    parser.add_argument("--three-m-results", type=Path, default=Path("results/study_3m"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    record = analyse(args.three_m, args.half_m, args.three_m_results)
    check(record)
    budgets = record["controls"]["budgets"]
    if budgets["3m"]["target_steps"] != [3_000_000]:
        raise SystemExit(f"this is the 3 M-step instrument, and the 3 M side reports "
                         f"{budgets['3m']['target_steps']}")

    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"wrote {out_path.as_posix()}")
    for key, row in record["paired_held_out"].items():
        print(f"  {key:22s} 500k {row['held_out_500k']:.3f} -> 3m {row['held_out_3m']:.3f} "
              f"({row['paired_diff']:+.3f}, p={row['p']:.3g}, below {row['seeds_below_500k']}/"
              f"{row['seeds']})")
    print(f"  half-M parity vs the committed study: "
          f"{record['controls']['half_m_parity_max_abs_delta_vs_committed_study']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
