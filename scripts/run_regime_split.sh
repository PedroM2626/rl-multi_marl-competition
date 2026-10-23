#!/usr/bin/env bash
# Close audit item 8's open half: the study's held-out win rates split by how each match was decided.
#
# § 8.1's two-regimes section reports that five of twenty replicates play a different game -- their
# matches end on the 90 s clock -- but the study's tables still pool the regimes, because the held-out
# recorder of run_experiment.py kept everything except the match duration. This campaign replays the
# study's own held-out protocol (argmax, seed+9000, fixed variant, no randomisation) over the twenty
# versioned checkpoint sets under data/training_replicates/, with the duration recorded per match, so
# the split can be derived instead of guessed from medians.
#
# Evaluation only -- no training. Two processes, one per tree, each replaying its own experiment's ten
# replicates serially; the same shape the readout campaign used, so the wall-clock figures are
# comparable. Expected output: 9,000 team-match rows (10 seeds x 150 matches x 3 teams per experiment)
# merged into results/study/eval_match_regimes.csv, then scripts/report_regimes.py derives
# results/study/regime_split.json and refuses to report unless the replayed win rates equal the
# published results/study/per_seed_metrics.csv on every cell.
#
# Not chained behind another campaign: run it on an otherwise idle host, or every cost figure in
# docs/09 § 9.5 is wrong for both this and whatever else is running.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# The repository runtime only: _runtime.sh verifies the JAX pin and the Rust extension and refuses
# .venv-check (jax 0.6.2, no Rust), which no measurement in docs/09 section 9.5 was taken on.
source "$ROOT/scripts/_runtime.sh"
TMP="${TEMP:-/tmp}"

echo "regime replay started at $(date -Is)"
for spec in "exp1 ." "exp2 ctde_arena"; do
  set -- $spec
  # One process per tree, serial over that experiment's seeds -- the readout campaign's shape. The
  # instrument resolves the versioned replicates at the repository root whichever tree it runs from.
  ( cd "$ROOT/$2" && "$PY" "$ROOT/$2/scripts/eval_regimes.py" --experiments "$1" \
      --seeds 1,2,3,4,5,6,7,8,9,10 --matches 150 --require-checkpoints \
      > "$TMP/regimes_${1}.log" 2>&1 ) &
done
wait
echo "  both trees done at $(date -Is)"

# Merge the two parts the way the readout campaign merges its arms: column identity, exact row count,
# and one protocol snapshot -- or the merge refuses, because a regime table is only comparable to the
# published study if every match behind it was played under the same rules.
"$PY" - "$ROOT/results/study" "$ROOT/results/study/parts/exp1_regimes" \
          "$ROOT/ctde_arena/results/study/parts/exp2_regimes" <<'PY'
import csv, json, sys
from pathlib import Path
out, first, second = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows, fields = [], None
for part in (first, second):
    rows_path = part / "match_rows.csv"
    if not rows_path.is_file():
        raise SystemExit(f"{rows_path} missing -- that tree's replay did not complete; see the regimes logs")
    with rows_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if fields is None:
            fields = reader.fieldnames
        elif reader.fieldnames != fields:
            raise SystemExit(f"{rows_path} has columns {reader.fieldnames}, expected {fields}")
        body = list(reader)
        if not body:
            raise SystemExit(f"{rows_path} has no rows")
        rows.extend(body)
if len(rows) != 9000:
    raise SystemExit(f"expected 9,000 team-match rows (2 experiments x 10 seeds x 150 matches x 3 teams), "
                     f"merged {len(rows)}")
out.mkdir(parents=True, exist_ok=True)
with (out / "eval_match_regimes.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

metas = []
for part in (first, second):
    meta_path = part / "part_summary.json"
    if not meta_path.is_file():
        raise SystemExit(f"{meta_path} missing -- the replay played but recorded no protocol")
    metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
if {json.dumps(m["protocol"], sort_keys=True) for m in metas} != {json.dumps(metas[0]["protocol"], sort_keys=True)}:
    raise SystemExit("the two trees replayed under different protocol snapshots; merging would hide that")
(out / "regime_summary.json").write_text(json.dumps({
    "instrument": "eval_regimes over the versioned study replicates",
    "readout": metas[0]["readout"],
    "checkpoint_set": metas[0]["checkpoint_set"],
    "protocol": metas[0]["protocol"],
    "experiments": sorted({m["experiments"][0] for m in metas if m["experiments"]}),
    "replicates": sum(m["replicates"] for m in metas),
    "match_rows": len(rows),
    "wall_seconds_by_part": [m.get("wall_seconds") for m in metas],
    "replayed_win_rates": {**metas[0]["replayed_win_rates"], **metas[1]["replayed_win_rates"]},
    "parts": [first.relative_to(ROOT).as_posix(), second.relative_to(ROOT).as_posix()],
}, indent=2), encoding="utf-8")
print(f"merged {len(rows)} rows -> {out / 'eval_match_regimes.csv'} and regime_summary.json")
PY

"$PY" scripts/report_regimes.py > "$TMP/regimes_report.txt" 2>&1 || {
  echo "report_regimes.py failed -- see $TMP/regimes_report.txt"; exit 1; }
tail -n 40 "$TMP/regimes_report.txt"
echo REGIMES_DONE
