#!/usr/bin/env bash
# Re-measure the study's twenty replicates under both readouts of the same policy: argmax, and sampled at a
# fixed softmax temperature.
#
# docs/08 section 8.1 established the divergence from artefacts the training runs had already written. This
# measures it directly: the same checkpoints, the same held-out protocol, one action rule different. Three
# arms, played over all ten seeds of both experiments:
#
#   T=0    argmax -- the published readout, replayed as a control. Its difference from results/study/ is
#          re-measurement noise, and it is the yardstick for the other two rows.
#   T=0.5  sampled, sharp
#   T=1    sampled, the temperature the policy was trained at (no extra scaling of the logits)
#
# Evaluation only -- no training. Cost, measured on this machine while five round-robin runs were already
# loaded: about 2 s per match, so 150 matches per replicate, 20 replicates and three temperatures is roughly
# 5,000 matches per tree per temperature and about an hour of wall clock with six concurrent processes.
#
# Chained behind the ten-block pool campaign for the same reason the other campaigns chain: overlapping runs
# do not change an answer, they make every cost figure in docs/09 section 9.5 wrong for both.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv-check/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"
TMP="${TEMP:-/tmp}"

while [ ! -f /tmp/blocks10.flag ]; do sleep 180; done
echo "ten-block pool complete; starting the readout re-measurement at $(date -Is)"

for T in 0 0.5 1; do
  tag=$(printf "%g" "$T")
  for spec in "exp1 ." "exp2 ctde_arena"; do
    set -- $spec
    # --runs-dir is absolute and always the root's data/runs: this study trained both experiments into it,
    # even though the child processes ran with experiment 2's tree as cwd. Left to the default, the ctde
    # copy would look in ctde_arena/data/runs -- which is where the *ablation* campaign wrote, not this one
    # -- and every replicate would be skipped as absent.
    ( cd "$ROOT/$2" && "$PY" "$ROOT/$2/scripts/eval_readout.py" --temperature "$T" --experiments "$1" \
        --runs-dir "$ROOT/data/runs" \
        --seeds 1,2,3,4,5,6,7,8,9,10 --matches 150 --out "results/readout/parts/${1}_T${tag}" \
        > "$TMP/readout_${1}_T${tag}.log" 2>&1 ) &
  done
  wait
  echo "  T=$tag done at $(date -Is)"
  # One CSV per arm, in the run_study schema, so report_study.py --compare-to reads it as just another arm.
  "$PY" - "$ROOT/results/readout/T${tag}" "$ROOT/results/readout/parts/exp1_T${tag}" \
            "$ROOT/ctde_arena/results/readout/parts/exp2_T${tag}" <<'PY'
import csv, sys
from pathlib import Path
out, first, second = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows, fields = [], None
for part in (first, second):
    handle_path = part / "per_seed_metrics.csv"
    if not handle_path.is_file():
        raise SystemExit(f"{handle_path} missing -- that arm did not complete; see the readout logs")
    with handle_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if fields is None:
            fields = reader.fieldnames
        elif reader.fieldnames != fields:
            raise SystemExit(f"{handle_path} has columns {reader.fieldnames}, expected {fields}")
        body = list(reader)
        if not body:
            raise SystemExit(f"{handle_path} has no rows")
        rows.extend(body)
if len(rows) != 60:
    raise SystemExit(f"expected 60 rows (10 seeds x 2 experiments x 3 paradigms), merged {len(rows)}")
out.mkdir(parents=True, exist_ok=True)
with (out / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"merged {len(rows)} rows from {first.name} and {second.name} -> {out / 'per_seed_metrics.csv'}")
PY
done

for T in 0 0.5 1; do
  tag=$(printf "%g" "$T")
  "$PY" scripts/report_study.py --results-dir "results/readout/T${tag}" --compare-to results/study \
    --label "readout T=${tag}" > "$TMP/readout_report_T${tag}.txt" 2>&1
done
echo READOUT_DONE > /tmp/readout.flag
