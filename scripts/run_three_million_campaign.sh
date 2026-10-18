#!/usr/bin/env bash
# Open-work item 7: does the study's ranking survive the 3 M-step budget the code defaults to?
#
# The item has read "deliberately not done" since the first audit, on the argument that a longer budget only
# refines an estimate of a quantity that is not stable -- the *direction* of the win-rate curve flipped when
# the replicate count changed. That argument is itself a claim about this arena, and it can be measured: train
# the same twenty replicates (same seeds, same budget knob except its value, same held-out protocol) at
# 3,000,000 steps instead of 500,000, and compare them against the committed study paired on seed number.
#
# Two questions, and they are different:
#   1. does the ranking change at the longer budget?  -- report_study's paired table answers that.
#   2. does CTDE-Comm's late improvement continue?    -- that needs the curve inside the run, which is what
#      scripts/analyse_budget.py reads out of each run's team_match_metrics.csv.
#
# Cost, scaled from the measured campaign: the 500 k x 20 study took 117.5 min of wall clock at twenty
# concurrent, so this is about six times that. Nothing else in the repository can run while it does: the
# throughput figures in docs/09 section 9.5 are quoted per concurrency level, and a second campaign on the
# same machine invalidates all of them.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv-check/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"
TMP="${TEMP:-/tmp}"
STEPS=3000000
OUTDIR="data/runs_3m"
RESULTS="results/study_3m"

# Re-launching must not re-train: fourteen hours is not a thing to lose to a typo. A run counts as trained
# when it wrote its evaluation summary, which is the last thing run_experiment.py writes.
trained=0
for seed in 1 2 3 4 5 6 7 8 9 10; do
  for e in exp1 exp2; do
    tree="$ROOT/$OUTDIR"; [ "$e" = exp2 ] && tree="$ROOT/ctde_arena/$OUTDIR"
    [ -f "$tree/${e}_seed${seed}/evaluation/run_summary.json" ] && trained=$((trained+1))
  done
done
if [ "$trained" = 20 ]; then
  echo "all 20 replicates already trained; skipping training at $(date -Is)"
else
  echo "$trained of 20 replicates trained so far; launching the 3 M-step study at $(date -Is)"
  # The budget is the only difference from the committed study: same seeds, same 150 held-out greedy matches,
  # same slot rotation, same configuration keys.
  "$PY" scripts/run_study.py --steps "$STEPS" --seeds 1,2,3,4,5,6,7,8,9,10 --jobs 20 \
    --eval-matches 150 --out "$OUTDIR" --results-dir "$RESULTS" \
    > "$TMP/study_3m.log" 2>&1
  echo "study exit $? at $(date -Is)"
  tail -6 "$TMP/study_3m.log"
  if ! grep -q "failures: none" "$TMP/study_3m.log"; then
    echo "the 3 M-step study did not finish cleanly; see $TMP/study_3m.log" >&2
    exit 1
  fi
  # Each run must record the budget it was trained at, or the comparison is anonymous.
  "$PY" - "$OUTDIR" "$STEPS" <<'PY' || exit 1
import json, pathlib, sys
root = pathlib.Path(".")
out_name, steps = sys.argv[1], int(sys.argv[2])
bad, seen = [], 0
for tree in (root, root / "ctde_arena"):
    for run_dir in sorted((tree / out_name).glob("*_seed*")):
        summary = run_dir / "evaluation" / "run_summary.json"
        if not summary.is_file():
            bad.append(f"{run_dir}: no summary")
            continue
        seen += 1
        record = json.loads(summary.read_text(encoding="utf-8"))
        if record.get("target_steps") != steps or record.get("completed_steps") != steps:
            bad.append(f"{run_dir}: {record.get('completed_steps')}/{record.get('target_steps')}")
print(f"checked {seen} run summaries")
if seen != 20 or bad:
    raise SystemExit(f"expected 20 runs all trained to {steps} steps; problems: {bad[:5]}")
PY
fi

"$PY" scripts/report_study.py --results-dir "$RESULTS" --compare-to results/study \
  > "$TMP/study_3m_report.txt" 2>&1
echo "report exit $? at $(date -Is)"

# The curve question, and the controls the record has to carry for it to mean anything: the budget each run
# actually trained, the seeds the two sides share, and whether the 500 k side is the committed study's own
# numbers rather than a re-measurement of them.
"$PY" scripts/analyse_budget.py --three-m "$OUTDIR" --half-m data/runs \
  --three-m-results "$RESULTS" --out "$RESULTS/budget_analysis.json" \
  > "$TMP/study_3m_analysis.txt" 2>&1
echo "analysis exit $? at $(date -Is)"
tail -20 "$TMP/study_3m_analysis.txt"

if [ ! -s "$RESULTS/budget_analysis.json" ]; then
  echo "no analysis record was written; see $TMP/study_3m_analysis.txt" >&2
  exit 1
fi
echo THREE_M_DONE > /tmp/three_m.flag
echo "3 M-step campaign complete at $(date -Is)"
