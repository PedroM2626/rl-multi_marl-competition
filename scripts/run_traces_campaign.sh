#!/usr/bin/env bash
# Re-train the twenty study replicates so the study has an optimiser trace to be analysed.
#
# docs/11 open work item 4 asks whether a run that ends at 0.000 held-out win rate looked different from the
# inside than one that ends at 0.607. It cannot be answered from the committed artefacts: the traces are
# written by run_experiment.py only since the capture landed, and the study's forty replicates predate it.
# This campaign replays the same seeds, the same 500 k budget and the same evaluation settings into
# data/runs_traces/, which is scratch — nothing committed is touched.
#
# Two controls ride along, because the analysis is worth nothing unless the re-training is the same training:
# every produced checkpoint is hashed against data/training_replicates/, and every re-trained held-out rate
# is compared with results/study/. analyse_traces.py writes both into its record, so a result quoted from a
# re-training that did *not* reproduce the weights has to say so out loud.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv-check/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"
TMP="${TEMP:-/tmp}"

SEEDS=1,2,3,4,5,6,7,8,9,10

if [ -d "$ROOT/data/runs_traces" ] && [ -n "$(ls -A "$ROOT/data/runs_traces" 2>/dev/null)" ]; then
  echo "data/runs_traces already exists and is not empty; delete it to re-run" >&2
  exit 1
fi

echo "re-training 20 replicates with traces, at $(date -Is)"
"$PY" scripts/run_study.py --steps 500000 --seeds "$SEEDS" --jobs 20 --eval-matches 150 \
  --out data/runs_traces --results-dir results/traces > "$TMP/traces_study.log" 2>&1
tail -6 "$TMP/traces_study.log"
if grep -q "failures: \[" "$TMP/traces_study.log" && ! grep -q "failures: none" "$TMP/traces_study.log"; then
  echo "at least one replicate failed to train; see $TMP/traces_study.log" >&2
  exit 1
fi

missing=0
for experiment in exp1 exp2; do
  for seed in 1 2 3 4 5 6 7 8 9 10; do
    directory="$ROOT/data/runs_traces/${experiment}_seed${seed}"
    [ -d "$directory" ] || directory="$ROOT/ctde_arena/data/runs_traces/${experiment}_seed${seed}"
    [ -f "$directory/checkpoints/training_log.json" ] \
      || { echo "  MISSING trace for ${experiment}_seed${seed}"; missing=1; }
  done
done
[ "$missing" = 0 ] || { echo "refusing to analyse a partial trace set" >&2; exit 1; }

"$PY" scripts/analyse_traces.py --runs-dir data/runs_traces --out results/trace_analysis.json \
  > "$TMP/trace_analysis.log" 2>&1
tail -20 "$TMP/trace_analysis.log"
echo TRACES_DONE > /tmp/traces.flag
