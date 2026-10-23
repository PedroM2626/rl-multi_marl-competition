#!/usr/bin/env bash
# Take the no_dr arm from three seeds to ten, then match its common-pool replay to the headline table.
#
# Section 8.13's no_dr pool is three blocks against the round-robin's ten, and it stays that way only because
# the arm trained three seeds -- the evaluation cannot be extended past what exists. This campaign trains
# seeds 4 to 10 of both experiments under the same single key change (DOMAIN_RANDOMIZATION=false, 500 k
# steps, same seeds numbering as the committed study so the slot rotation pairs seed by seed), versions the
# resulting checkpoints, and then replays the arm over ten blocks.
#
# What it buys: the finding that domain randomisation buys stability rather than level is currently an
# extrapolation from three draws per arm, where the headline pool's between-block SD rests on ten. With ten
# blocks the same statement can be made at the same replication count as § 8.10, and the reordering it
# reports -- CTDE-Comm from first to fifth -- stops being a three-network observation.
#
# Chained behind the traces campaign so neither campaign's wall clock is measured over the other.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/_runtime.sh"
TMP="${TEMP:-/tmp}"

NEW_SEEDS=4,5,6,7,8,9,10
PAIRS=("4 4" "5 5" "6 6" "7 7" "8 8" "9 9" "10 10")
OUTDIR="results/cross_arena_campaign"

while [ ! -f /tmp/traces.flag ]; do sleep 300; done
echo "traces complete; training no_dr seeds 4-10 at $(date -Is)"

# Training is the expensive, once-only part (170.8 min measured for these 14 runs). Re-launching the
# campaign to redo the pool replay must not retrain it, so the block is skipped when every run of the
# extension already wrote its evaluation summary.
trained=0
for s in 4 5 6 7 8 9 10; do
  for e in exp1 exp2; do
    tree="$ROOT/data/runs_ablation_no_dr"; [ "$e" = exp2 ] && tree="$ROOT/ctde_arena/data/runs_ablation_no_dr"
    [ -f "$tree/${e}_seed${s}/evaluation/run_summary.json" ] && trained=$((trained+1))
  done
done
if [ "$trained" = 14 ]; then
  echo "all 14 extension runs already trained; skipping training at $(date -Is)"
else
  echo "$trained of 14 extension runs trained so far; running the study"
  env DOMAIN_RANDOMIZATION=false "$PY" scripts/run_study.py --steps 500000 --seeds "$NEW_SEEDS" --jobs 14 \
    --eval-matches 150 --out data/runs_ablation_no_dr --results-dir results/ablation_no_dr \
    > "$TMP/no_dr_extension.log" 2>&1
  tail -6 "$TMP/no_dr_extension.log"
  if ! grep -q "failures: none" "$TMP/no_dr_extension.log"; then
    echo "the no_dr extension did not finish cleanly; see $TMP/no_dr_extension.log" >&2
    exit 1
  fi
fi

# Version the new sets before playing them: run_cross_arena.py reads data/campaign_replicates/, and the
# manifest's hidden width is what its capacity guard checks. Refusing to replay unversioned weights is the
# point -- a pool whose checkpoints are not in the repository is not reproducible.
"$PY" scripts/version_campaign_checkpoints.py > "$TMP/no_dr_versioning.log" 2>&1
tail -4 "$TMP/no_dr_versioning.log"
sets=$("$PY" -c 'import json;print(len(json.load(open("data/campaign_replicates/MANIFEST.json"))["arms"]["no_dr"]["sets"]))')
if [ "$sets" != 20 ]; then
  echo "the no_dr arm is versioned at $sets sets, expected 20 (10 seeds x 2 experiments); the versioning " \
       "script derives seeds from disk, so check $TMP/no_dr_versioning.log" >&2
  exit 1
fi

# One bounded pool, not a wave per block. Each run is single-threaded and measured 3 min per triple of
# architectures (20 triples per run), so five runs at a time -- the shape the earlier campaigns used --
# leaves 27 of this machine's 32 cores idle and stretches the 35 pending runs to seven hours. With a pool
# of 12 the same work is under two and a half hours. Parallelism cannot change a result: every run carries
# its own harness seed and writes its own directory, and the merge below reads only the files that exist.
PAR=${NODR_PARALLEL:-12}
pending=()
for pair in "${PAIRS[@]}"; do
  set -- $pair
  for H in 1 2 3 4 5; do
    dir="no_dr__e1s$1_e2s${2}_h${H}"
    if [ -f "$ROOT/$OUTDIR/$dir/matches.csv" ]; then
      echo "  $dir already played, skipping"
      continue
    fi
    pending+=("$1 $2 $H")
  done
done
echo "replaying ${#pending[@]} no_dr runs, $PAR at a time, at $(date -Is)"

running=0
for spec in "${pending[@]}"; do
  set -- $spec
  "$PY" "$ROOT/scripts/run_cross_arena.py" --campaign no_dr --exp1-replicate "$1" --exp2-replicate "$2" \
      --seed "$3" --out "$OUTDIR/no_dr__e1s$1_e2s$2_h$3" > "$TMP/nodr10_$1_$2_$3.log" 2>&1 &
  running=$((running+1))
  if [ "$running" -ge "$PAR" ]; then
    wait -n
    running=$((running-1))
    echo "  a no_dr run finished, $running still running at $(date -Is)"
  fi
done
wait
echo "  all no_dr runs replayed at $(date -Is)"

missing=0
for pair in "${PAIRS[@]}"; do
  set -- $pair
  for H in 1 2 3 4 5; do
    [ -f "$ROOT/$OUTDIR/no_dr__e1s$1_e2s$2_h${H}/matches.csv" ] \
      || { echo "  MISSING no_dr__e1s$1_e2s$2_h${H}"; missing=1; }
  done
done
[ "$missing" = 0 ] || { echo "refusing to merge an incomplete no_dr pool" >&2; exit 1; }

merged=$("$PY" scripts/combine_cross_arena.py "$ROOT"/$OUTDIR/no_dr__* \
  --out results/cross_arena/no_dr_blocks.csv)
echo "$merged"
case "$merged" in
  *"merged 10 training blocks x 5 harness seeds = 50 runs -> 90000 team-matches"*) ;;
  *) echo "the no_dr merge is not ten blocks over five seeds; see the line above" >&2; exit 1 ;;
esac
"$PY" scripts/report_cross_arena.py --in results/cross_arena/no_dr_blocks.csv \
  --json-out results/cross_arena/no_dr_blocks.json > "$TMP/nodr10_report.txt" 2>&1
echo NO_DR10_DONE > /tmp/no_dr10.flag
