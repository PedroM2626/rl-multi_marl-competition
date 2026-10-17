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

PY="$ROOT/.venv-check/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"
TMP="${TEMP:-/tmp}"

NEW_SEEDS=4,5,6,7,8,9,10
PAIRS=("4 4" "5 5" "6 6" "7 7" "8 8" "9 9" "10 10")
OUTDIR="results/cross_arena_campaign"

while [ ! -f /tmp/traces.flag ]; do sleep 300; done
echo "traces complete; training no_dr seeds 4-10 at $(date -Is)"

env DOMAIN_RANDOMIZATION=false "$PY" scripts/run_study.py --steps 500000 --seeds "$NEW_SEEDS" --jobs 14 \
  --eval-matches 150 --out data/runs_ablation_no_dr --results-dir results/ablation_no_dr \
  > "$TMP/no_dr_extension.log" 2>&1
tail -6 "$TMP/no_dr_extension.log"
if ! grep -q "failures: none" "$TMP/no_dr_extension.log"; then
  echo "the no_dr extension did not finish cleanly; see $TMP/no_dr_extension.log" >&2
  exit 1
fi

# Version the new sets before playing them: run_cross_arena.py reads data/campaign_replicates/, and the
# manifest's hidden width is what its capacity guard checks.
"$PY" scripts/version_campaign_checkpoints.py > "$TMP/no_dr_versioning.log" 2>&1
tail -4 "$TMP/no_dr_versioning.log"

for pair in "${PAIRS[@]}"; do
  set -- $pair; S=$1; M=$2
  for H in 1 2 3 4 5; do
    dir="no_dr__e1s${S}_e2s${M}_h${H}"
    if [ -f "$ROOT/$OUTDIR/$dir/matches.csv" ]; then
      echo "  $dir already played, skipping"
      continue
    fi
    ( "$PY" "$ROOT/scripts/run_cross_arena.py" --campaign no_dr --exp1-replicate "$S" --exp2-replicate "$M" \
        --seed "$H" --out "$OUTDIR/$dir" > "$TMP/nodr10_${S}_${M}_${H}.log" 2>&1 ) &
  done
  wait
  echo "  no_dr block exp1_seed${S}+exp2_seed${M} done at $(date -Is)"
done

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
