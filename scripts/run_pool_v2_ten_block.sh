#!/usr/bin/env bash
# Take the arena-v2 pool from six training blocks to ten, so it matches the headline table.
#
# Section 8.12's rule-vs-nothing-else control was measured when the committed round-robin had six blocks, and
# it is still six: 18,000 matches against the ten-block pool's 30,000. That mismatch is now the open item 13
# of docs/11's list, and it is the cheaper kind -- the checkpoints exist, only matches are missing. Playing
# the four remaining blocks puts the two instruments on the same population of trained networks, which is
# what lets the section say that a rule change reorders the table rather than that it reorders *a* table.
#
# Chained behind the readout campaign for the reason every other campaign here chains: overlapping runs do
# not change an answer, they make each campaign's wall-clock figure wrong for both, and those figures are how
# the next campaign gets scoped.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv-check/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"
TMP="${TEMP:-/tmp}"

PAIRS=("5 6" "6 8" "7 9" "8 10")
OUTDIR="results/cross_arena_v2_blocks"

while [ ! -f /tmp/readout.flag ]; do sleep 180; done
echo "readout complete; starting the ten-block v2 extension at $(date -Is)"

# Pre-flight, about a minute: the class of failure this guards against is a campaign that "completes" having
# played nothing, because the interpreter resolved a relative script path against the wrong root.
"$PY" "$ROOT/scripts/run_cross_arena.py" --exp1-replicate 5 --exp2-replicate 6 --seed 99 \
  --matches-per-triple 1 --out "$TMP/preflight_v2_ten" > "$TMP/v2ten_preflight.log" 2>&1
if [ ! -s "$TMP/preflight_v2_ten/matches.csv" ]; then
  echo "pre-flight played no matches; see $TMP/v2ten_preflight.log" >&2
  exit 1
fi
echo "pre-flight ok, starting at $(date -Is)"

# The three keys arena v2 changed, exported rather than written to disk, exactly as in
# run_pool_v2_campaign.sh: dotenv loads .env with override=False, so the real environment wins and the
# committed .env stays the study's instrument.
export REWARD_ALIVE_BONUS=0.0015 REWARD_ALIVE_DECAY_HALF_LIFE_S=30 TERMINATION_CAP_AWARDS_WIN=false
for pair in "${PAIRS[@]}"; do
  set -- $pair; S=$1; M=$2
  for H in 1 2 3 4 5; do
    dir="v2__e1s${S}_e2s${M}_h${H}"
    if [ -f "$ROOT/$OUTDIR/$dir/matches.csv" ]; then
      echo "  $dir already played, skipping (delete it to re-run)"
      continue
    fi
    ( "$PY" "$ROOT/scripts/run_cross_arena.py" --exp1-replicate "$S" --exp2-replicate "$M" --seed "$H" \
        --out "$OUTDIR/$dir" > "$TMP/v2ten_${S}_${M}_${H}.log" 2>&1 ) &
  done
  wait
  echo "  block exp1_seed${S}+exp2_seed${M} done at $(date -Is)"
done
unset REWARD_ALIVE_BONUS REWARD_ALIVE_DECAY_HALF_LIFE_S TERMINATION_CAP_AWARDS_WIN

missing=0
for dir in "$ROOT"/$OUTDIR/v2__*; do
  [ -f "$dir/matches.csv" ] || { echo "  MISSING $dir"; missing=1; }
done
if [ "$missing" = 1 ]; then
  echo "refusing to merge: at least one run directory has no matches.csv" >&2
  exit 1
fi

merged=$("$PY" scripts/combine_cross_arena.py "$ROOT"/$OUTDIR/* --out results/cross_arena/v2_blocks.csv)
echo "$merged"
case "$merged" in
  *"merged 10 training blocks x 5 harness seeds = 50 runs -> 90000 team-matches"*) ;;
  *) echo "the merge did not report ten v2 blocks over five seeds; see the line above" >&2; exit 1 ;;
esac

"$PY" scripts/report_cross_arena.py --in results/cross_arena/v2_blocks.csv \
  --json-out results/cross_arena/v2_blocks.json > "$TMP/v2ten_report.txt" 2>&1
echo V2_TEN_DONE > /tmp/v2_ten.flag
