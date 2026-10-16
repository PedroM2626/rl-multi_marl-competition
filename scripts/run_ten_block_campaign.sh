#!/usr/bin/env bash
# Extend the cross-experiment round-robin from six training blocks to ten.
#
# Section 8.10's between-block SD is estimated on six draws and docs/10 item 12 says of the replication
# ladder -- 1, 3, 6 blocks giving 2, 8, 5 corrected-significant pairs -- that "separations appear and dissolve
# with replication in both directions", i.e. n = 6 is enough to reverse a ranking and not enough to settle
# one. The study trained ten seeds per experiment and all sixty of those checkpoints are now versioned under
# `data/training_replicates/`, so the remaining four blocks are evaluation time on networks that already
# exist: 20 new runs and 12,000 matches, added to the 30 directories already on disk rather than re-playing
# the six blocks that are done.
#
# Chained behind the v2/no_dr pool campaign. Running them together would not change either answer, but it
# would make every wall-clock figure in docs/09 section 9.5 wrong for both, and those numbers are the reason
# the next campaign gets scoped the way it is.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv-check/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"
TMP="${TEMP:-/tmp}"

# Every exp-1 seed pairs with exactly one exp-2 seed and no seed repeats, so ten blocks use all twenty study
# replicates and a pairing effect stays distinguishable from a block effect only in the sense that no pairing
# is ever crossed -- which is the caveat docs/10 item 12 keeps.
PAIRS=("5 6" "6 8" "7 9" "8 10")

while [ ! -f /tmp/no_dr_pool.flag ]; do sleep 180; done
echo "pool campaigns complete; starting the ten-block extension at $(date -Is)"

for pair in "${PAIRS[@]}"; do
  set -- $pair; S=$1; M=$2
  for H in 1 2 3 4 5; do
    dir="e1s${S}_e2s${M}_h${H}"
    if [ -f "$ROOT/results/cross_arena_blocks/$dir/matches.csv" ]; then
      echo "  $dir already played, skipping (delete it to re-run)"
      continue
    fi
    ( "$PY" "$ROOT/scripts/run_cross_arena.py" --exp1-replicate "$S" --exp2-replicate "$M" --seed "$H" \
        --out "results/cross_arena_blocks/$dir" > "$TMP/blk10_${S}_${M}_${H}.log" 2>&1 ) &
  done
  wait
  echo "  block exp1_seed${S}+exp2_seed${M} done at $(date -Is)"
done

missing=0
for pair in "${PAIRS[@]}"; do
  set -- $pair
  for H in 1 2 3 4 5; do
    [ -f "$ROOT/results/cross_arena_blocks/e1s$1_e2s$2_h${H}/matches.csv" ] || { echo "  MISSING e1s$1_e2s$2_h${H}"; missing=1; }
  done
done
if [ "$missing" = 1 ]; then
  echo "refusing to merge: at least one of the twenty new runs wrote no matches.csv" >&2
  exit 1
fi

merged=$("$PY" scripts/combine_cross_arena.py results/cross_arena_blocks/* \
  --out results/cross_arena/blocks10.csv)
echo "$merged"
case "$merged" in
  *"merged 10 training blocks x 5 harness seeds = 50 runs -> 90000 team-matches"*) ;;
  *) echo "the merge did not report ten blocks over five seeds; see the line above" >&2; exit 1 ;;
esac

"$PY" scripts/report_cross_arena.py --in results/cross_arena/blocks10.csv \
  --json-out results/cross_arena/blocks10.json > "$TMP/blk10_report.txt" 2>&1
echo TEN_BLOCKS_DONE > /tmp/blocks10.flag
