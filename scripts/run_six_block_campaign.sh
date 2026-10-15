#!/usr/bin/env bash
# Extend the cross-experiment round-robin from three training blocks to six, chained behind the ablations.
#
# Chapter 8's own caveat says n = 3 blocks is enough to reverse a ranking and not enough to settle one, and
# the checkpoints for the other blocks already exist under the git-ignored data/runs/. This is evaluation
# only -- no training -- so it is the cheapest way to narrow the between-block interval that every
# architecture claim in this repository currently rests on.
#
# It waits for the ablation campaign so the measured throughput in the documentation stays true for all
# three campaigns; running them together would slow each and make every cost figure wrong at once.
set -u
cd "$(dirname "$0")/.."

while [ ! -f /tmp/ablations.flag ]; do sleep 180; done
echo "ablations complete; starting the six-block round-robin at $(date -Is)"

PY=.venv-check/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/Scripts/python.exe

# The three blocks already versioned are (1,7) (10,5) (9,2). These three add a different experiment-1 seed
# and a different experiment-2 seed each, so no exp1 x exp2 pairing repeats and a pairing effect cannot
# stand in for an architecture effect across the six.
for pair in "2 1" "3 3" "4 4"; do
  set -- $pair; S=$1; M=$2
  for H in 1 2 3 4 5; do
    ( cd ctde_arena \
      && "$PY" ../../scripts/run_cross_arena.py --exp1-replicate "$S" --exp2-replicate "$M" --seed "$H" \
           --out "results/cross_arena_blocks/e1s${S}_e2s${M}_h${H}" \
      > "/tmp/blk_${S}_${M}_${H}.log" 2>&1 ) &
  done
  wait
  echo "block exp1_seed${S}+exp2_seed${M} done at $(date -Is)"
done

# Rebuild the pooled table over all six blocks. The three original run directories are still on disk, and
# combine refuses to pool runs whose columns differ, so the merged file is the six-block superset under a
# new name rather than an overwrite of the versioned three-block one.
"$PY" scripts/combine_cross_arena.py results/cross_arena_blocks/* --out results/cross_arena/blocks6.csv
"$PY" scripts/report_cross_arena.py --in results/cross_arena/blocks6.csv \
  --json-out results/cross_arena/blocks6.json

echo SIX_BLOCKS_DONE > /tmp/blocks6.flag
