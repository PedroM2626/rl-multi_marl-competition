#!/usr/bin/env bash
# Extend the cross-experiment round-robin from three training blocks to six.
#
# Chapter 8's own caveat says n = 3 blocks is enough to reverse a ranking and not enough to settle one, and
# the checkpoints for the other blocks exist on this machine under data/runs/ (git-ignored). This is
# evaluation only -- no training -- so it is the cheapest way to narrow the between-block interval that every
# architecture claim in this repository rests on.
#
# A first attempt at this script produced nothing and did not say so: it ran `cd ctde_arena` inside a
# subshell and then invoked a repository-root-relative interpreter path, which no longer resolved, so all 15
# runs failed in under a second and the merge silently re-emitted the 15 old run directories as "blocks6".
# Paths are absolute now, every run directory is checked for a matches.csv before anything is merged, and a
# missing run aborts the campaign instead of being dropped from it.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

while [ ! -f /tmp/ablations.flag ]; do sleep 180; done
echo "ablations complete; starting the six-block round-robin at $(date -Is)"

PY="$ROOT/.venv-check/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"

# The three blocks already versioned are (1,7) (10,5) (9,2). These three add a different experiment-1 seed
# and a different experiment-2 seed each, so no exp1 x exp2 pairing repeats and a pairing effect cannot
# stand in for an architecture effect across the six.
PAIRS=("2 1" "3 3" "4 4")
for entry in "${PAIRS[@]}"; do
  S="${entry%% *}"; M="${entry##* }"
  for H in 1 2 3 4 5; do
    ( cd "$ROOT/ctde_arena" \
      && "$PY" "$ROOT/scripts/run_cross_arena.py" --exp1-replicate "$S" --exp2-replicate "$M" --seed "$H" \
           --out "results/cross_arena_blocks/e1s${S}_e2s${M}_h${H}" \
      > "/tmp/blk_${S}_${M}_${H}.log" 2>&1 ) &
  done
  wait
  echo "block exp1_seed${S}+exp2_seed${M} submitted at $(date -Is)"
done

missing=0
for entry in "${PAIRS[@]}"; do
  S="${entry%% *}"; M="${entry##* }"
  for H in 1 2 3 4 5; do
    dir="$ROOT/results/cross_arena_blocks/e1s${S}_e2s${M}_h${H}/matches.csv"
    if [ ! -f "$dir" ]; then
      echo "[FAIL] no matches.csv for exp1_seed${S}+exp2_seed${M} harness ${H}; last lines of its log:"
      tail -3 "/tmp/blk_${S}_${M}_${H}.log"
      missing=$((missing + 1))
    fi
  done
done
if [ "$missing" -gt 0 ]; then
  echo "aborting: $missing of 15 new runs produced nothing -- merging would silently re-emit the old blocks"
  exit 1
fi

# Rebuild the pooled table over all six blocks. combine refuses to pool runs whose columns differ, and it
# prints the block count, which is what the check below reads.
merged=$("$PY" "$ROOT/scripts/combine_cross_arena.py" "$ROOT"/results/cross_arena_blocks/* \
         --out results/cross_arena/blocks6.csv)
echo "$merged"
case "$merged" in
  *"merged 6 training blocks"*) ;;
  *) echo "aborting: expected 6 training blocks in the merge, got: $merged"; exit 1;;
esac

"$PY" "$ROOT/scripts/report_cross_arena.py" --in results/cross_arena/blocks6.csv \
  --json-out results/cross_arena/blocks6.json

echo SIX_BLOCKS_DONE > /tmp/blocks6.flag
