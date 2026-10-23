#!/usr/bin/env bash
# Ablations the documentation listed as "not done", chained onto the arena-v2 campaign.
#
# Each arm differs from the committed study by exactly one configuration key, uses the same seeds 1-3 so
# it is comparable against the same three seeds of results/study, and writes to its own results directory
# so nothing committed is touched.
#
# As launched it covers BOTH experiments -- run_study.py defaults to --experiments exp1,exp2, so each arm is
# 6 runs and the campaign is 18, roughly 4 hours of wall clock at three concurrent. An earlier version of
# this comment claimed experiment 1 only, which is not what ran; the results in
# [docs/08 section 8.13](../docs/08_results.md) are the wider ones.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/_runtime.sh"

while [ ! -f /tmp/study_v2.flag ]; do sleep 120; done
echo "arena v2 study complete; starting ablations at $(date -Is)"

run_arm() {
  local name="$1"; shift
  echo "=== $name ==="
  env "$@" "$PY" scripts/run_study.py \
    --steps 500000 --seeds 1,2,3 --jobs 3 --eval-matches 150 \
    --out "data/runs_ablation_$name" --results-dir "results/ablation_$name" \
    > "/tmp/ablation_$name.log" 2>&1
  echo "$name exit $? at $(date -Is)" >> "/tmp/ablation_$name.log"
}

# Baseline for all three is results/study (domain randomisation on, hidden dim 128).
run_arm no_dr      DOMAIN_RANDOMIZATION=false
run_arm narrow_64  RL_HIDDEN_DIM=64
run_arm wide_256   RL_HIDDEN_DIM=256

echo ABLATIONS_DONE > /tmp/ablations.flag
