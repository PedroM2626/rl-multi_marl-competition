#!/usr/bin/env bash
# Chain the arena-v2 study onto the end of the training-block round-robin.
#
# Both campaigns want most of the machine's 32 cores, and the v2 study is 20 concurrent
# half-million-step replicates. Running them together would slow both and make the measured
# throughput in the documentation wrong for both, so this waits for the round-robin to finish
# and then launches v2 with its own results directory.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/_runtime.sh"

PENDING=15
while :; do
  done_count=0
  for log in /tmp/blk_*.log; do
    [ -f "$log" ] || continue
    if grep -q "^wrote " "$log" 2>/dev/null; then done_count=$((done_count + 1)); fi
  done
  if [ "$done_count" -ge "$PENDING" ]; then break; fi
  sleep 60
done

echo "round-robin blocks complete at $(date -Is); starting arena v2 study"

# Arena v2: the alive bonus decays with a 30 s half-life and starts an order of magnitude
# smaller, and a match that runs to the clock is a draw. Everything else is unchanged, so the
# only difference between this and results/study is the two things chapter 8 named as remedies.
export REWARD_ALIVE_BONUS=0.0015
export REWARD_ALIVE_DECAY_HALF_LIFE_S=30
export TERMINATION_CAP_AWARDS_WIN=false

"$PY" scripts/run_study.py \
  --steps 500000 \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --jobs 20 \
  --eval-matches 150 \
  --out data/runs_v2 \
  --results-dir results/study_v2 \
  > /tmp/study_v2.log 2>&1
echo "arena v2 study exit $? at $(date -Is)" >> /tmp/study_v2.log
echo V2_DONE > /tmp/study_v2.flag
