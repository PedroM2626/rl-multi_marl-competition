#!/usr/bin/env bash
# Separate what the two single-key remedies do to training from what they do to the readout.
#
# SPAWN_CYCLE_PER_MATCH and ARENA_LAYOUT act in the held-out evaluation as well as in training, so an arm's
# smaller between-seed spread (docs/08 § 8.14) could be the instrument rather than the policies. Nothing is
# trained here: the same checkpoints are re-read at T = 0 (150 held-out matches, the study's protocol) with the
# key flipped in the evaluation only.
#
#   spawn_cycle_on    spawn_cycle checkpoints, cycling on      -- control: must reproduce the published arm
#   spawn_cycle_off   spawn_cycle checkpoints, cycling off     -- what training under cycling left
#   jax_on            JAX-replication checkpoints, cycling on  -- what the cycled readout does alone
#   symmetric_on      symmetric checkpoints, symmetric arena   -- control
#   symmetric_off     symmetric checkpoints, committed arena   -- what training in the symmetric arena left
#   jax_symmetric     JAX-replication checkpoints, symmetric arena -- what the symmetric readout does alone
#   v3_off            v3 checkpoints under the v3 profile with cycling off
#
# Usage:  bash scripts/run_remedy_readout.sh            (from the repository root)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_runtime.sh"
TMP="${TEMP:-/tmp}"
OUT="$ROOT/results/readout_remedies"
V3_PLAY=(RL_PARAM_TARGET=-1 COMM_ROUNDS=2 COMM_MASK_DEAD=true VD_LOCAL_CRITIC=true OBS_OBSTACLE_RAYS=8
         ARENA_LAYOUT=symmetric RL_OPPONENT_POOL_PROB=0.2)
ARMS=()

read_arm() {  # <name> <runs dir> <env...>
  local name=$1 runs=$2; shift 2
  ARMS+=("$name")
  [ -s "$OUT/$name/per_seed_metrics.csv" ] && { echo "$name: already read"; return 0; }
  env "$@" "$PY" scripts/eval_readout.py --temperature 0 --runs-dir "$runs" --experiments exp1 \
    --require-checkpoints --out "$OUT/$name/exp1" > "$TMP/remedy_readout_${name}_exp1.log" 2>&1 &
  ( cd ctde_arena && env "$@" "$PY" scripts/eval_readout.py --temperature 0 --runs-dir "$runs" --experiments exp2 \
    --require-checkpoints --out "$OUT/$name/exp2" ) > "$TMP/remedy_readout_${name}_exp2.log" 2>&1 &
}
read_arm spawn_cycle_on data/runs_jax_spawn_cycle SPAWN_CYCLE_PER_MATCH=true
read_arm spawn_cycle_off data/runs_jax_spawn_cycle SPAWN_CYCLE_PER_MATCH=false
read_arm jax_on data/runs_jax SPAWN_CYCLE_PER_MATCH=true
read_arm v3_off data/runs_v3 "${V3_PLAY[@]}" SPAWN_CYCLE_PER_MATCH=false
read_arm symmetric_on data/runs_jax_symmetric ARENA_LAYOUT=symmetric
read_arm symmetric_off data/runs_jax_symmetric ARENA_LAYOUT=legacy
read_arm jax_symmetric data/runs_jax ARENA_LAYOUT=symmetric
wait

# One per_seed_metrics.csv per arm, both experiments, in the schema report_study.py reads.
for name in "${ARMS[@]}"; do
  [ -s "$OUT/$name/per_seed_metrics.csv" ] && continue
  for e in exp1 exp2; do
    [ -s "$OUT/$name/$e/per_seed_metrics.csv" ] || { echo "$name/$e did not finish; see $TMP" >&2; exit 1; }
  done
  { cat "$OUT/$name/exp1/per_seed_metrics.csv"; tail -n +2 "$OUT/$name/exp2/per_seed_metrics.csv"; } \
    > "$OUT/$name/per_seed_metrics.csv"
done
echo "remedy readout complete at $(date -Is)"
