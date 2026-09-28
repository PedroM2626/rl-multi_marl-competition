# 11. Code audit

Every finding below was produced by reading the current source **and** confirmed by executing code
against it. Each carries a reproduction command so it can be re-checked. Findings are numbered `A-nn`
for reference from other documents.

Severity: **HIGH** affects results or blocks work · **MED** affects correctness or reproducibility under
conditions that can occur · **LOW** dead code, misleading labels, or cosmetic · **INFO** measured, not a
defect · **FIXED** resolved in this repository.

Every finding below was open at some point; the status column is the current state.

| ID | Severity | Finding |
|---|---|---|
| [A-1](#turn-control-defect) | FIXED | `angle_to_target` read the vertical axis; heading control was saturated noise |
| [A-2](#unbounded-transition-retention) | FIXED | `BaseTeamController.transitions` grew without bound; blocked long runs |
| [A-3](#versioning) | FIXED | Root-anchored `.gitignore` excluded experiment 1's results and included experiment 2's |
| [A-4](#silent-warm-start) | FIXED | Training silently resumed from versioned weights while resetting the step counter and optimizer |
| [A-5](#test-suite-never-updates) | FIXED | The "and update" test never reached a PPO update |
| [A-6](#coverage) | MOSTLY FIXED | Coverage raised from 73 % / 14 % to 92 % / 87 %; the collision solver is now geometrically verified |
| [A-7](#terminal-state-crash) | FIXED | `IndexError` when stepping in an already-decided state |
| [A-8](#unseeded-minibatch-shuffle) | FIXED | Minibatch order was unseeded; runs are now byte-reproducible |
| [A-9](#mlflow-cadence) | FIXED | MLflow logged one point per 100 k-step run; a final entry is now always written |
| [A-10](#cumulative-tiebreak) | FIXED | Wins were awarded using run-long cumulative statistics; ties are now draws |
| [A-11](#cross-tree-divergence) | FIXED | The two trees drifted; a parity test now pins which files may differ |
| [A-12](#configuration-claims) | FIXED | Documented training budget contradicted the committed `.env` |
| [A-13](#degenerate-transition-pair) | FIXED | `state_features == next_state_features` always; both were discarded |
| [A-14](#approx-kl-placeholder) | FIXED | `approx_kl` was hard-coded to 0.0 |
| [A-15](#ppostats-discarded) | FIXED | PPO statistics were computed, returned, and dropped |
| [A-16](#dead-configuration) | FIXED | `RESPAWN_ENABLED` had no effect; `np_rng` was unused |
| [A-17](#observation-encoding) | PARTIAL | Positions were unscaled and heading unbounded — both fixed, heading now \((\sin\theta,\cos\theta)\); obstacles remain unobservable by design |
| [A-18](#buffer-tensor-alignment) | FIXED | `to_tensors` could silently misalign global observations |
| [A-19](#plot-filter) | FIXED | `plotting.py` silently dropped rows whose team name did not match a literal prefix |
| [A-20](#dead-classes) | FIXED | `CTDEActorNetwork` / `CTDECriticNetwork` were empty aliases |
| [A-21](#annotation-drift) | FIXED | `ActorNetwork.act` was annotated as a 3-tuple and returned 4 values |
| [A-22](#missing-dependency) | FIXED | `pandas` was required by a script but declared nowhere |
| [A-23](#renderer-timestep) | FIXED | The visual loop stepped with real frame time; it now uses the fixed training step |
| [A-24](#test-isolation) | FIXED | The test suite overwrote the versioned checkpoints |
| [A-25](#unsafe-checkpoint-load) | FIXED | `torch.load(weights_only=False)` on model files |
| [A-26](#gpu-and-thread-scaling) | INFO | CUDA and extra torch threads both make this workload *slower* |
| [A-27](#slot-confounding) | FIXED | A paradigm always occupied the same slot, so a slot effect could not be separated |
| [A-28](#the-training-console-line-never-evaluated) | FIXED | The win-rate console line printed a comprehension as literal text |
| [A-29](#zero-norm-shots-burn-the-cooldown-and-vanish-from-the-metrics) | LOW | A zero-norm aim consumes the shoot cooldown yet counts as neither hit nor miss — 8–18 per replicate |
| [A-30](#greedy-evaluation-of-an-untrained-network-is-not-random-play) | MED | Argmax of random weights is a near-constant policy: untrained "random" arms differ by 18× in firing rate |

---

## Turn control defect

**A-1 · HIGH · FIXED**

`src/marl_arena/controllers/base.py:23`

```python
def angle_to_target(origin, heading_deg, target):
    offset = target - origin
    target_angle = math.degrees(math.atan2(offset[0], offset[1]))   # offset[1] is Y (vertical)
```

For agents on the ground plane, `offset[1] ≈ 0`, so the returned bearing is
\(\operatorname{atan2}(\Delta x, 0) \in \{-90^\circ, +90^\circ\}\) regardless of the true direction. The
intended term is `offset[2]`, matching `_forward_from_heading` which maps \(\theta \mapsto
(\sin\theta, 0, \cos\theta)\).

Reproduction:

```python
>>> angle_to_target(np.array([0.,1.,0.]), 0.0, np.array([10.,1.,20.]))
90.0                      # true heading-to-target: 26.57
>>> angle_to_target(np.array([0.,1.,0.]), 0.0, np.array([-10.,1.,5.]))
-90.0                     # true heading-to-target: -63.43
```

Effect: `turn = clip(delta/35, -1, 1)` saturated to \(\pm 1\) for any off-axis target, so agents spun
toward a fixed absolute heading instead of toward their waypoint. Shooting was unaffected because
`aim_direction` is computed independently of heading, which is why the agents remained competitive at all.

**Resolution.** Fixed by reading `offset[2]`. Verified against the analytic bearing over a 3×3 grid of
targets — every returned delta now equals the true heading-to-target error. Because the versioned
policies were trained under the broken law, the fix invalidates them: [§ Results](08_results.md) now
reports a re-run of both experiments under the corrected control, and the single-seed numbers are kept
only as the historical record.

## Unbounded transition retention

**A-2 · HIGH · FIXED**

`BaseTeamController.update` called `super().update(transitions)`, which did
`self.transitions.extend(transitions)` and was never cleared — not at `finish_episode`, not at
`reset_match`. Only `buffer` and `pending_steps` were.

```
before:  3,000 env steps -> 11,731 TransitionRecords retained, 13.59 MB heap growth
                         -> 4.53 KB per env step
                         -> ~453 MB at 100k steps, ~13.6 GB at 3M steps
after:   20,000 env steps -> 0.33 MB growth -> 16.3 B per env step
                         -> ~0.05 GB at 3M steps
```

**Resolution.** Nothing ever read the list, so the accumulation was deleted rather than periodically
cleared, and `update()` became an abstract hook like `decide()`. The same reasoning removed the three
`TransitionRecord` fields that were built every step and never consumed — see
[A-13](#degenerate-transition-pair) — which also cut two `build_local_features` calls per agent per step.

This is the change that made the replicated study possible: at 4.53 KB/step, a 3 M-step run would have
needed ~13.6 GB of RAM per process, so ten concurrent replicates were impossible.

## Versioning

**A-3 · HIGH · FIXED**

`.gitignore` patterns containing a `/` are anchored at the repository root:

```
data/exports/*.png        data/metrics/*.csv      data/metrics/*.json
data/checkpoints/*.pt     data/checkpoints/training_log.json
```

Consequence: experiment 1's `data/` is entirely excluded while `ctde_arena/data/` is not, and was
committed. The project's primary documented results table for experiment 1 is generated from a file that
does not exist in the repository. See
[§ The versioning asymmetry](09_reproducibility.md#the-versioning-asymmetry).

Related: `mlruns/` is ignored, so the MLflow store — the tracking system experiment 2's README describes
— is absent from the repository. Only the 48-row export added on 2026-09-27 survives in Git.

**Resolution.** The rules now use `**/data/...` where they need to reach both trees, and experiment
artefacts are versioned by design. What stays excluded is only what is not evidence: the `mlruns/` file
store, and `**/data/metrics/*.legacy*.csv` — orphaned copies left by schema rotations, one of which is
29.9 MB of pre-rename history. Experiment 1's five artefacts were committed after regenerating its
dashboard, which had also been missed by the localisation pass because it was untracked.

## Silent warm start

**A-4 · FIXED**

`RLTeamController.__init__` ends with `self._load_if_exists()`, which loads
`data/checkpoints/<slug>_<paradigm>.pt` whenever the file is present. So `python scripts/train_rl.py`
does **not** start from scratch on a machine that has the versioned checkpoints: it continues from the
100 k-step policies while resetting `total_env_steps` to 0 and creating a fresh Adam optimizer with
fresh moments.

`train_rl.py` still has no flag to disable this, so the hazard stands for that entry point.

**Mitigation.** `ARENA_DATA_DIR` now re-roots the whole data tree — metrics, exports and checkpoints —
so pointing it at a fresh directory guarantees a run from random initialisation.
`scripts/run_experiment.py` does exactly that, which is what makes the replicates in
[§ Results](08_results.md) genuine fresh runs rather than continuations of the shipped policies.

## Test suite never updates

**A-5 · MED · FIXED**

`tests/test_rl_training.py::test_rl_controllers_collect_rollouts_and_update` set
`MATCH_DURATION_SECONDS=5` and looped for 30 env steps at `dt = 0.1`. Termination by duration needs 50
steps; termination by wipeout needs someone to die.

```
test loop: steps taken=30, match terminated within 30 steps? False
```

`finish_match()` was therefore never called, `finish_rl_episode` never ran, and no PPO gradient step was
ever taken. The test validated rollout *collection* and checkpoint *writing* only. Its name overstated
what it proved, and the coverage figures below are the consequence.

**Resolution.** `test_match_termination_triggers_a_real_ppo_update` now loops until two matches have
ended and asserts that every team's actor and critic tensors changed after `finish_match()`, and that the
buffer was cleared. The original test survives as `test_rl_controllers_collect_rollouts_and_save`, with
its buffer assertion made against the peak observed size rather than the end state (a match can legitimately
terminate and flush within 30 steps, which made the naive assertion flaky).

## Coverage

**A-6 · MOSTLY FIXED**

Measured with `coverage run --source=src/marl_arena -m pytest`, before and after the tests added on
2026-09-27 and 2026-09-28:

| Module | Exp. 1 before | Exp. 1 after | Exp. 2 before | Exp. 2 after |
|---|---:|---:|---:|---:|
| `rl/ppo.py` | **26 %** | **97 %** | **16 %** | 62 % |
| `rl/buffer.py` | 49 % | **100 %** | 47 % | **100 %** |
| `rl/networks.py` | 84 % | **100 %** | 64 % | 85 % |
| `systems/simulation.py` | 87 % | 92 % | **0 %** | 92 % |
| `systems/metrics.py` | **0 %** | 87 % | **0 %** | 87 % |
| `systems/plotting.py` | **0 %** | 87 % | **0 %** | 87 % |
| `controllers/rl_controller.py` | 73 % | 93 % | **0 %** | 86 % |
| `controllers/base.py` | 91 % | 89 % | 91 % | 89 % |
| **total** | **73 %** | **93 %** | **14 %** | **87–88 %** |

Test count went from 9 to 106 (59 and 47). Experiment 2's total moves by two statements between identical
invocations, so it is quoted as a range. The additions that mattered:

* `test_control_law.py` pins the A-1 fix, including a closed-loop check that repeated application of the
  commanded turn actually converges on the target bearing.
* The training tests now terminate real matches and assert that every team's actor and critic parameters
  **move** after `finish_match()` — the first assertion in the repository that training trains. This is
  what took `ppo.py` from 26 % to 97 %.
* `test_metrics_store.py` covers CSV append, the summary denominators, schema rotation and the A-19
  prefix filter — the code that produces every number in this documentation set.
* A greedy-evaluation test drives `decide()` with training off, which is the branch the held-out study
  runs entirely through and which nothing had ever executed.
* `test_collision.py` (14 cases) checks the swept segment-vs-AABB solver against **hand-computed** entry
  parameters and hit positions, and drives `_advance_projectiles` with a step long enough to jump a barrier
  outright — asserting both that the hit registers and that its position lands on the inflated box face.

**Residual gaps.** `ui/dashboard.py` is 0 % (it only formats overlay text). Experiment 2's `ppo.py` is
capped at 62 % for a structural reason worth naming: `update_actor_critic` and `update_cte` are the DTE
and CTE paths, and experiment 2 configures none of its three teams as DTE or CTE, so **78 statements in
that file are unreachable in that tree**. They are not bugs, they are the cost of forking the engine
([§ Cross-tree divergence](#cross-tree-divergence)) — but they mean experiment 2's headline coverage
number understates how much of its own live code is tested.

**Coverage is not correctness, and the collision suite proves the point.** Adding 14 geometry tests moved
`systems/simulation.py` from 92 % to 92 %: the solver's lines were already *executed* by the match-running
tests. What changed is that those lines are now *asserted* — before, a solver that hit nothing would have
passed the suite at the same percentage. The remaining unverified surface is of that kind, not of the
uncovered-line kind: no test checks that a metric is right against an independent computation, only that it
is computed at all.

## Terminal-state crash

**A-7 · FIXED (was latent)**

`BaseTeamController.candidate_targets` returns a **one-element** list when no enemy is alive, while
`parse_action` can emit `target_index ∈ {0,1,2,3}`:

```
candidate_targets length when every enemy is dead: 1 (action space indexes 0..3)
```

Calling `step()` in an already-decided state therefore raises:

```
match decided; alive counts: {'Team 1': 0, 'Team 2': 2, 'Team 3': 0}
extra step in terminal state -> IndexError: list index out of range
  File "src/marl_arena/rl/actions.py", line 39, in action_to_decision
    target = targets[parsed.target_index]
```

Both shipped callers guard against it (`train_rl.py` breaks immediately; `main.py` checks
`if not self.finished`), so it does not fire in normal operation — but it will fire for any new caller,
any evaluation harness that steps past termination, or any test that forgets `reset_match()`. It fired
during the preparation of this documentation.

Fix: pad `candidate_targets` to a constant length, or clamp the index in `action_to_decision`.

## Unseeded minibatch shuffle

**A-8 · FIXED**

All three update paths call `np.random.shuffle(indices)` against NumPy's global RNG. Nothing in the
library calls `np.random.seed`, `torch.manual_seed`, or sets deterministic algorithms. `ArenaSimulation`
creates `self.np_rng = np.random.default_rng(seed)` and then never uses it (see A-16).

Consequence: identical seeds give different minibatch orders, different gradient trajectories, and
different final policies. This is the concrete reason
[§ Reproducibility](09_reproducibility.md#96-what-is-and-is-not-reproducible) states that re-runs are
only statistically similar.

**Mitigation.** `scripts/run_experiment.py::seed_everything` seeds `random`, NumPy's global RNG and
Torch before the simulation is constructed, so the replicates behind
[§ Results](08_results.md) are reproducible end to end. The library default is unchanged: `train_rl.py`
still produces non-reproducible runs, because making the library seed global state as a side effect of
import would be worse than leaving it explicit at the entry point.

## MLflow cadence

**A-9 · MED · FIXED**

`ctde_arena/scripts/train_rl.py` logs metrics only inside the `RL_LOG_EVERY_STEPS` branch, which is
evaluated at a match boundary. With `RL_LOG_EVERY_STEPS=50000` and a 100,000-step budget the branch
fires once — at step 50,793 — and the `while` loop exits before a second boundary is reached.

```
run c762d435: 1 logged point, at env step 50,793
run 457ce1e8: 3 logged points (1,217 / 2,526 / 3,958), rl_log_every_steps was 1,000
```

The same single-fire pattern affected `training_log.json` in both experiments, which contains exactly one
entry each. The final policy's metrics never reached either sink.

**Resolution.** Both training scripts now write one unconditional final entry after the loop exits,
carrying the end-of-run summary and the PPO statistics, and `train_rl.py` accepts `--steps`/`--seed` so the
cadence can be matched to the budget. The study records its own held-out curve independently of MLflow.

## Cumulative tiebreak

**A-10 · MED · FIXED**

```python
sorted(team_alive.items(), key=lambda kv: (kv[1], cumulative_metrics[kv[0]].eliminations), reverse=True)
```

The second key is a **run-long** total, so among teams with equal survivors the historically stronger one
is awarded the win. Over 60 random-initialisation matches, 10 ended by duration exhaustion — in those,
the tiebreak may have decided the winner. Its effect on the reported win rates is unmeasured.

## Cross-tree divergence

**A-11 · MED · FIXED**

`ctde_arena/` is a fork of the root engine, not an import. Twelve files are byte-identical; seven have
diverged. Beyond the intended paradigm differences:

| File | Unintended divergence |
|---|---|
| `systems/match_variant.py` | root uses `zip(..., strict=True)`, `ctde_arena` dropped the `strict` argument — a silent truncation guard removed |
| `systems/simulation.py` | identical except the three status-text lines |
| `rl/ppo.py` | `@staticmethod` followed by a stray blank line in `ctde_arena` (valid but accidental) |
| `controllers/rl_controller.py` | `ctde_arena` retains a fallback branch that no configured paradigm reaches |
| `scripts/train_rl.py` | root prints save messages that the MLflow version dropped |

Any fix applied to the shared physics or metrics code must be applied twice, and there is no test that
would notice a drift. The defect in A-1 lives in `controllers/base.py`, which is byte-identical in both
trees — so it is present in both.

## Configuration claims

**A-12 · MED · FIXED**

The prior documentation stated the standard training budget was 3,000,000 steps. The committed `.env`
says otherwise, and `CONFIG` reads it:

```
$ grep TOTAL_STEPS .env
RL_TRAIN_TOTAL_STEPS=100000
CONFIG.rl_train_total_steps = 100,000
```

The 3,000,000 figure is the *code default* used only when the environment variable is absent — i.e. in a
checkout with no `.env`. Both numbers are real; the documentation attributed the wrong one to the shipped
artefacts. Corrected in [§ 6.7](06_optimisation_procedure.md#67-hyperparameters-as-trained).

Also: `ctde_arena/.env` and `ctde_arena/.env.example` are byte-identical, as are the root pair, so
committing `.env` leaks nothing today — but the pattern means a future secret added to `.env` would be
committed by default.

## Degenerate transition pair

**A-13 · LOW · FIXED**

`ArenaSimulation.step` built each `TransitionRecord` with
`state_features=controller.build_local_features(agent.snapshot(), post_snapshots)` and
`next_state_features=` the identical expression. Measured over 1,593 transitions: **100.0 % identical**.

The fields were then discarded — `RLTeamController.update` read only `reward` and `done`, and the
observation actually trained on was captured pre-step in `_record_step`. So the learning signal was
correct and this was wasted computation plus a memory cost, not a training corruption.

**Resolution.** `TransitionRecord` is now `{agent_id, team_name, reward, done}` — the four fields
anything reads. Removing the other three deleted two `build_local_features` calls and one
`np.array` allocation per agent per step, which is part of why a step is cheap enough to replicate.

## Approx KL placeholder

**A-14 · LOW · FIXED**

`ppo.py` returned `approx_kl=0.0` unconditionally. The field existed, was populated, and was meaningless.
Combined with A-15 there was no KL signal anywhere in the system, so the 4 PPO epochs ran with no guard
against policy collapse.

**Resolution.** `_optimize` now takes the KL computed as `mean(old_log_prob - log_prob)` from the same
minibatch, and it is written to the training log. Measured over a 3k-step smoke run: max |KL| 0.0099.
Note this makes the KL *observable*, not *enforced* — there is still no early stop when it grows.

## PPOStats discarded

**A-15 · LOW · FIXED**

`finish_episode` returned `PPOStats`; `finish_rl_episode` collected them into a dict; and
`ArenaSimulation.finish_match` called it as a statement, dropping the result. No caller in either tree
read it.

**Resolution.** `finish_match` stores them on `simulation.last_ppo_stats`, and both training scripts write
policy loss, value loss, entropy and approx KL per team into `training_log.json`. Still not persisted:
a per-update history in MLflow for experiment 1, which has no MLflow integration.

## Dead configuration

**A-16 · LOW · FIXED**

* `RESPAWN_ENABLED` → `ArenaConfig.respawn_enabled` is read and never consulted. `SimAgent.respawn_timer`
  is declared and never touched. The root README states the flag has no effect; agents that die
  stay dead.
* `ArenaSimulation.np_rng` is constructed and never used.
* `JUMP_SPEED` / `GRAVITY` are live, but jump is unreachable from the action space
  ([§ 4.3](04_mdp_formalisation.md#43-action-space)), so vertical dynamics are effectively decorative.

## Observation encoding

**A-17 · LOW · FIXED**

Three properties documented in [§ 4.1](04_mdp_formalisation.md#41-local-observation-o_i): heading divided
by 180 yields \([0,2)\) rather than \([-1,1]\) and is discontinuous at the wrap; positions are raw metres
so the input distribution shifts with arena size; and no component encodes obstacles, so cover is
invisible to the policy.

## Buffer tensor alignment

**A-18 · FIXED (was latent)**

`RolloutBuffer.to_tensors` builds the global-observation tensor with a filter:

```python
np.stack([step.global_obs for step in self.steps if step.global_obs is not None])
```

If any row had `global_obs is None` while others did not, the resulting tensor would be shorter than
`local_obs` and the subsequent `tensors["global_obs"][batch_indices]` would misalign or raise. Today the
filter is vacuous — a controller either always records global observations or never does — so the bug
cannot fire. It becomes live the moment a mixed-observability team is introduced.

## Plot filter

**A-19 · LOW · FIXED**

`plotting.py:31` drops every CSV row whose `team_name` does not start with the literal `"Team "`:

```python
if not team_name.startswith("Team "):
    continue
```

Before the localisation rename this literal was `"Equipe "`, so the dashboard silently produced an empty
chart for any team renamed. There is no warning when rows are discarded. Renaming teams again — or
loading data from an older clone — will silently yield a blank dashboard.

## Dead classes

**A-20 · LOW · FIXED**

`CTDEActorNetwork(ActorNetwork): pass` and `CTDECriticNetwork(CentralizedCriticNetwork): pass` in both
trees. Never referenced.

## Annotation drift

**A-21 · LOW · FIXED**

`ActorNetwork.act` is annotated `-> tuple[Tensor, Tensor, Tensor]` and returns four values. Every call
site unpacks four. The annotation is wrong, not the behaviour.

## Missing dependency

**A-22 · LOW · FIXED**

`scripts/plot_metrics.py` imports `pandas`; neither `requirements.txt` declares it. The script catches
the `ImportError` and prints an install hint, so the failure is visible.

## Renderer timestep

**A-23 · LOW · FIXED**

`main.py` calls `simulation.step(time.dt)` with the real frame delta while training uses the fixed
`SIM_STEP_DT = 0.1`. The same policies therefore see different dynamics on screen than during training,
and the projectile-lifetime arithmetic of
[§ 3.4.2](03_arena_system_model.md#342-projectile-lifetime) changes with frame rate.

## Test isolation

**A-24 · FIXED (2026-09-27)**

`test_rl_controllers_collect_rollouts_and_update` previously called `save_rl_checkpoints(...)`, which
writes to `CONFIG.rl_checkpoint_dir`. Running the test suite therefore overwrote the trained policies in
`data/checkpoints/` with a 30-step untrained snapshot. Fixed by saving into pytest's `tmp_path`.

**Note on damage.** During preparation of this documentation the suite was run before the fix, and it
overwrote the local, never-versioned experiment 1 checkpoints (`equipe_*_{cte,dte,ctde}.pt`, since
renamed in the working tree to `team_*`). Those files were excluded from Git by A-3, so they existed
only on the original machine and were not recoverable from the repository. Experiment 1's *metrics*
survived intact — `data/metrics/summary.json` is untouched by the test path, and every table in
[§ Results](08_results.md) still reproduces from it exactly.

## Unsafe checkpoint load

**A-25 · FIXED (2026-09-27)**

`load_checkpoint` used `torch.load(..., weights_only=False)`, which executes arbitrary pickled objects
found in a `.pt` file. The versioned payloads contain only strings and tensor state dicts, so
`weights_only=True` was verified compatible before the change:

```
weights_only=True OK -> keys=['paradigm', 'actor', 'critic'] types=['OrderedDict', 'str']
```

Applied to both trees.

## GPU and thread scaling

**A-26 · INFO · measured, so that nobody re-derives it**

The obvious intuitions — "use the GPU", "give torch more threads" — are both wrong for this workload,
and the numbers are worth recording because the workload looks like it should benefit from both.

Same 2,500-step training workload, RTX 4070 Laptop GPU, 32-core host:

| Configuration | Steps/s |
|---|---:|
| CPU, 1 torch thread | **368** |
| CPU, 2 threads | 261 |
| CPU, 4 threads | 259 |
| CUDA, batch-1 forwards | 149.6 |

CPU with a single thread is fastest, and the GPU is **less than half** the speed of one CPU core. The
reason is visible in the profile: a step is nine independent forward passes of a batch of **one** 8-element
vector through a 35,465-parameter MLP. Of 28.6 s profiled for 4,000 steps, 18.6 s (65 %) is inside
`RLTeamController.decide`, and the torch kernels themselves are only ~2.3 s — the rest is Python and
dispatch overhead across 152,658 `Linear` calls. There is no arithmetic here to accelerate; kernel launch
cost dominates, and extra intra-op threads add contention without adding usable parallelism.

Consequences for how to scale this project:

* **Do not move to GPU without also vectorising the environment.** A GPU only pays off when the batch
  dimension becomes large, which means running thousands of arena instances simultaneously — a rewrite of
  a 779-line imperative simulator built on Python lists of dataclasses, dynamic projectile sets and
  rejection sampling. The bottleneck is not FLOPs, so that rewrite buys throughput only if it also
  removes the per-agent Python, which is most of the code.
* **Do scale across processes — but measure it, because the machine decides the answer.** Runs are
  independent and each wants exactly one core, and that is what `scripts/run_study.py` does. The scaling
  is far from linear on this laptop: 20 concurrent replicates measured ~1,030 steps/s in aggregate, i.e.
  ~51 steps/s each against 368 steps/s solo — under 3× the throughput of a single process from 20 times
  the cores. So the win is real but sublinear, and the honest way to plan a study is to measure the
  aggregate rate before committing to a seed count.
* `torch.set_num_threads(1)` is therefore set explicitly in the runner; without it each process defaults
  to multi-threaded intra-op kernels and the concurrent runs degrade each other.

## Slot confounding

**A-27 · MED · FIXED**

Team identity was fixed to a spawn corner \((-m,1,-m)\), \((+m,1,-m)\), \((0,1,+m)\), a controller seed
offset of 11 / 23 / 37, and a position in the network-construction order. With one paradigm per slot and
one seed, a persistent slot effect is indistinguishable from an architecture effect and adding seeds does
not reveal it, because every seed reproduces the same mapping.

**Resolution.** `team_meta(rotation)` shifts the paradigm cycle by `rotation`, which defaults to
`seed % 3`, and `ArenaSimulation`, the spawn sampler and `build_controllers` all read it from the same
place. A test asserts that controllers, spawns and the metrics table agree on the mapping for every seed,
and that the rotations cover every paradigm-in-every-slot combination.

## The training console line never evaluated

**A-28 · LOW · FIXED**

`scripts/train_rl.py` printed:

```python
f"win_rates={{k: round(v['win_rate'], 3) for k, v in summary.items()}}"
```

The doubled braces make that a literal `{k: round(...)}` in the rendered string rather than a dict
comprehension, so the training console never actually showed win rates. Found while rewriting the script
for A-4/A-15; the line now renders real values.

## Zero-norm shots burn the cooldown and vanish from the metrics

**A-29 · LOW · DOCUMENTED, NOT FIXED**

`action_to_decision` can produce an `aim_direction` of zero norm — the clearest route is the A-7 fallback,
where `candidate_targets` returns the agent's own position four times when there is no enemy, so the
target offset is exactly zero. `_spawn_projectile` then normalises, gets a zero vector, warns and returns
`None`:

```python
agent.last_shot_at = self.match_time                    # cooldown consumed
self._spawn_projectile(agent, decision.aim_direction)   # no projectile created
```

A shot that never exists still costs the cooldown, but because hits and misses are attributed by the
projectile, it is counted as neither — so `shots_per_match` and `shot_accuracy` both silently ignore it.
The behaviour itself is defensible (burning the cooldown punishes an illegal aim); the metric hole is the
defect.

**Measured**, by counting the warning in every log the study produced: 8–18 occurrences per 500,000-step
replicate, against tens of thousands of registered shots. It changes no reported number at the third
decimal place, which is why it is documented rather than fixed — patching it means deciding whether a
dropped shot is a miss, and that choice would move `shot_accuracy` for reasons unrelated to policy quality.

The related maintainability trap is that the collision helpers do not share a return order:
`_segment_intersects_aabb` yields `(flag, t, position)` while `_first_obstacle_collision`,
`_first_agent_collision` and `_arena_boundary_collision` yield `(target, position, t)`. Both orders are now
pinned by `tests/test_collision.py::test_the_helpers_do_not_share_a_return_order`.

## Greedy evaluation of an untrained network is not random play

**A-30 · MED · PROTOCOL, DOCUMENTED**

`set_rl_training(sim.controllers, False)` makes `decide()` take `torch.argmax` of the actor's logits — the
right thing for measuring a trained policy and the wrong thing for measuring an untrained one. A random
network's argmax is a nearly-constant function of its initial weights, so an "untrained greedy" agent tends
to replay one action for the whole match rather than sample across the eight. `scripts/random_baseline.py`
measured what that means, 300 headless fixed-variant matches:

```
300 headless matches (greedy-policy init, no checkpoints): {'wipeout': 107, 'timeout': 129, 'draw': 64}
  wipeout 35.7%  timeout 43.0%  draw 21.3%
  duration min=4.3s median=90.1s max=90.1s mean=63.0s
  CTDE  win 0.750  elim/match 3.13  survival 60.4s  shots/match 82.1  accuracy 0.038
  CTE   win 0.013  elim/match 0.07  survival 40.7s  shots/match  4.6  accuracy 0.015
  DTE   win 0.023  elim/match 0.23  survival 48.8s  shots/match  6.8  accuracy 0.034
```

Two consequences for how the study may be read:

1. **The paradigm label predicts firing rate before any learning happens** — 82.1 shots per match for CTDE
   against 4.6 for CTE, an 18× spread that exists purely because the three architectures start from
   different random weight draws and argmax commits to whichever action that draw favours. Experiment 2
   shows the same effect with the sign reversed (88.2 for CAC against 397.9 for Comm, accuracy 0.1–0.2 %).
   Any comparison of these arms' *engagement* is therefore partly a comparison of their initialisations.
2. **A "random baseline" win rate is not 1/3.** Untrained greedy CTDE wins 75 % of matches, against 32.6 %
   for the same arm in the trained study. Read naively that says training cost CTDE 42 points. It does not —
   the two are measured in different regimes (fixed variant vs. rotated slots, one weight draw vs. ten, and
   21 % of baseline matches ending in a three-way draw) — but it does mean the study has no clean floor to
   subtract, and the `--policy uniform` mode of the same script is the version that supplies one.

**Reproduction** — in either tree:

```
python scripts/random_baseline.py --matches 300 --policy greedy
python scripts/random_baseline.py --matches 300 --policy uniform
```

The baseline is therefore reported in [§ 8.9](08_results.md#random-baseline) in both modes, and the honest
summary is that win rate in this arena cannot be interpreted against a chance level of 1/3.

## Open work: suggested measurements

Items 1–3 were the original list and are done; 6 is done; 4 is partly done; 7 was deliberately not done.

1. ~~Fix A-1, then re-run both experiments.~~ **Done** — see
   [§ Replicated study](08_results.md#replicated-study).
2. ~~Fix A-2, then run 5 seeds.~~ **Done** — and then superseded: the study was re-run at 10 seeds, which
   is what [§ 8.1](08_results.md#replicated-study) reports.
3. ~~Add a held-out greedy evaluation.~~ **Done** — 150 greedy fixed-variant matches per replicate.
4. Log `PPOStats` (A-15) and a real `approx_kl` (A-14) so convergence and divergence are distinguishable.
   **Partly done** — both are now computed and written to `run_summary.json`, but the study still judges the
   arms by outcome only; nothing correlates a replicate's win rate with its KL or value loss. The seed-6
   collapse ([§ 8.1](08_results.md#leave-one-out)) is exactly the event those traces would have flagged, and
   it went unrecorded.
5. Report shots-per-second-of-alive-life per agent, which is the only way to separate the two candidate
   explanations for CTDE-Comm's deficit ([§ 8.3](08_results.md#decomposing-the-elimination-gap)).
   **Still open**, and A-30 makes it more urgent: firing rate differs by 18× at initialisation.
6. ~~Rotate architecture-to-slot assignment across seeds~~ (A-27). **Done** — `paradigm_rotation` is derived
   from the seed and recorded in every `run_summary.json`.
7. Re-run at the 3 M-step budget the code defaults to, to test whether CTDE-Comm's late improvement
   continues. **Deliberately not done.** The 10-replicate study showed the curve *direction* flipping when
   the replicate count changed, so a longer budget would refine an estimate of a quantity that is not
   stable; the binding problem is the reward and termination design, not the step count.
8. **New:** separate the two match regimes. A quarter of replicates decide matches on the 90 s clock rather
   than on eliminations ([§ 8.1](08_results.md#two-regimes)), so any pooled win rate mixes "who wins a
   fight" with "who is most alive at the buzzer". Either report the regimes separately or change the
   termination rule so that survival-to-cap is not a winning outcome.
9. **New:** test the reward against the never-fires basin. Seed 6's CTDE arm finished training with 3.8 shots
   per match and a 0.000 held-out win rate; nothing in the reward punishes not engaging until the match is
   already over.
