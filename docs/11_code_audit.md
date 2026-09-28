# 11. Code audit

Every finding below was produced by reading the current source **and** confirmed by executing code
against it. Each carries a reproduction command so it can be re-checked. Findings are numbered `A-nn`
for reference from other documents.

Severity: **HIGH** affects results or blocks work · **MED** affects correctness or reproducibility under
conditions that can occur · **LOW** dead code, misleading labels, or cosmetic · **FIXED** resolved during
the 2026-09-27 localisation commit.

| ID | Severity | Finding |
|---|---|---|
| [A-1](#turn-control-defect) | HIGH | `angle_to_target` reads the vertical axis; heading control is saturated noise |
| [A-2](#unbounded-transition-retention) | HIGH | `BaseTeamController.transitions` grows without bound; blocks the 3 M-step plan |
| [A-3](#versioning) | HIGH | Root-anchored `.gitignore` excludes experiment 1's results and includes experiment 2's |
| [A-4](#silent-warm-start) | MED | Training silently resumes from versioned weights while resetting the step counter and optimizer |
| [A-5](#test-suite-never-updates) | MED | The "and update" test never reaches a PPO update |
| [A-6](#coverage) | MED | 73 % / 14 % statement coverage; `ppo.py` is 26 % / 16 % |
| [A-7](#terminal-state-crash) | MED | `IndexError` when stepping in an already-decided state |
| [A-8](#unseeded-minibatch-shuffle) | MED | Minibatch order is unseeded; runs are not reproducible |
| [A-9](#mlflow-cadence) | MED | MLflow logging fires once per 100 k-step run; no learning curve exists |
| [A-10](#cumulative-tiebreak) | MED | Wins are awarded using run-long cumulative statistics |
| [A-11](#cross-tree-divergence) | MED | The two trees are diverged forks of one engine |
| [A-12](#configuration-claims) | MED | Documented training budget contradicts the committed `.env` |
| [A-13](#degenerate-transition-pair) | LOW | `state_features == next_state_features` always; both are discarded |
| [A-14](#approx-kl-placeholder) | LOW | `approx_kl` hard-coded to 0.0 |
| [A-15](#ppostats-discarded) | LOW | PPO statistics are computed, returned, and dropped |
| [A-16](#dead-configuration) | LOW | `RESPAWN_ENABLED` has no effect; `np_rng` is unused |
| [A-17](#observation-encoding) | LOW | Heading normalises to \([0,2)\); obstacles are unobservable; positions are unscaled |
| [A-18](#buffer-tensor-alignment) | LOW | `to_tensors` can silently misalign global observations |
| [A-19](#plot-filter) | LOW | `plotting.py` silently drops rows whose team name does not match a literal prefix |
| [A-20](#dead-classes) | LOW | `CTDEActorNetwork` / `CTDECriticNetwork` are empty aliases |
| [A-21](#annotation-drift) | LOW | `ActorNetwork.act` is annotated as a 3-tuple and returns 4 values |
| [A-22](#missing-dependency) | LOW | `pandas` is required by a script but declared nowhere |
| [A-23](#renderer-timestep) | LOW | The visual loop steps with real frame time, training steps with fixed `dt` |
| [A-24](#test-isolation) | FIXED | The test suite overwrote the versioned checkpoints |
| [A-25](#unsafe-checkpoint-load) | FIXED | `torch.load(weights_only=False)` on model files |

---

## Turn control defect

**A-1 · HIGH · not fixed — deliberately**

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

Effect: `turn = clip(delta/35, -1, 1)` saturates to \(\pm 1\) for any off-axis target, so agents spin
toward a fixed absolute heading instead of toward their waypoint. Shooting is unaffected because
`aim_direction` is computed independently of heading, which is why the agents remain competitive at all.

**Why it was not fixed here.** The versioned policies were trained under this control law. Patching it
invalidates every checkpoint and every number in [§ Results](08_results.md). It is a one-line change
whose only honest companion is a full re-run of both experiments. The fix belongs in a new experiment,
not in a documentation commit.

## Unbounded transition retention

**A-2 · HIGH · blocks the stated next step**

`BaseTeamController.update` calls `super().update(transitions)`, which does
`self.transitions.extend(transitions)` and is never cleared — not at `finish_episode`, not at
`reset_match`. Only `buffer` and `pending_steps` are cleared.

```
3,000 env steps -> 11,731 TransitionRecords retained, 13.59 MB attributed heap growth
                -> 4.53 KB per env step
extrapolated    -> ~453 MB at 100k steps (the versioned runs, which survived it)
                -> ~13.6 GB at 3M steps (the documented plan)
```

Each retained record also holds two NumPy arrays, so the per-record cost is dominated by the 8-element
float arrays plus Python object overhead. The list is read by nothing.

Fix: drop `super().update()` from `RLTeamController.update`, or clear `self.transitions` in
`finish_episode`. This is the cheapest change that unlocks the 3 M-step study.

## Versioning

**A-3 · HIGH · repository structure**

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

## Silent warm start

**A-4 · MED**

`RLTeamController.__init__` ends with `self._load_if_exists()`, which loads
`data/checkpoints/<slug>_<paradigm>.pt` whenever the file is present. So `python scripts/train_rl.py`
does **not** start from scratch on a machine that has the versioned checkpoints: it continues from the
100 k-step policies while resetting `total_env_steps` to 0 and creating a fresh Adam optimizer with
fresh moments.

There is no flag to disable this. A reproduction run and a fresh run are indistinguishable from the
command line, and the resulting `training_log.json` will claim 100,000 steps of training that did not
happen from random initialisation.

Fix: a `--from-scratch` flag, or refuse to load when `RL_TRAIN_TOTAL_STEPS` is being counted from zero.

## Test suite never updates

**A-5 · MED**

`tests/test_rl_training.py::test_rl_controllers_collect_rollouts_and_update` sets
`MATCH_DURATION_SECONDS=5` and loops for 30 env steps at `dt = 0.1`. Termination by duration needs 50
steps; termination by wipeout needs someone to die.

```
test loop: steps taken=30, match terminated within 30 steps? False
```

`finish_match()` is therefore never called, `finish_rl_episode` never runs, and no PPO gradient step is
ever taken. The test validates rollout *collection* and checkpoint *writing* only. Its name overstates
what it proves, and the coverage figures below are the consequence.

## Coverage

**A-6 · MED**

Measured with `coverage run --source=src`:

| Module | Experiment 1 | Experiment 2 |
|---|---:|---:|
| `rl/ppo.py` | **26 %** | **16 %** |
| `rl/buffer.py` | 49 % | 47 % |
| `systems/simulation.py` | 87 % | **0 %** |
| `systems/metrics.py` | **0 %** | **0 %** |
| `systems/plotting.py` | **0 %** | **0 %** |
| `controllers/rl_controller.py` | 73 % | **0 %** |
| `systems/match_variant.py` | 98 % | **0 %** |
| `ui/dashboard.py` | 0 % | 0 % |
| **total** | **73 %** | **14 %** |

Experiment 2's two tests exercise only the shape of two new network classes; the entire simulator,
controller and training path is untouched by them.

Not covered anywhere, in either tree:

* `RolloutBuffer.compute_returns` — the GAE recursion, i.e. the core learning signal;
* every `PPOTrainer.update_*` method — the policy gradient step;
* `_segment_intersects_aabb` and the four collision queries — no geometric ground truth is asserted;
* `MetricsStore` — CSV append, schema rotation, summary writing;
* `export_metric_dashboard` — including the filter in A-19;
* `ArenaSimulation.finish_match` — winner assignment and row construction;
* `main.py`, both `train_rl.py` scripts, both `plot_metrics.py` scripts.

## Terminal-state crash

**A-7 · MED · latent**

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

**A-8 · MED**

All three update paths call `np.random.shuffle(indices)` against NumPy's global RNG. Nothing in the
codebase calls `np.random.seed`, `torch.manual_seed`, or sets deterministic algorithms. `ArenaSimulation`
creates `self.np_rng = np.random.default_rng(seed)` and then never uses it (see A-16).

Consequence: identical seeds give different minibatch orders, different gradient trajectories, and
different final policies. This is the concrete reason
[§ Reproducibility](09_reproducibility.md#96-what-is-and-is-not-reproducible) states that re-runs are
only statistically similar.

## MLflow cadence

**A-9 · MED**

`ctde_arena/scripts/train_rl.py` logs metrics only inside the `RL_LOG_EVERY_STEPS` branch, which is
evaluated at a match boundary. With `RL_LOG_EVERY_STEPS=50000` and a 100,000-step budget the branch
fires once — at step 50,793 — and the `while` loop exits before a second boundary is reached.

```
run c762d435: 1 logged point, at env step 50,793
run 457ce1e8: 3 logged points (1,217 / 2,526 / 3,958), rl_log_every_steps was 1,000
```

The same single-fire pattern affects `training_log.json` in both experiments, which contains exactly one
entry each. The final policy's metrics never reached either sink.

## Cumulative tiebreak

**A-10 · MED**

```python
sorted(team_alive.items(), key=lambda kv: (kv[1], cumulative_metrics[kv[0]].eliminations), reverse=True)
```

The second key is a **run-long** total, so among teams with equal survivors the historically stronger one
is awarded the win. Over 60 random-initialisation matches, 10 ended by duration exhaustion — in those,
the tiebreak may have decided the winner. Its effect on the reported win rates is unmeasured.

## Cross-tree divergence

**A-11 · MED**

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

**A-12 · MED**

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

**A-13 · LOW**

`ArenaSimulation.step` builds each `TransitionRecord` with
`state_features=controller.build_local_features(agent.snapshot(), post_snapshots)` and
`next_state_features=` the identical expression. Measured over 1,593 transitions: **100.0 % identical**.

The fields are then discarded — `RLTeamController.update` reads only `reward` and `done`, and the
observation actually trained on was captured pre-step in `_record_step`. So the learning signal is
correct and this is wasted computation plus a memory cost, not a training corruption. It is still a trap
for anyone who later starts consuming `TransitionRecord`.

## Approx KL placeholder

**A-14 · LOW**

`ppo.py:67` returns `approx_kl=0.0` unconditionally. The field exists, is populated, and is meaningless.
Combined with A-15 there is no KL signal anywhere in the system, so the 4 PPO epochs run with no guard
against policy collapse.

## PPOStats discarded

**A-15 · LOW**

`finish_episode` returns `PPOStats`; `finish_rl_episode` collects them into a dict; and
`ArenaSimulation.finish_match` calls it as a statement, dropping the result. No caller in either tree
reads it. The prior README's "next steps" item — "PPO statistics are computed but not persisted" — was
accurate and is now referenced here as A-15.

## Dead configuration

**A-16 · LOW**

* `RESPAWN_ENABLED` → `ArenaConfig.respawn_enabled` is read and never consulted. `SimAgent.respawn_timer`
  is declared and never touched. The README correctly notes respawn is unimplemented; agents that die
  stay dead.
* `ArenaSimulation.np_rng` is constructed and never used.
* `JUMP_SPEED` / `GRAVITY` are live, but jump is unreachable from the action space
  ([§ 4.3](04_mdp_formalisation.md#43-action-space)), so vertical dynamics are effectively decorative.

## Observation encoding

**A-17 · LOW**

Three properties documented in [§ 4.1](04_mdp_formalisation.md#41-local-observation-o_i): heading divided
by 180 yields \([0,2)\) rather than \([-1,1]\) and is discontinuous at the wrap; positions are raw metres
so the input distribution shifts with arena size; and no component encodes obstacles, so cover is
invisible to the policy.

## Buffer tensor alignment

**A-18 · LOW · latent**

`RolloutBuffer.to_tensors` builds the global-observation tensor with a filter:

```python
np.stack([step.global_obs for step in self.steps if step.global_obs is not None])
```

If any row had `global_obs is None` while others did not, the resulting tensor would be shorter than
`local_obs` and the subsequent `tensors["global_obs"][batch_indices]` would misalign or raise. Today the
filter is vacuous — a controller either always records global observations or never does — so the bug
cannot fire. It becomes live the moment a mixed-observability team is introduced.

## Plot filter

**A-19 · LOW**

`plotting.py:31` drops every CSV row whose `team_name` does not start with the literal `"Team "`:

```python
if not team_name.startswith("Team "):
    continue
```

Before the localisation rename this literal was `"Equipe "`, so the dashboard silently produced an empty
chart for any team renamed. There is no warning when rows are discarded. Renaming teams again — or
loading data from an older clone — will silently yield a blank dashboard.

## Dead classes

**A-20 · LOW**

`CTDEActorNetwork(ActorNetwork): pass` and `CTDECriticNetwork(CentralizedCriticNetwork): pass` in both
trees. Never referenced.

## Annotation drift

**A-21 · LOW**

`ActorNetwork.act` is annotated `-> tuple[Tensor, Tensor, Tensor]` and returns four values. Every call
site unpacks four. The annotation is wrong, not the behaviour.

## Missing dependency

**A-22 · LOW**

`scripts/plot_metrics.py` imports `pandas`; neither `requirements.txt` declares it. The script catches
the `ImportError` and prints an install hint, so the failure is visible.

## Renderer timestep

**A-23 · LOW**

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

## Open work: suggested measurements

Derived directly from the gaps above, in decreasing value per unit of effort:

1. Fix A-1, then re-run both experiments — every existing number is conditioned on the defect.
2. Fix A-2, then run 5 seeds × 3 M steps (≈ 22 h CPU) and report Wilson intervals per seed.
3. Add a held-out greedy evaluation: 200 fixed-variant matches per pairing, which is the measurement the
   headline tables claim to be.
4. Log `PPOStats` (A-15) and a real `approx_kl` (A-14) so convergence and divergence are distinguishable.
5. Report shots-per-second-of-alive-life per agent, which is the only way to separate the two candidate
   explanations for CTDE-Comm's deficit ([§ 8.3](08_results.md#83-decomposing-the-elimination-gap)).
6. Rotate architecture-to-slot assignment across seeds (A-11 /
   [§ 10.1](10_threats_to_validity.md#slot-and-seed-confounding)).
