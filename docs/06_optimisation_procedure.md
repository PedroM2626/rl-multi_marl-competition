# 6. Optimisation procedure

Source: `src/marl_arena/rl/ppo.py`, `src/marl_arena/rl/buffer.py`,
`src/marl_arena/controllers/rl_controller.py::finish_episode`.

## 6.1 Objective

Each team owns one `PPOTrainer` and one Adam optimizer over the concatenation of its actor and critic
parameters. Per minibatch:

\[
\mathcal{L} = \mathcal{L}^{\text{policy}}_{\text{clip}}
+ c_v\,\mathcal{L}^{\text{value}}
- c_H\,\mathbb{E}[\mathcal{H}[\pi]]
\qquad c_v = 0.5,\ c_H = 0.01
\]

with the standard two-sided PPO surrogate
([Schulman et al., 2017](12_glossary_and_references.md#references)):

\[
\mathcal{L}^{\text{policy}}_{\text{clip}}
= -\,\mathbb{E}\!\left[\min\!\bigl(r_t \hat{A}_t,\ \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\bigr)\right],
\qquad r_t = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t)},\ \epsilon = 0.2 .
\]

\(\mathcal{L}^{\text{value}}\) is the plain MSE between the critic output and the GAE return. Gradients
are clipped to a global norm of 0.5 after the backward pass.

## 6.2 Advantage estimation

`RolloutBuffer.compute_returns(gamma, gae_lambda, last_value)` implements GAE
([Schulman et al., 2016](12_glossary_and_references.md#references)) over the flat buffer:

```python
values = [step.value for step in steps] + [last_value]
for t in reversed(range(len(steps))):
    mask  = 1 - done[t]
    delta = reward[t] + gamma * values[t+1] * mask - values[t]
    A[t]  = delta + gamma * gae_lambda * mask * A[t+1]
returns = A + values[:-1]
```

then normalises: \(\hat A \leftarrow (\hat A - \mu)/(\sigma + 10^{-8})\).

\(\gamma = 0.99\), \(\lambda = 0.95\).

### The buffer is not a trajectory

This is the central methodological caveat of the implementation, and it is worth being explicit about.
`RolloutBuffer` accumulates one row per **(decision step, agent)** pair of a single team, in the order
the agents were processed. GAE is then applied along that flat list, so the term
\(\text{values}[t+1]\) used to compute agent \(i\)'s advantage at step \(t\) is the value estimate of
**whichever row happens to follow it** — typically a *different agent* at the same step, or the same
agent one step later.

Measured on one 859-step match of Experiment 1:

| Team | Paradigm | Buffer rows |
|---|---|---:|
| Team 1 | CTE | 1,157 |
| Team 2 | DTE | 122 |
| Team 3 | CTDE | 952 |

Rows are ordered `step-major` (all alive agents of a step, then the next step). With three agents,
roughly two thirds of the \(\delta_t \to A_{t+1}\) links in the recursion cross an agent boundary
rather than a time boundary. The result is not a valid single-agent GAE estimate; it behaves more like
a batch advantage estimator with an arbitrary sequential chaining, and the chaining order is a function
of who was alive.

Two further consequences:

* **Buffer size is proportional to survival.** Team 2's 122 rows against Team 1's 1,157 is not a
  difference in how much it acted — it is a difference in how long its agents lived. The arm that dies
  early receives proportionally fewer gradient updates per match. Since survival is close to the
  outcome being measured, **training signal volume is confounded with performance**, and the
  confound runs in the direction of rewarding whoever is already winning.
* **A single-row buffer is benign.** \(\sigma\) of a one-element advantage array is 0.0, but the
  numerator is also 0, so normalisation yields `[0.]` rather than an overflow. Verified:
  `np.std([0.7]) == 0.0` and `(a - a.mean())/(a.std()+1e-8) == [0.]`.

## 6.3 Update schedule

There is no fixed rollout length. The sequence is:

```
match begins → every step appends rows to each team's buffer
             → step() returns True (terminal)
             → finish_match() → finish_rl_episode(controllers)
             → per controller: bootstrap last_value, run PPO update, clear buffer
             → reset_match()
```

so **one PPO update per match**, over a batch whose size varies by two orders of magnitude between
matches and between teams, followed by `buffer.clear()`. Data is never reused across matches: there is
no replay buffer, so the algorithm is strictly on-policy-within-a-match.

`last_value` for bootstrapping is taken from the **last row of the buffer** — i.e. from the last agent
that acted in the last step — and applied as the terminal value for the whole chain.

`ppo_epochs = 4` passes over the buffer, shuffled with `np.random.shuffle`, minibatch size 256. The
shuffle uses NumPy's **global** RNG, which is never seeded
([§ Reproducibility](09_reproducibility.md#96-what-is-and-is-not-reproducible)).

## 6.4 The three update paths

| Path | Actor input | Critic input | Used by |
|---|---|---|---|
| `update_actor_critic` | \(o_i\) (8) | none — the actor's own value head | DTE |
| `update_ctde` | \(o_i\) (8) | \(s\) (36) | CTDE, CTDE-CAC |
| `update_cte` | \([s, \text{slot}]\) (39) | \(s\) (36) | CTE |
| `update_ctde_vd` | — delegates verbatim to `update_ctde` — | | CTDE-VD |
| `update_ctde_comm` | team obs (24) + slot | \(s\) (36) | CTDE-Comm |

Two things follow. First, **`update_ctde_vd` is a one-line alias**:

```python
def update_ctde_vd(self, actor, critic, buffer, last_value, gamma, gae_lambda):
    return self.update_ctde(actor, critic, buffer, last_value, gamma, gae_lambda)
```

so the entire difference between the VD and CAC arms in Experiment 2 is the critic's `forward`
function. That makes Experiment 2's VD-vs-CAC contrast a clean architectural comparison at the level of
the value network — and, given the capacity difference of
[§ 5.4](05_network_architectures.md#54-valuedecompositioncriticnetwork--additive-per-agent-critic-ctde-vd),
not a clean one at the level of capacity.

Second, `update_ctde_comm` is a near-copy of `update_ctde` with the extra slot tensor. The three paths
share 95 % of their code; the duplication is a maintenance hazard that has already produced a divergence
between the two trees ([§ Code audit](11_code_audit.md#cross-tree-divergence)).

## 6.5 What the implementation does not do

Each of these is a standard PPO ingredient that the naming of the code suggests but the code omits.
They are listed because they bound what the results can be read as.

| Omitted | Where it would matter | Evidence |
|---|---|---|
| **KL monitoring / early stopping** | With 4 epochs over a variable-size batch and no KL guard, a short match (122 rows → one minibatch) and a long match (1,157 rows → five minibatches) get very different effective step counts | `PPOStats.approx_kl` is hard-coded to `0.0` at `ppo.py:67` |
| **Value-function clipping** | Deviation from MAPPO practice. Measured returns in a typical match span \([-1.50, +2.72]\), so this is not visibly harmful here | `nn.functional.mse_loss(values, returns_batch)` only |
| **Reward / return normalisation** | MAPPO's reported stability is largely attributed to this; here the mean reward per buffer row is \(+0.012\) for a surviving team against informative events of \(\pm1.5\), so the signal-to-noise ratio of the value target is low | absent |
| **Reporting of training loss** | Without policy/value/entropy curves there is no way to tell a converged policy from a diverged one | `PPOStats` is constructed and returned, then **discarded**: `finish_rl_episode` returns a dict that no caller reads |
| **Recurrent or history-conditioned policies** | Agents have no memory, so a partially observed state cannot be disambiguated over time | MLPs only |
| **Learning-rate schedule** | 3 M-step runs at a constant 3e-4 Adam | absent |
| **Optimizer state persistence** | Training cannot be resumed; a restart silently resets Adam moments | [§ 5.7](05_network_architectures.md#57-checkpoint-contents) |

The `PPOStats` row deserves emphasis because it is the one that was *intended*: the dataclass exists,
`_optimize` fills it, `finish_episode` returns it, `finish_rl_episode` collects it into a dict — and
then `simulation.finish_match()` throws the dict away. The README's "next steps" item about persisting
PPO statistics is accurate: the plumbing stops one function short of being useful.

## 6.6 Exploration

Exploration is a single Categorical entropy bonus with weight \(c_H = 0.01\) over 8 actions. There is
no intrinsic reward, no noise injection, and no annealing of \(c_H\). Given that the action space
collapses to "pick one of four waypoints, maybe shoot"
([§ 4.3](04_mdp_formalisation.md#43-action-space)), the exploration problem is small; the credit
assignment problem is not.

## 6.7 Hyperparameters as trained

Every value below is what the versioned runs used, read from the committed `.env` and confirmed against
the 43 parameters recorded in the MLflow store
([`ctde_arena/data/mlflow_export/mlflow_runs.json`](../ctde_arena/data/mlflow_export/mlflow_runs.json)).

| Group | Parameter | Value |
|---|---|---|
| Optimiser | Adam lr | 3e-4 |
| | clip eps | 0.2 |
| | value coef | 0.5 |
| | entropy coef | 0.01 |
| | max grad norm | 0.5 |
| | PPO epochs | 4 |
| | batch size | 256 |
| Returns | \(\gamma\) | 0.99 |
| | \(\lambda\) | 0.95 |
| Networks | hidden dim | 128 (2 hidden layers, Tanh) |
| Environment | `RANDOM_SEED` | 7 |
| | `SIM_STEP_DT` | 0.1 s |
| | `ARENA_SIZE` | 32 (eval) / 28–36 (train) |
| | `MATCH_DURATION_SECONDS` | 90 (eval) / 60–120 (train) |
| | `DOMAIN_RANDOMIZATION` | true |
| Budget | `RL_TRAIN_TOTAL_STEPS` | **100,000 in the committed `.env`** |
| | `RL_SAVE_EVERY_STEPS` | 100,000 |
| | `RL_LOG_EVERY_STEPS` | 50,000 |
| | `RL_METRICS_EVERY_MATCHES` | 10 |
| | device | cpu |

> The code default for `RL_TRAIN_TOTAL_STEPS` is 3,000,000 and several documents state that this is
> "the standard configuration". The committed `.env` overrides it to 100,000, which is what the
> versioned artefacts were produced with. The discrepancy is resolved in
> [§ Code audit](11_code_audit.md#configuration-claims) — the `.env` is authoritative for the shipped
> results.
