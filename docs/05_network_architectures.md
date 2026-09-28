# 5. Network architectures

Source: `src/marl_arena/rl/networks.py` (both trees), wired by
`src/marl_arena/controllers/rl_controller.py::_build_networks`.

Every network is built from one helper:

```python
def _mlp(input_dim, output_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )
```

so all backbones are **two Tanh-hidden-layer MLPs of width `RL_HIDDEN_DIM` = 128** with no output
nonlinearity. No normalisation layers, no dropout, no residual connections.

Parameter counts below were measured by instantiating each module, not derived by hand.

## Overview

| Team | Paradigm | Actor | Critic | Actor params | Critic params | Total |
|---|---|---|---|---:|---:|---:|
| 1 (exp. 1) | **CTE** | `CentralizedActorNetwork` | `CentralizedCriticNetwork` | 39,176 | 37,889 | **77,065** |
| 2 (exp. 1) | **DTE** | `ActorNetwork` (integrated value head) | — | 35,337 | 0 | **35,337** |
| 3 (exp. 1) | **CTDE** | `ActorNetwork` | `CentralizedCriticNetwork` | 35,337 | 37,889 | **73,226** |
| 1 (exp. 2) | **CTDE-VD** | `ActorNetwork` | `ValueDecompositionCriticNetwork` | 35,337 | 17,281 | **52,618** |
| 2 (exp. 2) | **CTDE-CAC** | `ActorNetwork` | `CentralizedCriticNetwork` | 35,337 | 37,889 | **73,226** |
| 3 (exp. 2) | **CTDE-Comm** | `CommActorNetwork` | `CentralizedCriticNetwork` | 37,388 | 37,889 | **75,277** |

> **Capacity is not matched across arms.** DTE carries 35 k parameters while CTE carries 77 k — a
> 2.2× difference. Any CTE-vs-DTE reading of Experiment 1 is confounded by model size, and the
> VD-vs-CAC contrast in Experiment 2 is confounded in the opposite direction (VD has less than half
> the critic capacity). See
> [§ Threats to validity](10_threats_to_validity.md#101-internal-validity).

## 5.1 `ActorNetwork` — local actor

Used by DTE, CTDE, CTDE-VD and CTDE-CAC.

```
trunk    : Linear(8,128) → Tanh → Linear(128,128) → Tanh → Linear(128,128) → Tanh
policy   : Linear(128, 8)        → Categorical logits
value    : Linear(128, 1)        → V(o_i)
```

Input is the 8-dim local observation of [§ 4.1](04_mdp_formalisation.md#41-local-observation-o_i).
`act()` returns `(action, log_prob, entropy, value)`; `evaluate()` returns
`(log_prob, entropy, value)`.

For **DTE** the value head *is* the critic: policy and value share a trunk, which is the classic
integrated actor-critic. For **CTDE/VD/CAC** the value head exists but is unused — the critic below
supplies the baseline. Those wasted 129 parameters are harmless, but they mean DTE and CTDE differ in
*two* ways (shared vs separate value function, and 35 k vs 73 k parameters), not one.

## 5.2 `CentralizedActorNetwork` — joint-state actor (CTE)

```
input    : concat(global_obs[36], agent_slot[3]) = 39
trunk    : Linear(39,128) → Tanh → Linear(128,128) → Tanh → Linear(128,128) → Tanh
policy   : Linear(128, 8)
```

`agent_slot` is one-hot over the team's three agents, so a single network parameterises all three
actors and is told which one is acting. Execution reads the **full 36-dim global vector**, which is why
this arm is centralised at *execution* time and is not deployable to independent agents.

## 5.3 `CentralizedCriticNetwork` — joint-state critic

```
input    : global_obs[36]
trunk    : Linear(36,128) → Tanh → Linear(128,128) → Tanh → Linear(128,128) → Tanh
value    : Linear(128, 1) → V(s)
```

One scalar per team. Note it does not receive the joint *action*, so it estimates \(V^{\pi}(s)\) for the
team's own policy while the other two teams are part of \(s\) — the usual non-stationarity of
concurrent MARL training, unmitigated here.

## 5.4 `ValueDecompositionCriticNetwork` — additive per-agent critic (CTDE-VD)

```python
self.agent_critic = _mlp(4, 1, hidden_dim)          # ONE shared network
V_tot(s) = Σ_{k∈{0,1,2}} agent_critic(s[4k : 4k+4])
```

with `agent_indices` fixed per team: `[0,1,2]` for Team 1, `[3,4,5]` for Team 2, `[6,7,8]` for Team 3,
selected by an `if/elif` on `self.team_name`.

Each per-agent term consumes the 4-dim slice \([p_x, p_z, \theta/180, \mathbb 1_{\text{alive}}]\) of
the **global** vector for one of its own team's agents. Two deviations from VDN are therefore built in:

1. **The summands are functions of global-state slices, not of local observations \(o_i\).** The
   documentation elsewhere writes \(V_{\text{tot}}(s) = \sum_i V_i(o_i)\); what the code computes is
   \(\sum_i V_i(\text{slice}_i(s))\). The slices are drawn from a vector the executing actors never
   see, so the decomposition does **not** yield a decentralisable greedy policy — it is a
   *parameter-sharing and input-sparsity* choice for the critic, not a factorisation that supports
   decentralised action selection.
2. **No \(\sum_i \max_{a_i} Q_i\) consistency term** and no monotonicity constraint, because there is
   no \(Q\) at all — this is a \(V\) decomposition feeding GAE, which is a different use of the
   additive assumption than VDN's.

What it does deliver is a 17,281-parameter critic against CAC's 37,889, and an input that excludes the
opponent agents entirely. Both are plausible reasons for its behaviour in
[§ Results](08_results.md), and both are confounds rather than clean tests of decomposition.

## 5.5 `CommActorNetwork` — one-round differentiable communication (CTDE-Comm)

```
msg_net     : _mlp(8, 4, 128)     → message m_k ∈ R^4 from ally k's local observation
policy_head : _mlp(8+4, 8, 128)   → logits from [o_i , c_i]
```

The forward pass is executed for the **whole team at once** and the acting agent's result is selected
by the one-hot slot vector:

\[
\begin{aligned}
m_k &= \text{msg\_net}(o_k) & k &= 1,2,3\\[2pt]
c_1 &= \tfrac12(m_2 + m_3),\quad c_2 = \tfrac12(m_1 + m_3),\quad c_3 = \tfrac12(m_1 + m_2)\\[2pt]
\tilde{o} &= \textstyle\sum_k s_k o_k,\qquad \tilde{c} = \textstyle\sum_k s_k c_k\\[2pt]
\text{logits} &= \text{policy\_head}([\tilde{o}, \tilde{c}])
\end{aligned}
\]

where \(s\) is `agent_slot`. Self-exclusion from \(c_i\) is correct, and the mean-pooling matches
CommNet. The differences that matter:

* **One round only.** Information cannot travel \(i \to j \to k\).
* **The team observation is assembled by the controller**, sorted by `agent_id`, with dead allies
  replaced by an all-zero 8-vector. A dead ally therefore emits a message \(m = \text{msg\_net}(0)\),
  which is a *non-zero constant* that biases the receiver's channel — CommNet masks absent agents; this
  implementation does not. With 3-agent teams and frequent early deaths, this is a systematic effect on
  the arm that is already last.
* **The buffer stores `team_obs` (24-dim) as the "local" observation** for this team, so its rollout
  tensors have a different shape from every other arm.
* The message is 4-dimensional against an 8-dimensional observation, so the channel is
  lower-dimensional than the private signal it modulates.

## 5.6 Dead code

`CTDEActorNetwork` and `CTDECriticNetwork` are `pass` subclasses of `ActorNetwork` and
`CentralizedCriticNetwork`, referenced nowhere. They are listed here so that a reader searching for
"the CTDE network" does not mistake them for the live implementation — the CTDE arm uses plain
`ActorNetwork` + `CentralizedCriticNetwork`.

## 5.7 Checkpoint contents

`save_checkpoint` writes `{"paradigm": str, "actor": state_dict, "critic": state_dict}` and nothing
else, verified by loading the three versioned policies:

```
team_1_ctde-vd.pt   : top-level=['paradigm', 'actor', 'critic'] paradigm=CTDE-VD   actor tensors=10
team_2_ctde-cac.pt  : top-level=['paradigm', 'actor', 'critic'] paradigm=CTDE-CAC  actor tensors=10
team_3_ctde-comm.pt : top-level=['paradigm', 'actor', 'critic'] paradigm=CTDE-Comm actor tensors=12
```

No optimizer state, no step counter, no seed, no configuration snapshot. Consequences:

* training **cannot be resumed** — `train_rl.py` restarts from a fresh Adam state and a step count of
  zero even though `_load_if_exists()` restores the weights;
* a checkpoint does not record how long it was trained for, so provenance depends entirely on the
  accompanying `training_log.json`;
* `torch.load` is now called with `weights_only=True` (verified compatible with these payloads), which
  removes the arbitrary-code-execution surface that `weights_only=False` implies for model files of
  unknown origin.
