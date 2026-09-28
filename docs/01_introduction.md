# 1. Introduction and research questions

## 1.1 Problem

Multi-agent reinforcement learning (MARL) algorithms differ less in the loss they optimise than in
**where information is allowed to flow** — at training time, at execution time, or through an explicit
communication channel. Three axes organise the design space:

1. **Training-time centralisation.** Does the value function see the joint state of all agents, or
   only each agent's local observation?
2. **Execution-time centralisation.** Does action selection see the joint state, or only local
   observations? (This is the constraint that decides whether a policy survives deployment on
   independent robots.)
3. **Value decomposition.** Is the team value a single function of the joint state, or a structured
   sum of per-agent contributions?

The standard claim in the literature is a trade-off: centralised training reduces the variance of the
learning signal, decentralised execution is what makes a policy deployable, and value decomposition
tries to have both. Explicit communication is a fourth option that changes the *architecture* rather
than the *training regime*.

These claims are usually demonstrated on different environments, by different papers, with different
hyperparameters — which makes them hard to compare directly. This repository exists to run them
against each other **inside one engine**, with one simulator, one reward function, one observation
encoding and one PPO implementation, so that the only thing that varies is the information-flow
architecture.

## 1.2 Setting

The test bed is a 3D free-for-all battle arena: **three teams of three agents** (nine agents total)
compete in a walled, obstacle-filled arena. Each agent can move forward/backward, turn, and fire
projectiles at enemies. An agent that is hit is eliminated and takes no further action for the rest of
the match. A match ends when at most one team still has living agents, or when the duration limit is
reached. Each team is controlled by one policy network trained with PPO, and the three teams in a
match implement three different architectures — so the architectures compete directly, and the
competitive outcome is itself the measurement.

The engine is written from scratch in NumPy; the 3D rendering layer is Ursina. Training and the
simulation are decoupled: the same engine runs headless for training and drives the visual window for
inspection.

## 1.3 The two experiments

| | Experiment 1 (root) | Experiment 2 (`ctde_arena/`) |
|---|---|---|
| Question | *Which training/execution paradigm wins?* | *Within CTDE, which architecture wins?* |
| Contenders | CTE, DTE, CTDE | CTDE-VD, CTDE-CAC, CTDE-Comm |
| Axis varied | Centralisation at train and execution time | Value decomposition vs centralised critic vs learned communication |
| Reference style | — | VDN, MAPPO, CommNet |
| Extra infrastructure | — | MLflow tracking, Dockerfile |
| Versioned results | Yes | Yes |
| Replicated across seeds | **10** (500 k steps each, slot-rotated) | **10** (500 k steps each, slot-rotated) |

Experiment 2 is a self-contained fork of the same engine, not an import of it. The two trees share
most files byte-for-byte but have diverged; the divergence is catalogued in
[§ Code audit](11_code_audit.md#cross-tree-divergence) because it is a live reproducibility risk.

## 1.4 Research questions

* **RQ1.** Does centralised training with decentralised execution (CTDE) outperform fully centralised
  execution (CTE) and fully decentralised actor-critic (DTE) in this arena, under matched
  hyperparameters and a fixed step budget?
* **RQ2.** Among CTDE variants, does a decomposed value function (VD), a centralised actor-critic
  critic (CAC), or an explicit differentiable communication channel (Comm) yield the highest win rate?
* **RQ3.** Is the observed ranking stable — across seeds, across training length, and across the
  match-to-match variance of the arena?

RQ1 and RQ2 are answered in [§ Results](08_results.md#replicated-study) from a ten-seed, 500 000-step
replicated study with held-out greedy evaluation and the paradigm-to-slot assignment rotated by seed. RQ3 is answered in two parts: **stability across seeds
and across the arena's match-to-match variance is measured**, and the per-seed spread is reported rather
than pooled away; **stability across training length is only partially addressed**, because the budget is
500 k steps rather than the 3 M the code defaults to, so a paradigm still improving at the end of the run is
reported as "not yet settled" rather than "no".

The repository previously answered these questions from a single 100 k-step run per experiment, measured
during training. That evidence is retained and labelled historical in chapter 8, and the reasons it should
not be relied on are set out in [§ Threats to validity](10_threats_to_validity.md).

## 1.5 Contributions

1. A self-contained, dependency-light MARL arena (NumPy physics + PPO, no external RL framework) in
   which six distinct information-flow architectures share one simulator, one reward and one
   observation encoding.
2. A from-scratch implementation of all six architectures in ~1,900 lines of model and training code,
   including a swept-AABB projectile collision solver that is exact at the discrete step level
   ([§ Collision detection](03_arena_system_model.md#343-collision-detection)).
3. Versioned training artefacts for Experiment 2: policies, per-match metric CSVs, aggregate summary,
   dashboard, and the exact set of points that reached MLflow.
4. A written account of what the implementation *actually computes*, as opposed to what the
   architecture names conventionally imply — several divergences are documented in
   [§ Optimisation procedure](06_optimisation_procedure.md#65-what-the-implementation-does-not-do) and
   [§ Code audit](11_code_audit.md). This is the part that constrains how the results may be read.

## 1.6 Scope boundaries

Out of scope, deliberately or as unfinished work:

* No comparison against published baselines on a shared benchmark; the arena is bespoke, so absolute
  win rates are not comparable to any figure in the literature.
* No evaluation against a fixed pool of opponents or scripted baselines — the only comparison available is
  between the three arms inside one run, so a ranking cannot be separated from the co-adaptation dynamics
  that produced it.
* No ablation of domain randomisation, reward coefficients, or network width.
* Respawn is configured but not implemented; the flag has no effect
  ([§ Code audit](11_code_audit.md#dead-configuration)).
* No recurrent policies, no opponent modelling, no self-play league. Teams train concurrently against
  each other's moving policies, which is a non-stationary environment none of the three algorithms
  accounts for.
* The 3D renderer is for human inspection only; no measurement reported here depends on it.

## 1.7 Reading note

[§ Results](08_results.md) carries two bodies of evidence. The **replicated study** — ten seeds per
experiment at 500 k steps, with held-out greedy evaluation of the final policies and the paradigm-to-slot
assignment rotated across seeds — is the primary result
and is what the conclusions should be drawn from. The **historical single-seed runs** at 100 k steps are
reported alongside it because they are what the repository originally shipped, but they were produced
under a defective heading controller
([A-1, now fixed](11_code_audit.md#turn-control-defect)) and with no replication, so they are a record of
what was measured, not evidence for a claim. Read the audit before citing either.
