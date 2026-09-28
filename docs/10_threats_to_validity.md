# 10. Threats to validity

This chapter is the limitations section. It is organised by the standard four categories, and every
entry states what would have to be done to remove the threat, so it doubles as a research agenda.

## 10.1 Internal validity

Threats that could make the observed differences an artefact of the experiment's construction rather
than a property of the architectures.

### Confounded training signal volume

Buffer size per match is proportional to how long a team's agents stay alive, because dead agents
produce no decisions and therefore no rollout rows. Measured on one match: CTE 1,157 rows, DTE 122,
CTDE 952. A team that is winning gets more gradient updates. **The treatment and the outcome feed back
into each other**, so "CTDE learned better" and "CTDE got to learn more" are not separable.

*Remedy:* fixed-length rollouts collected per agent, or importance-weighted minibatching so that update
volume is decoupled from survival.

### Unmatched model capacity

Parameter totals range from 35,337 (DTE) to 77,065 (CTE) — a 2.2× spread — and the arms differ in
whether the value function shares a trunk with the policy. In Experiment 2, CTDE-VD's critic is 17,281
parameters against CTDE-CAC's 37,889.

*Remedy:* match total trainable parameters per team, e.g. by widening the smaller network.

### Slot confounding (fixed) <a name="slot-and-seed-confounding"></a>

Team identity is fixed to a spawn corner, a controller seed offset (11/23/37), and an initialisation
position in the construction order. No permutation of architecture-to-slot assignment was run. With one
seed, an architecture-vs-slot effect cannot be separated from an architecture effect.

*Remedy:* rotate which slot each architecture occupies across seeds, and report the per-slot means.

### GAE over an interleaved buffer

The advantage recursion chains rows that belong to different agents at the same timestep
([§ 6.2](06_optimisation_procedure.md#the-buffer-is-not-a-trajectory)). This is a defect common to all
six arms, so it does not favour one architecture — but it means none of the arms is the algorithm its
name describes, and it invalidates any attempt to compare against published PPO/MAPPO curves.

*Remedy:* maintain one buffer per agent, or sort the buffer by (agent, time) before computing returns.

### Non-stationary opponents

All three teams train concurrently against each other. A team's win rate depends on which two
co-adapting opponents it faced at that moment. There is no league, no past-opponent sampling, and no
fictitious self-play.

*Remedy:* evaluate each final policy against a fixed pool of the others, or against scripted baselines.

### The turn control defect (fixed, but it splits the evidence base)

`angle_to_target` read the wrong axis, so the low-level heading controller was driven by a near-constant
signal ([A-1](11_code_audit.md#turn-control-defect)). Because aiming and shooting do not depend on
heading, the agents still functioned — but every policy trained before the fix learned in a world where
turning was effectively noise, which makes their "survival" metric partly a measure of tumbling rather
than tactics.

The fix is in place and the replicated study ran under it. The consequence for reading this
documentation is that **the two bodies of evidence are not comparable**: the historical single-seed
100 k-step runs and the 10-seed 500 k-step study differ in control law, budget, seed count, slot rotation and whether the
measurement was taken during training or held out. Only the study supports conclusions; the historical
runs are a record of what the repository used to claim.

### Cumulative tie-breaking (fixed)

Wins on a timeout used to be awarded using *run-long* cumulative eliminations as the tiebreak, giving
historically stronger teams a persistent edge in otherwise-flat matches. Ranking now uses within-match
kills, and a match whose top two teams are indistinguishable is recorded as a draw with no win awarded
([A-10](11_code_audit.md#cumulative-tiebreak)). The draw rate under the current code is measured in
[§ Results](08_results.md#replicated-study).

## 10.2 Construct validity

Do the measurements mean what they are labelled as?

### "CTDE-VD" is not value decomposition over local observations

The documented formula is \(V_{\text{tot}}(s) = \sum_i V_i(o_i)\). The implementation computes
\(\sum_i V_i(\text{slice}_i(s))\), where the slices come from the **global** vector the executing actors
never see ([§ 5.4](05_network_architectures.md#valuedecompositioncriticnetwork)).
The arm is therefore a *sparse-input, weight-shared centralised critic*, not a decentralisable
factorisation. Conclusions about VDN cannot be drawn from it.

### "CTDE-Comm" is one round of mean-pooled communication

CommNet's published form stacks multiple rounds. With one round the channel cannot propagate a hop, so
the arm does not test the hypothesis that communication needs time to bootstrap — it tests a strictly
weaker architecture. The claim "communication requires >500k steps" that appears in the project's own
documentation is therefore not supported by this implementation, which is a different model than the
one the claim is about.

### Dead allies inject a constant message

`CommActorNetwork` receives zero-filled observations for dead teammates, and `msg_net(0)` is a
non-zero learned constant that enters every receiver's channel. The Comm arm's behaviour is partly a
function of how many teammates happen to be dead — a nuisance variable no other arm has.

### Training metrics presented as results

`summary.json` is a cumulative average over a non-stationary trajectory, and every recorded match used
stochastic action sampling. Neither the final policy nor a greedy-action evaluation is measured
([§ 7.2](07_experimental_protocol.md#72-procedure)). The headline "win rate" is thus an average over
~460 different policies, not a property of the checkpoint that ships.

### Domain randomisation is unmeasured

Randomisation is applied during training and disabled during evaluation, with no ablation. Its
contribution is unknown.

### Win rate is not the same construct as competence

Because a win can be awarded by timeout with nobody dead, and because the tiebreak is cumulative,
"win rate" mixes combat effectiveness with survival duration and historical advantage.

## 10.3 External validity

* The arena is bespoke. No published benchmark, so no figure here is comparable to the literature.
* Nine agents, three teams, discrete 8-action space, no obstacles in the observation. Conclusions about
  "MARL" in general are not supported; at most, about these six architectures in this geometry.
* The reward is dominated by a per-step alive bonus of the same order as a kill
  ([§ 4.4.1](04_mdp_formalisation.md#441-reward-scale-versus-horizon)). A different reward scale would
  plausibly reorder the arms.
* Everything is CPU-simulated at \(\Delta t = 0.1\) s with a 0.4 s projectile lifetime. Nothing about
  the transfer of these policies to a real-time or real-robot setting has been examined.

## 10.4 Statistical validity

The replicated study in [§ Results](08_results.md#replicated-study) removed the two largest threats that
applied to the original single-seed evidence, and left three that no amount of replication fixes.

**Now addressed:**

1. ~~**n = 1 seed per experiment.**~~ Five seeds per experiment, so between-seed variance is estimable
   and a paradigm that only wins on one seed is visible as such.
2. ~~**46 recorded matches per team.**~~ 150 held-out greedy matches per replicate, i.e. 750 per paradigm
   per experiment, pooled with Wilson intervals reported per seed and in aggregate.
3. ~~**Training metrics presented as results.**~~ The study measures the final policy under greedy
   action selection on the fixed evaluation variant, which is what the headline tables always claimed to
   report.
4. ~~**The cumulative denominator is not independent.**~~ Pooled estimates are now over held-out matches
   of a frozen policy, so the Bernoulli approximation is defensible; Fisher's exact test is used instead
   of a normal-approximation \(z\)-test because the counts are small.

**Still open:**

5. **Exactly-one-winner coupling.** Within a match the three outcomes are negatively correlated
   (\(\sum_i \text{winner}_i = 1\)). Pooling across seeds does not break this, so a two-sample test
   between two arms of the same match is not strictly valid — it is anti-conservative for the loser and
   conservative for the winner. Reporting all three arms' win rates together, as done here, is the honest
   presentation; the third is determined by the other two.
6. ~~**Slot is not rotated.**~~ **Fixed** — `paradigm_rotation = seed % 3`, so across the 20 runs each
   paradigm occupies each slot 6 or 7 times. Note what this does *not* buy: rotation removes the
   confound, it does not remove the slot's effect, so residual slot variance now appears inside the
   between-seed SD rather than hiding inside the architecture contrast.
7. **Budget still short of the hypothesis being tested.** The claim that CTDE-Comm needs more steps than
   the others is tested at 500 k steps; if it is still improving at the end of that budget, the answer is
   "not yet", not "no". The evidence now points the other way: at 500 k steps three of six arms are best
   at their *earliest* checkpoint ([§ 8.1](08_results.md#replicated-study)).
8. **No effect sizes with uncertainty on the secondary metrics.** Eliminations per match, survival and
   accuracy are reported as pooled means without intervals.
9. **Win rate does not isolate fighting.** A quarter of replicates decide most matches on the 90 s clock,
   and the untrained greedy baseline shows 41 % three-way draws and 88–398 shots per match at 0.1 %
   accuracy ([§ 8.9](08_results.md#random-baseline)), so the construct being measured is partly
   survival-to-buzzer. The termination rule, not the sample size, is what would have to change.
10. **No fixed-opponent evaluation.** Arms are compared only against each other, never against a frozen
    pool or scripted opponents, so absolute strength is unmeasured
    ([§ 10.3](10_threats_to_validity.md)).

*Next concrete steps:* change what win rate measures (items 9 and 10) before adding replicates. Rotating
the slot — the threat a modest re-run used to fix — has been done, and the ranking it might have exposed
turned out to be reversible by a single degenerate replicate anyway
([§ 8.1](08_results.md#leave-one-out)).

## 10.5 Reporting validity

* The prior Portuguese-language README asserted a diagnosis of the CTDE-Comm deficit ("messages start as
  noise") that the accuracy decomposition in [§ 8.3](08_results.md#decomposing-the-elimination-gap)
  does not support: Comm's accuracy is fine, its firing volume is not.
* The README also stated the default training budget was 3 M steps; the committed `.env` says 100,000.
* Both have been corrected in this documentation set, and the corrections are itemised in
  [§ Configuration claims](11_code_audit.md#configuration-claims).
