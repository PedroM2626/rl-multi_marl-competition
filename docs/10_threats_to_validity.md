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

Parameter totals range from 35,465 (DTE) to 79,369 (CTE) — a 2.24× spread — and the arms differ in
whether the value function shares a trunk with the policy. In Experiment 2, CTDE-VD's critic is 17,409
parameters against CTDE-CAC's 39,041.

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

### ~~Domain randomisation is unmeasured~~ Measured, and it is not a neutral control

Randomisation is applied during training and disabled during evaluation. Turning it off
([§ 8.13](08_results.md#ablations)) moves a paradigm's held-out win rate by up to 0.253, inverts the ranking
of both experiments, and roughly triples mean survival time -- so its contribution was not merely unknown, it
was larger than the effect the study set out to measure. Two consequences remain open: the ablated arms were
evaluated only against their own co-adapted teammates, so the common-opponent
([§ 8.10](08_results.md#cross-arena)) and random-pool ([§ 8.11](08_results.md#absolute-strength)) results are still
single-configuration measurements; and survival changing that much means part of what `no_dr` "wins" is the
buzzer, which [§ 8.12](08_results.md#arena-v2) shows is a separate large effect.

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
* **The pinned environment does not reproduce the recorded runs.** Measured directly:
  [§ 9.1](09_reproducibility.md#verification-environment-actually-used) shows the arena's geometry and its
  randomisation streams are identical under the pinned NumPy 2.2.6, but the PPO update diverges at the
  1e-6 level and every saved weight differs after 4,000 steps. A replicate is therefore not reproducible
  across library versions — an external-validity problem for the *specific* numbers and independent
  corroboration of the study's central claim, that this system's outcome differences are smaller than the
  perturbations that produce them.

## 10.4 Statistical validity

The replicated study in [§ Results](08_results.md#replicated-study) removed the two largest threats that
applied to the original single-seed evidence, and left three that no amount of replication fixes.

**Now addressed:**

1. ~~**n = 1 seed per experiment.**~~ Ten seeds per experiment, so between-seed variance is estimable
   and a paradigm that only wins on one seed is visible as such — which is exactly what happened to the
   project's headline claim ([§ 8.1](08_results.md#reversal)).
2. ~~**46 recorded matches per team.**~~ 150 held-out greedy matches per replicate, i.e. 1,500 per paradigm
   per experiment, pooled with Wilson intervals reported per seed and in aggregate.
3. ~~**Training metrics presented as results.**~~ The study measures the final policy under greedy
   action selection on the fixed evaluation variant, which is what the headline tables always claimed to
   report. **This remedy is only half right, and the other half is now a threat in its own right** — see
   item 11.
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
9. **~~Win rate does not isolate fighting.~~ Measured; removing the route does not resolve the ranking.**
   A quarter of replicates decide most matches on the 90 s clock, and the untrained greedy baseline shows
   41 % three-way draws and 88–398 shots per match at 0.1 % accuracy
   ([§ 8.9](08_results.md#random-baseline)), so the construct being measured was partly survival-to-buzzer.
   The remedy was to stop awarding a win for surviving, and it has now been run: under
   [§ 8.12](08_results.md#arena-v2) 44 % of experiment 1's and 39 % of experiment 2's held-out matches have
   **no winner at all**, no arm separates (paired *p* ≥ 0.439), and experiment 1's first place moves from DTE
   to CTDE inside noise. The confound was real and larger than the duration medians suggested; it was not
   the reason the study cannot rank the paradigms.
10. **~~The reward scale favours survival over engagement.~~ Measured; the behavioural half is refuted.**
    At \(\gamma = 0.99\) the maximum discounted alive bonus (1.5) exceeds one elimination (1.2), so an agent
    can earn more by avoiding fights than by winning them
    ([§ 4.4.1](04_mdp_formalisation.md#441-reward-scale-versus-horizon)). Arena v2 cut the bonus ten-fold and
    decayed it with a 30 s half-life. Shots per match did rise for CTDE (101.2 → 140.3) and DTE (105.2 →
    119.0), but fell for CTE (121.7 → 107.4), and eliminations fell for both DTE (1.95 → 1.72) and CTE
    (2.22 → 2.03). Cheap survival does not reliably buy aggression, so the incentive diagnosis stands and
    the prediction built on it does not.
11. **The readout is not the trained policy — now measured, and it changes experiment 2's winner.** Greedy
   held-out win rate — the headline response variable of every table in
   [§ 8.1](08_results.md#replicated-study) — has 2.5× the spread of the
   sampled policy it summarises (SD 0.177 vs 0.072 across 60 arm-replicate cells) and preserves its
   within-replicate ranking in only 35.6 % of comparable pairs. The remedy was applied and the threat
   quantified: all sixty cells were replayed at T = 0 (bit-identical to the published study, so the
   instrument contributes no noise), at T = 0.5 and at T = 1
   ([§ Readout re-measured](08_results.md#readout-re-measured)). Two results matter. **Experiment 2's ordering
   is a readout artefact**: CTDE-VD leads at 0.367 under argmax and is last at 0.323 under T = 0.5, with
   CTDE-CAC gaining +0.043 (*p* = 0.020) — the only cells in the study whose shift reaches an uncorrected
   0.05, and neither clears its Bonferroni threshold. **The variance is not the readout's**: between-seed SD
   spans 0.128–0.220 under both sampled temperatures against 0.122–0.218 greedy, so the instability this
   document keeps reporting is training variance, and dual-reporting buys interpretability rather than
   power. What remains open is the choice the study still makes: the published tables are the greedy arm, so
   every experiment-2 ranking in them carries the qualification above.
12. **~~The round-robin tests six checkpoints, not six architectures.~~ Closed at ten; the threat moved from
    the sample to the finding.** The single-checkpoint version represented each arm by its promoted replicate,
    so its two corrected-significant results were statements about particular trained networks — and that is
    exactly how it turned out: replication moved plain CTDE from last to third and back, moved CTE from third
    to last, and made pairs lose their threshold as soon as one checkpoint set was excluded. The pool now
    covers **all ten training blocks the study can produce** (30,000 matches, every replicate of both
    experiments exactly once), and it did not stabilise the table: separations went 2 → 8 → 5 → 4 across the
    one-, three-, six- and ten-block pools, DTE fell from second to fifth, CTDE-VD from fifth to second, and
    three of the five pairs the six-block pool separated lost correction. What is left after ten blocks is a
    single claim about a single arm — CTDE-Comm above four of the other five, robust to deleting any block —
    and the honest reading of the middle of the table is that **these ten networks do not order the five
    non-communication architectures**, with between-block SD 0.026–0.086 against 0.009–0.013 across match
    streams. There is no eleventh block: the sample of trained replicates is exhausted, so a further move on
    this axis costs matches per block, seeds per training run, or a different configuration
    ([§ 8.10's limits](08_results.md#cross-arena-limits)). The pairing caveat also survives at ten: each block
    pairs one experiment-1 seed with exactly one experiment-2 seed, no pairing is crossed, so an exp-1 × exp-2
    pairing effect cannot be separated from a block effect at this n.
13. **~~No fixed-opponent evaluation, and no absolute reference point.~~ Partly closed.** Two instruments now
    exist. The cross-experiment round-robin ([§ 8.10](08_results.md#cross-arena)) puts all six arms on a common
    pool of *trained* opponents, which removes the co-adaptation confound for evaluation. And
    [§ 8.11](08_results.md#absolute-strength) puts each arm against a fixed pool of **uniform-random** teams,
    which is a floor that does not depend on who else was in the run — with a null matched slot by slot,
    because the arena's three spawn positions are not equally winnable even when all three teams are random
    ([§ 8.9](08_results.md#random-baseline): 0.257 / 0.270 / 0.463). Over six checkpoint sets the two
    scales rank the six architectures the same way (Spearman +0.77, same arm first and same arm last), which
    is corroboration the peer-relative ranking previously had none of; the statistic is +0.41 and +0.94 at
    one and three sets, and −0.04 only when a one-set measurement is compared against a three-set peer table,
    which is why the corroboration is always stated with its replication count. Recomputed against the
    ten-block pool — mismatched counts on the two sides, so a stress test rather than the headline figure —
    the same six-set absolute measurement gives Spearman **+0.657** with the same arm first and the same arm
    last: the direction survives a peer pool that reordered the middle, its magnitude is not the one the
    committed artefact records. What
    remains open: the random pool is the only non-self-referential opponent set here — there are no scripted
    or human policies, no other game, and no published baseline, so no number in this repository is
    comparable to the literature in absolute terms; and **training** itself is still three arms co-adapting
    inside one run, so the confound is removed at measurement time only.

*Next concrete steps:* the objective-side remedies (items 9 and 10) have now been measured
([§ 8.12](08_results.md#arena-v2)) and did not resolve the ranking, and the sample-side one has been taken to
its end — the round-robin now plays **all ten training blocks the study can produce** (item 12), which was
evaluation time on checkpoints that already existed. What is left is the measurement: dual-report the readout
at a fixed temperature (item 11, [§ Readout](08_results.md#readout-divergence)), whose re-measurement is
running; replay the configurations that have not been played on the common pool at the headline replication
count (item 13 of [the audit's open work](11_code_audit.md#open-work-suggested-measurements)); and
give the arms opponents that are not other trained policies at all. Four of the remedies a previous version
of this list proposed have now been carried out — rotating the slot, evaluating on a common opponent pool,
replicating that evaluation over several sets of checkpoints, and changing the objective (items 9 and 10) —
and each of them changed a conclusion that was in print: the rotation helped expose that a single replicate
could reverse the ranking, the common pool reversed the round-robin's own first verdict once its seeds were
pooled, pooling *training* replicates on top of that moved two architectures to opposite ends of the same
table, and turning the buzzer-win into a draw showed that 44 % and 39 % of the held-out matches had only
ever been "won" by the clock
([§ 8.10](08_results.md#cross-arena), [§ 8.12](08_results.md#arena-v2)).

## 10.5 Reporting validity

* The prior Portuguese-language README asserted a diagnosis of the CTDE-Comm deficit ("messages start as
  noise") that the accuracy decomposition in [§ 8.3](08_results.md#decomposing-the-elimination-gap)
  does not support: Comm's accuracy is fine, its firing volume is not.
* The README also stated the default training budget was 3 M steps; the committed `.env` says 100,000.
* Both have been corrected in this documentation set, and the corrections are itemised in
  [§ Configuration claims](11_code_audit.md#configuration-claims).
