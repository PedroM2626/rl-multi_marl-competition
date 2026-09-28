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

### Slot and seed confounding

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

### The turn control defect

`angle_to_target` reads the wrong axis, so the low-level heading controller is driven by a near-constant
signal ([§ Turn control defect](11_code_audit.md#turn-control-defect)). Because aiming and shooting do
not depend on heading, the agents still function — but every policy learned in a world where turning is
effectively noise. This is a shared defect, not a differential one, so it does not directly bias the
comparison; it does mean the reported absolute performance understates what these architectures would
achieve in a correctly actuated arena, and it makes the "survival" metric partly a measure of tumbling
rather than tactics.

*Remedy:* one-line fix, then re-run both experiments. This is the highest-value change available.

### Cumulative tie-breaking

Wins on a timeout are awarded using *run-long* cumulative eliminations as the tiebreak
([§ 3.5](03_arena_system_model.md#35-episode-structure-and-termination)), giving historically stronger
teams a persistent edge in otherwise-flat matches. Its magnitude is unmeasured.

*Remedy:* break ties on within-match statistics, or record a draw.

## 10.2 Construct validity

Do the measurements mean what they are labelled as?

### "CTDE-VD" is not value decomposition over local observations

The documented formula is \(V_{\text{tot}}(s) = \sum_i V_i(o_i)\). The implementation computes
\(\sum_i V_i(\text{slice}_i(s))\), where the slices come from the **global** vector the executing actors
never see ([§ 5.4](05_network_architectures.md#54-valuedecompositioncriticnetwork--additive-per-agent-critic-ctde-vd)).
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

The dominant limitation, and the one that most changes the reading of
[§ Results](08_results.md):

1. **n = 1 seed per experiment.** No variance estimate across runs is available at all.
2. **46 recorded matches per team.** Wilson 95 % intervals are ±13 percentage points wide, while the
   claimed gaps are 11–17 points. In Experiment 1 **no** pairwise comparison reaches \(p < 0.05\).
3. **The cumulative denominator is not independent.** Treating 460 matches as Bernoulli trials makes the
   same comparisons significant at \(p < 10^{-5}\) — a 10-order-of-magnitude swing from an assumption
   choice. This is the clearest demonstration that the current evidence base cannot support the
   conclusions drawn from it.
4. **Exactly-one-winner coupling.** Within a match the three outcomes are negatively correlated
   (\(\sum_i \text{winner}_i = 1\)), so standard two-proportion tests are conservative in one direction
   and wrong in another.
5. **No multiple-comparison correction is applied** to the six pairwise tests across the two
   experiments; the Bonferroni threshold would be 0.0167, which does not change any verdict in
   Experiment 2 and does not rescue any in Experiment 1.
6. **No effect sizes with uncertainty.** Eliminations per match and survival time are reported as point
   estimates only.

*Remedy, concretely:* 5 seeds × 3 M steps × 2 experiments ≈ 22.5 hours of CPU time on this machine
([§ 9.5](09_reproducibility.md#95-cost-of-a-run)), plus a held-out greedy evaluation of 200 matches per
pairing. That is the minimum credible study, and it is affordable.

## 10.5 Reporting validity

* The prior Portuguese-language README asserted a diagnosis of the CTDE-Comm deficit ("messages start as
  noise") that the accuracy decomposition in [§ 8.3](08_results.md#83-decomposing-the-elimination-gap)
  does not support: Comm's accuracy is fine, its firing volume is not.
* The README also stated the default training budget was 3 M steps; the committed `.env` says 100,000.
* Both have been corrected in this documentation set, and the corrections are itemised in
  [§ Configuration claims](11_code_audit.md#configuration-claims).
