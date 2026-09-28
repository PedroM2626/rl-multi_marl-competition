# 2. Related work

This chapter positions the two experiments in the MARL literature. It is a positioning exercise, not a
survey: the point is to make clear which claims this repository is *testing*, which it is *assuming*,
and which it is *not* entitled to make because the published versions of those claims were established
on different environments.

Full bibliographic entries are in [§ References](12_glossary_and_references.md#references).

## 2.1 The decentralised-execution problem

A team of agents that each observe only part of the world is formally a decentralized partially
observed Markov decision process (Dec-POMDP). Optimal control of a Dec-POMDP is NEXP-complete
[Bernstein et al., 2002](12_glossary_and_references.md#references), which is the theoretical reason
practical MARL restricts itself to policy classes that do not maintain a joint belief state.

Two families of restriction dominate:

* **Centralised training, decentralised execution (CTDE).** Learn with access to a global critic, but
  constrain each actor to depend only on its own observation, so the deployed policy is a set of
  independent controllers. Formalised for continuous control by MADDPG
  ([Lowe et al., 2017](12_glossary_and_references.md#references)), which is the template Experiment 1's
  "CTDE" follows.
* **Value decomposition.** Constrain the joint value function to a factored form over per-agent
  utilities, so that greedy per-agent action selection is consistent with joint optimisation. VDN
  ([Sunehag et al., 2018](12_glossary_and_references.md#references)) uses an additive factorisation;
  QMIX ([Rashid et al., 2018](12_glossary_and_references.md#references)) relaxes it to monotonic;
  QTRAN ([Wang et al., 2020](12_glossary_and_references.md#references)) removes the monotonicity
  assumption at the cost of consistency regularisers.

A review of the paradigm and its failure modes is
[Papoudakis et al., 2021](12_glossary_and_references.md#references).

## 2.2 Independent learning as the control condition

Training each agent as a single-agent learner against a non-stationary environment — "independent
learners" — goes back to [Tan (1993)](12_glossary_and_references.md#references), who found that
cooperative information sharing helps in some gridworlds and not others. In the StarCraft Multi-Agent
Challenge, independent PPO proved a much stronger baseline than originally reported
[de Witt et al., 2020](12_glossary_and_references.md#references). Experiment 1's **DTE** arm is this
control condition: local actor, local value head, no global critic. Its inclusion is deliberate — the
hypothesis being tested is not "CTDE beats nothing" but "a global critic buys win rate over a local
value head at matched capacity".

The non-stationarity that makes independent learning fragile is a consequence of every team improving
simultaneously; see [Hernandez-Leal et al., 2019](12_glossary_and_references.md#references) for a
critique. This repository does nothing about it: there is no replay buffer shared across teams, no
fictitious self-play, and no population of past policies. All three teams chase each other's moving
targets within one run.

## 2.3 The three CTDE variants of Experiment 2

| Variant | Nearest published method | What is shared with it | What differs here |
|---------|--------------------------|------------------------|-------------------|
| **CTDE-VD** | VDN ([Sunehag et al., 2018](12_glossary_and_references.md#references)) | Additive decomposition of team value into per-agent terms | VDN decomposes *action-value* \(Q_i(o_i,a_i)\) over discrete actions; here the decomposition is over *state-value* terms \(V_i\) and the per-agent input is a slice of the global state, not a local observation |
| **CTDE-CAC** | MAPPO ([Yu et al., 2022](12_glossary_and_references.md#references)) | Per-agent PPO actors, one centralised critic over the joint state | MAPPO parameter-shares across agents and normalises returns with running averages; neither is done here |
| **CTDE-Comm** | CommNet ([Sukhbaatar et al., 2016](12_glossary_and_references.md#references)) | Continuous, differentiable, average-pooled messages trained end-to-end with the policy | CommNet stacks \(d\) communication *rounds*; here there is exactly one round, and the message is produced from a partial team observation vector |

The one-round, mean-pooled form is the important simplification. With a single round, the channel
\(c_i = \frac{1}{|N(i)|}\sum_{j \ne i} m_j\) is commutative and associative by construction and cannot
propagate information more than one hop. Multi-hop structure — which is where communication tends to
pay off — is not representable. See
[§ Architectures](05_network_architectures.md#commactornetwork) for the exact algebra implemented.

The alternative family of learned communication is discrete signalling trained with REINFORCE-style
gradients, e.g. DIAL ([Foerster et al., 2016](12_glossary_and_references.md#references)) and the
targeted-dropout variant TarMAC. This repository uses the continuous, backpropagable branch.

## 2.4 Credit assignment

VDN-style decomposition is motivated by credit assignment: with a single scalar team reward, an agent
needs to know which of its own actions earned it. Counterfactual methods attack the same problem
differently — COMA ([Foerster et al., 2018](12_glossary_and_references.md#references)) uses a
counterfactual baseline that marginalises the acting agent's own action out of a centralised critic.

Here, credit assignment is handled by the *reward function alone*: a hit gives +1.2 to the shooter and
−1.0 to the victim, so an agent's own contribution is directly observable in its own reward. That makes
the VD-vs-CAC comparison in Experiment 2 less sharp than it would be under a shared team-only reward,
because the incentive to solve credit assignment is largely removed by the design
([§ Reward](04_mdp_formalisation.md#44-reward)). This is a protocol caveat, not a defect — but it is the
reason the VD–CAC gap in the historical single-seed run (2.2 points of win rate) should not be read as
evidence about decomposition in general. The study's own VD–CAC comparison is in
[§ Results](08_results.md#replicated-study).

## 2.5 Optimisation algorithm

All six policies are trained with PPO
([Schulman et al., 2017](12_glossary_and_references.md#references)) with clipped surrogate objectives,
an entropy bonus in the A3C style ([Mnih et al., 2016](12_glossary_and_references.md#references)),
gradient-norm clipping (analysed in [Pascanu et al., 2013](12_glossary_and_references.md#references)),
and advantages from GAE ([Schulman et al., 2016](12_glossary_and_references.md#references)). PPO
replaces the trust-region constraint of TRPO
([Schulman et al., 2015](12_glossary_and_references.md#references)) with a first-order clip, which is
what makes the shared-optimizer-per-team structure in [§ Optimisation](06_optimisation_procedure.md)
straightforward.

Two implementation facts limit how strongly the results can be called "PPO": the approximate KL used
for early stopping is never computed (the field is hard-coded to zero), and the advantage baseline is
bootstrapped once per *match* rather than from a fixed-length rollout. Details and consequences in
[§ What the implementation does not do](06_optimisation_procedure.md#65-what-the-implementation-does-not-do).

## 2.6 Domain randomisation

Training randomises arena size, duration, speeds, weapon range and cooldown, and obstacle count. This
is the standard sim-to-robustness device of
[Tobin et al., 2017](12_glossary_and_references.md#references) and
[Peng et al., 2018](12_glossary_and_references.md#references). In this repository the motivation is not
transfer — there is no real robot — but regularisation: without it, a policy can memorise the fixed
default geometry. The evaluation protocol, however, uses the *fixed* variant, so randomisation is a
training-time treatment whose effect is never measured
([§ Threats to validity](10_threats_to_validity.md#102-construct-validity)).

## 2.7 What is therefore novel here, and what is not

**Not novel.** The arena is bespoke and its win rates are not comparable to any published benchmark;
the algorithms are textbook; none of the six architectures introduces a methodological contribution.
SMAC ([Samvelyan et al., 2019](12_glossary_and_references.md#references)) and its successors already
provide the standard comparative venue.

**The useful part.** Placing CTE, DTE, CTDE, VDN, CAC and CommNet under one reward, one observation
encoding, one optimiser and one engine turns a cross-paper comparison into a within-engine one, at the
cost of external validity. That is a legitimate pedagogical and diagnostic instrument — and, given the
defects catalogued in [§ Code audit](11_code_audit.md), an example of why within-engine comparisons
must be audited before the ranking they produce is trusted.
