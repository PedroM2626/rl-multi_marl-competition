# 12. Glossary, symbols and references

## 12.1 Terminology

| Term | Meaning in this repository |
|---|---|
| **Arm** | One architecture under comparison. Six arms total: CTE, DTE, CTDE, CTDE-VD, CTDE-CAC, CTDE-Comm. |
| **Team** | The slot an arm occupies in a match: `Team 1`, `Team 2`, `Team 3`. Renamed from `Equipe N` on 2026-09-27 ([provenance](09_reproducibility.md#localisation-rename)). |
| **Paradigm** | Synonym for arm as stored in the `paradigm` column of every artefact. |
| **Match** | One episode, terminated by wipeout or duration. The unit of a PPO update. |
| **Recorded match** | A match written to the CSVs — one in every `RL_METRICS_EVERY_MATCHES = 10`. |
| **Cumulative match** | Any match played; the denominator behind `summary.json`. |
| **Variant** | One geometry/physics configuration. Fixed for evaluation, sampled per match during training. |
| **Waypoint action** | The 8-element discrete action: one of four target roles × a shoot bit. Movement and aiming are executed by a hand-written controller on top of it. |
| **Slot vector** | One-hot encoding of which of a team's three agents is acting, used by CTE and CTDE-Comm. |
| **Global observation** | The 36-float concatenation of per-agent \([p_x, p_z, \theta/180, \mathbb{1}_{\text{alive}}]\) for all nine agents. |
| **Local observation** | The 8-float per-agent vector of [§ 4.1](04_mdp_formalisation.md#41-local-observation-o_i). |
| **Wipeout** | A match ending because at most one team has living agents. |

## 12.2 Symbols

| Symbol | Definition | Where |
|---|---|---|
| \(o_i\) | 9-dim local observation of agent \(i\) | [§ 4.1](04_mdp_formalisation.md#41-local-observation-o_i) |
| \(s\) | 45-dim global observation | [§ 4.2](04_mdp_formalisation.md#42-global-observation-s) |
| \(a\) | Action index in \(\{0,\dots,7\}\) | [§ 4.3](04_mdp_formalisation.md#43-action-space) |
| \(\pi_\theta\) | Stochastic categorical policy | [§ 6.1](06_optimisation_procedure.md#61-objective) |
| \(V_\phi(s)\) | Critic value estimate | [§ 5.3](05_network_architectures.md#53-centralizedcriticnetwork--joint-state-critic) |
| \(r_i(t)\) | Reward of agent \(i\) at step \(t\) | [§ 4.4](04_mdp_formalisation.md#44-reward) |
| \(\gamma\) | Discount factor, 0.99 | [§ 6.7](06_optimisation_procedure.md#67-hyperparameters-as-trained) |
| \(\lambda\) | GAE trace coefficient, 0.95 | [§ 6.2](06_optimisation_procedure.md#62-advantage-estimation) |
| \(\epsilon\) | PPO clip range, 0.2 | [§ 6.1](06_optimisation_procedure.md#61-objective) |
| \(c_v,\ c_H\) | Value and entropy loss coefficients, 0.5 / 0.01 | [§ 6.1](06_optimisation_procedure.md#61-objective) |
| \(\hat A_t\) | Normalised GAE advantage | [§ 6.2](06_optimisation_procedure.md#62-advantage-estimation) |
| \(\hat f(\theta)\) | Forward unit vector from heading | [§ 3.2](03_arena_system_model.md#32-kinematics) |
| \(L\) | Arena side length | [§ 3.1](03_arena_system_model.md#31-entities-and-geometry) |
| \(W, E, H, X, S, M\) | Wins, eliminations, hits, misses, survival sum, matches played | [§ 7.4](07_experimental_protocol.md#74-metric-definitions) |
| \(m_k,\ c_i\) | CommNet message and pooled channel | [§ 5.5](05_network_architectures.md#55-commactornetwork--one-round-differentiable-communication-ctde-comm) |

## 12.3 Module map

| Path | Responsibility | Documented in |
|---|---|---|
| `main.py` | Ursina 3D front end; loads checkpoints, runs the visual loop, records metrics per match | [A-23](11_code_audit.md#renderer-timestep) |
| `scripts/train_rl.py` | Headless training loop, checkpoint and log cadence; MLflow in experiment 2 | [§ 7.2](07_experimental_protocol.md#72-procedure) |
| `scripts/plot_metrics.py` | Standalone summary → CSV/PNG using pandas | [A-22](11_code_audit.md#missing-dependency) |
| `src/marl_arena/config.py` | Frozen dataclass reading every tunable from `.env`; creates the data directories | [§ 6.7](06_optimisation_procedure.md#67-hyperparameters-as-trained) |
| `src/marl_arena/models.py` | Dataclasses: snapshots, `TeamMetrics.as_summary`, `MatchResult` | [§ 7.4](07_experimental_protocol.md#74-metric-definitions) |
| `src/marl_arena/systems/match_variant.py` | `TEAM_NAMES`, `PARADIGM_CYCLE`, `team_meta(rotation)`, obstacle specs, fixed and randomised variant sampling | [§ 3.6](03_arena_system_model.md#36-match-variants-and-domain-randomisation) |
| `src/marl_arena/systems/simulation.py` | The engine: kinematics, combat, collision, termination, trajectory recording | chapters 3 and 4 |
| `src/marl_arena/systems/metrics.py` | CSV append with schema rotation, `summary.json`, dashboard trigger | [§ 7.5](07_experimental_protocol.md#75-artefact-schemas) |
| `src/marl_arena/systems/plotting.py` | Cumulative four-panel comparison from the team CSV | [A-19](11_code_audit.md#plot-filter) |
| `src/marl_arena/controllers/base.py` | Feature builders, waypoint heuristics, `angle_to_target` | [A-1](11_code_audit.md#turn-control-defect) |
| `src/marl_arena/controllers/rl_controller.py` | One `RLTeamController` per team: network construction, action dispatch, rollout recording, episode finish | chapters 5 and 6 |
| `src/marl_arena/rl/networks.py` | The five live network classes plus two empty aliases | chapter 5 |
| `src/marl_arena/rl/buffer.py` | `RolloutStep`, `RolloutBuffer`, GAE, tensor conversion | [§ 6.2](06_optimisation_procedure.md#62-advantage-estimation) |
| `src/marl_arena/rl/ppo.py` | `PPOTrainer` and the four update paths | [§ 6.4](06_optimisation_procedure.md#64-the-three-update-paths) |
| `src/marl_arena/rl/actions.py` | Action parsing, decision materialisation, checkpoint I/O | [A-25](11_code_audit.md#unsafe-checkpoint-load) |
| `src/marl_arena/ui/dashboard.py` | Builds the in-game overlay string | — |

## References

Only works cited in this documentation set. ArXiv identifiers are given where the preprint is the
conventionally cited version. The bibliographic details below were written from knowledge of the
literature rather than verified against the publishers during this session; they should be checked
before being quoted in a formal publication.

**Policy optimisation and advantage estimation**

* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). *Proximal Policy Optimization
  Algorithms*. arXiv:1707.06347.
* Schulman, J., Moritz, P., Levine, S., Jordan, M. I., Abbeel, P. (2016). *High-Dimensional Continuous
  Control Using Generalized Advantage Estimation*. ICLR. arXiv:1506.02438.
* Schulman, J., Levine, S., Abbeel, P., Jordan, M. I., Moritz, P. (2015). *Trust Region Policy
  Optimization*. ICML. arXiv:1502.05477.
* Mnih, V., Badia, A. P., Mirza, M. et al. (2016). *Asynchronous Methods for Deep Reinforcement
  Learning* (A3C). ICML. arXiv:1602.01545.
* Pascanu, R., Mikolov, T., Bengio, Y. (2013). *On the difficulty of training recurrent neural
  networks* — the standard analysis behind gradient-norm clipping. ICML.

**Multi-agent paradigms**

* Lowe, M., Wu, Y., Tamar, A., Harl, J., Abbeel, P., Levine, I. (2017). *Multi-Agent Actor-Critic for
  Mixed Cooperative-Competitive Environments* (MADDPG). NeurIPS. arXiv:1706.02268.
* Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Wu, Y. (2022). *The Surprising
  Effectiveness of PPO in Cooperative, Multi-Agent Games* (MAPPO). NeurIPS. arXiv:2103.01955.
* Papoudakis, G., Christianos, F., Albrecht, S. V. (2021). *Cooperative Multi-Agent
  Reinforcement Learning: The Centralized Training with Decentralized Execution Paradigm — A Short
  Survey*. arXiv:2103.04932.
* Bernstein, D. S., Givan, R., Immerman, N., Zilberstein, D. (2002). *The Complexity of Decentralized
  Control of Markov Decision Processes*. Mathematics of Operations Research 27(4), 817–840.
* Oliehoek, F. A., Amato, C. (2016). *A Concise Introduction to Decentralized POMDPs*. Springer.
* Tan, M. (1993). *Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents*. ICML.
* de Witt, C. S. et al. (2020). *Is Independent Learning
  All You Need in the StarCraft Multi-Agent Challenge?* arXiv:2011.09533.
* Hernandez-Leal, P., Kartal, B., Taylor, M. E. (2019). *A survey and critique of multiagent deep
  reinforcement learning*. Autonomous Agents and Multi-Agent Systems 33. arXiv:1807.05875.

**Value decomposition**

* Sunehag, P., Lever, G., Gruslys, A. et al. (2018). *Value-Decomposition Networks For Cooperative
  Multi-Agent Learning*. AAMAS. arXiv:1706.05296.
* Rashid, T., Samvelyan, M., de Witt, C. S., Farquhar, G., Foerster, J., Whiteson, S. (2018). *QMIX:
  Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning*. ICML.
  arXiv:1803.11485.
* Wang, T., Yuan, J., Ren, X., Lin, W., Zhang, J. (2020). *QTRAN: Learning to Factorize with
  Transformation for Cooperative Multi-Agent Reinforcement Learning*. ICML. arXiv:1902.09405.
* Foerster, J., Farquhar, G., Afouras, T., Nardelli, N., Whiteson, S. (2018). *Counterfactual
  Multi-Agent Policy Gradients* (COMA). AAAI. arXiv:1705.08926.

**Learned communication**

* Sukhbaatar, S., Szlam, A., Fergus, R. (2016). *Learning Multiagent Communication with Backpropagation*
  (CommNet). NeurIPS. arXiv:1605.07736.
* Foerster, J., Assael, I. A., de Freitas, N., Abbeel, D. (2016). *Learning to Communicate with Deep
  Multi-Agent Reinforcement Learning* (DIAL). NeurIPS. arXiv:1605.06607.

**Benchmarks**

* Samvelyan, M., Rashid, T., de Witt, C. S. et al. (2019). *The StarCraft Multi-Agent Challenge*.
  AAMAS. arXiv:1902.09083.

**Domain randomisation**

* Tobin, J., Fong, R., Schneider, J., Zaremba, W. (2017). *Domain randomization for transferring deep
  neural networks from simulation to the real world*. IROS. arXiv:1703.06907.
* Peng, X. B., Andrychowicz, M., Zaremba, W., Abbeel, P. (2018). *Sim-to-Real Transfer of Robotic
  Control with Dynamics Randomization*. ICRA. arXiv:1710.06537.

**Exploration in multi-agent settings**

* Mahajan, A., Rashid, T., Samvelyan, M., Whiteson, S. (2019). *MAVEN: Multi-Agent Variational
  Exploration*. NeurIPS. arXiv:1907.04543.
