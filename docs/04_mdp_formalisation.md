# 4. MDP formalisation

Each team is treated as one learning entity with three parameter-sharing-free agents; each agent
selects an action from the same policy network. Formally this is a partially observed Markov game with
nine players partitioned into three coalitions, but the implementation reduces it to three PPO agents
that each optimise a per-team objective.

## 4.1 Local observation \(o_i\)

`BaseTeamController.build_local_features` returns an 8-dimensional vector
(`LOCAL_OBS_DIM = 8`):

| # | Component | Definition | Units / range |
|---|---|---|---|
| 1 | \(p_x\) | own position, x | metres, \(\pm L/2\) |
| 2 | \(p_z\) | own position, z | metres, \(\pm L/2\) |
| 3 | \(\theta\) | heading / 180 | \([0, 2)\) |
| 4 | \(\Delta^{e}_{x}\) | nearest living enemy x − own x | metres |
| 5 | \(\Delta^{e}_{z}\) | nearest living enemy z − own z | metres |
| 6 | \(d^{e}\) | \(\lVert\)enemy − own\(\rVert_2\) (3D) | metres |
| 7 | \(\Delta^{a}_{x}\) | ally centroid x − own x | metres |
| 8 | \(\Delta^{a}_{z}\) | ally centroid z − own z | metres |

Three properties of this encoding shape everything downstream:

* **Obstacles are not observable.** No component encodes wall or obstacle geometry. Agents can only
  infer obstacles from being blocked or shoved, which is not in the state either. Path planning around
  cover is therefore impossible to learn; the arena's cover is, from the policy's point of view,
  invisible.
* **Only the *nearest* enemy is represented.** The other five enemies are absent from \(o_i\).
  "Decentralised execution" here means decentralised *and* strongly partial.
* **Scale is raw metres.** Components 1, 2, 4–8 are unnormalised, so their magnitude grows with arena
  size (28–36 under randomisation) and the network sees a drifting input distribution. Component 3 is
  divided by 180 but heading lives in \([0, 360)\), so it occupies \([0, 2)\) rather than \([-1, 1]\)
  and is discontinuous at the wrap — a heading of 359° reads as 1.994 and 1° as 0.017.

## 4.2 Global observation \(s\)

`build_global_features` returns 36 floats (`GLOBAL_OBS_DIM = 36`): the concatenation, over all nine
agents sorted lexicographically by `agent_id` (`"1-1" … "3-3"`), of

\[
\bigl[\,p_x,\ p_z,\ \theta/180,\ \mathbb{1}[\text{alive}]\,\bigr].
\]

The critic therefore sees positions, facing, and life status of everyone — but **not** velocities,
not projectiles in flight, not cooldowns, and not obstacles. The action taken by each agent is also
absent. This is a weak "state": it is a snapshot of the nine agents only, so the value function
cannot condition on the most immediate predictor of danger, an incoming projectile.

The lexicographic ordering is what makes the fixed index slices in the VDN critic
([§ Architectures](05_network_architectures.md#54-valuedecompositioncriticnetwork--additive-per-agent-critic-ctde-vd)) land on the right
agents. It holds because team and slot indices are single digits; it would break at ten agents per
team.

## 4.3 Action space

The policy emits one of `NUM_ACTIONS = 8` discrete actions, decoded as a pair
(`parse_action`):

\[
a \in \{0,\dots,7\},\qquad
\text{target} = \lfloor a / 2 \rfloor,\quad
\text{shoot} = a \bmod 2 .
\]

The target index selects one of four **waypoints** from `candidate_targets`:

| \(a/2\) | Role | Waypoint |
|---|---|---|
| 0 | `engage` | nearest enemy's position |
| 1 | `flank` | nearest enemy + \((3.5, 0, -3.5)\) |
| 2 | `support` | ally centroid |
| 3 | `evade` | own position − \(4 \cdot \hat{d}_{\text{enemy}}\) |

The waypoint is then converted to a `StepDecision` by a **fixed, hand-written controller**
(`make_decision_from_target`):

```
direction      = normalise(target − position)
turn_delta     = angle_to_target(position, heading_deg, target)
turn           = clip(turn_delta / 35, −1, 1)
move           = 1.0 if ‖target − position‖ > 1.4 else 0.0
jump           = (|turn_delta| < 20°) and (rng.random() < 0.015)
shoot          = shoot_bit and (distance to nearest enemy ≤ shoot_range)
aim_direction  = direction
```

This is the single most consequential design fact in the repository, so it is worth stating plainly:

> **The learned policy controls two things: which of four waypoints to pursue, and whether to fire.**
> Locomotion, aiming and shooting are executed by hand-written heuristics on top of that choice.

Three specific consequences:

1. **No backward motion.** `move ∈ {0.0, 1.0}`; the agent can advance or stop, never reverse. Turning
   is the only way to change direction of travel.
2. **Jump is not in the action space.** It fires from a 1.5 % coin flip conditioned on the turn error,
   so it is uncontrollable noise, not a learned skill. The `JUMP_SPEED` / `GRAVITY` configuration and
   the vertical dynamics of [§ 3.2](03_arena_system_model.md#32-kinematics) are therefore almost
   entirely decorative.
3. **Aiming bypasses facing.** `aim_direction` is the straight-line vector to the waypoint, so a shot
   travels toward the target regardless of where the agent points. Hit probability depends on the
   projectile geometry of [§ 3.4](03_arena_system_model.md#34-combat), not on marksmanship.

### 4.3.1 The turn control defect

`angle_to_target` computes the bearing as `atan2(offset[0], offset[1])`. The second argument is the
**vertical** component, which is ≈ 0 for all on-ground pairs; the intended argument is `offset[2]`
(the \(z\) component). The returned delta is therefore driven by \(\operatorname{atan2}(\Delta x,
\approx 0) \in \{-90^\circ, +90^\circ\}\) and is almost independent of the actual bearing.

Measured directly against the correct bearing:

```
target=( -10.0,1.0,  5.0)  delta returned= -90.00   true heading-to-target= -63.43
target=(   0.0,1.0,  5.0)  delta returned=   0.00   true heading-to-target=   0.00
target=(  10.0,1.0,  5.0)  delta returned=  90.00   true heading-to-target=  63.43
target=( -10.0,1.0, 20.0)  delta returned= -90.00   true heading-to-target= -26.57
target=(  10.0,1.0, 20.0)  delta returned=  90.00   true heading-to-target=  26.57
```

The controller saturates `turn` to a constant \(\pm 1\) for any target off the \(z\) axis, so agents
spin toward a fixed absolute heading instead of toward their waypoint. The full analysis and the reason
it was **not** patched are in
[§ Turn control defect](11_code_audit.md#turn-control-defect).

## 4.4 Reward

`ArenaSimulation._reward_for_agent` is evaluated once per alive agent per step, after projectiles have
resolved:

\[
r_i(t) =
\underbrace{\begin{cases} +0.015 & \text{if } \text{alive}_i(t) \\ -0.5 & \text{otherwise}\end{cases}}_{\text{survival / death}}
\;+\; \underbrace{1.2\,\mathbb{1}[\text{scored a hit this step}]}_{\text{elimination}}
\;-\; \underbrace{1.0\,\mathbb{1}[\text{was hit this step}]}_{\text{being eliminated}}
\]

Verified by direct evaluation:

| alive | was hit | scored | \(r\) |
|---|---|---|---|
| yes | no | no | **+0.015** |
| no | no | no | **−0.500** |
| yes | no | yes | **+1.215** |
| yes | yes | no | **−0.985** |
| yes | yes | yes | **+0.215** |

Notes on the shape of this function:

* A dead agent produces **no** transitions at all — `step()` only records transitions for agents that
  had a decision this step — so the −0.5 death penalty is applied exactly once, on the step the agent
  died, and the −1.0 "was hit" term is applied to the victim on that same step. The victim of a kill
  therefore receives −1.5 in one step.
* Missing a shot is **free**. There is no penalty for firing and no reward for a hit that does not
  eliminate, so the only gradient pressure on shooting comes from the +1.2 elimination bonus. With
  accuracy between 8 % and 13 %, the expected return per shot is \(0.1 \times 1.2 \approx 0.12\) — a
  small, high-variance signal.
* There is no team-level reward, no win bonus, and no shaping term for position or cover.

### 4.4.1 Reward scale versus horizon

With \(\gamma = 0.99\) and \(\Delta t = 0.1\) s, the effective horizon is
\(1/(1-\gamma) = 100\) steps \(= 10\) s. The maximum discounted alive bonus an agent can collect is

\[
\sum_{t=0}^{\infty} 0.015\,\gamma^{t} = \frac{0.015}{1-\gamma} = 1.5 ,
\]

against +1.2 for a single elimination. **Staying alive is worth more discounted return than killing**,
and the two terms are the same order of magnitude. Since mean survival is 7.5 – 10.2 s across all six
teams ([§ Results](08_results.md)) — i.e. almost exactly the discount horizon — the return is
dominated by the survival term.

This is a genuine confound for both research questions: the architectures under comparison differ in
how well they *estimate value*, but the measured outcome is largely a function of *how long agents
stood around*, which the alive bonus rewards directly regardless of what the critic looks like.

## 4.5 Episode and transition bookkeeping

The per-step pipeline in `ArenaSimulation.step(dt)` is:

1. advance obstacles;
2. snapshot all agents → `snapshots` (pre-decision);
3. for each **alive** agent, `controller.decide(agent.snapshot(), snapshots, context)`;
4. apply turn / jump / move, integrate gravity, accumulate survival time;
5. fire allowed shots (cooldown-gated);
6. advance projectiles, which mutates `alive`, `kills`, `hits`, `misses` and fills `hit_status` /
   `scored_status`;
7. build `TransitionRecord`s per team and call `controller.update(transitions)`;
8. append a trajectory row;
9. return whether the match is over.

`TransitionRecord` carries `state_features`, `action_features`, `reward`, `next_state_features` and
`done`. Two facts about it:

* **The recorded state pair is degenerate.** Both `state_features` and `next_state_features` are
  computed from `controller.build_local_features(agent.snapshot(), post_snapshots)` *after* the step,
  so they are the same vector. Measured over 1,593 sampled transitions: **100.0 % identical**.
* **The degeneracy is inert.** `RLTeamController.update()` reads only `transition.reward` and
  `transition.done`; the observation actually trained on was captured earlier, in `_record_step`, from
  the *pre-step* observation. So the learning signal uses the correct \(o_t\), and the two 8-element
  arrays in the record are computed, stored, and discarded — wasted work and a memory leak
  ([§ Code audit](11_code_audit.md#unbounded-transition-retention)).

`action_features` is `[move, turn, shoot_bit, jump_bit]`, recorded but never used as a training target.
