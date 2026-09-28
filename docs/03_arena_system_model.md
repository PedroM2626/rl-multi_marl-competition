# 3. Arena system model

Source: `src/marl_arena/systems/simulation.py`, `src/marl_arena/systems/match_variant.py`,
`src/marl_arena/models.py`. The two experiment trees implement the same model; where they differ it is
stated.

## 3.1 Entities and geometry

The world is a rectangular arena on the \(XZ\) plane with \(Y\) up. All state is held in plain
dataclasses and NumPy arrays; there is no physics engine.

| Constant | Value | Meaning |
|---|---|---|
| `AGENT_RADIUS` | 0.58 | Horizontal half-extent used for collisions and arena clamping |
| `AGENT_HALF_HEIGHT` | 1.1 | Vertical half-extent of the agent AABB |
| `PROJECTILE_RADIUS` | 0.12 | Padding applied to every surface a projectile may hit |
| `PROJECTILE_SPEED_MULTIPLIER` | 2.5 | Projectile speed = \(2.5 \times\) weapon range |
| `FLOAT_EPSILON` | \(10^{-6}\) | Degeneracy tolerance in the segment/AABB solver |

An agent is an axis-aligned box of \(1.16 \times 2.2 \times 1.16\) centred on its position; a
projectile is a point inflated by `PROJECTILE_RADIUS`. Agents stand at \(y = 1.0\) on the ground.

**Nine agents**, three per team, are created in `reset_match()`:

```
agent_id  = f"{team_index+1}-{agent_index+1}"      # "1-1" … "3-3"
position  = team_spawn_center + AGENT_FORMATION_OFFSETS[agent_index]
heading   = rng.uniform(0, 360)                    # degrees, independent per agent
```

with formation offsets \((-1.7,0,0)\), \((1.7,0,0)\), \((0,0,1.7)\). Team spawn centres in the fixed
variant are at \((\mp m, 1, -m)\) and \((0, 1, m)\) where \(m = 0.32 \times \texttt{arena\_size}\)
(\(m = 10.24\) at the default size 32).

**Obstacles** are axis-aligned boxes, either static or oscillating:

\[
p(t) = p_{\text{base}} + \hat{a}\, A \sin(\phi + s t)
\]

where \(\hat a\) is the unit movement axis, \(A\) the amplitude, \(s\) the speed and \(\phi\) the phase
offset. Static obstacles have \(A = 0\). The default variant contains **nine** obstacles: three
`fixed_barrier`, two `moving_obstacle` (both sweeping horizontally), four `restricted_passage`.

## 3.2 Kinematics

Heading \(\theta\) (degrees) maps to a forward vector

\[
\hat f(\theta) = (\sin\theta,\ 0,\ \cos\theta)
\]

and one step of size \(\Delta t\) applies, in order:

1. **Turn.** \(\theta \leftarrow (\theta + u_{\text{turn}}\, v_{\text{turn}}\, \Delta t) \bmod 360\),
   with \(u_{\text{turn}} \in [-1,1]\) and \(v_{\text{turn}} =\) `agent_turn_speed` (110°/s default).
2. **Jump impulse.** If the policy commanded a jump and \(y \le 1.02\), then
   \(\dot y \leftarrow\) `jump_speed` (6.3).
3. **Translate.** \(p^+ = p + \hat f(\theta)\, u_{\text{move}}\, v_{\text{move}}\, \Delta t\), then
   resolve against obstacles and walls (§ 3.3).
4. **Gravity.** \(\dot y \leftarrow \dot y - g\,\Delta t\); \(y \leftarrow y + \dot y\,\Delta t\);
   clamp to \(y = 1.0\) and zero \(\dot y\) on landing.

Survival time accumulates by \(\Delta t\) for every agent still alive at the end of the step.

> Note on the renderer. Training steps with the fixed \(\Delta t =\) `SIM_STEP_DT` \(= 0.1\) s. The
> visual `main.py` calls `simulation.step(time.dt)` with the **real frame delta**, so the same policies
> experience a different time discretisation on screen. Results reported here never use the visual
> loop, but do not expect the window to reproduce a training match exactly.

## 3.3 Movement resolution

`_resolve_movement(current, desired)` is a deterministic cascade, evaluated in this order:

1. Clamp `desired` into the arena: \(|x|, |z| \le \tfrac{L}{2} - r_{\text{agent}}\).
2. If the clamped point overlaps no obstacle, accept it.
3. Try the same point with the \(z\) component restored to `current.z` (slide along \(x\)); accept if
   free.
4. Try the same point with \(x\) restored (slide along \(z\)); accept if free.
5. For each overlapping obstacle, push out along whichever axis has the smaller penetration; accept
   the first such correction that lands in free space.
6. Otherwise stay put.

Overlap is tested with an inflated AABB against a probe box of side \(2 \times 2.2\)
(`agent_clear_radius = 2.2` is used at spawn-sampling time; runtime uses `AGENT_RADIUS`).

Additionally, `_push_agents_out_of_obstacles()` runs **before** decisions each step: a
`moving_obstacle` can sweep into a standing agent, and the agent is ejected along the minimum-penetration
axis with a 0.05 margin. Agents are therefore never permanently trapped, but they can be shoved.

## 3.4 Combat

### 3.4.1 Firing

A shot is created only if all of the following hold at the decision step:

* the agent is alive and the action's `shoot` bit is set;
* at least one enemy is alive (`nearest_enemy` returns non-`None`);
* the nearest enemy is within `shoot_range` — this gate is applied in
  `action_to_decision`, so out-of-range shots are silently converted to `shoot = False`;
* the cooldown has elapsed: \(t_{\text{now}} - t_{\text{last shot}} \ge\) `shoot_cooldown` (0.45 s).

The projectile spawns at \(p + \hat d (r_{\text{agent}} + r_{\text{proj}} + 0.05)\) where \(\hat d\) is
the **normalised 3D direction toward the chosen target point**, not the agent's heading. Aiming is
therefore independent of facing; facing only affects movement. The spawn point is clamped inside the
arena.

### 3.4.2 Projectile lifetime

Speed is \(2.5 \times\) range and the maximum travelled distance is the range, so

\[
\tau_{\text{life}} = \frac{\text{range}}{2.5 \times \text{range}} = 0.4\ \text{s}
\]

**independently of the range value.** At \(\Delta t = 0.1\) s a projectile exists for exactly four
steps and covers 25 % of its range per step. Verified by direct computation:

```
$ python -c "r=20.0; v=r*2.5; print(r/v, r/v/0.1, v*0.1, v*0.1/r)"
0.4 4.0 5.0 0.25
```

The consequence for the results is important: weapon range controls *reach*, not *hang time*, and a
shot can only ever hit something in the first four steps after it is fired. Hit rates of 8–13 %
([§ Results](08_results.md)) are measured against that geometry.

### 3.4.3 Collision detection

Each step advances a projectile along the segment \(s \to e\) and finds the **earliest** intersection
among three candidate sets, using a slab test of a segment against an AABB
(`_segment_intersects_aabb`), which returns the entry parameter \(t \in [0,1]\) and the hit point:

* enemy agents — AABB inflated by `PROJECTILE_RADIUS`, excluding same-team agents and the shooter;
* obstacles — AABB inflated by `PROJECTILE_RADIUS`;
* arena boundary planes at \(|x|, |z| = L/2\).

Because the full segment is tested rather than the endpoint, a projectile cannot tunnel through an
agent or a wall between steps. Ties resolve in favour of whatever has the smaller \(t\); an agent hit
must beat the current best by more than `FLOAT_EPSILON` to supersede an obstacle.

Outcomes:

| Event | Effect |
|---|---|
| Agent hit | Target `alive = False`; shooter `kills += 1`, `hits += 1`; team `eliminations += 1`, `shots_hit += 1`; projectile consumed |
| Obstacle or wall hit | Team `shots_missed += 1`; a visual explosion is queued in `pending_obstacle_hits`; projectile consumed |
| Range exhausted | Team `shots_missed += 1`; projectile consumed |
| Degenerate state (non-finite or zero-norm vector) | `ValueError` caught per projectile, a warning is emitted, counted as a miss |

Friendly fire is impossible by construction. A projectile that kills its own shooter's would-be target
after the target is already dead cannot occur, because dead agents are excluded from the candidate set.

### 3.4.4 Miss attribution

`_register_projectile_miss` looks the shooter up by id. If the shooter has been removed from
`self.agents` — which cannot happen within a match, since the list is only rebuilt at reset — the miss
is dropped with a warning. The same defensive path exists for hits. These branches are unreachable in
normal operation and are therefore **not covered by the test suite**
([§ Code audit](11_code_audit.md#coverage)).

## 3.5 Episode structure and termination

A *match* is the episode. `step()` returns `True` on the step that ends it:

\[
\text{terminal} \iff \bigl|\{\text{teams with } \ge 1 \text{ living agent}\}\bigr| \le 1
\ \lor\ t \ge \texttt{match\_duration\_seconds}
\]

`finish_match()` then assigns the win:

```python
sorted_alive = sorted(team_alive.items(),
                      key=lambda kv: (kv[1], cumulative_metrics[kv[0]].eliminations),
                      reverse=True)
winner = sorted_alive[0][0]
```

Two properties of this rule matter for interpretation:

1. **A timeout awards the win to whoever is still standing**, even if nobody died. With the default
   90 s limit and a median recorded match length of 9.1 s (Experiment 1) / 7.6 s (Experiment 2),
   timeouts are rare but not absent: over 60 headless matches from random initialisation, 50 ended by
   wipeout and 10 by duration exhaustion.
2. **Ties are broken by *cumulative* eliminations across the whole run**, not by performance inside the
   tied match. A team that accumulated more kills earlier in training wins otherwise-flat ties. This is
   a persistent advantage bias whose size is not measured
   ([§ Threats to validity](10_threats_to_validity.md#102-construct-validity)).

## 3.6 Match variants and domain randomisation

`create_default_variant()` produces the fixed evaluation geometry. `sample_training_variant()`
produces a randomised one, drawing independently and uniformly:

| Quantity | Range |
|---|---|
| `arena_size` | 28 – 36 |
| `match_duration_seconds` | 60 – 120 |
| `agent_move_speed` | 3.5 – 5.5 |
| `agent_turn_speed` | 90 – 130 |
| `shoot_range` | 16 – 24 |
| `shoot_cooldown` | 0.35 – 0.6 |
| obstacle count | 5 – 10 |

Obstacle types are drawn from a pool weighted 2 : 1 : 1 toward `fixed_barrier` and
`restricted_passage`; positions are rejection-sampled inside \(\pm 0.42 L\), rejecting overlaps with an
0.8 margin and anything within \(|x| < 2 \wedge |z| < 2\) of the centre. Team spawns are chosen from
seven candidate corners with jitter \(\pm 2\), requiring pairwise separation \(> 0.38 L\) and a clear
2.2-radius probe; if fewer than three survive 60 attempts, the remainder are placed uniformly in
\(\pm 6\), which **can** land inside an obstacle — the runtime ejection in § 3.3 is what keeps that
playable.

A new variant is sampled at every `reset_match()`, so consecutive matches differ in geometry, and the
variant id is recorded in `team_match_metrics.csv` for post-hoc stratification.

## 3.7 Randomness and reproducibility

Three independent streams exist:

| Stream | Seeded from | Used for |
|---|---|---|
| `self.rng` (`random.Random`) | `RANDOM_SEED` = 7 | Variant sampling, initial headings, controller jitter |
| `self.np_rng` (`numpy.random.default_rng`) | same | **created but never used** |
| per-controller `self.rng` | `seed + offset`, offsets 11 / 23 / 37 | The random jump decision |
| `np.random.shuffle` inside PPO | global NumPy state, **never seeded** | Minibatch order |

The last row is the one that breaks exact reproducibility: two runs with `RANDOM_SEED=7` still diverge
in minibatch order. See [§ Reproducibility](09_reproducibility.md#96-what-is-and-is-not-reproducible).
