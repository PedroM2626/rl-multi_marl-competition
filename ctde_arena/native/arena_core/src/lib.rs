use pyo3::prelude::*;

type V3 = (f64, f64, f64);

const PROJECTILE_RADIUS: f64 = 0.12;
const AGENT_HALF_HEIGHT: f64 = 1.1;
const PROJECTILE_EPSILON: f64 = 1e-6;

#[inline]
fn sub(a: V3, b: V3) -> V3 {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

#[inline]
fn add_scaled(a: V3, b: V3, scale: f64) -> V3 {
    (a.0 + b.0 * scale, a.1 + b.1 * scale, a.2 + b.2 * scale)
}

#[inline]
fn clamp_position(position: V3, half: f64) -> V3 {
    (
        position.0.max(-half).min(half),
        position.1,
        position.2.max(-half).min(half),
    )
}

#[inline]
fn obstacle_blocked(position: V3, obstacles: &[(V3, V3)], radius: f64) -> bool {
    obstacles.iter().any(|(center, size)| {
        (position.0 - center.0).abs() <= size.0 * 0.5 + radius
            && (position.2 - center.2).abs() <= size.2 * 0.5 + radius
    })
}

fn resolve_movement(current: V3, desired: V3, half: f64, obstacles: &[(V3, V3)], radius: f64) -> V3 {
    let desired = clamp_position(desired, half);
    if !obstacle_blocked(desired, obstacles, radius) {
        return desired;
    }

    let slide_x = clamp_position((desired.0, desired.1, current.2), half);
    if !obstacle_blocked(slide_x, obstacles, radius) {
        return slide_x;
    }

    let slide_z = clamp_position((current.0, desired.1, desired.2), half);
    if !obstacle_blocked(slide_z, obstacles, radius) {
        return slide_z;
    }

    for (center, size) in obstacles {
        let dx = desired.0 - center.0;
        let dz = desired.2 - center.2;
        let overlap_x = size.0 * 0.5 + radius - dx.abs();
        let overlap_z = size.2 * 0.5 + radius - dz.abs();
        if overlap_x > 0.0 && overlap_z > 0.0 {
            let corrected = if overlap_x < overlap_z {
                (desired.0 + if dx >= 0.0 { overlap_x } else { -overlap_x }, desired.1, desired.2)
            } else {
                (desired.0, desired.1, desired.2 + if dz >= 0.0 { overlap_z } else { -overlap_z })
            };
            let corrected = clamp_position(corrected, half);
            if !obstacle_blocked(corrected, obstacles, radius) {
                return corrected;
            }
        }
    }
    current
}

#[pyfunction]
fn push_agents_out(
    positions: Vec<V3>,
    alive: Vec<bool>,
    obstacle_positions: Vec<V3>,
    obstacle_sizes: Vec<V3>,
    arena_size: f64,
    agent_radius: f64,
) -> PyResult<Vec<V3>> {
    if positions.len() != alive.len() || obstacle_positions.len() != obstacle_sizes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("mismatched simulation array lengths"));
    }
    let half = arena_size * 0.5 - agent_radius;
    let obstacles: Vec<(V3, V3)> = obstacle_positions.into_iter().zip(obstacle_sizes).collect();
    let mut corrected = positions.clone();
    for (index, position) in positions.iter().copied().enumerate() {
        if !alive[index] {
            continue;
        }
        if position.0 >= -half
            && position.0 <= half
            && position.2 >= -half
            && position.2 <= half
            && !obstacle_blocked(position, &obstacles, agent_radius)
        {
            continue;
        }

        let result = resolve_movement(position, position, half, &obstacles, agent_radius);
        let close = (0..3).all(|axis| {
            let (a, b) = match axis {
                0 => (result.0, position.0),
                1 => (result.1, position.1),
                _ => (result.2, position.2),
            };
            (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
        });
        if close {
            let mut pushed = position;
            for (center, size) in &obstacles {
                let dx = pushed.0 - center.0;
                let dz = pushed.2 - center.2;
                let overlap_x = size.0 * 0.5 + agent_radius - dx.abs();
                let overlap_z = size.2 * 0.5 + agent_radius - dz.abs();
                if overlap_x > 0.0 && overlap_z > 0.0 {
                    if overlap_x < overlap_z {
                        pushed.0 += if dx >= 0.0 { overlap_x + 0.05 } else { -(overlap_x + 0.05) };
                    } else {
                        pushed.2 += if dz >= 0.0 { overlap_z + 0.05 } else { -(overlap_z + 0.05) };
                    }
                    pushed = clamp_position(pushed, half);
                }
            }
            corrected[index] = pushed;
        } else {
            corrected[index] = result;
        }
    }
    Ok(corrected)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn move_agents(
    positions: Vec<V3>,
    headings: Vec<f64>,
    vertical_velocities: Vec<f64>,
    alive: Vec<bool>,
    moves: Vec<f64>,
    turns: Vec<f64>,
    jumps: Vec<bool>,
    obstacle_positions: Vec<V3>,
    obstacle_sizes: Vec<V3>,
    arena_size: f64,
    agent_radius: f64,
    move_speed: f64,
    turn_speed: f64,
    dt: f64,
    gravity: f64,
    jump_speed: f64,
) -> PyResult<(Vec<V3>, Vec<f64>, Vec<f64>)> {
    let n = positions.len();
    if headings.len() != n || vertical_velocities.len() != n || alive.len() != n
        || moves.len() != n || turns.len() != n || jumps.len() != n
        || obstacle_positions.len() != obstacle_sizes.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err("mismatched simulation array lengths"));
    }
    let half = arena_size * 0.5 - agent_radius;
    let obstacles: Vec<(V3, V3)> = obstacle_positions.into_iter().zip(obstacle_sizes).collect();
    let mut next_positions = positions.clone();
    let mut next_headings = headings.clone();
    let mut next_vertical = vertical_velocities.clone();

    for index in 0..n {
        let mut position = positions[index];
        if alive[index] {
            let heading = (headings[index] + turns[index] * turn_speed * dt).rem_euclid(360.0);
            next_headings[index] = heading;
            if jumps[index] && position.1 <= 1.02 {
                next_vertical[index] = jump_speed;
            }
            let radians = heading.to_radians();
            let forward = (radians.sin(), 0.0, radians.cos());
            let distance = moves[index] * move_speed * dt;
            let desired = add_scaled(position, forward, distance);
            position = resolve_movement(position, desired, half, &obstacles, agent_radius);
        }

        if position.1 <= 1.0 + 1e-4 {
            position.1 = 1.0;
            if next_vertical[index] < 0.0 {
                next_vertical[index] = 0.0;
            }
        }
        next_vertical[index] -= gravity * dt;
        position.1 += next_vertical[index] * dt;
        if position.1 <= 1.0 {
            position.1 = 1.0;
            next_vertical[index] = 0.0;
        }
        next_positions[index] = position;
    }
    Ok((next_positions, next_headings, next_vertical))
}

#[inline]
fn segment_aabb(start: V3, end: V3, min_corner: V3, max_corner: V3) -> Option<(f64, V3)> {
    let direction = sub(end, start);
    let mut t_enter: f64 = 0.0;
    let mut t_exit: f64 = 1.0;
    for axis in 0..3 {
        let (s, delta, min, max) = match axis {
            0 => (start.0, direction.0, min_corner.0, max_corner.0),
            1 => (start.1, direction.1, min_corner.1, max_corner.1),
            _ => (start.2, direction.2, min_corner.2, max_corner.2),
        };
        if delta.abs() <= PROJECTILE_EPSILON {
            if s < min || s > max {
                return None;
            }
            continue;
        }
        let inv_delta = 1.0 / delta;
        let mut t0 = (min - s) * inv_delta;
        let mut t1 = (max - s) * inv_delta;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_enter = t_enter.max(t0);
        t_exit = t_exit.min(t1);
        if t_enter - t_exit > PROJECTILE_EPSILON {
            return None;
        }
    }
    let hit_t = t_enter.max(0.0).min(1.0);
    Some((hit_t, add_scaled(start, direction, hit_t)))
}

fn boundary_collision(start: V3, end: V3, arena_half: f64) -> Option<(f64, V3)> {
    let direction = sub(end, start);
    let mut best: Option<(f64, V3)> = None;
    for (delta, start_axis) in [(direction.0, start.0), (direction.2, start.2)] {
        if delta.abs() <= PROJECTILE_EPSILON {
            continue;
        }
        for boundary in [-arena_half, arena_half] {
            let hit_t = (boundary - start_axis) / delta;
            if hit_t <= PROJECTILE_EPSILON || hit_t > 1.0 + PROJECTILE_EPSILON {
                continue;
            }
            let position = add_scaled(start, direction, hit_t);
            if position.0.abs() <= arena_half + PROJECTILE_EPSILON
                && position.2.abs() <= arena_half + PROJECTILE_EPSILON
                && best.map_or(true, |(best_t, _)| hit_t < best_t)
            {
                best = Some((hit_t, position));
            }
        }
    }
    best
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn advance_projectiles(
    positions: Vec<V3>,
    velocities: Vec<V3>,
    shooter_indices: Vec<isize>,
    projectile_teams: Vec<usize>,
    ages: Vec<f64>,
    distance_travelled: Vec<f64>,
    max_distances: Vec<f64>,
    agent_positions: Vec<V3>,
    agent_alive: Vec<bool>,
    agent_teams: Vec<usize>,
    obstacle_positions: Vec<V3>,
    obstacle_sizes: Vec<V3>,
    dt: f64,
    arena_size: f64,
) -> PyResult<(Vec<V3>, Vec<f64>, Vec<f64>, Vec<usize>, Vec<(usize, u8, isize, V3, f64)>, Vec<bool>)> {
    let n = positions.len();
    if velocities.len() != n || shooter_indices.len() != n || projectile_teams.len() != n
        || ages.len() != n || distance_travelled.len() != n || max_distances.len() != n
        || agent_positions.len() != agent_alive.len() || agent_positions.len() != agent_teams.len()
        || obstacle_positions.len() != obstacle_sizes.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err("mismatched simulation array lengths"));
    }

    let obstacles: Vec<(V3, V3)> = obstacle_positions.into_iter().zip(obstacle_sizes).collect();
    let mut alive = agent_alive;
    let mut result_positions = positions.clone();
    let mut result_ages = ages.clone();
    let mut result_distances = distance_travelled.clone();
    let mut survivors = Vec::with_capacity(n);
    // (projectile index, kind: 1 agent hit / 2 obstacle or boundary / 3 range or invalid,
    //  target agent index or -1, event position, distance travelled this step)
    let mut events = Vec::new();
    let arena_half = arena_size * 0.5;

    for p in 0..n {
        let start = positions[p];
        let velocity = velocities[p];
        let speed = (velocity.0 * velocity.0 + velocity.1 * velocity.1 + velocity.2 * velocity.2).sqrt();
        if speed <= PROJECTILE_EPSILON {
            events.push((p, 4, -1, start, 0.0));
            continue;
        }
        let remaining = (max_distances[p] - distance_travelled[p]).max(0.0);
        if remaining <= PROJECTILE_EPSILON {
            events.push((p, 3, -1, start, 0.0));
            continue;
        }
        let travel = (speed * dt).min(remaining);
        let direction = (velocity.0 / speed, velocity.1 / speed, velocity.2 / speed);
        let end = add_scaled(start, direction, travel);
        result_ages[p] += dt;

        let mut best_kind = 0u8;
        let mut best_t = f64::INFINITY;
        let mut best_position = end;
        let mut target_index: isize = -1;

        for (center, size) in &obstacles {
            let half = (size.0 * 0.5 + PROJECTILE_RADIUS, size.1 * 0.5 + PROJECTILE_RADIUS, size.2 * 0.5 + PROJECTILE_RADIUS);
            let min = (center.0 - half.0, center.1 - half.1, center.2 - half.2);
            let max = (center.0 + half.0, center.1 + half.1, center.2 + half.2);
            if let Some((t, hit_pos)) = segment_aabb(start, end, min, max) {
                if t < best_t {
                    best_kind = 2;
                    best_t = t;
                    best_position = hit_pos;
                    target_index = -1;
                }
            }
        }

        for a in 0..agent_positions.len() {
            if !alive[a] || agent_teams[a] == projectile_teams[p] || shooter_indices[p] == a as isize {
                continue;
            }
            let center = agent_positions[a];
            let half = (0.58 + PROJECTILE_RADIUS, AGENT_HALF_HEIGHT + PROJECTILE_RADIUS, 0.58 + PROJECTILE_RADIUS);
            let min = (center.0 - half.0, center.1 - half.1, center.2 - half.2);
            let max = (center.0 + half.0, center.1 + half.1, center.2 + half.2);
            if let Some((t, hit_pos)) = segment_aabb(start, end, min, max) {
                if t < best_t - PROJECTILE_EPSILON {
                    best_kind = 1;
                    best_t = t;
                    best_position = hit_pos;
                    target_index = a as isize;
                }
            }
        }

        if let Some((t, hit_pos)) = boundary_collision(start, end, arena_half) {
            if t < best_t - PROJECTILE_EPSILON {
                best_kind = 2;
                best_t = t;
                best_position = hit_pos;
                target_index = -1;
            }
        }

        if best_kind == 1 {
            alive[target_index as usize] = false;
            result_positions[p] = best_position;
            result_distances[p] += travel * best_t;
            events.push((p, 1, target_index, best_position, travel * best_t));
        } else if best_kind == 2 {
            result_positions[p] = best_position;
            result_distances[p] += travel * best_t;
            events.push((p, 2, -1, best_position, travel * best_t));
        } else {
            result_positions[p] = end;
            result_distances[p] += travel;
            if result_distances[p] >= max_distances[p] - PROJECTILE_EPSILON {
                events.push((p, 5, -1, end, travel));
            } else {
                survivors.push(p);
            }
        }
    }

    Ok((result_positions, result_ages, result_distances, survivors, events, alive))
}

#[pymodule]
fn marl_arena_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(push_agents_out, module)?)?;
    module.add_function(wrap_pyfunction!(move_agents, module)?)?;
    module.add_function(wrap_pyfunction!(advance_projectiles, module)?)?;
    Ok(())
}
