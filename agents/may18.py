import math
from typing import NamedTuple

import numpy as np

BOARD_SIZE: float = 100.0
CENTER: np.ndarray = np.array([50.0, 50.0])
SUN_RADIUS: float = 10.0
ROTATION_RADIUS_LIMIT: float = 50.0
MAX_SHIP_SPEED: float = 6.0
COMET_SPEED: float = 4.0

def is_orbiting(planet_xy, planet_radius):
    planet_xy = np.asarray(planet_xy, dtype=float)
    planet_radius = np.asarray(planet_radius, dtype=float)
    orbital_r = np.linalg.norm(planet_xy - CENTER, axis=-1)
    return orbital_r + planet_radius < ROTATION_RADIUS_LIMIT


def effective_omega(planet_xy, planet_radius, game_omega):
    rotates = is_orbiting(planet_xy, planet_radius)
    return np.where(rotates, game_omega, 0.0)


def predict_position_after(current_xy, omega, delta_steps):
    current_xy = np.asarray(current_xy, dtype=float)
    omega = np.asarray(omega, dtype=float)
    delta_steps = np.asarray(delta_steps, dtype=float)
    angle = omega * delta_steps
    rel = current_xy - CENTER
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rx = rel[..., 0] * cos_a - rel[..., 1] * sin_a
    ry = rel[..., 0] * sin_a + rel[..., 1] * cos_a
    rotated = np.stack([rx, ry], axis=-1)
    return rotated + CENTER

def fleet_speed(num_ships):
    n = np.asarray(num_ships, dtype=float)
    n = np.maximum(n, 1.0)
    raw = 1.0 + (MAX_SHIP_SPEED - 1.0) * (np.log(n) / np.log(1000.0)) ** 1.5
    return np.minimum(raw, MAX_SHIP_SPEED)


def fleet_position_after(fleet_xy, fleet_angle, ship_speed, delta_steps):
    fleet_xy = np.asarray(fleet_xy, dtype=float)
    fleet_angle = np.asarray(fleet_angle, dtype=float)
    ship_speed = np.asarray(ship_speed, dtype=float)
    delta_steps = np.asarray(delta_steps, dtype=float)
    dx = np.cos(fleet_angle) * ship_speed * delta_steps
    dy = np.sin(fleet_angle) * ship_speed * delta_steps
    return fleet_xy + np.stack([dx, dy], axis=-1)


def path_hits_sun(from_xy, to_xy):
    from_xy = np.asarray(from_xy, dtype=float)
    to_xy = np.asarray(to_xy, dtype=float)
    seg = to_xy - from_xy
    seg_len_sq = np.sum(seg * seg, axis=-1)
    rel = CENTER - from_xy
    safe_len_sq = np.where(seg_len_sq > 0, seg_len_sq, 1.0)
    t = np.sum(rel * seg, axis=-1) / safe_len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = from_xy + (seg * t[..., None])
    dist = np.linalg.norm(closest - CENTER, axis=-1)
    return dist < SUN_RADIUS


def fleet_collision_step(
    fleet_xy, fleet_angle, ship_speed,
    target_current_xy, target_radius, target_omega,
    max_steps=500, step_resolution=0.5,
):
    fleet_xy = np.asarray(fleet_xy, dtype=float)
    fleet_angle = np.asarray(fleet_angle, dtype=float)
    ship_speed = np.asarray(ship_speed, dtype=float)
    target_current_xy = np.asarray(target_current_xy, dtype=float)
    target_radius = np.asarray(target_radius, dtype=float)
    target_omega = np.asarray(target_omega, dtype=float)

    ts = np.arange(0.0, max_steps + step_resolution, step_resolution)
    fleet_dx = np.cos(fleet_angle)[..., None] * ship_speed[..., None] * ts
    fleet_dy = np.sin(fleet_angle)[..., None] * ship_speed[..., None] * ts
    fleet_pos = fleet_xy[..., None, :] + np.stack([fleet_dx, fleet_dy], axis=-1)
    target_pos = predict_position_after(
        target_current_xy[..., None, :], target_omega[..., None], ts,
    )
    dist = np.linalg.norm(fleet_pos - target_pos, axis=-1)
    hit = dist < target_radius[..., None]
    any_hit = np.any(hit, axis=-1)
    first_idx = np.argmax(hit, axis=-1)
    return np.where(any_hit, ts[first_idx], np.inf)


def first_planet_collision(
    fleet_xy, fleet_angle, ship_speed,
    planet_xys, planet_radii, planet_omegas,
    max_steps=500, step_resolution=0.5, exclude_idx=-1,
):
    fleet_xy = np.asarray(fleet_xy, dtype=float)
    planet_xys = np.asarray(planet_xys, dtype=float)
    planet_radii = np.asarray(planet_radii, dtype=float)
    planet_omegas = np.asarray(planet_omegas, dtype=float)
    m = planet_xys.shape[0]
    if m == 0:
        return -1, float("inf")
    fleet_xys = np.broadcast_to(fleet_xy, (m, 2))
    fleet_angles = np.full((m,), float(fleet_angle))
    speeds = np.full((m,), float(ship_speed))
    steps = fleet_collision_step(
        fleet_xys, fleet_angles, speeds,
        planet_xys, planet_radii, planet_omegas,
        max_steps=max_steps, step_resolution=step_resolution,
    )
    if 0 <= exclude_idx < m:
        steps = steps.copy()
        steps[exclude_idx] = np.inf
    if not np.any(np.isfinite(steps)):
        return -1, float("inf")
    idx = int(np.argmin(steps))
    return idx, float(steps[idx])

class InterceptSolution(NamedTuple):
    angle: np.ndarray
    arrival_step: np.ndarray
    feasible: np.ndarray


def solve_intercept(
    from_xy, target_current_xy, target_omega, ship_speed,
    max_steps=500, tol=0.5, max_iters=40,
    source_radius=0.0, first_step_lag=0,
):
    from_xy = np.asarray(from_xy, dtype=float)
    target_current_xy = np.asarray(target_current_xy, dtype=float)
    target_omega = np.asarray(target_omega, dtype=float)
    ship_speed = np.asarray(ship_speed, dtype=float)
    source_radius = np.asarray(source_radius, dtype=float)

    safe_speed = np.maximum(ship_speed, 1e-12)
    radius_pad = (source_radius + 0.1) / safe_speed

    delta = target_current_xy - from_xy
    dt = np.linalg.norm(delta, axis=-1) / safe_speed

    def _rot_time(dt_val):
        return np.maximum(dt_val - radius_pad - float(first_step_lag), 0.0)

    converged = np.zeros_like(dt, dtype=bool)
    for _ in range(max_iters):
        target_pos = predict_position_after(target_current_xy, target_omega, _rot_time(dt))
        new_delta = target_pos - from_xy
        new_dt = np.linalg.norm(new_delta, axis=-1) / safe_speed
        converged = np.abs(new_dt - dt) < tol
        dt = new_dt
        if np.all(converged):
            break

    target_pos = predict_position_after(target_current_xy, target_omega, _rot_time(dt))
    aim_vec = target_pos - from_xy
    angle = np.arctan2(aim_vec[..., 1], aim_vec[..., 0])
    sun_safe = ~path_hits_sun(from_xy, target_pos)
    feasible = converged & (dt <= max_steps) & sun_safe
    arrival = np.where(feasible, dt, np.inf)
    return InterceptSolution(angle=angle, arrival_step=arrival, feasible=feasible)


def solve_intercept_comet(from_xy, comet_path, path_index, ship_speed, speed_tol_frac=0.1):
    from_xy = np.asarray(from_xy, dtype=float)
    comet_path = np.asarray(comet_path, dtype=float)
    ship_speed = float(ship_speed)
    if ship_speed <= 0 or len(comet_path) == 0:
        return InterceptSolution(
            angle=np.array(0.0), arrival_step=np.array(np.inf), feasible=np.array(False),
        )
    remaining = len(comet_path) - int(path_index)
    if remaining <= 1:
        return InterceptSolution(
            angle=np.array(0.0), arrival_step=np.array(np.inf), feasible=np.array(False),
        )
    tol = speed_tol_frac * ship_speed
    for dt in range(1, remaining):
        target_pos = comet_path[int(path_index) + dt]
        dist = float(np.linalg.norm(target_pos - from_xy))
        required = dist / dt
        if abs(required - ship_speed) > tol:
            continue
        if bool(path_hits_sun(from_xy, target_pos)):
            continue
        aim = target_pos - from_xy
        angle = float(math.atan2(float(aim[1]), float(aim[0])))
        return InterceptSolution(
            angle=np.array(angle), arrival_step=np.array(float(dt)), feasible=np.array(True),
        )
    return InterceptSolution(
        angle=np.array(0.0), arrival_step=np.array(np.inf), feasible=np.array(False),
    )

NEUTRAL_OWNER = -1

def ships_needed_to_capture(garrison, production, owner_id, target_player_id, arrival_step):
    garrison = np.asarray(garrison, dtype=float)
    production = np.asarray(production, dtype=float)
    owner_id = np.asarray(owner_id, dtype=int)
    arrival_step = np.asarray(arrival_step, dtype=float)
    is_friendly = owner_id == target_player_id
    is_neutral = owner_id == NEUTRAL_OWNER
    needed_neutral = garrison + 1
    needed_enemy = garrison + production * arrival_step + 1
    needed = np.where(is_neutral, needed_neutral, needed_enemy)
    needed = np.where(is_friendly, 0.0, needed)
    return np.ceil(needed).astype(int)

AGENT_CODE = "AIN"
AGENT_NAME = "All-In (may18 launch-safety champion)"

HORIZON = 30
MIN_RESERVE = 0
DEFENSE_HORIZON = 80
SCORE_FLOOR = 2.4
ENEMY_DENIAL_BONUS = 2.0
LAUNCH_SAFETY_FLOOR = 0.4
LAUNCH_SAFETY_SCALE = 30.0


def _obs_field(obs, name, default):
    return obs.get(name, default) if isinstance(obs, dict) else getattr(obs, name, default)


def _comet_path_map(obs):
    groups = _obs_field(obs, "comets", [])
    out: dict[int, tuple] = {}
    for group in groups:
        idx = int(group["path_index"] if isinstance(group, dict) else group.path_index)
        pids = group["planet_ids"] if isinstance(group, dict) else group.planet_ids
        paths = group["paths"] if isinstance(group, dict) else group.paths
        for i, pid in enumerate(pids):
            path = np.asarray(paths[i], dtype=float)
            remaining = max(0, len(path) - idx)
            out[int(pid)] = (path, idx, remaining)
    return out


def agent(obs):
    planets = _obs_field(obs, "planets", [])
    fleets = _obs_field(obs, "fleets", [])
    my_player = _obs_field(obs, "player", 0)
    omega = _obs_field(obs, "angular_velocity", 0.0)
    comet_ids = set(_obs_field(obs, "comet_planet_ids", []))

    rows = list(planets)
    if not rows:
        return []

    ids = np.array([p[0] for p in rows], dtype=int)
    owners = np.array([p[1] for p in rows], dtype=int)
    xys = np.array([[p[2], p[3]] for p in rows], dtype=float)
    radii = np.array([p[4] for p in rows], dtype=float)
    ships = np.array([p[5] for p in rows], dtype=int)
    prods = np.array([p[6] for p in rows], dtype=int)
    omegas = effective_omega(xys, radii, omega)
    is_comet = np.array([int(pid) in comet_ids for pid in ids])
    # Comets follow precomputed paths, not omega-based rotation; treat as static
    # obstacles for the planet-collision sweep.
    omegas = np.where(is_comet, 0.0, omegas)
    comet_paths = _comet_path_map(obs)

    mine_mask = owners == my_player
    target_mask = ~mine_mask
    if not np.any(mine_mask):
        return []
    mine_idx = np.where(mine_mask)[0]

    # ---- Defensive reserves: how much garrison each of my planets must keep ----
    reserves: dict[int, int] = {}
    for f in fleets:
        if f[1] == my_player:
            continue
        fleet_xy = np.array([f[2], f[3]], dtype=float)
        fleet_angle = float(f[4])
        fleet_n = int(f[6])
        speed = float(fleet_speed(fleet_n))
        best_i = -1
        best_t = math.inf
        for i in mine_idx:
            t = float(fleet_collision_step(
                fleet_xy, fleet_angle, speed,
                xys[i], radii[i], omegas[i],
                max_steps=DEFENSE_HORIZON,
            ))
            if t < best_t:
                best_t = t
                best_i = int(i)
        if best_i < 0:
            continue
        # (1) Sun-doomed: enemy fleet dies in the sun before arrival.
        arrival_xy = fleet_position_after(fleet_xy, fleet_angle, speed, best_t)
        if bool(path_hits_sun(fleet_xy, arrival_xy)):
            continue
        grow = int(prods[best_i]) * int(round(best_t))
        shortfall = fleet_n - grow
        # (2) Planet-doomed: short by more than current ships; planet falls regardless.
        if shortfall > int(ships[best_i]):
            continue
        reserves[int(ids[best_i])] = max(reserves.get(int(ids[best_i]), 0), shortfall)

    # ---- (3) Reinforce doomed high-production planets ----
    ships_avail = ships.copy()
    reinforce_moves: list[list] = []
    reinforced: set[int] = set()
    for f in fleets:
        if f[1] == my_player:
            continue
        fxy = np.array([f[2], f[3]], dtype=float)
        fang = float(f[4])
        fn = int(f[6])
        sp = float(fleet_speed(fn))
        bi = -1
        bt = math.inf
        for i in mine_idx:
            t = float(fleet_collision_step(
                fxy, fang, sp,
                xys[i], radii[i], omegas[i],
                max_steps=DEFENSE_HORIZON,
            ))
            if t < bt:
                bt = t
                bi = int(i)
        if bi < 0 or int(prods[bi]) < 2:
            continue
        if bool(path_hits_sun(fxy, fleet_position_after(fxy, fang, sp, bt))):
            continue
        gr = int(prods[bi]) * int(round(bt))
        sf = fn - gr
        if sf <= int(ships_avail[bi]):
            continue
        tpid = int(ids[bi])
        if tpid in reinforced:
            continue
        gap = sf - int(ships_avail[bi]) + 1
        best_k = -1
        best_k_angle = 0.0
        best_k_t = math.inf
        for k in mine_idx:
            if int(k) == bi:
                continue
            spare = int(ships_avail[k]) - MIN_RESERVE
            if spare < gap:
                continue
            send_speed = float(fleet_speed(gap))
            sol = solve_intercept(xys[k], xys[bi], omegas[bi], send_speed)
            if not bool(sol.feasible) or float(sol.arrival_step) >= bt:
                continue
            if float(sol.arrival_step) < best_k_t:
                best_k_t = float(sol.arrival_step)
                best_k = int(k)
                best_k_angle = float(sol.angle)
        if best_k >= 0:
            reinforce_moves.append([int(ids[best_k]), best_k_angle, int(gap)])
            ships_avail[best_k] -= gap
            reinforced.add(tpid)

    if not np.any(target_mask):
        return reinforce_moves
    target_idx = np.where(target_mask)[0]

    # ---- Launch-safety: distance from each of my planets to nearest enemy ----
    # Threats = enemy-owned planets + enemy fleet CURRENT POSITIONS. The latter
    # is the may18 promotion delta (~+100 Elo).
    enemy_owned_mask = (owners >= 0) & (owners != my_player) & (~is_comet)
    threats: list[np.ndarray] = []
    if np.any(enemy_owned_mask):
        threats.append(xys[enemy_owned_mask])
    enemy_fleet_xys = np.array(
        [[f[2], f[3]] for f in fleets if f[1] != my_player], dtype=float,
    )
    if len(enemy_fleet_xys) > 0:
        threats.append(enemy_fleet_xys)
    if threats:
        all_threats = np.vstack(threats)
        d_mine = np.linalg.norm(
            xys[mine_idx, None, :] - all_threats[None, :, :], axis=2
        ).min(axis=1)
        safety_by_mine = np.clip(
            d_mine / LAUNCH_SAFETY_SCALE, LAUNCH_SAFETY_FLOOR, 1.0
        )
    else:
        safety_by_mine = np.ones(len(mine_idx), dtype=float)
    safety_lookup = {int(mi): float(safety_by_mine[k]) for k, mi in enumerate(mine_idx)}

    # ---- Offensive scoring ----
    candidates = []
    for j in target_idx:
        target_is_comet = bool(is_comet[j])
        comet_info = comet_paths.get(int(ids[j])) if target_is_comet else None
        if target_is_comet and (comet_info is None or comet_info[2] <= 1):
            continue
        for i in mine_idx:
            reserve = max(reserves.get(int(ids[i]), 0), MIN_RESERVE)
            available = int(ships_avail[i]) - reserve
            if available < 1:
                continue
            speed = float(fleet_speed(available))

            if target_is_comet:
                path, pidx, lifetime = comet_info
                sol = solve_intercept_comet(xys[i], path, pidx, speed)
            else:
                sol = solve_intercept(xys[i], xys[j], omegas[j], speed)
            if not bool(sol.feasible):
                continue
            arrival = float(sol.arrival_step)

            # Sweep filter: another planet may eclipse the target before we arrive.
            first_idx, _ = first_planet_collision(
                xys[i], float(sol.angle), speed,
                xys, radii, omegas,
                max_steps=int(arrival) + 5,
                exclude_idx=int(i),
            )
            if first_idx != int(j):
                continue

            needed = int(ships_needed_to_capture(
                int(ships[j]), int(prods[j]), int(owners[j]),
                int(my_player), arrival,
            ))
            if needed <= 0 or needed > available:
                continue

            if target_is_comet:
                rem_after = max(0, comet_info[2] - 1 - int(arrival))
                effective_horizon = min(HORIZON, rem_after)
            else:
                effective_horizon = HORIZON
            score = (int(prods[j]) * effective_horizon - needed) / max(arrival, 1.0)
            # (5) Enemy denial bonus.
            if int(owners[j]) >= 0:
                score *= ENEMY_DENIAL_BONUS
            # (7) Launch-safety penalty.
            score *= safety_lookup[int(i)]
            # (4) Score floor.
            if score < SCORE_FLOOR:
                continue
            # (6) AGRO send-full-available: angle was solved for fleet_speed(available);
            # under-sending slows the fleet and breaks the arrival math.
            candidates.append((score, int(j), int(i), float(sol.angle), available))

    candidates.sort(reverse=True)
    moves = list(reinforce_moves)
    used_planets: set[int] = {m[0] for m in reinforce_moves}
    locked_targets: set[int] = set()
    for score, j, i, angle, needed in candidates:
        if i in used_planets:
            continue
        if j in locked_targets:
            continue
        used_planets.add(i)
        locked_targets.add(j)
        moves.append([int(ids[i]), float(angle), int(needed)])
    return moves