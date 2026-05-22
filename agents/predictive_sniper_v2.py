import math
import random
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS

BOARD_SIZE = 100.0

def get(d, key, default):
    """Helper to read attributes safely from dict or SimpleNamespace."""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)

def get_fleet_speed(ships, max_speed=6.0):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000)
    if ratio < 0:
        ratio = 0.0
    return min(max_speed, 1.0 + (max_speed - 1.0) * (ratio ** 1.5))

def get_ships_for_speed(v, max_speed=6.0):
    """Calculates the minimum fleet size required to achieve at least speed v."""
    if v <= 1.0:
        return 1
    if v >= max_speed:
        return 1000  # Max speed is reached at ~1000 ships
    ratio = (v - 1.0) / (max_speed - 1.0)
    log_ships = math.log(1000) * (ratio ** (2.0 / 3.0))
    return math.ceil(math.exp(log_ships))

def point_to_segment_distance(p, v, w):
    """Minimum distance from point p to line segment v-w."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return math.hypot(p[0] - v[0], p[1] - v[1])
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return math.hypot(p[0] - projection[0], p[1] - projection[1])

def swept_pair_hit(A, B, P0, P1, r):
    """Continuous swept-pair check for linear paths."""
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0

def is_path_safe(origin_pos, target_pos):
    dist = point_to_segment_distance((CENTER, CENTER), origin_pos, target_pos)
    return dist >= SUN_RADIUS + 0.2

def get_planet_pos_at_t(planet, t, obs):
    """Predicts the position of a planet or comet t steps into the future."""
    comets = get(obs, "comets", [])
    if comets is None:
        comets = []
    for group in comets:
        planet_ids = get(group, "planet_ids", [])
        if planet_ids is None:
            planet_ids = []
        if planet.id in planet_ids:
            try:
                i = list(planet_ids).index(planet.id)
                paths = get(group, "paths", [])
                if paths is None:
                    paths = []
                p_path = paths[i]
                if p_path is None:
                    p_path = []
                path_index = get(group, "path_index", 0)
                if path_index is None:
                    path_index = 0
                idx = path_index + t
                if idx < len(p_path) and p_path[idx] is not None:
                    return p_path[idx][0], p_path[idx][1]
                else:
                    return None  # Comet expired
            except (ValueError, IndexError):
                pass

    dx = planet.x - CENTER
    dy = planet.y - CENTER
    orbital_r = math.hypot(dx, dy)
    if orbital_r + planet.radius < ROTATION_RADIUS_LIMIT:
        angular_velocity = get(obs, "angular_velocity", 0.03)
        init_angle = math.atan2(dy, dx)
        future_angle = init_angle + angular_velocity * t
        px = CENTER + orbital_r * math.cos(future_angle)
        py = CENTER + orbital_r * math.sin(future_angle)
        return px, py

    return planet.x, planet.y

def get_comet_remaining_steps(comet_id, obs):
    comets = get(obs, "comets", [])
    if comets is None:
        comets = []
    for group in comets:
        planet_ids = get(group, "planet_ids", [])
        if planet_ids is None:
            planet_ids = []
        if comet_id in planet_ids:
            try:
                i = list(planet_ids).index(comet_id)
                paths = get(group, "paths", [])
                if paths is None or i >= len(paths):
                    return 0
                p_path = paths[i]
                if p_path is None:
                    return 0
                path_index = get(group, "path_index", 0)
                if path_index is None:
                    path_index = 0
                return max(0, len(p_path) - path_index)
            except (ValueError, IndexError):
                pass
    return 999  # Not a comet

def get_fleet_target_and_arrival(fleet, planets, obs):
    max_speed = 6.0
    speed = get_fleet_speed(fleet.ships, max_speed)
    fx, fy = fleet.x, fleet.y
    angle = fleet.angle

    for t in range(1, 100):
        prev_fx = fx + math.cos(angle) * speed * (t - 1)
        prev_fy = fy + math.sin(angle) * speed * (t - 1)
        curr_fx = fx + math.cos(angle) * speed * t
        curr_fy = fy + math.sin(angle) * speed * t

        for p in planets:
            p_old = get_planet_pos_at_t(p, t - 1, obs)
            p_new = get_planet_pos_at_t(p, t, obs)
            if p_old is None or p_new is None:
                continue

            if swept_pair_hit((prev_fx, prev_fy), (curr_fx, curr_fy), p_old, p_new, p.radius):
                return p.id, t

        if not (0 <= curr_fx <= 100 and 0 <= curr_fy <= 100):
            break
        if point_to_segment_distance((CENTER, CENTER), (prev_fx, prev_fy), (curr_fx, curr_fy)) < SUN_RADIUS:
            break

    return None, None

def find_intercept(origin, target, speed, obs):
    for t in range(1, 100):
        pos = get_planet_pos_at_t(target, t, obs)
        if pos is None:
            break
        px, py = pos
        dist = math.hypot(px - origin.x, py - origin.y)
        if dist <= speed * t:
            return px, py, t
    dist = math.hypot(target.x - origin.x, target.y - origin.y)
    return target.x, target.y, max(1, int(math.ceil(dist / speed)))

def project_planet_timeline(planet, incoming_fleets, max_steps=100):
    arrivals_by_turn = {}
    for owner, ships, turn in incoming_fleets:
        arrivals_by_turn.setdefault(turn, []).append((owner, ships))

    timeline = {}
    curr_owner = planet.owner
    curr_ships = planet.ships
    timeline[0] = (curr_owner, curr_ships)

    for t in range(1, max_steps):
        if curr_owner != -1:
            curr_ships += planet.production

        if t in arrivals_by_turn:
            player_ships = {}
            for owner, ships in arrivals_by_turn[t]:
                player_ships[owner] = player_ships.get(owner, 0) + ships

            sorted_players = sorted(player_ships.items(), key=lambda item: item[1], reverse=True)
            top_player, top_ships = sorted_players[0]

            if len(sorted_players) > 1:
                second_ships = sorted_players[1][1]
                survivor_ships = top_ships - second_ships
                if sorted_players[0][1] == sorted_players[1][1]:
                    survivor_ships = 0
                survivor_owner = top_player if survivor_ships > 0 else -1
            else:
                survivor_owner = top_player
                survivor_ships = top_ships

            if survivor_ships > 0:
                if curr_owner == survivor_owner:
                    planet_garrison_new = curr_ships + survivor_ships
                else:
                    planet_garrison_new = curr_ships - survivor_ships
                    if planet_garrison_new < 0:
                        curr_owner = survivor_owner
                        planet_garrison_new = abs(planet_garrison_new)
                curr_ships = planet_garrison_new

        timeline[t] = (curr_owner, curr_ships)

    return timeline

def knapsack_expand(my_planets, targets, max_speed=6.0):
    """0-1 Knapsack opening solver to distribute initial ships optimally en-masse."""
    moves = []
    if len(my_planets) == 1:
        home = my_planets[0]
        available = home.ships - 1  # Leave at least 1 ship to keep home planet
        if available < 5:
            return moves

        candidates = []
        for t in targets:
            if t.owner == -1:
                dist = math.hypot(t.x - home.x, t.y - home.y)
                if is_path_safe((home.x, home.y), (t.x, t.y)):
                    cost = t.ships + 1
                    candidates.append((t, cost, t.production, dist))

        n = len(candidates)
        if n == 0:
            return moves

        dp = [0] * (available + 1)
        parent = [-1] * (available + 1)
        selected_item = [-1] * (available + 1)

        for i, (t, cost, value, dist) in enumerate(candidates):
            for w in range(available, cost - 1, -1):
                if dp[w - cost] + value > dp[w]:
                    dp[w] = dp[w - cost] + value
                    parent[w] = w - cost
                    selected_item[w] = i

        curr_w = available
        best_items = []
        while curr_w > 0 and selected_item[curr_w] != -1:
            best_items.append(candidates[selected_item[curr_w]])
            curr_w = parent[curr_w]

        for t, cost, value, dist in best_items:
            angle = math.atan2(t.y - home.y, t.x - home.x)
            moves.append([home.id, angle, cost])

    return moves

class SimComet:
    def __init__(self, planet_ids, paths, path_index):
        self.planet_ids = planet_ids
        self.paths = paths
        self.path_index = path_index

def simulate_state_forward(sim_planets, sim_fleets, angular_velocity, comets, comet_planet_ids, max_steps, max_speed=6.0):
    """A highly optimized rollout simulator running entirely on flat lists for maximum speed."""
    comet_pid_set = set(comet_planet_ids)

    sim_comets = []
    if comets is not None:
        for g in comets:
            pids = get(g, "planet_ids", [])
            if pids is None:
                pids = []
            sim_comets.append(SimComet(
                planet_ids=list(pids),
                paths=get(g, "paths", []) or [],
                path_index=int(get(g, "path_index", 0) or 0)
            ))

    for step in range(1, max_steps + 1):
        # 1. Expire comets
        expired_comet_pids = []
        for g in sim_comets:
            idx = g.path_index
            for i, pid in enumerate(g.planet_ids):
                if idx >= len(g.paths[i]):
                    expired_comet_pids.append(pid)
        if expired_comet_pids:
            expired_set = set(expired_comet_pids)
            sim_planets = [p for p in sim_planets if p[0] not in expired_set]
            for g in sim_comets:
                g.planet_ids = [pid for pid in g.planet_ids if pid not in expired_set]
            sim_comets = [g for g in sim_comets if g.planet_ids]

        # 2. Production
        for p in sim_planets:
            if p[1] != -1:
                p[5] += p[6]

        # 3. Planet movement
        planet_paths = {}
        for p in sim_planets:
            if p[0] in comet_pid_set:
                continue
            old_pos = (p[2], p[3])
            new_pos = old_pos
            dx = p[2] - CENTER
            dy = p[3] - CENTER
            r = math.hypot(dx, dy)
            if r + p[4] < ROTATION_RADIUS_LIMIT:
                init_angle = math.atan2(dy, dx)
                curr_angle = init_angle + angular_velocity
                new_pos = (
                    CENTER + r * math.cos(curr_angle),
                    CENTER + r * math.sin(curr_angle),
                )
            planet_paths[p[0]] = (old_pos, new_pos, True)

        for g in sim_comets:
            g.path_index += 1
            idx = g.path_index
            for i, pid in enumerate(g.planet_ids):
                planet = next((pl for pl in sim_planets if pl[0] == pid), None)
                if planet is None:
                    continue
                p_path = g.paths[i]
                old_pos = (planet[2], planet[3])
                if idx >= len(p_path):
                    planet_paths[pid] = (old_pos, old_pos, True)
                else:
                    new_pos = (p_path[idx][0], p_path[idx][1])
                    check = old_pos[0] >= 0
                    planet_paths[pid] = (old_pos, new_pos, check)

        # 4. Fleet movement
        fleets_to_remove = []
        combat_lists = {p[0]: [] for p in sim_planets}

        for f in sim_fleets:
            angle = f[4]
            ships = f[6]
            speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
            speed = min(speed, max_speed)
            old_pos = (f[2], f[3])
            f[2] += math.cos(angle) * speed
            f[3] += math.sin(angle) * speed
            new_pos = (f[2], f[3])

            hit_planet = False
            for p in sim_planets:
                path = planet_paths.get(p[0])
                if path is None or not path[2]:
                    continue
                p_old, p_new, _ = path
                if swept_pair_hit(old_pos, new_pos, p_old, p_new, p[4]):
                    combat_lists[p[0]].append(f)
                    fleets_to_remove.append(f)
                    hit_planet = True
                    break
            if hit_planet:
                continue

            if not (0 <= f[2] <= BOARD_SIZE and 0 <= f[3] <= BOARD_SIZE):
                fleets_to_remove.append(f)
                continue

            if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
                fleets_to_remove.append(f)
                continue

        # Update simulated planet positions
        for p in sim_planets:
            path = planet_paths.get(p[0])
            if path is not None:
                p[2], p[3] = path[1]

        sim_fleets = [f for f in sim_fleets if f not in fleets_to_remove]

        # 5. Resolve combat
        for pid, p_fleets in combat_lists.items():
            planet = next((pl for pl in sim_planets if pl[0] == pid), None)
            if not planet or not p_fleets:
                continue

            player_ships = {}
            for f in p_fleets:
                owner = f[1]
                player_ships[owner] = player_ships.get(owner, 0) + f[6]

            if not player_ships:
                continue

            sorted_players = sorted(player_ships.items(), key=lambda item: item[1], reverse=True)
            top_player, top_ships = sorted_players[0]

            if len(sorted_players) > 1:
                second_ships = sorted_players[1][1]
                survivor_ships = top_ships - second_ships
                if sorted_players[0][1] == sorted_players[1][1]:
                    survivor_ships = 0
                survivor_owner = top_player if survivor_ships > 0 else -1
            else:
                survivor_owner = top_player
                survivor_ships = top_ships

            if survivor_ships > 0:
                if planet[1] == survivor_owner:
                    planet[5] += survivor_ships
                else:
                    planet[5] -= survivor_ships
                    if planet[5] < 0:
                        planet[1] = survivor_owner
                        planet[5] = abs(planet[5])

    return sim_planets, sim_fleets

def evaluate_simulated_state(planets, fleets, player):
    """Scoring function evaluating total ships and weighted production of both players."""
    our_ships = 0
    our_prod = 0
    enemy_ships = 0
    enemy_prod = 0

    for p in planets:
        owner = p[1]
        ships = p[5]
        prod = p[6]
        if owner == player:
            our_ships += ships
            our_prod += prod
        elif owner != -1:
            enemy_ships += ships
            enemy_prod += prod

    for f in fleets:
        owner = f[1]
        ships = f[6]
        if owner == player:
            our_ships += ships
        else:
            enemy_ships += ships

    return (our_ships + 12.0 * our_prod) - (enemy_ships + 12.0 * enemy_prod)

def apply_action_to_state(action, planets_state, fleets_state, player, next_fleet_id):
    """Mutates simulation state list to dispatch a candidate fleet action."""
    for move in action:
        from_id, angle, ships = move
        planet = next((pl for pl in planets_state if pl[0] == from_id), None)
        if planet and planet[1] == player and planet[5] >= ships and ships > 0:
            planet[5] -= ships
            start_x = planet[2] + math.cos(angle) * (planet[4] + 0.1)
            start_y = planet[3] + math.sin(angle) * (planet[4] + 0.1)
            fleets_state.append([next_fleet_id, player, start_x, start_y, angle, from_id, ships])
            next_fleet_id += 1
    return fleets_state, next_fleet_id

def generate_candidate_actions(player, planets, fleets, obs, timelines, available_surplus, comet_planet_ids, is_opponent=False):
    """Reduces the branching factor down to high-quality candidate moves."""
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    candidates = [[]]  # Always include No-op

    if not my_planets:
        return candidates

    # 1. Greedy Neutral Captures (Prune: Only top 3 highest production neutral captures)
    neutral_targets = [t for t in targets if t.owner == -1]
    scored_neutrals = []
    for target in neutral_targets:
        # Comet Longevity Filter: Skip comets about to leave the board soon
        if target.id in comet_planet_ids:
            rem_steps = get_comet_remaining_steps(target.id, obs)
            if rem_steps < 25:
                continue

        min_dist = min((math.hypot(target.x - mine.x, target.y - mine.y) for mine in my_planets), default=float('inf'))
        score = target.production / (min_dist + 1.0)
        scored_neutrals.append((target, score))
    scored_neutrals.sort(key=lambda x: x[1], reverse=True)

    max_greedy = 2 if is_opponent else 3
    greedy_count = 0
    for target, score in scored_neutrals:
        if greedy_count >= max_greedy:
            break
        for mine in my_planets:
            ships_needed = target.ships + 1
            if available_surplus.get(mine.id, 0) >= ships_needed:
                speed = get_fleet_speed(ships_needed)
                intercept = find_intercept(mine, target, speed, obs)
                if intercept is not None:
                    tx, ty, arrival_turns = intercept
                    if is_path_safe((mine.x, mine.y), (tx, ty)):
                        angle = math.atan2(ty - mine.y, tx - mine.x)
                        candidates.append([[mine.id, angle, ships_needed]])
                        greedy_count += 1
                        break  # Only one source planet needs to capture this neutral target

    # 2. Frontier Doom Consolidation (Only generated for the ally to save opponent search space)
    if not is_opponent:
        enemy_planets = [p for p in planets if p.owner != player and p.owner != -1]
        if enemy_planets:
            avg_ex = sum(p.x for p in enemy_planets) / len(enemy_planets)
            avg_ey = sum(p.y for p in enemy_planets) / len(enemy_planets)
        else:
            avg_ex, avg_ey = CENTER, CENTER

        frontier = min(my_planets, key=lambda p: math.hypot(p.x - avg_ex, p.y - avg_ey))
        consolidation_moves = []
        for mine in my_planets:
            if mine.id != frontier.id and available_surplus.get(mine.id, 0) >= 10:
                send_amount = available_surplus[mine.id] // 2
                if is_path_safe((mine.x, mine.y), (frontier.x, frontier.y)):
                    angle = math.atan2(frontier.y - mine.y, frontier.x - mine.x)
                    consolidation_moves.append([mine.id, angle, send_amount])
        if consolidation_moves:
            candidates.append(consolidation_moves)

    # 3. Coordinated Attacks on Top Targets (Prune: Only top 2 targets for ally, top 1 for opponent)
    scored_targets = []
    for target in targets:
        # Comet Longevity Filter
        if target.id in comet_planet_ids:
            rem_steps = get_comet_remaining_steps(target.id, obs)
            if rem_steps < 30:
                continue

        min_dist = min((math.hypot(target.x - mine.x, target.y - mine.y) for mine in my_planets), default=float('inf'))
        neutral_bonus = 2.0 if target.owner == -1 else 1.0
        score = (target.production * 20.0 * neutral_bonus) / (min_dist + 1.0)
        scored_targets.append((target, score))
    scored_targets.sort(key=lambda x: x[1], reverse=True)

    max_coord = 1 if is_opponent else 2
    coord_count = 0
    for target, score in scored_targets:
        if coord_count >= max_coord:
            break

        # Calculate Dynamic Coordination Arrival Horizons based on actual distances
        my_distances = [math.hypot(target.x - p.x, target.y - p.y) for p in my_planets if available_surplus.get(p.id, 0) >= 5]
        if not my_distances:
            continue
        min_dist = min(my_distances)
        
        # Test horizons representing average travel speeds: speed = 2.5, 4.0, 5.5
        test_horizons = []
        for v_test in [2.5, 4.0, 5.5]:
            T_est = int(math.ceil(min_dist / v_test))
            if 5 <= T_est <= 60:
                test_horizons.append(T_est)
        test_horizons = sorted(list(set(test_horizons)))
        if not test_horizons:
            test_horizons = [20, 35]

        for T in test_horizons:
            t_pos = get_planet_pos_at_t(target, T, obs)
            if t_pos is None:
                continue
            tx, ty = t_pos

            contributors = []
            total_contrib = 0
            for p in my_planets:
                surp = available_surplus.get(p.id, 0)
                if surp < 5:
                    continue
                dist = math.hypot(tx - p.x, ty - p.y)
                req_v = dist / T
                min_ships = get_ships_for_speed(req_v)
                if min_ships <= surp and is_path_safe((p.x, p.y), (tx, ty)):
                    contributors.append((p, min_ships, tx, ty))
                    total_contrib += surp

            target_garrison = timelines[target.id][T - 1][1]
            target_prod = target.production if target.owner != -1 else 0
            ships_needed = target_garrison + target_prod + 1

            if total_contrib >= ships_needed:
                attack_moves = []
                remaining_needed = ships_needed
                for p, min_s, cx, cy in contributors:
                    if remaining_needed <= 0:
                        break
                    send_amount = max(min_s, min(available_surplus[p.id], remaining_needed))
                    angle = math.atan2(cy - p.y, cx - p.x)
                    attack_moves.append([p.id, angle, send_amount])
                    remaining_needed -= send_amount
                if attack_moves:
                    candidates.append(attack_moves)
                    coord_count += 1
                break

    limit = 4 if is_opponent else 8
    return candidates[:limit]

def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
    fleets = [Fleet(*f) for f in raw_fleets]

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    step = obs.get("step", 0) if isinstance(obs, dict) else obs.step
    comet_planet_ids = obs.get("comet_planet_ids", [])

    if not my_planets:
        return moves

    # --- Phase 1: FastExpand Knapsack Opening ---
    if step <= 30 and len(my_planets) == 1:
        opening_moves = knapsack_expand(my_planets, targets)
        if opening_moves:
            return opening_moves

    # --- Phase 2: In-Flight Timeline Forecast ---
    destinations = {}
    for f in fleets:
        target_pid, arrival_t = get_fleet_target_and_arrival(f, planets, obs)
        if target_pid is not None:
            destinations.setdefault(target_pid, []).append((f.owner, f.ships, arrival_t))

    timelines = {}
    for p in planets:
        incoming = destinations.get(p.id, [])
        timelines[p.id] = project_planet_timeline(p, incoming, max_steps=100)

    # Establish surplus baseline per planet
    available_surplus = {}
    for mine in my_planets:
        timeline = timelines[mine.id]
        min_garrison = mine.ships
        is_threatened = False
        for t in range(1, 40):
            owner, ships = timeline[t]
            if owner != player:
                is_threatened = True
                min_garrison = 0
                break
            if ships < min_garrison:
                min_garrison = ships
        if is_threatened:
            available_surplus[mine.id] = 0
        else:
            available_surplus[mine.id] = max(0, min_garrison - 10)

    # --- Phase 3: Border Threat Opponent Modeling (4-Player FFA aware) ---
    opponents = [i for i in range(4) if i != player]
    opponent_threats = {o: 0.0 for o in opponents}
    for opp in opponents:
        for opp_p in planets:
            if opp_p.owner == opp:
                min_d = min((math.hypot(opp_p.x - mine.x, opp_p.y - mine.y) for mine in my_planets), default=150.0)
                threat = (opp_p.production * 5.0 + opp_p.ships) / (min_d + 1.0)
                opponent_threats[opp] += threat
    opponent = max(opponent_threats, key=lambda o: opponent_threats[o])

    # Generate Candidate Action Sets
    ally_candidates = generate_candidate_actions(
        player, planets, fleets, obs, timelines, available_surplus, comet_planet_ids, is_opponent=False
    )
    
    # Establish opponent surplus baseline to model their choices
    opp_planets = [p for p in planets if p.owner == opponent]
    opp_surplus = {}
    for opp_p in opp_planets:
        timeline = timelines[opp_p.id]
        min_garrison = opp_p.ships
        is_threatened = False
        for t in range(1, 40):
            owner, ships = timeline[t]
            if owner != opponent:
                is_threatened = True
                min_garrison = 0
                break
            if ships < min_garrison:
                min_garrison = ships
        if is_threatened:
            opp_surplus[opp_p.id] = 0
        else:
            opp_surplus[opp_p.id] = max(0, min_garrison - 10)

    enemy_candidates = generate_candidate_actions(
        opponent, planets, fleets, obs, timelines, opp_surplus, comet_planet_ids, is_opponent=True
    )

    # --- Phase 4: 2-Ply Minimax Search Rollouts ---
    best_move = []
    best_score = -float('inf')
    next_fleet_id = get(obs, "next_fleet_id", len(fleets) + 1)
    angular_velocity = get(obs, "angular_velocity", 0.03)
    comets = get(obs, "comets", [])

    # Cache representation as flat primitive lists to skip named tuple overhead entirely during loops
    planets_list = [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production] for p in planets]
    fleets_list = [[f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships] for f in fleets]

    for ally_move in ally_candidates:
        min_score = float('inf')
        for enemy_move in enemy_candidates:
            # Deep clone flat lists directly (extremely fast)
            sim_planets = [[p[0], p[1], p[2], p[3], p[4], p[5], p[6]] for p in planets_list]
            sim_fleets = [[f[0], f[1], f[2], f[3], f[4], f[5], f[6]] for f in fleets_list]
            
            # Dispatch candidate moves directly on primitive state
            sim_fleets, temp_fid = apply_action_to_state(ally_move, sim_planets, sim_fleets, player, next_fleet_id)
            sim_fleets, temp_fid = apply_action_to_state(enemy_move, sim_planets, sim_fleets, opponent, temp_fid)

            # Run flat state rollout
            rollout_planets, rollout_fleets = simulate_state_forward(
                sim_planets, sim_fleets, angular_velocity, comets, comet_planet_ids, max_steps=40
            )

            # Evaluate final rollout state
            score = evaluate_simulated_state(rollout_planets, rollout_fleets, player)

            if score < min_score:
                min_score = score

        if min_score > best_score:
            best_score = min_score
            best_move = ally_move

    return best_move
