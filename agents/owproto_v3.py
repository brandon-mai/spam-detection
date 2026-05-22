import math
import kaggle_environments.envs.orbit_wars.orbit_wars as ow
import numpy as np
from collections import defaultdict

fleet_trajectories = []
reinforcement_trajectories = []
moving_planets = []
steps = 0

MAX_SPEED = 6.0
MIN_SHIPS_MINE_ATTACK = 5
MIN_SHIPS_TARGET_COOP_ATTACK = 20
COOP_PLANET_CAP = 8

FORMULA_DIST = 100
FORMULA_PROD_MULT = 15
FORMULA_ENEMY_BONUS_MULT = 10
FORMULA_TOTAL_SHIPS_PERCENT = 0.7

def get_fleet_speed(ships):
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, ships)) / math.log(1000)) ** 1.5

def collides(x1, y1, x2, y2, cx, cy, r):
    vec_x, vec_y = x2 - x1, y2 - y1
    vec_to_cx, vec_to_cy = cx - x1, cy - y1
    vec_length_sq = vec_x**2 + vec_y**2
    if vec_length_sq == 0:
        return (x1 - cx)**2 + (y1 - cy)**2 <= r**2
    closest_point = max(0, min(1, (vec_to_cx * vec_x + vec_to_cy * vec_y) / vec_length_sq))
    closest_x = x1 + closest_point * vec_x
    closest_y = y1 + closest_point * vec_y
    return (closest_x - cx)**2 + (closest_y - cy)**2 <= r**2

def get_planet_trajectories(p, vel):
    planet_trajectories = []
    angle = math.atan2(p.y - 50, p.x - 50)
    r = math.sqrt((p.x - 50)**2 + (p.y - 50)**2)
    for tick in range(1, 61):
        angle_t = angle + vel * tick
        planet_trajectories.append((50 + r * math.cos(angle_t), 50 + r * math.sin(angle_t)))
    return planet_trajectories

def path_collision(src, t, fleet_speed, angle, planets, vel, ticks=61):
    prev_x, prev_y = src.x, src.y
    other_planets = [p for p in planets if p.id != src.id and p.id != t.id]
    planet_params = []
    for p in other_planets:
        if p.id in moving_planets:
            angle_p = math.atan2(p.y - 50, p.x - 50)
            r = math.sqrt((p.x - 50)**2 + (p.y - 50)**2)
            planet_params.append((p.id, True, angle_p, r, p.radius))
        else:
            planet_params.append((p.id, False, p.x, p.y, p.radius))

    for tick in range(1, ticks):
        x = src.x + math.cos(angle) * fleet_speed * tick
        y = src.y + math.sin(angle) * fleet_speed * tick
        if collides(prev_x, prev_y, x, y, 50, 50, 10):
            return True
        for pid, is_moving, val1, val2, r_p in planet_params:
            if is_moving:
                angle_t = val1 + vel * tick
                p_x, p_y = 50 + val2 * math.cos(angle_t), 50 + val2 * math.sin(angle_t)
            else:
                p_x, p_y = val1, val2
            if collides(prev_x, prev_y, x, y, p_x, p_y, r_p):
                return True
        prev_x, prev_y = x, y
    return False

def find_angle_to_planet(p, t, ships, vel, planets, moving=False):
    fleet_speed = get_fleet_speed(ships)
    if moving:
        planet_trajectories = get_planet_trajectories(t, vel)
        for tick, (tx, ty) in enumerate(planet_trajectories, start=1):
            dx, dy = tx - p.x, ty - p.y
            dist_to_target = math.sqrt(dx**2 + dy**2) - p.radius
            if abs(fleet_speed * tick - dist_to_target) > t.radius:
                continue
            angle = math.atan2(dy, dx)
            if path_collision(p, t, fleet_speed, angle, planets, vel, ticks=tick + 1):
                return None, None
            return angle, tick
    else:
        angle = math.atan2(t.y - p.y, t.x - p.x)
        dist = math.sqrt((p.x - t.x)**2 + (p.y - t.y)**2)
        tick = math.floor(dist / fleet_speed)
        if path_collision(p, t, fleet_speed, angle, planets, vel, ticks=tick + 1):
            return None, None
        return angle, tick
    return None, None

def predict_total_ships(m, t, vel, base_ships, m_ships, planets, moving=False):
    total_ships = base_ships
    for _ in range(5):
        angle, arrive_tick = find_angle_to_planet(m, t, total_ships, vel, planets, moving=moving)
        if angle is None: return None, None, None
        new_total_ships = base_ships + arrive_tick * t.production if t.owner != -1 else base_ships
        if new_total_ships > m_ships: return None, None, None
        if new_total_ships == total_ships: break
        total_ships = new_total_ships
    return total_ships, angle, arrive_tick

def plan_coop_attack(attacking_planets, t, base_ships, vel, planets, moving=False):
    remainder = base_ships
    planned = []
    for a_p in attacking_planets:
        p = a_p["planet"]
        p_ships = min(a_p["ships"], remainder)
        if p_ships > 0:
            p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
        if p_ships <= 0: continue
        angle, arrive_tick = find_angle_to_planet(p, t, p_ships, vel, planets, moving=moving)
        if angle is None or arrive_tick is None:
            continue  # FIX: We do not subtract from remainder if collision fails!
        remainder -= p_ships
        planned.append([p, angle, p_ships, arrive_tick])
    return remainder, planned

def find_fast_reinforcement_ships(p_np, p, needed_by_tick, min_ships, max_ships, moving_planets, vel, planets):
    best_ships = None
    low, high = min_ships, max_ships
    while low <= high:
        mid = (low + high) // 2
        angle_np, arrive_tick = find_angle_to_planet(p_np, p, mid, vel, planets, moving=(p.id in moving_planets))
        if angle_np is not None and arrive_tick is not None and arrive_tick <= needed_by_tick:
            best_ships = mid
            high = mid - 1
        else:
            low = mid + 1
    return best_ships

def get_comet_life(comet_id, obs):
    for group in obs.get("comets", []):
        if comet_id in group.get("planet_ids", []):
            idx = group["planet_ids"].index(comet_id)
            paths = group.get("paths", [])
            if idx < len(paths):
                return max(0, len(paths[idx]) - group.get("path_index", 0))
    return 0

def get_custom_score(m, t, obs):
    dist = math.sqrt((m.x - t.x)**2 + (m.y - t.y)**2)
    min_ships = t.ships + 1
    eta = dist / get_fleet_speed(min_ships)

    if t.id in obs.get("comet_planet_ids", []):
        life = get_comet_life(t.id, obs)
        if eta >= life: return -float('inf')
        profit = (t.production * (life - eta)) - min_ships
        if profit <= 0: return -float('inf')
        return (profit * 10) - dist

    enemy_produced = eta * t.production if t.owner != -1 else 0
    enemy_bonus = t.production if t.owner != -1 else 0
    total_ships = min_ships + enemy_produced
    return ((FORMULA_DIST - dist) + (FORMULA_PROD_MULT * t.production)
            + (FORMULA_ENEMY_BONUS_MULT * enemy_bonus)
            - (FORMULA_TOTAL_SHIPS_PERCENT * total_ships) - (2 * eta))

def get_candidate_targets(m, targets, obs):
    candidate_targets = []
    for t in targets:
        score = get_custom_score(m, t, obs)
        if score != -float('inf'):
            candidate_targets.append((m, t, score))
    return sorted(candidate_targets, key=lambda x: x[2], reverse=True)

def build_crash_exploit_missions(lobs, under_attack, vel, exhausted_planets_id):
    moves, planned_trajectories = [], []
    if lobs.get("player_count", 0) < 4: return moves, planned_trajectories
    
    for target_id, data in under_attack.items():
        t = data["planet"]
        if t.owner == lobs.get("player"): continue
        
        enemy_fleets = [f for f in data["fleets"] if f["fleet"].owner not in (-1, lobs.get("player"))]
        by_eta = defaultdict(list)
        for f in enemy_fleets:
            by_eta[f["arrive_tick"]].append((f["fleet"].owner, f["fleet"].ships))
            
        for eta, forces in by_eta.items():
            owners = set(o for o, _ in forces)
            if len(owners) >= 2:
                # We have a crash! Determine total survivor forces
                owner_totals = defaultdict(int)
                for o, s in forces: owner_totals[o] += s
                sorted_forces = sorted(owner_totals.values(), reverse=True)
                survivor_ships = sorted_forces[0] - sorted_forces[1]
                
                capture_ships = survivor_ships + 1
                desired_arrival = eta + 1
                
                # Try to snipe this planet!
                for src in lobs.get("mine", []):
                    if src.id in exhausted_planets_id or src.ships < capture_ships + MIN_SHIPS_MINE_ATTACK:
                        continue
                    angle, arr = find_angle_to_planet(src, t, capture_ships, vel, lobs.get("planets"), moving=(t.id in moving_planets))
                    if angle is not None and abs(arr - desired_arrival) <= 2:
                        moves.append([src.id, angle, capture_ships])
                        exhausted_planets_id.add(src.id)
                        planned_trajectories.append({
                            "mine": src, "target": t, "angle": angle, "ships": capture_ships, "arrive_tick": arr
                        })
                        break # Successfully built snipe
    return moves, planned_trajectories

def get_planets_under_attack(mine, fleets, player, vel):
    mov_pl_traj = {m.id: get_planet_trajectories(m, vel) for m in mine if m.id in moving_planets}
    under_attack, seen = {}, set()
    enemy_fleets = [f for f in fleets if f.owner != player]
    
    for f in enemy_fleets:
        fleet_speed = get_fleet_speed(f.ships)
        prev_x, prev_y = f.x, f.y
        for tick in range(1, 61):
            next_x = f.x + math.cos(f.angle) * fleet_speed * tick
            next_y = f.y + math.sin(f.angle) * fleet_speed * tick
            for m in mine:
                m_x, m_y = mov_pl_traj[m.id][tick-1] if m.id in moving_planets else (m.x, m.y)
                if collides(prev_x, prev_y, next_x, next_y, m_x, m_y, m.radius):
                    if (m.id, f.id) not in seen:
                        if m.id not in under_attack: under_attack[m.id] = {"planet": m, "fleets": []}
                        under_attack[m.id]["fleets"].append({"fleet": f, "arrive_tick": tick})
                        seen.add((m.id, f.id))
            prev_x, prev_y = next_x, next_y
    return under_attack

def refresh_local_obs(obs):
    planets = [ow.Planet(*p) for p in obs.get("planets", [])]
    mine = [p for p in planets if p.owner == obs.get("player")]
    targets = [p for p in planets if p.owner != obs.get("player")]
    return {
        "planets": planets, "mine": mine, "targets": targets,
        "player": obs.get("player", -2), "fleets": [ow.Fleet(*f) for f in obs.get("fleets", [])],
        "player_count": len(set([p.owner for p in planets if p.owner != -1] + [f.owner for f in [ow.Fleet(*f) for f in obs.get("fleets", [])]]))
    }

def get_reinforcement_plans(mine, under_attack):
    reinforcement_plans = {}
    for p in mine:
        if p.id in under_attack:
            attacking_fleets = sorted(under_attack[p.id]["fleets"], key=lambda a: a["arrive_tick"])
            incoming = sorted([r for r in reinforcement_trajectories if r["target"].id == p.id], key=lambda r: r["arrive_tick"])
            
            p_available = p.ships
            prev_tick, r_idx = 0, 0
            for att in attacking_fleets:
                arr = att["arrive_tick"]
                p_available += (arr - prev_tick) * p.production
                while r_idx < len(incoming) and incoming[r_idx]["arrive_tick"] <= arr:
                    p_available += incoming[r_idx]["ships"]
                    r_idx += 1
                p_available -= att["fleet"].ships
                prev_tick = arr
                if p_available < 0:
                    reinforcement_plans[p] = {"ships_needed": max(MIN_SHIPS_MINE_ATTACK, abs(p_available)), "needed_by_tick": arr}
                    break
    return reinforcement_plans

def agent(obs):
    global steps, fleet_trajectories, reinforcement_trajectories, moving_planets
    moves = []
    
    if steps < 2:
        steps += 1
        return moves
    if steps == 2:
        initial_by_id = {i[0]: ow.Planet(*i) for i in obs.get("initial_planets", [])}
        moving_planets = [p[0] for p in obs.get("planets", []) if (p[2], p[3]) != (initial_by_id[p[0]].x, initial_by_id[p[0]].y)]
        steps = 3

    lobs = refresh_local_obs(obs)
    
    # Update fleet tracking
    for f_t in fleet_trajectories[:]:
        if any(f.from_planet_id == f_t["mine"].id and abs(f.angle - f_t["angle"]) < 1e-6 for f in lobs["fleets"]):
            f_t["arrive_tick"] = max(0, f_t["arrive_tick"] - 1)
        else: fleet_trajectories.remove(f_t)
            
    for r_t in reinforcement_trajectories[:]:
        r_t["arrive_tick"] -= 1
        if r_t["arrive_tick"] <= 0: reinforcement_trajectories.remove(r_t)

    under_attack = get_planets_under_attack(lobs["mine"], lobs["fleets"], lobs["player"], obs.angular_velocity)
    exhausted_planets_id = set()
    
    # FFA Crash Exploitation
    cmoves, ctraj = build_crash_exploit_missions(lobs, under_attack, obs.angular_velocity, exhausted_planets_id)
    moves.extend(cmoves)
    fleet_trajectories.extend(ctraj)
    
    if not lobs["targets"]: return moves

    # Priority Reinforcements
    sorted_plans = sorted(get_reinforcement_plans(lobs["mine"], under_attack).items(), key=lambda x: x[0].production, reverse=True)
    for p, plan in sorted_plans:
        if any(r["target"].id == p.id and r["arrive_tick"] >= 0 for r in reinforcement_trajectories): continue
        ships_needed, needed_by_tick = plan["ships_needed"], plan["needed_by_tick"]
        
        nearest = sorted([(m, math.hypot(m.x - p.x, m.y - p.y)) for m in lobs["mine"] if m.id != p.id and m.id not in exhausted_planets_id], key=lambda x: x[1])
        for p_np, _ in nearest:
            avail = p_np.ships - sum(r["ships"] for r in reinforcement_trajectories if r["mine"].id == p_np.id)
            if p_np.id in under_attack: avail = max(0, avail - sum(a["fleet"].ships for a in under_attack[p_np.id]["fleets"]))
            if avail < max(MIN_SHIPS_MINE_ATTACK, ships_needed): continue
            
            boosted = find_fast_reinforcement_ships(p_np, p, needed_by_tick, ships_needed, avail, moving_planets, obs.angular_velocity, lobs["planets"])
            if boosted is not None:
                angle_np, arr = find_angle_to_planet(p_np, p, boosted, obs.angular_velocity, lobs["planets"], moving=(p.id in moving_planets))
                moves.append([p_np.id, angle_np, boosted])
                exhausted_planets_id.add(p_np.id)
                reinforcement_trajectories.append({"mine": p_np, "target": p, "angle": angle_np, "ships": boosted, "arrive_tick": arr})
                break

    # Offensive Actions
    for m in sorted(lobs["mine"], key=lambda p: p.ships, reverse=True):
        if m.id in exhausted_planets_id or m.ships < MIN_SHIPS_MINE_ATTACK: continue
        avail = max(0, m.ships - sum(a["fleet"].ships for a in under_attack[m.id]["fleets"])) if m.id in under_attack else m.ships
        if avail < MIN_SHIPS_MINE_ATTACK: continue

        for _, t, _ in get_candidate_targets(m, lobs["targets"], obs)[:3]:
            safe_nearest = []
            for p, _ in sorted([(x, math.hypot(x.x - t.x, x.y - t.y)) for x in lobs["mine"] if x.id != m.id and x.id not in exhausted_planets_id], key=lambda k: k[1]):
                p_avail = p.ships - sum(a["fleet"].ships for a in under_attack[p.id]["fleets"]) if p.id in under_attack else p.ships
                if p_avail >= MIN_SHIPS_MINE_ATTACK: safe_nearest.append((p, 0, p_avail))
            
            en_route = sum(f["ships"] for f in fleet_trajectories if f["target"].id == t.id)
            needed = t.ships + 1 + (3 * t.production if t.owner != -1 else 0)
            if len(lobs["mine"]) < len(lobs["planets"]) * 0.75 and en_route >= needed: continue
            
            base_ships = max(MIN_SHIPS_MINE_ATTACK, needed - en_route)
            moving = t.id in moving_planets
            
            if avail >= base_ships:
                tot, ang, arr = predict_total_ships(m, t, obs.angular_velocity, base_ships, avail, lobs["planets"], moving)
                if ang is not None:
                    moves.append([m.id, ang, tot])
                    exhausted_planets_id.add(m.id)
                    fleet_trajectories.append({"mine": m, "target": t, "angle": ang, "ships": tot, "arrive_tick": arr})
            elif avail < base_ships and len(lobs["mine"]) > 1 and t.ships >= MIN_SHIPS_TARGET_COOP_ATTACK:
                rem, planned = plan_coop_attack([{"planet": m, "ships": avail}] + [{"planet": p, "ships": s} for p, _, s in safe_nearest[:COOP_PLANET_CAP]], t, base_ships, obs.angular_velocity, lobs["planets"], moving)
                if rem <= 0:
                    for move in planned:
                        fleet_trajectories.append({"mine": move[0], "target": t, "angle": move[1], "ships": move[2], "arrive_tick": move[3]})
                        exhausted_planets_id.add(move[0].id)
                        moves.append([move[0].id, move[1], move[2]])
    return moves
