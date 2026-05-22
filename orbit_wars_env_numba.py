import numpy as np
import math
from numba import njit, prange

MAX_PLANETS = 60 # Increased to accommodate comets
MAX_FLEETS = 1000

@njit(cache=True)
def distance(p1_x, p1_y, p2_x, p2_y):
    return math.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)

@njit(cache=True)
def point_to_segment_distance(px, py, vx, vy, wx, wy):
    """Minimum distance from point p to line segment v-w."""
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    if l2 == 0.0:
        return distance(px, py, vx, vy)
    t = max(0.0, min(1.0, ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2))
    proj_x = vx + t * (wx - vx)
    proj_y = vy + t * (wy - vy)
    return distance(px, py, proj_x, proj_y)

@njit(cache=True)
def swept_pair_hit(Ax, Ay, Bx, By, P0x, P0y, P1x, P1y, r):
    """True iff a fleet moving A->B and a planet moving P0->P1 come within r
    of each other for some t in [0, 1]."""
    d0x = Ax - P0x
    d0y = Ay - P0y
    dvx = (Bx - Ax) - (P1x - P0x)
    dvy = (By - Ay) - (P1y - P0y)
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

@njit(cache=True)
def get_fleet_speed(ships, max_speed=6.0):
    if ships < 1:
        ships = 1
    return 1.0 + (max_speed - 1.0) * (math.log(float(ships)) / math.log(1000.0)) ** 1.5

@njit(cache=True)
def step_simulation(planets_state, fleets_state, angular_velocity):
    """
    planets_state: [num_planets, 8] - id, owner, x, y, radius, ships, production, is_moving
    fleets_state: [num_fleets, 7] - id, owner, x, y, angle, from_planet_id, ships
    """
    num_planets = planets_state.shape[0]
    num_fleets = fleets_state.shape[0]

    fleet_survives = np.ones(num_fleets, dtype=np.int32)
    planet_incoming_attackers = np.zeros((num_planets, 4), dtype=np.float32)

    # Pre-calculate planet paths for this tick
    # shape: [num_planets, 4] -> old_x, old_y, new_x, new_y
    planet_paths = np.zeros((num_planets, 4), dtype=np.float32)
    for j in range(num_planets):
        p_x = planets_state[j, 2]
        p_y = planets_state[j, 3]
        planet_paths[j, 0] = p_x
        planet_paths[j, 1] = p_y
        
        is_moving = int(planets_state[j, 7])
        if is_moving == 1:
            dx = p_x - 50.0
            dy = p_y - 50.0
            r = math.sqrt(dx**2 + dy**2)
            angle = math.atan2(dy, dx)
            # Numba simulation tracks current rotation, we don't have initial positions
            # so we just add angular velocity
            current_angle = angle + angular_velocity
            planet_paths[j, 2] = 50.0 + r * math.cos(current_angle)
            planet_paths[j, 3] = 50.0 + r * math.sin(current_angle)
        else:
            planet_paths[j, 2] = p_x
            planet_paths[j, 3] = p_y

    for i in range(num_fleets):
        f_owner = int(fleets_state[i, 1])
        f_x = fleets_state[i, 2]
        f_y = fleets_state[i, 3]
        f_angle = fleets_state[i, 4]
        f_ships = fleets_state[i, 6]
        
        f_speed = get_fleet_speed(f_ships)
        next_x = f_x + math.cos(f_angle) * f_speed
        next_y = f_y + math.sin(f_angle) * f_speed
        
        hit_planet = False
        # 1. Planet collisions (swept continuous)
        for j in range(num_planets):
            p_old_x = planet_paths[j, 0]
            p_old_y = planet_paths[j, 1]
            p_new_x = planet_paths[j, 2]
            p_new_y = planet_paths[j, 3]
            p_r = planets_state[j, 4]
            
            if p_r <= 0.0: continue # ignore dead/empty slots
            
            if swept_pair_hit(f_x, f_y, next_x, next_y, p_old_x, p_old_y, p_new_x, p_new_y, p_r):
                planet_incoming_attackers[j, f_owner] += f_ships
                fleet_survives[i] = 0
                hit_planet = True
                break
                
        if hit_planet:
            continue
            
        # 2. Out of bounds
        if next_x < 0 or next_x > 100 or next_y < 0 or next_y > 100:
            fleet_survives[i] = 0
            continue
            
        # 3. Sun collision
        if point_to_segment_distance(50.0, 50.0, f_x, f_y, next_x, next_y) < 10.0:
            fleet_survives[i] = 0
            continue

        # If survived
        fleets_state[i, 2] = next_x
        fleets_state[i, 3] = next_y

    # Production
    for j in range(num_planets):
        owner = int(planets_state[j, 1])
        production = planets_state[j, 6]
        if owner != -1 and planets_state[j, 4] > 0:
            planets_state[j, 5] += production
            
    # Apply planet movement
    for j in range(num_planets):
        planets_state[j, 2] = planet_paths[j, 2]
        planets_state[j, 3] = planet_paths[j, 3]

    # Combat Resolution
    for j in range(num_planets):
        attackers = planet_incoming_attackers[j]
        if np.sum(attackers) == 0:
            continue
            
        p_owner = int(planets_state[j, 1])
        
        # Sort attackers manually for Numba
        # Elements are (owner, ships)
        # Using a fixed size array for the 4 players
        att_counts = np.zeros((4, 2), dtype=np.float32)
        count = 0
        for owner in range(4):
            if attackers[owner] > 0:
                att_counts[count, 0] = owner
                att_counts[count, 1] = attackers[owner]
                count += 1
                
        if count == 0:
            continue
            
        # Sort descending by ships
        for a in range(count):
            for b in range(a + 1, count):
                if att_counts[b, 1] > att_counts[a, 1]:
                    temp_owner = att_counts[a, 0]
                    temp_ships = att_counts[a, 1]
                    att_counts[a, 0] = att_counts[b, 0]
                    att_counts[a, 1] = att_counts[b, 1]
                    att_counts[b, 0] = temp_owner
                    att_counts[b, 1] = temp_ships
                    
        if count > 1:
            largest_owner = int(att_counts[0, 0])
            largest_ships = att_counts[0, 1]
            second_ships = att_counts[1, 1]
            
            surviving_ships = largest_ships - second_ships
            if largest_ships == second_ships:
                surviving_ships = 0
                
            survivor_owner = largest_owner if surviving_ships > 0 else -1
            
            if surviving_ships > 0:
                if p_owner == survivor_owner:
                    planets_state[j, 5] += surviving_ships
                else:
                    planets_state[j, 5] -= surviving_ships
                    if planets_state[j, 5] < 0:
                        planets_state[j, 1] = survivor_owner
                        planets_state[j, 5] = -planets_state[j, 5]
        else:
            largest_owner = int(att_counts[0, 0])
            largest_ships = att_counts[0, 1]
            if p_owner == largest_owner:
                planets_state[j, 5] += largest_ships
            else:
                planets_state[j, 5] -= largest_ships
                if planets_state[j, 5] < 0:
                    planets_state[j, 1] = largest_owner
                    planets_state[j, 5] = -planets_state[j, 5]

    # Filter out dead fleets
    new_fleets_state = fleets_state[fleet_survives == 1]
    return planets_state, new_fleets_state
