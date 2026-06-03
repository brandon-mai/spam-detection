import math
import numpy as np
from numba import njit, int32, float32, bool_

@njit(cache=True)
def jit_point_to_segment_dist(px, py, x1, y1, x2, y2):
    """
    Calculates the minimum perpendicular distance from a planet/sun coordinate 
    to a fleet's linear movement vector segment.
    """
    l2 = (x1 - x2)**2 + (y1 - y2)**2
    if l2 == 0:
        return math.hypot(px - x1, py - y1)
    
    t = max(0.0, min(1.0, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return math.hypot(px - proj_x, py - proj_y)

@njit(cache=True)
def jit_predict_orbit_positions(init_x, init_y, ang_vel, ticks):
    """
    Extrapolates circular orbital trajectories around the center sun (50, 50)
    across a forward time horizon.
    """
    cx, cy = 50.0, 50.0
    dx = init_x - cx
    dy = init_y - cy
    radius = math.hypot(dx, dy)
    
    if radius == 0:
        return cx, cy
        
    current_angle = math.atan2(dy, dx)
    future_angle = current_angle + (ang_vel * ticks)
    
    future_x = cx + radius * math.cos(future_angle)
    future_y = cy + radius * math.sin(future_angle)
    return future_x, future_y

@njit(cache=True)
def jit_resolve_fleet_targets(fleet_matrix, planet_matrix, ang_vel):
    """
    Project every active fleet along its heading vector. 
    Perform continuous collision sweeping against static and rotating planets.
    Returns: Array of size [Num_Fleets, 2] containing [Target_Planet_ID, Arrival_Tick].
    """
    num_fleets = fleet_matrix.shape[0]
    num_planets = planet_matrix.shape[0]
    targets = np.zeros((num_fleets, 2), dtype=np.int32)
    
    for f in range(num_fleets):
        fx, fy = fleet_matrix[f, 0], fleet_matrix[f, 1]
        heading = fleet_matrix[f, 2]
        ships = fleet_matrix[f, 3]
        
        if ships <= 0:
            targets[f, 0] = -1
            targets[f, 1] = -1
            continue
            
        true_speed = 1.0 + 5.0 * (math.log(ships) / math.log(1000)) ** 1.5
        
        # Simple Raycast sweeping up to 200 ticks
        found = False
        for tick in range(1, 201):
            next_fx = fx + true_speed * tick * math.cos(heading)
            next_fy = fy + true_speed * tick * math.sin(heading)
            
            # Check collision with any planet
            for p in range(num_planets):
                px, py = planet_matrix[p, 0], planet_matrix[p, 1]
                pradius = planet_matrix[p, 2]
                
                if ang_vel != 0:
                    px, py = jit_predict_orbit_positions(px, py, ang_vel, tick)
                    
                dist = jit_point_to_segment_dist(px, py, fx + true_speed * (tick-1) * math.cos(heading), 
                                                 fy + true_speed * (tick-1) * math.sin(heading),
                                                 next_fx, next_fy)
                
                if dist <= pradius:
                    targets[f, 0] = p
                    targets[f, 1] = tick
                    found = True
                    break
            
            if found:
                break
                
        if not found:
            targets[f, 0] = -1
            targets[f, 1] = -1
            
    return targets

@njit(cache=True)
def jit_step_combat(planet_owners, planet_garrisons, arriving_targets, arriving_owners, arriving_ships):
    """
    Aggregates incoming multi-player forces per node, executes top-minus-second 
    reduction arithmetic, and resolves surface flips.
    """
    num_planets = planet_owners.shape[0]
    num_fleets = arriving_targets.shape[0]
    
    next_owners = planet_owners.copy()
    next_garrisons = planet_garrisons.copy()
    
    for p in range(num_planets):
        # Accumulate forces by player (0=p1, 1=p2, 2=p3, 3=p4, 4=neutral)
        forces = np.zeros(5, dtype=np.int32)
        forces[planet_owners[p]] = planet_garrisons[p]
        
        for f in range(num_fleets):
            if arriving_targets[f] == p:
                owner = arriving_owners[f]
                ships = arriving_ships[f]
                forces[owner] += ships
                
        # Find top 2 forces
        top_idx = -1
        top_force = -1
        second_force = -1
        
        for idx in range(5):
            f_val = forces[idx]
            if f_val > top_force:
                second_force = top_force
                top_force = f_val
                top_idx = idx
            elif f_val > second_force:
                second_force = f_val
                
        if top_force > second_force:
            next_owners[p] = top_idx
            next_garrisons[p] = top_force - second_force
        else:
            # Tie between top 2 -> revert to neutral with 0 garrison, or stay as is?
            # Standard Kaggle Orbit Wars tie rule: if tie for first, planet becomes neutral with 0 ships.
            # But wait, if tie for first is 0, it means no one is there. 
            # If top_force == 0, it's empty neutral.
            if top_force == 0:
                next_owners[p] = 4
                next_garrisons[p] = 0
            else:
                next_owners[p] = 4 # Neutral
                next_garrisons[p] = 0
                
    return next_owners, next_garrisons

@njit(cache=True)
def jit_build_graph_features(planet_matrix, fleet_matrix, player_idx):
    """
    Accelerated function to build node (V) and edge (E) features.
    planet_matrix: [N, 6] -> [owner, x, y, radius, garrison, prod]
    fleet_matrix: [M, 4] -> [x, y, heading, ships]
    """
    num_planets = planet_matrix.shape[0]
    V = np.zeros((num_planets, 13), dtype=np.float32)
    E = np.zeros((num_planets, num_planets, 4), dtype=np.float32)
    
    for i in range(num_planets):
        owner = planet_matrix[i, 0]
        x = planet_matrix[i, 1]
        y = planet_matrix[i, 2]
        radius = planet_matrix[i, 3]
        garrison = planet_matrix[i, 4]
        prod = planet_matrix[i, 5]
        
        V[i, 0] = radius
        V[i, 1] = prod
        V[i, 2] = garrison
        
        if owner == player_idx:
            V[i, 3] = 1.0
        elif owner == 4 or owner == -1:
            V[i, 7] = 1.0
        else:
            rel_idx = int((owner - player_idx - 1) % 3)
            V[i, 4 + rel_idx] = 1.0
            
        V[i, 8] = 0.0 # Is_Comet
        V[i, 12] = prod * 20.0 # Net_Garrison_Delta (stub)
        
    for i in range(num_planets):
        for j in range(num_planets):
            dx = planet_matrix[i, 1] - planet_matrix[j, 1]
            dy = planet_matrix[i, 2] - planet_matrix[j, 2]
            dist = math.hypot(dx, dy)
            E[i, j, 0] = dist
            if i != j and dist > 0:
                hit_sun = jit_point_to_segment_dist(50.0, 50.0, planet_matrix[i, 1], planet_matrix[i, 2], planet_matrix[j, 1], planet_matrix[j, 2])
                E[i, j, 1] = 1.0 if hit_sun < 10.0 else 0.0
                
    if fleet_matrix.shape[0] > 0:
        targets = jit_resolve_fleet_targets(fleet_matrix, planet_matrix[:, 1:4], 0.0)
        for f in range(fleet_matrix.shape[0]):
            tgt = targets[f, 0]
            if tgt != -1 and tgt < num_planets:
                # Add to E[tgt, tgt, 2] as a stub for inbound fleet mass
                E[tgt, tgt, 2] += fleet_matrix[f, 3]
                
    return V, E
