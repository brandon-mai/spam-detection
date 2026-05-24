import numpy as np
import math
import sys
import os
import logging

# Silence stderr cleanly using file descriptor redirection
stderr_fd = sys.stderr.fileno()
dup_stderr = os.dup(stderr_fd)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, stderr_fd)
logging.disable(logging.CRITICAL)

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
finally:
    logging.disable(logging.NOTSET)
    os.dup2(dup_stderr, stderr_fd)
    os.close(devnull)
    os.close(dup_stderr)
from orbit_wars_env_numba import point_to_segment_distance

MAX_PLANETS = 60
MAX_FLEETS = 100

def extract_tensors(obs, local_state_tracking):
    """
    Constructs the Planet Matrix, Fleet Matrix, and Global Context Vector.
    local_state_tracking: dictionary mapping planet_id -> current ships 
                          (dynamically updated during autoregressive loop).
    """
    player_id = obs.get("player", 0)
    step = obs.get("step", 0)
    
    raw_planets = obs.get("planets", [])
    raw_fleets = obs.get("fleets", [])
    comet_planet_ids = set(obs.get("comet_planet_ids", []))
    
    # 1. Global Context Vector
    total_ships = np.zeros(4, dtype=np.float32)
    for p in raw_planets:
        owner = p[1]
        ships = local_state_tracking.get(p[0], p[5]) # Use dynamic ships
        if owner != -1:
            total_ships[owner] += ships
    for f in raw_fleets:
        owner = f[1]
        total_ships[owner] += f[6]
        
    world_ships = np.sum(total_ships)
    if world_ships == 0: world_ships = 1
    my_share = total_ships[player_id] / world_ships
    
    active_comets_count = len([p for p in raw_planets if p[0] in comet_planet_ids])
    
    global_vec = np.array([
        step / 500.0, 
        my_share, 
        len(raw_fleets) / 500.0, 
        active_comets_count / 10.0
    ], dtype=np.float32)
    
    # 2. Planet Matrix
    planet_matrix = np.zeros((MAX_PLANETS, 14), dtype=np.float32)
    
    # Incorporate Ghost Nodes (future comets from obs.comets paths)
    ghost_nodes = []
    # Kaggle obs.comets structure: [{"planet_ids": [...], "paths": [...], "path_index": -1}, ...]
    for group in obs.get("comets", []):
        if group.get("path_index", 0) < 0: # Hasn't spawned yet
            for i, p_id in enumerate(group["planet_ids"]):
                path = group["paths"][i]
                if path:
                    first_pos = path[0]
                    # We estimate it spawns in a few ticks
                    ghost_nodes.append({
                        "id": p_id,
                        "owner": -1,
                        "x": first_pos[0],
                        "y": first_pos[1],
                        "radius": 1.0,
                        "ships": 0, # Don't know initial ships for sure, but usually low
                        "production": 1,
                        "is_comet": True,
                        "turns_exp": len(path)
                    })
    
    all_planets = []
    for p in raw_planets:
        is_comet = p[0] in comet_planet_ids
        turns_exp = 50 if is_comet else 0 # Approximate if active
        all_planets.append({
            "id": p[0],
            "owner": p[1],
            "x": p[2],
            "y": p[3],
            "radius": p[4],
            "ships": local_state_tracking.get(p[0], p[5]),
            "production": p[6],
            "is_comet": is_comet,
            "turns_exp": turns_exp
        })
    all_planets.extend(ghost_nodes)
    
    for idx, p in enumerate(all_planets):
        if idx >= MAX_PLANETS: break
        planet_matrix[idx, 0] = p["x"] / 100.0
        planet_matrix[idx, 1] = p["y"] / 100.0
        planet_matrix[idx, 2] = p["radius"] / 10.0
        
        # Ownership
        owner = p["owner"]
        if owner == player_id:
            planet_matrix[idx, 3] = 1.0
        elif owner == -1:
            planet_matrix[idx, 7] = 1.0
        else:
            # Map other players to fixed slots (4, 5, 6) relative to me
            opp_slot = 4 + (owner if owner < player_id else owner - 1)
            if opp_slot <= 6:
                planet_matrix[idx, opp_slot] = 1.0
                
        planet_matrix[idx, 8] = p["ships"] / 1000.0
        planet_matrix[idx, 9] = p["production"] / 5.0
        planet_matrix[idx, 10] = 0.0 # inc friendly
        planet_matrix[idx, 11] = 0.0 # inc enemy
        planet_matrix[idx, 12] = 1.0 if p["is_comet"] else 0.0
        planet_matrix[idx, 13] = p["turns_exp"] / 500.0
        
    # 3. Fleet Matrix
    fleet_matrix = np.zeros((MAX_FLEETS, 9), dtype=np.float32)
    for idx, f in enumerate(raw_fleets):
        if idx >= MAX_FLEETS: break
        fleet_matrix[idx, 0] = f[2] / 100.0
        fleet_matrix[idx, 1] = f[3] / 100.0
        fleet_matrix[idx, 2] = math.cos(f[4])
        fleet_matrix[idx, 3] = math.sin(f[4])
        fleet_matrix[idx, 4] = f[6] / 1000.0
        
        # Ownership one-hot
        owner = f[1]
        if owner == player_id:
            fleet_matrix[idx, 5] = 1.0
        else:
            opp_slot = 6 + (owner if owner < player_id else owner - 1)
            if opp_slot <= 8:
                fleet_matrix[idx, opp_slot] = 1.0

    return planet_matrix, fleet_matrix, global_vec, all_planets

def get_autoregressive_mask(player_id, all_planets, local_state_tracking):
    """
    Returns boolean mask of shape (MAX_PLANETS * MAX_PLANETS * 4 + 1).
    Takes dynamically updated local_state_tracking into account.
    """
    num_actions = MAX_PLANETS * MAX_PLANETS * 4 + 1
    mask = np.zeros(num_actions, dtype=np.bool_)
    
    # PASS action is always valid (index 0 or last index. Let's use last index)
    PASS_INDEX = num_actions - 1
    mask[PASS_INDEX] = True
    
    MIN_SHIPS_MINE_ATTACK = 5
    
    for s_idx, s in enumerate(all_planets):
        if s_idx >= MAX_PLANETS: break
        if s["owner"] != player_id:
            continue
            
        current_ships = local_state_tracking.get(s["id"], s["ships"])
        
        for t_idx, t in enumerate(all_planets):
            if t_idx >= MAX_PLANETS: break
            
            # Simple analytical pathing
            angle = math.atan2(t["y"] - s["y"], t["x"] - s["x"])
            dist = math.sqrt((t["x"] - s["x"])**2 + (t["y"] - s["y"])**2)
            next_x = s["x"] + math.cos(angle) * dist
            next_y = s["y"] + math.sin(angle) * dist
            
            # Intersects Sun
            if point_to_segment_distance(50.0, 50.0, s["x"], s["y"], next_x, next_y) < 10.0:
                continue
                
            for bin_idx in range(4):
                alloc_pct = [0.25, 0.50, 0.75, 1.0][bin_idx]
                ships_to_send = int(current_ships * alloc_pct)
                
                if ships_to_send >= MIN_SHIPS_MINE_ATTACK:
                    idx = s_idx * (MAX_PLANETS * 4) + t_idx * 4 + bin_idx
                    mask[idx] = True
                    
    return mask

