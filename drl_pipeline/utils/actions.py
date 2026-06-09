import torch
import math
from utils.orbit_lite.intercept_aim import intercept_angle

def process_atomic_drl_action(source_id, target_id, quota_index, movement, obs_tensors, player_id):
    if source_id == target_id or target_id == 50:
        return []
        
    device = obs_tensors["planets"].device
    
    planets = obs_tensors["planets"]
    sgarrison = planets[source_id, 4].item()
    sprod = planets[source_id, 5].item()
    
    safety_floor = max(10, sprod * 3)
    available_ships = sgarrison - safety_floor
    if available_ships <= 0:
        return []
        
    quota_map = {0: 0.25, 1: 0.50, 2: 1.00}
    quota = quota_map[int(quota_index)]
    ship_payload = math.floor(available_ships * quota)
    if ship_payload <= 0:
        return []
        
    src_t = torch.tensor([source_id], device=device, dtype=torch.long)
    tgt_t = torch.tensor([target_id], device=device, dtype=torch.long)
    ships_t = torch.tensor([ship_payload], device=device, dtype=torch.float32)
    
    aim = intercept_angle(
        movement,
        source_slots=src_t,
        target_slots=tgt_t,
        fleet_sizes=ships_t,
    )
    
    if not aim["viable"][0].item():
        return []
        
    launch_angle = aim["angle"][0].item()
    return [[int(source_id), float(launch_angle), int(ship_payload)]]
