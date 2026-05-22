import os
import sys
import numpy as np
import math

try:
    sys.path.append(os.path.dirname(__file__))
except NameError:
    pass

from rl_env_wrapper import extract_tensors, get_autoregressive_mask, MAX_PLANETS

# Dummy Entity Transformer model to demonstrate the architecture
class DummyEntityTransformer:
    def __init__(self):
        # We output a token per combination
        self.output_dim = MAX_PLANETS * MAX_PLANETS * 4 + 1
        self.w = np.random.randn(MAX_PLANETS * 14 + MAX_PLANETS * 9 + 4, self.output_dim)
        
    def predict(self, planet_matrix, fleet_matrix, global_vec, mask):
        x = np.concatenate([planet_matrix.flatten(), fleet_matrix.flatten()[:MAX_PLANETS*9], global_vec])
        logits = np.dot(x, self.w)
        logits[~mask] = -np.inf
        return np.argmax(logits)

model = DummyEntityTransformer()

def agent(obs):
    player_id = obs.get("player", 0)
    raw_planets = obs.get("planets", [])
    
    my_planets = [p for p in raw_planets if p[1] == player_id]
    
    # Track ships dynamically to deplete them during the autoregressive loop
    local_state_tracking = {p[0]: p[5] for p in raw_planets}
    
    actions_taken = 0
    MAX_USEFUL_ACTIONS = max(1, len(my_planets) * 4)
    PASS_TOKEN_INDEX = MAX_PLANETS * MAX_PLANETS * 4
    
    moves = []
    
    while actions_taken < MAX_USEFUL_ACTIONS:
        planet_matrix, fleet_matrix, global_vec, all_planets = extract_tensors(obs, local_state_tracking)
        action_mask = get_autoregressive_mask(player_id, all_planets, local_state_tracking)
        
        # Inference
        action_token = model.predict(planet_matrix, fleet_matrix, global_vec, action_mask)
        
        if action_token == PASS_TOKEN_INDEX:
            break
            
        # Decode action
        bin_idx = action_token % 4
        rem = action_token // 4
        t_idx = rem % MAX_PLANETS
        s_idx = rem // MAX_PLANETS
        
        if s_idx < len(all_planets) and t_idx < len(all_planets):
            s = all_planets[s_idx]
            t = all_planets[t_idx]
            
            alloc_pct = [0.25, 0.50, 0.75, 1.0][bin_idx]
            current_ships = local_state_tracking.get(s["id"], 0)
            ships_to_send = int(current_ships * alloc_pct)
            
            if ships_to_send > 0:
                angle = math.atan2(t["y"] - s["y"], t["x"] - s["x"])
                moves.append([s["id"], angle, ships_to_send])
                
                # Dynamic state depletion
                local_state_tracking[s["id"]] -= ships_to_send
                
        actions_taken += 1
        
    return moves
