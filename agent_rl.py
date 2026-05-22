import os
import sys
import numpy as np
import math

try:
    sys.path.append(os.path.dirname(__file__))
except NameError:
    pass

import jax
import jax.numpy as jnp
from flax import serialization
from rl_env_wrapper import extract_tensors, get_autoregressive_mask, MAX_PLANETS
from models import EntityTransformer

# Initialize the real EntityTransformer
model = EntityTransformer()

# Generate dummy variables to establish the parameter tree structure
dummy_p = jnp.zeros((1, 60, 14))
dummy_f = jnp.zeros((1, 1000, 9))
dummy_g = jnp.zeros((1, 4))
variables = model.init(jax.random.PRNGKey(0), dummy_p, dummy_f, dummy_g)

# Load the trained weights from the binary msgpack file
weights_path = os.path.join(os.path.dirname(__file__), "model_weights.msgpack")
if os.path.exists(weights_path):
    with open(weights_path, "rb") as f:
        trained_params = serialization.from_bytes(variables['params'], f.read())
else:
    print(f"WARNING: Checkpoint {weights_path} not found! Using random initialization.")
    trained_params = variables['params']

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
        
        # CPU JAX Inference
        logits, _ = model.apply(
            {'params': trained_params}, 
            jnp.expand_dims(planet_matrix, 0), 
            jnp.expand_dims(fleet_matrix, 0), 
            jnp.expand_dims(global_vec, 0)
        )
        logits = np.array(logits[0])
        logits[~action_mask] = -np.inf
        action_token = int(np.argmax(logits))
        
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
