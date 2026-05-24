import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import math

# Add root dir to sys.path so we can import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EntityTransformer
from rl_env_wrapper import extract_tensors, get_autoregressive_mask, MAX_PLANETS
from flax import serialization

# Global state to prevent recompilation and reloading per step
_model = None
_params = None
_infer = None
_local_state_tracking = {}
_last_step = -1
_last_mtime = 0.0

def initialize():
    global _model, _params, _infer, _last_mtime
    champion_path = "champion.msgpack"
    
    if not os.path.exists(champion_path):
        return
        
    global _model, _params, _infer, _last_mtime
    champion_path = "champion.msgpack"
    
    if not os.path.exists(champion_path):
        return
        
    current_mtime = os.path.getmtime(champion_path)
    if _model is not None and current_mtime <= _last_mtime:
        return
        
    _last_mtime = current_mtime
    
    if _model is None:
        _model = EntityTransformer()
        
    with open(champion_path, "rb") as f:
        bytes_data = f.read()
    
    rng = jax.random.PRNGKey(0)
    dummy_p = jnp.zeros((1, 60, 14))
    dummy_f = jnp.zeros((1, 100, 9))
    dummy_g = jnp.zeros((1, 4))
    variables = _model.init(rng, dummy_p, dummy_f, dummy_g)
    
    _params = serialization.from_bytes(variables['params'], bytes_data)
    
    @jax.jit
    def infer_fn(params, p, f, g):
        return _model.apply({"params": params}, p, f, g)
        
    _infer = infer_fn
    
def agent(obs, config):
    global _local_state_tracking, _last_step
    initialize()
    
    # Kaggle obs is a dict. If it's a new episode/tick, reset local tracking
    step = obs.get("step", 0)
    if step != _last_step:
        _local_state_tracking = {p[0]: p[5] for p in obs.get("planets", [])}
        _last_step = step
        
    player_id = obs.get("player", 0)
    PASS_TOKEN_INDEX = MAX_PLANETS * MAX_PLANETS * 4
    
    actions_taken = 0
    MAX_USEFUL_ACTIONS = min(5, max(1, len([p for p in obs.get("planets", []) if p[1] == player_id]) * 4))
    moves = []
    
    # Hoist extraction
    p_mat_base, f_mat, g_vec, all_planets = extract_tensors(obs, _local_state_tracking)
    
    while actions_taken < MAX_USEFUL_ACTIONS:
        p_mat = p_mat_base.copy()
        for idx, p in enumerate(all_planets):
            p_mat[idx, 8] = _local_state_tracking.get(p["id"], p["ships"]) / 1000.0
            
        mask = get_autoregressive_mask(player_id, all_planets, _local_state_tracking)
        
        logits, _ = _infer(
            _params, 
            np.expand_dims(p_mat, 0), 
            np.expand_dims(f_mat, 0), 
            np.expand_dims(g_vec, 0)
        )
        
        logits = np.array(logits[0])
        logits[~mask] = -np.inf
        
        # Stochastic selection for the champion to prevent Challenger overfitting
        probs = np.exp(logits - np.max(logits)) # stability
        probs = probs / np.sum(probs)
        
        if np.isnan(probs).any():
            action = PASS_TOKEN_INDEX
        else:
            action = np.random.choice(len(probs), p=probs)
        
        if action == PASS_TOKEN_INDEX:
            break
            
        bin_idx = action % 4
        rem = action // 4
        t_idx = rem % MAX_PLANETS
        s_idx = rem // MAX_PLANETS
        
        if s_idx < len(all_planets) and t_idx < len(all_planets):
            s = all_planets[s_idx]
            t = all_planets[t_idx]
            alloc_pct = [0.25, 0.50, 0.75, 1.0][bin_idx]
            current_ships = _local_state_tracking.get(s["id"], 0)
            ships_to_send = int(current_ships * alloc_pct)
            
            if ships_to_send > 0:
                angle = math.atan2(t["y"] - s["y"], t["x"] - s["x"])
                moves.append([s["id"], angle, ships_to_send])
                _local_state_tracking[s["id"]] -= ships_to_send
                
        actions_taken += 1
        
    return moves
