import os
import sys
import math
import numpy as np

# Suppress jax logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "drl_pipeline"))

from orbit_physics_jit import jit_build_graph_features
from gat_model import OrbitGATModel

_model = None
_params = None

def init_model():
    global _model, _params
    _model = OrbitGATModel()
    
    rng = jax.random.PRNGKey(42)
    dummy_V = jnp.zeros((1, 50, 13))
    dummy_E = jnp.zeros((1, 50, 50, 4))
    dummy_mode = jnp.zeros((1, 1))
    
    variables = _model.init(rng, dummy_V, dummy_E, dummy_mode, true_source=jnp.array([0]), true_target=jnp.array([0]))
    _params = variables['params']
    
    ckpt_path = os.path.join(project_root, "drl_pipeline", "checkpoints", "bc_model_4p_finetuned.msgpack")
    with open(ckpt_path, "rb") as f:
        _params = from_bytes(_params, f.read())

@jax.jit
def predict_action(params, V, E, mode):
    logits_S, logits_T, logits_Q = _model.apply({'params': params}, V, E, mode)
    src = jnp.argmax(logits_S[0])
    tgt = jnp.argmax(logits_T[0])
    quota = jnp.argmax(logits_Q[0])
    return src, tgt, quota

def predict_future_position(target_id, future_tick, planet_matrix):
    """Extrapolates circular orbital path coordinates for interception."""
    cx, cy = 50.0, 50.0
    px, py = planet_matrix[target_id, 1], planet_matrix[target_id, 2]
    
    # Check if target is a moving planet (heuristic default speed check)
    # Standard orbit settings apply to non-static assets
    dx, dy = px - cx, py - cy
    radius = math.hypot(dx, dy)
    if radius == 0:
        return px, py
        
    # Orbit speed estimation matching game configuration mechanics
    orbit_speed = 0.01 if target_id % 2 == 0 else 0.015 
    current_angle = math.atan2(dy, dx)
    future_angle = current_angle + (orbit_speed * future_tick)
    
    return cx + radius * math.cos(future_angle), cy + radius * math.sin(future_angle)

def process_atomic_drl_action(source_id, target_id, quota_index, planet_matrix, current_step):
    if source_id == target_id or target_id == 50:
        return []
        
    sx, sy = planet_matrix[source_id, 1], planet_matrix[source_id, 2]
    sgarrison = planet_matrix[source_id, 4]
    sprod = planet_matrix[source_id, 5]
    
    safety_floor = max(10, sprod * 3)
    available_ships = sgarrison - safety_floor
    if available_ships <= 0:
        return []
        
    quota_map = {0: 0.25, 1: 0.50, 2: 1.00}
    quota = quota_map[int(quota_index)]
    ship_payload = math.floor(available_ships * quota)
    if ship_payload <= 0:
        return []
        
    # PHYSICAL SPEED CALCULATION: Speed is strictly logarithmic based on mass
    true_speed = 1.0 + 5.0 * (math.log(ship_payload) / math.log(1000)) ** 1.5
    
    # Fixed-Point Intercept Loop
    estimated_eta = 20
    for _ in range(5):
        future_tx, future_ty = predict_future_position(target_id, estimated_eta, planet_matrix)
        distance = math.hypot(future_tx - sx, future_ty - sy)
        estimated_eta = max(1, math.ceil(distance / true_speed))
        
    final_tx, final_ty = predict_future_position(target_id, estimated_eta, planet_matrix)
    launch_angle = math.atan2(final_ty - sy, final_tx - sx)
    
    return [[int(source_id), float(launch_angle), int(ship_payload)]]

def agent(obs, config):
    global _model, _params
    if _model is None:
        init_model()
        
    planets_data = obs.get("planets", [])
    fleets_data = obs.get("fleets", [])
    
    planets_items = [(int(k), v) for k, v in planets_data.items()] if isinstance(planets_data, dict) else list(enumerate(planets_data))
    fleets_items = [(int(k), v) for k, v in fleets_data.items()] if isinstance(fleets_data, dict) else list(enumerate(fleets_data))
    
    num_planets = 50
    planet_matrix = np.zeros((num_planets, 6), dtype=np.float32)
    
    # CORRECTION: Explicitly structure fleet matrix columns to match the 4-column feature builder specification
    fleet_matrix = np.zeros((len(fleets_items), 4), dtype=np.float32)
    
    for pid, pdata in planets_items:
        owner, px, py, radius, garrison, prod = pdata[1:7] if len(pdata) == 7 else pdata[0:6]
        planet_matrix[pid, 0] = owner
        planet_matrix[pid, 1] = px
        planet_matrix[pid, 2] = py
        planet_matrix[pid, 3] = radius
        planet_matrix[pid, 4] = garrison
        planet_matrix[pid, 5] = prod
        
    for i, (fid, fdata) in enumerate(fleets_items):
        owner, fx, fy, heading, ships = fdata[1:6] if len(fdata) == 6 else fdata[0:5]
        fleet_matrix[i, 0] = fx
        fleet_matrix[i, 1] = fy
        fleet_matrix[i, 2] = heading
        fleet_matrix[i, 3] = ships
        
    player_id = obs.get("player", 0)
    num_agents = config.get("num_agents", 4) if isinstance(config, dict) else getattr(config, "num_agents", 4)
        
    mode_flag = 0.0 if num_agents == 2 else 1.0
    mode_jnp = jnp.array([[mode_flag]])
    
    actions_to_launch = []
    current_step = obs.get("step", 0)
    
    # Autoregressive intra-tick action assignment loop
    for _ in range(5):
        V, E = jit_build_graph_features(planet_matrix, fleet_matrix, player_id)
        V_jnp = jnp.expand_dims(jnp.array(V), 0)
        E_jnp = jnp.expand_dims(jnp.array(E), 0)
        
        src, tgt, quota = predict_action(_params, V_jnp, E_jnp, mode_jnp)
        src, tgt, quota = int(src), int(tgt), int(quota)

        print(f"DEBUG - Raw Model Output -> Src ID: {int(src)}, Tgt ID: {int(tgt)}, Quota Bin: {int(quota)}")
        # Check what the safety governor sees right after the model picks them
        sgarrison = planet_matrix[int(src), 4]
        sprod = planet_matrix[int(src), 5]
        safety_floor = max(10, sprod * 3)
        print(f"DEBUG - Safety Check -> Src Garrison: {sgarrison}, Calculated Floor: {safety_floor}, Safe Available: {sgarrison - safety_floor}")
        
        if src == tgt or tgt == 50:
            break
            
        act = process_atomic_drl_action(src, tgt, quota, planet_matrix, current_step)
        if len(act) == 0:
            break
            
        actions_to_launch.append(act[0])
        planet_matrix[src, 4] -= act[0][2] # Decrement remaining forces in real-time
        
    return actions_to_launch