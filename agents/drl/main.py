import os
import sys
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils.gat_model import OrbitGATModel
from utils.features import build_graph_features
from utils.actions import process_atomic_drl_action

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
    
    ckpt_path = os.path.join(current_dir, "../../drl_pipeline/checkpoints/bc_model_2p.ckpt")
    with open(ckpt_path, "rb") as f:
        _params = from_bytes(_params, f.read())

@jax.jit
def predict_action(params, V, E, mode):
    logits_S, logits_T, logits_Q = _model.apply({'params': params}, V, E, mode)
    src = jnp.argmax(logits_S[0])
    tgt = jnp.argmax(logits_T[0])
    quota = jnp.argmax(logits_Q[0])
    return src, tgt, quota

def agent(obs, config):
    global _model, _params
    if _model is None:
        init_model()
        
    player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    num_agents = config.get("num_agents", 4) if isinstance(config, dict) else getattr(config, "num_agents", 4)
        
    mode_flag = 0.0 if num_agents == 2 else 1.0
    mode_jnp = jnp.array([[mode_flag]])
    
    actions_to_launch = []
    
    for _ in range(5):
        V, E, movement, obs_tensors = build_graph_features(obs, config)
        
        V_jnp = jnp.expand_dims(jnp.array(V), 0)
        E_jnp = jnp.expand_dims(jnp.array(E), 0)
        
        src, tgt, quota = predict_action(_params, V_jnp, E_jnp, mode_jnp)
        src, tgt, quota = int(src), int(tgt), int(quota)
        
        if src == tgt or tgt >= 50:
            break
            
        act = process_atomic_drl_action(src, tgt, quota, movement, obs_tensors, player_id)
        if len(act) == 0:
            break
            
        actions_to_launch.append(act[0])
        
        # Deduct ships from obs for the next auto-regressive loop
        ship_payload = act[0][2]
        src_str = str(src)
        if "planets" in obs and src_str in obs["planets"]:
            pdata = obs["planets"][src_str]
            # pdata is [owner, x, y, radius, garrison, prod] or [id, owner, x, y, radius, garrison, prod]
            if len(pdata) == 7:
                obs["planets"][src_str][4] -= ship_payload
            else:
                obs["planets"][src_str][4] -= ship_payload
        
    return actions_to_launch
