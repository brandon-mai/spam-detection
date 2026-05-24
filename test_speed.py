from kaggle_environments import make
from models import EntityTransformer
import jax
import jax.numpy as jnp
import numpy as np
import time
from rl_env_wrapper import extract_tensors, get_autoregressive_mask, MAX_PLANETS

def test():
    env = make('orbit_wars', debug=False)
    # trainer = env.train([None, "random"])
    trainer = env.train([None, "agents/owproto_v2.py"])
    obs = trainer.reset()
    
    model = EntityTransformer()
    rng = jax.random.PRNGKey(0)
    dummy_p = jnp.zeros((1, 60, 14))
    dummy_f = jnp.zeros((1, 1000, 9))
    dummy_g = jnp.zeros((1, 4))
    variables = model.init(rng, dummy_p, dummy_f, dummy_g)
    current_weights = variables['params']
    
    @jax.jit
    def infer(params, p, f, g):
        return model.apply({"params": params}, p, f, g)
        
    # dummy call to compile
    infer(current_weights, dummy_p, dummy_f, dummy_g)
    
    start = time.time()
    steps = 0
    done = False
    
    local_state_tracking = {p[0]: p[5] for p in obs.get("planets", [])}
    player_id = obs.get("player", 0)
    PASS_TOKEN_INDEX = MAX_PLANETS * MAX_PLANETS * 4
    
    while not done:
        actions_taken = 0
        MAX_USEFUL_ACTIONS = max(1, len([p for p in obs.get("planets", []) if p[1] == player_id]) * 4)
        moves = []
        
        while actions_taken < MAX_USEFUL_ACTIONS:
            p_mat, f_mat, g_vec, all_planets = extract_tensors(obs, local_state_tracking)
            mask = get_autoregressive_mask(player_id, all_planets, local_state_tracking)
            
            logits, value = infer(
                current_weights, 
                np.expand_dims(p_mat, 0), 
                np.expand_dims(f_mat, 0), 
                np.expand_dims(g_vec, 0)
            )
            
            logits = np.array(logits[0])
            logits[~mask] = -np.inf
            
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            
            if np.isnan(probs).any():
                action = PASS_TOKEN_INDEX
            else:
                action = np.random.choice(len(probs), p=probs)
                
            if action == PASS_TOKEN_INDEX:
                break
                
            # Decode action
            bin_idx = action % 4
            rem = action // 4
            t_idx = rem % MAX_PLANETS
            s_idx = rem // MAX_PLANETS
            if s_idx < len(all_planets) and t_idx < len(all_planets):
                s = all_planets[s_idx]
                t = all_planets[t_idx]
                alloc_pct = [0.25, 0.50, 0.75, 1.0][bin_idx]
                current_ships = local_state_tracking.get(s["id"], 0)
                ships_to_send = int(current_ships * alloc_pct)
                if ships_to_send > 0:
                    angle = np.arctan2(t["y"] - s["y"], t["x"] - s["x"])
                    moves.append([s["id"], angle, ships_to_send])
                    local_state_tracking[s["id"]] -= ships_to_send
                    
            actions_taken += 1
            
        next_obs, reward, done, info = trainer.step(moves)
        obs = next_obs
        local_state_tracking = {p[0]: p[5] for p in obs.get("planets", [])}
        steps += 1
        
        if steps % 50 == 0:
            print(f"Step {steps}, Time elapsed: {time.time() - start:.2f}s")
            
    print(f"Finished episode in {time.time() - start:.2f}s, Reward: {reward}")

if __name__ == '__main__':
    test()
