import os
import multiprocessing as mp
import numpy as np
import math

# Force JAX to use CPU strictly for rollout workers to prevent GPU OOM
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import logging

# Suppress Kaggle Environments open_spiel debug logs
logging.getLogger('kaggle_environments').setLevel(logging.ERROR)
from rl_env_wrapper import extract_tensors, get_autoregressive_mask, MAX_PLANETS
from models import EntityTransformer

def worker_process(worker_id, weights_queue, trajectory_queue, num_episodes=10):
    """
    Independent worker process that runs CPU-based JAX inference 
    and Numba physics simulation.
    """
    model = EntityTransformer()
    # Initialize empty variables for weights
    current_weights = None
    
    # We will just play against 'random' for phase 1 bootstrap
    from kaggle_environments import make
    env = make('orbit_wars', debug=False)
    
    for ep in range(num_episodes):
        # Check if new weights are available
        while not weights_queue.empty():
            current_weights = weights_queue.get()
            
        if current_weights is None:
            # If no weights yet, skip or wait
            continue
            
        trainer = env.train([None, "random"])
        obs = trainer.reset()
        
        trajectory = {
            'planet_matrices': [],
            'fleet_matrices': [],
            'global_vecs': [],
            'masks': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'logprobs': [],
            'dones': []
        }
        
        done = False
        local_state_tracking = {p[0]: p[5] for p in obs.get("planets", [])}
        player_id = obs.get("player", 0)
        PASS_TOKEN_INDEX = MAX_PLANETS * MAX_PLANETS * 4
        
        step_reward = 0.0
        
        while not done:
            actions_taken = 0
            MAX_USEFUL_ACTIONS = max(1, len([p for p in obs.get("planets", []) if p[1] == player_id]) * 4)
            moves = []
            
            # Autoregressive Loop for a single environment step
            while actions_taken < MAX_USEFUL_ACTIONS:
                p_mat, f_mat, g_vec, all_planets = extract_tensors(obs, local_state_tracking)
                mask = get_autoregressive_mask(player_id, all_planets, local_state_tracking)
                
                # CPU JAX Inference
                # We add batch dim manually
                logits, value = model.apply(
                    {'params': current_weights}, 
                    np.expand_dims(p_mat, 0), 
                    np.expand_dims(f_mat, 0), 
                    np.expand_dims(g_vec, 0)
                )
                
                logits = np.array(logits[0])
                value = np.array(value[0])
                logits[~mask] = -np.inf
                
                # Sample action
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)
                
                # Handle possible NaNs from all -inf (shouldnt happen since Pass is always valid)
                if np.isnan(probs).any():
                    action = PASS_TOKEN_INDEX
                else:
                    action = np.random.choice(len(probs), p=probs)
                
                logprob = np.log(probs[action] + 1e-8)
                
                # Store trajectory
                trajectory['planet_matrices'].append(p_mat)
                trajectory['fleet_matrices'].append(f_mat)
                trajectory['global_vecs'].append(g_vec)
                trajectory['masks'].append(mask)
                trajectory['actions'].append(action)
                trajectory['values'].append(value)
                trajectory['logprobs'].append(logprob)
                
                # We will assign rewards and dones after the step
                # For autoregressive steps inside a tick, reward is 0 and done is False
                # except for the final pass token which gets the actual step reward.
                if actions_taken > 0:
                    trajectory['rewards'].append(0.0) # Penalty for extra steps could go here
                    trajectory['dones'].append(False)
                
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
                        angle = math.atan2(t["y"] - s["y"], t["x"] - s["x"])
                        moves.append([s["id"], angle, ships_to_send])
                        local_state_tracking[s["id"]] -= ships_to_send
                        
                actions_taken += 1
                
            # Step Environment
            next_obs, reward, done, info = trainer.step(moves)
            if reward is None: reward = 0.0
            
            # The reward for the final step of the autoregressive loop
            trajectory['rewards'].append(float(reward))
            trajectory['dones'].append(done)
            
            obs = next_obs
            local_state_tracking = {p[0]: p[5] for p in obs.get("planets", [])}
            
        # Send trajectory back
        trajectory_queue.put(trajectory)
