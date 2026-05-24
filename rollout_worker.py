import os
import multiprocessing as mp
import numpy as np
import math

# Force JAX to use CPU strictly for rollout workers to prevent GPU OOM
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import sys
import logging
import jax

# Suppress Kaggle Environments open_spiel debug logs
logging.getLogger('kaggle_environments').setLevel(logging.ERROR)
from rl_env_wrapper import extract_tensors, get_autoregressive_mask, MAX_PLANETS
from models import EntityTransformer

import importlib.util
import inspect
from kaggle_environments import make

def load_agent(filepath):
    spec = importlib.util.spec_from_file_location("opponent", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent

def call_agent(agent_fn, obs, config):
    sig = inspect.signature(agent_fn)
    if len(sig.parameters) == 1:
        return agent_fn(obs)
    return agent_fn(obs, config)

def worker_process(worker_id, weights_queue, trajectory_queue, episodes_per_worker):
    """
    Independent worker process that runs CPU-based JAX inference 
    and Numba physics simulation.
    """
    infer = jax.jit(EntityTransformer().apply)
    
    current_opponent_path = None
    opponent_agent = None
    env = make("orbit_wars")
    
    while True:
        # Block until new weights are available
        msg = weights_queue.get()
        
        if msg is None:
            # Poison pill received, shut down worker
            break
            
        current_weights = msg["weights"]
        new_opponent = msg["opponent"]
        if new_opponent != current_opponent_path:
            opponent_agent = load_agent(new_opponent)
            current_opponent_path = new_opponent
            
        for ep in range(episodes_per_worker):
            env.reset()
            obs = env.state[0].observation
            config = env.configuration
            
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
                MAX_USEFUL_ACTIONS = min(5, max(1, len([p for p in obs.get("planets", []) if p[1] == player_id]) * 4))
                moves = []
                
                # Hoist extraction outside the autoregressive loop
                p_mat_base, f_mat, g_vec, all_planets = extract_tensors(obs, local_state_tracking)
                
                # Autoregressive Loop for a single environment step
                while actions_taken < MAX_USEFUL_ACTIONS:
                    # Patch the ships column with local tracking
                    p_mat = p_mat_base.copy()
                for idx, p in enumerate(all_planets):
                    p_mat[idx, 8] = local_state_tracking.get(p["id"], p["ships"]) / 1000.0
                    
                mask = get_autoregressive_mask(player_id, all_planets, local_state_tracking)
                
                # CPU JAX Inference
                # We add batch dim manually
                logits, value = infer(
                    current_weights, 
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
                    if msg.get("greedy", False):
                        action = np.argmax(probs)
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
                
            # Get opponent moves using loaded agent
            opp_obs = env.state[1].observation
            opp_moves = call_agent(opponent_agent, opp_obs, config)
            
            # Step Environment manually
            step_actions = [moves, opp_moves] if player_id == 0 else [opp_moves, moves]
            state = env.step(step_actions)
            
            next_obs = state[0].observation
            reward = state[0].reward
            done = state[0].status == 'DONE' or state[1].status == 'DONE'
            if reward is None: reward = 0.0
            
            # Strict Zero-Sum Reward Shaping (The Tie-Bug Fix)
            if done:
                my_score = sum([p[5] for p in next_obs.get('planets', []) if p[1] == player_id]) + \
                           sum([f[6] for f in next_obs.get('fleets', []) if f[1] == player_id])
                opp_score = sum([p[5] for p in next_obs.get('planets', []) if p[1] != player_id]) + \
                            sum([f[6] for f in next_obs.get('fleets', []) if f[1] != player_id])
                
                if my_score <= opp_score:
                    reward = -1.0  # Tie or Loss
                else:
                    reward = 1.0   # Win
            
            # The reward for the final step of the autoregressive loop
            trajectory['rewards'].append(float(reward))
            trajectory['dones'].append(done)
            
            obs = next_obs
            local_state_tracking = {p[0]: p[5] for p in obs.get("planets", [])}
            
        # Send trajectory back
        trajectory_queue.put(trajectory)
