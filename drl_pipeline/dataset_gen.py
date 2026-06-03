import os
import sys
import json
import glob
import random
import logging
import math
import concurrent.futures
from tqdm.auto import tqdm
import numpy as np

# Suppress all output (including C++ level) during kaggle_environments import
fd_out = sys.stdout.fileno()
fd_err = sys.stderr.fileno()
saved_out = os.dup(fd_out)
saved_err = os.dup(fd_err)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, fd_out)
os.dup2(devnull, fd_err)
logging.disable(logging.WARNING)

try:
    from kaggle_environments import make
finally:
    os.dup2(saved_out, fd_out)
    os.dup2(saved_err, fd_err)
    os.close(devnull)
    os.close(saved_out)
    os.close(saved_err)
    logging.disable(logging.NOTSET)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from drl_pipeline.orbit_physics_jit import jit_resolve_fleet_targets, jit_point_to_segment_dist, jit_build_graph_features

TARGET_TRANSITIONS = 500000
MAX_STEPS = 500
OUTPUT_FILE_2P = "drl_pipeline/expert_dataset_2p.npz"
OUTPUT_FILE_4P = "drl_pipeline/expert_dataset_4p.npz"
AGENT_PATH = "agents/vkhydras_final.py"
MAX_WORKERS = os.cpu_count() or 4

def get_other_agents():
    files = glob.glob("agents/*.py")
    exclude = ["agents\\nearest_planet_sniper.py", "agents/nearest_planet_sniper.py", "agents\\__init__.py", "agents/__init__.py"]
    return [f for f in files if f not in exclude]

OTHER_AGENTS = get_other_agents()

def simulate_match(game_idx, num_players):
    import logging
    logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").setLevel(logging.WARNING)
    logging.getLogger("kaggle_environments").setLevel(logging.WARNING)
    env = make("orbit_wars", configuration={"episodeSteps": MAX_STEPS})
    
    # 70% chance self-play, 30% against others
    agents = [AGENT_PATH] * num_players
    vkhydras_indices = [0, 1, 2, 3][:num_players]
    
    if random.random() < 0.30 and OTHER_AGENTS:
        # Pick a random other agent
        other = random.choice(OTHER_AGENTS)
        # Randomly assign the other agent to some slots, ensuring at least one vkhydras_final remains
        num_others = random.randint(1, num_players - 1)
        other_indices = random.sample(range(num_players), num_others)
        for idx in other_indices:
            agents[idx] = other
        vkhydras_indices = [i for i in range(num_players) if i not in other_indices]
        
    env.run(agents)
    
    records = []
    for step_idx, step_data in enumerate(env.steps):
        if step_idx == 0: continue
        
        # Only record from the perspective of vkhydras_final
        for player_idx in vkhydras_indices:
            agent_state = step_data[player_idx]
            obs = agent_state.observation
            action = agent_state.action
            
            if action is None or len(action) == 0:
                action = []
                
            # Extract features using the physics engine
            planets_data = obs.get("planets", [])
            fleets_data = obs.get("fleets", [])
            if isinstance(planets_data, dict):
                planets_items = [(int(k), v) for k, v in planets_data.items()]
            else:
                planets_items = enumerate(planets_data)
                
            if isinstance(fleets_data, dict):
                fleets_items = [(int(k), v) for k, v in fleets_data.items()]
            else:
                fleets_items = enumerate(fleets_data)
            
            num_planets_total = 50
            num_planets = num_planets_total
            V = np.zeros((num_planets_total, 13), dtype=np.float32)
            E = np.zeros((num_planets_total, num_planets_total, 4), dtype=np.float32)
            
            # Build basic matrix for python processing
            planet_matrix = np.zeros((num_planets_total, 6), dtype=np.float32)
            fleet_matrix = np.zeros((len(fleets_data), 4), dtype=np.float32)
            
            has_planets = False
            for pid, pdata in planets_items:
                if len(pdata) == 7:
                    owner, px, py, radius, garrison, prod = pdata[1:7]
                else:
                    owner, px, py, radius, garrison, prod = pdata[0:6]
                    
                if owner == player_idx:
                    has_planets = True
                    
                planet_matrix[pid, 0] = owner
                planet_matrix[pid, 1] = px
                planet_matrix[pid, 2] = py
                planet_matrix[pid, 3] = radius
                planet_matrix[pid, 4] = garrison
                planet_matrix[pid, 5] = prod
                
            for i, (fid, fdata) in enumerate(fleets_items):
                if len(fdata) == 6:
                    owner, fx, fy, heading, ships = fdata[1:6]
                else:
                    owner, fx, fy, heading, ships = fdata[0:5]
                    
                fleet_matrix[i, 0] = fx
                fleet_matrix[i, 1] = fy
                fleet_matrix[i, 2] = heading
                fleet_matrix[i, 3] = ships
                
            if not has_planets:
                continue
                
            # Run accelerated feature builder
            V, E = jit_build_graph_features(planet_matrix, fleet_matrix, player_idx)
            
            # Format action: [source, target, quota_index]
            parsed_action = [0, 50, 0] # NO_OP
            if len(action) > 0:
                act = action[0]
                src = int(act[0])
                ships = act[2]
                garrison = V[src, 2]
                
                quota_idx = 2
                if garrison > 0:
                    frac = ships / garrison
                    if frac <= 0.35: quota_idx = 0
                    elif frac <= 0.75: quota_idx = 1
                    else: quota_idx = 2
                    
                # Calculate target destination using raycast on physics engine
                heading = act[1]
                fleet_dummy = np.zeros((1, 4), dtype=np.float32)
                fleet_dummy[0, 0] = planet_matrix[src, 1]
                fleet_dummy[0, 1] = planet_matrix[src, 2]
                fleet_dummy[0, 2] = heading
                fleet_dummy[0, 3] = ships
                
                # We need planet subset [N, 3] -> [px, py, radius]
                p_sub = planet_matrix[:, 1:4]
                targets = jit_resolve_fleet_targets(fleet_dummy, p_sub, 0.0)
                tgt = targets[0, 0]
                if tgt == -1:
                    tgt = 50 # Missing / Offboard
                    
                parsed_action = [src, tgt, quota_idx]
                
            record = {
                "V": V.astype(np.float16),
                "E": E.astype(np.float16),
                "src": parsed_action[0],
                "tgt": parsed_action[1],
                "quota": parsed_action[2]
            }
            records.append(record)
    return records

def gather_dataset(mode, target_file):
    print(f"Generating {mode}P Dataset: {TARGET_TRANSITIONS} transitions...")
    
    all_V = np.zeros((TARGET_TRANSITIONS, 50, 13), dtype=np.float16)
    all_E = np.zeros((TARGET_TRANSITIONS, 50, 50, 4), dtype=np.float16)
    all_src = np.zeros(TARGET_TRANSITIONS, dtype=np.int32)
    all_tgt = np.zeros(TARGET_TRANSITIONS, dtype=np.int32)
    all_quota = np.zeros(TARGET_TRANSITIONS, dtype=np.int32)
    
    total_transitions = 0
    game_idx = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = set()
        for _ in range(MAX_WORKERS * 2):
            futures.add(executor.submit(simulate_match, game_idx, mode))
            game_idx += 1
            
        pbar = tqdm(total=TARGET_TRANSITIONS, desc=f"{mode}P Transitions")
        while total_transitions < TARGET_TRANSITIONS and futures:
            done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try:
                    records = future.result()
                    for r in records:
                        if total_transitions < TARGET_TRANSITIONS:
                            all_V[total_transitions] = r["V"]
                            all_E[total_transitions] = r["E"]
                            all_src[total_transitions] = r["src"]
                            all_tgt[total_transitions] = r["tgt"]
                            all_quota[total_transitions] = r["quota"]
                            total_transitions += 1
                            pbar.update(1)
                except Exception as e:
                    print(f"{mode}P Game failed: {e}")
                    
                if total_transitions < TARGET_TRANSITIONS:
                    not_done.add(executor.submit(simulate_match, game_idx, mode))
                    game_idx += 1
            futures = not_done
            
        pbar.close()
        
    print(f"Saving {mode}P Dataset to {target_file}...")
    np.savez_compressed(target_file, 
                        V=all_V, 
                        E=all_E, 
                        src=all_src, 
                        tgt=all_tgt, 
                        quota=all_quota)
    print("Done!")

def generate_datasets():
    print(f"Discovered {len(OTHER_AGENTS)} other agents for 30% sampling.")
    gather_dataset(2, OUTPUT_FILE_2P)
    gather_dataset(4, OUTPUT_FILE_4P)
    print("All datasets generated successfully.")

if __name__ == "__main__":
    generate_datasets()


