import os
import sys
import json
import glob
import os
import sys
import random
import logging
import concurrent.futures
from tqdm.auto import tqdm

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

NUM_GAMES_2P = 2000
NUM_GAMES_4P = 2000
MAX_STEPS = 500
OUTPUT_FILE_2P = "drl_pipeline/expert_dataset_2p.jsonl"
OUTPUT_FILE_4P = "drl_pipeline/expert_dataset_4p.jsonl"
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
                
            record = {
                "game_id": game_idx,
                "step": obs.get("step", step_idx),
                "player": obs.get("player", player_idx),
                "obs": obs,
                "action": action
            }
            records.append(record)
    return records

def generate_datasets():
    print(f"Discovered {len(OTHER_AGENTS)} other agents for 30% sampling.")
    print(f"Generating 2P Dataset: {NUM_GAMES_2P} games...")
    
    # Generate 2P
    with open(OUTPUT_FILE_2P, "w") as f2p:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures_2p = {executor.submit(simulate_match, i, 2): i for i in range(NUM_GAMES_2P)}
            for future in tqdm(concurrent.futures.as_completed(futures_2p), total=NUM_GAMES_2P, desc="2P Games"):
                try:
                    for record in future.result():
                        f2p.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"2P Game failed: {e}")
                    
    print(f"Generating 4P Dataset: {NUM_GAMES_4P} games...")
    
    # Generate 4P
    with open(OUTPUT_FILE_4P, "w") as f4p:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures_4p = {executor.submit(simulate_match, i, 4): i for i in range(NUM_GAMES_4P)}
            for future in tqdm(concurrent.futures.as_completed(futures_4p), total=NUM_GAMES_4P, desc="4P Games"):
                try:
                    for record in future.result():
                        f4p.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"4P Game failed: {e}")

    print("Datasets generated successfully.")

if __name__ == "__main__":
    generate_datasets()


