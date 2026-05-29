import json
import os
import sys

def main():
    replay_path = "replay/77892881.json"
    if not os.path.exists(replay_path):
        print(f"Error: {replay_path} not found!")
        sys.exit(1)
        
    print(f"Analyzing Kaggle Replay: {replay_path}...")
    with open(replay_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    rewards = data.get("rewards", [])
    statuses = data.get("statuses", [])
    steps = data.get("steps", [])
    
    print("\n--- Match Summary ---")
    print(f"Total Steps: {len(steps)}")
    print(f"Statuses: {statuses}")
    print(f"Final Rewards: {rewards}")
    
    # Track statistics turn by turn
    num_players = len(rewards)
    
    # We want to trace planet counts and total ship counts for each player
    # Let's look at key turning points in the game
    print("\n--- Game Turning Points & Timeline ---")
    
    for t_idx, step in enumerate(steps):
        # The observation is in the first active player's state
        # Find player that has observation
        obs = None
        for player_state in step:
            if "observation" in player_state:
                obs = player_state["observation"]
                break
        if obs is None:
            continue
            
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        step_num = obs.get("step", 0)
        
        # Calculate stats for this step
        player_planets = {i: 0 for i in range(num_players)}
        player_planet_ships = {i: 0 for i in range(num_players)}
        player_fleet_ships = {i: 0 for i in range(num_players)}
        player_production = {i: 0 for i in range(num_players)}
        
        # Count neutral too
        neutral_planets = 0
        neutral_ships = 0
        
        for p in planets:
            owner = int(p[1])
            ships = int(p[5])
            prod = int(p[6])
            if owner == -1:
                neutral_planets += 1
                neutral_ships += ships
            else:
                player_planets[owner] += 1
                player_planet_ships[owner] += ships
                player_production[owner] += prod
                
        for f in fleets:
            owner = int(f[1])
            ships = int(f[6])
            player_fleet_ships[owner] += ships
            
        total_ships = {
            i: player_planet_ships[i] + player_fleet_ships[i]
            for i in range(num_players)
        }
        
        # Print snapshots at regular intervals, or if someone gets eliminated
        if step_num in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, len(steps)-1]:
            print(f"\n[Turn {step_num}]")
            for p_idx in range(num_players):
                status_str = f"P{p_idx}: Planets={player_planets[p_idx]} (Prod={player_production[p_idx]}) | Ships={total_ships[p_idx]} (Garrison={player_planet_ships[p_idx]}, Fleets={player_fleet_ships[p_idx]})"
                print(status_str)
            print(f"Neutral: Planets={neutral_planets} | Ships={neutral_ships}")
            
        # Detect fleet launch details or anomalies (e.g. sun collisions or out-of-bounds)
        # We can look at the actions of players
        actions = []
        for p_idx, player_state in enumerate(step):
            action = player_state.get("action")
            if action:
                actions.append((p_idx, action))
                
    # Detect who was hellburner (usually the code was playing as player 0 or 1, check ELO/rewards)
    print("\n--- Failure Mode Diagnostics ---")
    # Identify which player experienced a massive loss in ELO/rewards
    loser_idx = rewards.index(min(rewards))
    winner_idx = rewards.index(max(rewards))
    print(f"Winner: Player {winner_idx} ({rewards[winner_idx]})")
    print(f"Loser: Player {loser_idx} ({rewards[loser_idx]})")
    
if __name__ == "__main__":
    main()
