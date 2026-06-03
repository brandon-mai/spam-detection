import json
import os
import math

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def main():
    replay_path = "replay/77892881.json"
    with open(replay_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    steps = data.get("steps", [])
    
    # Trace planet owners and populations turn-by-turn to detect who captured what
    prev_owners = {}
    prev_ships = {}
    
    print("--- Capture & Event Logger ---")
    
    for t_idx, step in enumerate(steps):
        obs = None
        for player_state in step:
            if "observation" in player_state:
                obs = player_state["observation"]
                break
        if obs is None:
            continue
            
        planets = obs.get("planets", [])
        step_num = obs.get("step", 0)
        
        current_owners = {p[0]: int(p[1]) for p in planets}
        current_ships = {p[0]: int(p[5]) for p in planets}
        
        if prev_owners:
            # Check for captures
            for pid, owner in current_owners.items():
                prev_owner = prev_owners.get(pid)
                if prev_owner is not None and prev_owner != owner:
                    print(f"[Turn {step_num}] PLANET CAPTURED: Planet {pid} changed from P{prev_owner} to P{owner} (Garrison was {prev_ships.get(pid)} -> now {current_ships[pid]})")
                    
        prev_owners = current_owners.copy()
        prev_ships = current_ships.copy()
        
        # Check if any fleets crashed into the sun or went out of bounds
        # In a real step, we can look at active fleets
        # If a fleet existed in turn t but vanished in t+1 without hitting a planet, it was deleted (sun/out of bounds)
        # Let's count fleet events if needed
        
    # Let's check early game decisions
    print("\n--- Early Game Launch Analysis (Turns 0-5) ---")
    for t_idx in range(6):
        if t_idx >= len(steps):
            break
        print(f"\n[Turn {t_idx}]")
        for p_idx, player_state in enumerate(steps[t_idx]):
            action = player_state.get("action")
            if action:
                print(f"  P{p_idx} Action: {action}")
                
if __name__ == "__main__":
    main()
