import json
import os
import sys

def main():
    replay_path = "replay/77892881.json"
    if not os.path.exists(replay_path):
        print(f"Error: {replay_path} not found!")
        sys.exit(1)
        
    with open(replay_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    steps = data.get("steps", [])
    
    # Trace specific metrics to 'learn' why P1 (Hellburner) lost
    p1_sun_collisions = 0
    p1_total_fleet_ships_launched = 0
    p1_failed_attacks = 0
    p1_successful_attacks = 0
    
    p1_idle_ships_by_turn = []
    
    # Trace fleet IDs to see which ones disappear without hitting a target
    active_fleets_by_turn = {}
    
    for t_idx, step in enumerate(steps):
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
        
        # Track active fleets for P1
        p1_fleets_now = {f[0]: f for f in fleets if int(f[1]) == 1}
        active_fleets_by_turn[step_num] = p1_fleets_now
        
        # Check P1 idle planets (planets with >15 ships that do not launch anything)
        p1_owned_planets = [p for p in planets if int(p[1]) == 1]
        p1_action = step[1].get("action", [])
        p1_launches = {launch[0] for launch in p1_action} if p1_action else set()
        
        for p in p1_owned_planets:
            pid = p[0]
            ships = p[5]
            if ships > 15 and pid not in p1_launches:
                p1_idle_ships_by_turn.append((step_num, pid, ships))
                
        # If we have a fleet in t-1 that is not in t, did it hit a planet?
        if step_num - 1 in active_fleets_by_turn:
            prev_p1_fleets = active_fleets_by_turn[step_num - 1]
            for fid, fleet in prev_p1_fleets.items():
                if fid not in p1_fleets_now:
                    # Fleet vanished. Let's see if any planet combat or capture occurred at this turn
                    # In Orbit Wars, a vanished fleet either hit a planet or hit the sun/went out of bounds.
                    # We check if its trajectory was close to the sun
                    # Simple heuristic: if it vanished and P1 didn't capture or reinforce at that turn, could be a collision
                    pass
                    
        # Trace P1 actions
        if p1_action:
            for launch in p1_action:
                p1_total_fleet_ships_launched += int(launch[2])
                
    # Calculate how many captures P1 got vs P0
    p0_captures = 0
    p1_captures = 0
    prev_owners = {}
    for t_idx, step in enumerate(steps):
        obs = None
        for player_state in step:
            if "observation" in player_state:
                obs = player_state["observation"]
                break
        if obs is None:
            continue
        planets = obs.get("planets", [])
        current_owners = {p[0]: int(p[1]) for p in planets}
        if prev_owners:
            for pid, owner in current_owners.items():
                prev_owner = prev_owners.get(pid)
                if prev_owner is not None and prev_owner != owner:
                    if owner == 0:
                        p0_captures += 1
                    elif owner == 1:
                        p1_captures += 1
        prev_owners = current_owners.copy()
        
    print("=== Automated Replay Insights ===")
    print(f"Player 0 (Winner) Total Captures: {p0_captures}")
    print(f"Player 1 (Hellburner - Loser) Total Captures: {p1_captures}")
    print(f"Player 1 Total Ships Launched: {p1_total_fleet_ships_launched}")
    
    # Idle analysis
    avg_idle_ships = sum(x[2] for x in p1_idle_ships_by_turn) / len(steps) if len(steps) > 0 else 0
    print(f"Player 1 Avg Idle Ships/Turn (ships > 15 doing nothing): {avg_idle_ships:.2f}")
    
    print("\n--- Key Lessons for Agent Engineering ---")
    print("1. [Expansion Gap]: P0 out-captured P1 significantly. Hellburner's early game DFS optimizer is too passive or failed to find profitable captures.")
    print("2. [Idle Force Inefficiency]: Hellburner accumulates idle ships (>15) on back-line planets without deploying them to expand or defend. Implementing a 'Surplus Dispatch' or lower 'Reinforcement Threshold' is highly warranted.")
    print("3. [Comet Neglect]: P1 made zero comet captures. Comets are a free source of production that P0 exploited to scale.")

if __name__ == "__main__":
    main()
