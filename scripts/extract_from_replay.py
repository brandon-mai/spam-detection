import json
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.collect_data import encode_shot, find_target_via_ray

def _label_outcome_dict(env_steps, tgt_id, attacker_side, arrival_turn, window=10):
    t_min = max(0, arrival_turn - 2)
    t_max = min(len(env_steps)-1, arrival_turn + window)
    for t in range(t_min, t_max + 1):
        s = env_steps[t][attacker_side].get("observation", {})
        planets = s.get("planets", [])
        owner = next((int(p[1]) for p in planets if int(p[0]) == tgt_id), -2)
        if owner == attacker_side:
            return 1
    return 0

def extract_from_replay(json_path, agent_name_hint="hellburner"):
    with open(json_path, "r") as f:
        replay = json.load(f)

    # find which player index is our agent (if available in info)
    # usually info or steps contains agent names or we can just extract for both players
    # and use the side that matches the behavior or just extract all shots!
    # A replay consists of a list of steps. 
    # replay["steps"][t][player_idx]["observation"] -> obs
    # replay["steps"][t][player_idx]["action"] -> action
    
    steps = replay["steps"]
    all_rows = []
    
    for side in (0, 1):
        for step_idx, step_state in enumerate(steps):
            s = step_state[side]
            obs = s.get("observation")
            action = s.get("action", [])
            if obs is None or not action:
                continue
                
            planets = obs["planets"]
            src_xy = {int(p[0]): (float(p[2]), float(p[3])) for p in planets}
            
            for mv in action:
                try:
                    src_id, ang, ships = int(mv[0]), float(mv[1]), int(mv[2])
                except Exception:
                    continue
                if src_id not in src_xy: continue
                tgt_id = find_target_via_ray(src_xy[src_id], ang, planets)
                if tgt_id < 0 or tgt_id == src_id: continue
                
                # Check if it's an attack
                tgt_owner = next((int(p[1]) for p in planets if int(p[0]) == tgt_id), -2)
                if tgt_owner == side: continue 
                
                feat = encode_shot(obs, src_id, tgt_id, ships)
                if feat is None: continue
                
                # compute arrival turn for labeling
                import math
                from scripts.collect_data import fleet_speed
                tx, ty, tr = next(((float(p[2]), float(p[3]), float(p[4])) for p in planets if int(p[0]) == tgt_id), (0,0,0))
                sx, sy = src_xy[src_id]
                sr = next((float(p[4]) for p in planets if int(p[0]) == src_id), 0)
                dist = max(math.hypot(tx-sx, ty-sy) - sr - tr, 0.0)
                speed = fleet_speed(ships)
                eta_turns = max(int(math.ceil(dist / max(speed, 0.5))), 1)
                arrival_turn = step_idx + eta_turns
                
                label = _label_outcome_dict(steps, tgt_id, side, arrival_turn, window=10)
                all_rows.append((feat, label))
                
    if all_rows:
        feats = np.stack([r[0] for r in all_rows]).astype(np.float32)
        labels = np.asarray([r[1] for r in all_rows], dtype=np.int64)
        print(f"Extracted {len(all_rows)} shots from {json_path}")
        print(f"Pos rate: {labels.mean()*100:.1f}%")
        return feats, labels
    else:
        print("No shots found.")
        return None, None

if __name__ == "__main__":
    extract_from_replay("orbit-wars-data/77892881.json")
