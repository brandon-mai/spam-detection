import os
import math
import time
import multiprocessing as mp
import numpy as np
from kaggle_environments import make

BOARD = 100.0; MAX_SPEED = 6.0

def fleet_speed(ships):
    if ships <= 0: return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5

def find_target_via_ray(src_xy, send_angle, planets, ray_horizon=200.0, perp_margin=1.0):
    sx, sy = src_xy; fx, fy = math.cos(send_angle), math.sin(send_angle)
    best_pid, best_perp = -1, 1e9
    for p in planets:
        pid, _, px, py, pr, _, _ = p
        pid = int(pid); px = float(px); py = float(py); pr = float(pr)
        dx = px - sx; dy = py - sy
        t = dx * fx + dy * fy
        if t <= 0 or t > ray_horizon: continue
        perp = abs(dx * fy - dy * fx)
        if perp <= pr + perp_margin and perp < best_perp:
            best_perp = perp; best_pid = pid
    return best_pid

def label_outcome(env_steps, target_id, side, arrival_turn, window=10):
    end_t = min(arrival_turn + window, len(env_steps) - 1)
    start_t = min(arrival_turn, end_t)
    for t in range(start_t, end_t + 1):
        s = env_steps[t][side].observation
        if s is None: continue
        for p in s["planets"]:
            if int(p[0]) == target_id and int(p[1]) == side: return 1
    return 0

FEATURE_DIM = 24
def encode_shot(obs, src_id, target_id, ships_sent):
    pdict = {int(p[0]): p for p in obs["planets"]}
    if src_id not in pdict or target_id not in pdict: return None
    src = pdict[src_id]; tgt = pdict[target_id]
    me = int(obs.get("player", 0))
    fleets = obs.get("fleets", [])
    planets = obs["planets"]
    my_ships_total = sum(int(p[5]) for p in planets if int(p[1]) == me)
    enemy_ships_total = sum(int(p[5]) for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    my_planets = sum(1 for p in planets if int(p[1]) == me)
    enemy_planets = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    sx, sy, sr, sships = float(src[2]), float(src[3]), float(src[4]), int(src[5])
    tx, ty, tr, tships = float(tgt[2]), float(tgt[3]), float(tgt[4]), int(tgt[5])
    sprod, tprod = float(src[6]), float(tgt[6])
    dx, dy = tx - sx, ty - sy
    dist = max(math.hypot(dx, dy) - sr - tr, 0.0)
    speed = fleet_speed(ships_sent)
    eta = dist / max(speed, 0.5)
    own_self = 1.0 if int(tgt[1]) == me else 0.0
    own_neutral = 1.0 if int(tgt[1]) < 0 else 0.0
    own_enemy = 1.0 if (int(tgt[1]) >= 0 and int(tgt[1]) != me) else 0.0
    ship_frac = ships_sent / max(sships, 1)
    ally_n = sum(1 for f in fleets if int(f[1]) == me)
    ally_s = sum(int(f[6]) for f in fleets if int(f[1]) == me)
    enemy_n = sum(1 for f in fleets if int(f[1]) != me)
    enemy_s = sum(int(f[6]) for f in fleets if int(f[1]) != me)
    turn = int(obs.get("step", 0))
    return np.array([
        sships/100.0, sprod/5.0, sr/4.0,
        tships/100.0, tprod/5.0, tr/4.0,
        own_self, own_neutral, own_enemy,
        ships_sent/100.0, ship_frac,
        dist/BOARD, eta/60.0, speed/MAX_SPEED,
        ally_n/10.0, ally_s/100.0, enemy_n/10.0, enemy_s/100.0,
        turn/500.0, my_ships_total/200.0, enemy_ships_total/200.0,
        (my_ships_total - enemy_ships_total)/200.0,
        my_planets/20.0, enemy_planets/20.0,
    ], dtype=np.float32)

def collect_one_game(args):
    teacher_path, opponent_path, seed, side, game_id = args
    paths = [teacher_path, opponent_path] if side == 0 else [opponent_path, teacher_path]
    env = make("orbit_wars", configuration={"randomSeed": seed}, debug=False)
    try:
        env.run(paths)
    except Exception as e:
        return [], game_id, str(e)
    rows = []
    for step_idx, st in enumerate(env.steps):
        s = st[side]
        obs = s.observation
        action = s.action or []
        if obs is None or not action: continue
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
            tgt_owner = next((int(p[1]) for p in planets if int(p[0]) == tgt_id), -2)
            if tgt_owner == side: continue 
            feat = encode_shot(obs, src_id, tgt_id, ships)
            if feat is None: continue
            tx, ty, tr = next(((float(p[2]), float(p[3]), float(p[4])) for p in planets if int(p[0]) == tgt_id), (0,0,0))
            sx, sy = src_xy[src_id]
            sr = next((float(p[4]) for p in planets if int(p[0]) == src_id), 0)
            dist = max(math.hypot(tx-sx, ty-sy) - sr - tr, 0.0)
            speed = fleet_speed(ships)
            eta_turns = max(int(math.ceil(dist / max(speed, 0.5))), 1)
            arrival_turn = step_idx + eta_turns
            label = label_outcome(env.steps, tgt_id, side, arrival_turn, window=10)
            rows.append((feat, label, game_id, step_idx))
    return rows, game_id, None

if __name__ == "__main__":
    OPPONENTS_DIR = "agents"
    os.makedirs(OPPONENTS_DIR, exist_ok=True)
    
    OPPONENT_CODES = {
        "nearest_sniper.py": '''
import math
def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    moves = []
    for src in planets:
        sid, owner, sx, sy, sr, ships, _ = src
        if int(owner) != int(me) or int(ships) < 2: continue
        best_pid, best_d, best_pxy = -1, 1e9, None
        for tgt in planets:
            tid, towner, tx, ty, tr, tships, _ = tgt
            if int(tid) == int(sid): continue
            if int(towner) == int(me): continue
            d = math.hypot(float(tx)-float(sx), float(ty)-float(sy))
            if d < best_d:
                best_d, best_pid, best_pxy = d, int(tid), (float(tx), float(ty), int(tships))
        if best_pid < 0 or best_pxy is None: continue
        ang = math.atan2(best_pxy[1]-float(sy), best_pxy[0]-float(sx))
        send = min(int(ships)-1, best_pxy[2]+1)
        if send > 0: moves.append([int(sid), float(ang), int(send)])
    return moves
''',
        "weakest_first.py": '''
import math
def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    moves = []
    for src in planets:
        sid, owner, sx, sy, sr, ships, _ = src
        if int(owner) != int(me) or int(ships) < 5: continue
        candidates = []
        for tgt in planets:
            tid, towner, tx, ty, tr, tships, _ = tgt
            if int(tid) == int(sid) or int(towner) == int(me): continue
            candidates.append((int(tships), float(tx), float(ty), int(tid)))
        if not candidates: continue
        candidates.sort()
        ts, tx, ty, tid = candidates[0]
        ang = math.atan2(ty-float(sy), tx-float(sx))
        send = min(int(ships)-1, ts+2)
        if send > 0: moves.append([int(sid), float(ang), int(send)])
    return moves
''',
        "production_first.py": '''
import math
def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    moves = []
    for src in planets:
        sid, owner, sx, sy, sr, ships, _ = src
        if int(owner) != int(me) or int(ships) < 5: continue
        best = None
        for tgt in planets:
            tid, towner, tx, ty, tr, tships, prod = tgt
            if int(tid) == int(sid) or int(towner) == int(me): continue
            score = float(prod) - 0.05 * int(tships)
            if best is None or score > best[0]:
                best = (score, float(tx), float(ty), int(tid), int(tships))
        if best is None: continue
        _, tx, ty, tid, tships = best
        ang = math.atan2(ty-float(sy), tx-float(sx))
        send = min(int(ships)-1, tships+3)
        if send > 0: moves.append([int(sid), float(ang), int(send)])
    return moves
''',
        "defender.py": '''
import math
def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    fleets = obs["fleets"] if isinstance(obs, dict) else obs.fleets
    mine = [p for p in planets if int(p[1]) == int(me)]
    if not mine: return []
    threats = set()
    for f in fleets:
        if int(f[1]) == int(me): continue
        for p in mine:
            if math.hypot(float(f[2])-float(p[2]), float(f[3])-float(p[3])) < 40:
                threats.add(int(p[0]))
                break
    moves = []
    for src in mine:
        sid, _, sx, sy, sr, ships, _ = src
        if int(ships) < 8: continue
        if int(sid) in threats:
            best = None
            for p in mine:
                if int(p[0]) == int(sid) or int(p[0]) not in threats: continue
                d = math.hypot(float(p[2])-float(sx), float(p[3])-float(sy))
                if best is None or d < best[0]:
                    best = (d, float(p[2]), float(p[3]), int(p[0]))
            if best is not None:
                _, tx, ty, tid = best
                ang = math.atan2(ty-float(sy), tx-float(sx))
                send = max(int(ships)//2, 1)
                moves.append([int(sid), float(ang), int(send)])
            continue
        best = None
        for tgt in planets:
            tid, towner, tx, ty, tr, tships, _ = tgt
            if int(tid) == int(sid) or int(towner) == int(me): continue
            d = math.hypot(float(tx)-float(sx), float(ty)-float(sy))
            if best is None or d < best[0]:
                best = (d, float(tx), float(ty), int(tid), int(tships))
        if best is None: continue
        _, tx, ty, tid, tships = best
        ang = math.atan2(ty-float(sy), tx-float(sx))
        send = min(int(ships)-1, tships+2)
        if send > 0: moves.append([int(sid), float(ang), int(send)])
    return moves
''',
    }

    OPPONENT_PATHS = []
    for fname, code in OPPONENT_CODES.items():
        p = os.path.join(OPPONENTS_DIR, fname)
        with open(p, 'w') as f:
            f.write(code)
        OPPONENT_PATHS.append(p)
        print(f"  wrote {p}")

    TEACHER = "agents/hellburner_v2.py"
    OPPONENT_PATHS.append(TEACHER)
    print(f"  added teacher self-play: {TEACHER}")

    SEEDS = list(range(101, 106))  # 5 seeds 
    SELFPLAY_SEEDS = list(range(101, 116))  # 15 self-play seeds
    jobs = []
    gid = 0
    for opp in OPPONENT_PATHS:
        for seed in (SELFPLAY_SEEDS if opp == TEACHER else SEEDS):
            for side in (0, 1):
                gid += 1
                jobs.append((TEACHER, opp, seed, side, gid))
    print(f"Jobs: {len(jobs)} games")

    all_rows = []
    failed = 0
    t0 = time.time()
    
    # We will use mp.Pool
    with mp.Pool(processes=min(8, os.cpu_count() or 1)) as pool:
        for i, (rows, gid_, err) in enumerate(pool.imap_unordered(collect_one_game, jobs)):
            if err is not None:
                failed += 1
                print(f"  [WARN] game {gid_} failed: {err[:80]}")
            else:
                all_rows.extend(rows)
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(jobs)} games, rows={len(all_rows)}, t={time.time()-t0:.0f}s", flush=True)

    print(f"\\nDone: {len(all_rows)} shots collected ({failed} games failed)")
    if len(all_rows) > 0:
        feats = np.stack([r[0] for r in all_rows]).astype(np.float32)
        labels = np.asarray([r[1] for r in all_rows], dtype=np.float32)
        meta_game = np.asarray([r[2] for r in all_rows], dtype=np.int32)
        pos_rate = labels.mean()
        print(f"  features: {feats.shape}, labels: {labels.shape}")
        print(f"  positive rate: {pos_rate*100:.1f}%")
        
        np.savez_compressed("shot_dataset.npz", features=feats, labels=labels.astype(np.int64), meta_game=meta_game)
        print(f"  saved shot_dataset.npz")
    else:
        print("No rows collected. Exiting.")
