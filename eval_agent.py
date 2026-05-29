import multiprocessing as mp
from kaggle_environments import make
import sys

def play_one(args):
    seed, side = args
    agent1 = "agents/hellburner_v2_ml.py"
    agent2 = "agents/hellburner_v2.py"
    paths = [agent1, agent2] if side == 0 else [agent2, agent1]
    env = make("orbit_wars", configuration={"randomSeed": seed}, debug=False)
    try:
        env.run(paths)
        res = [s.reward for s in env.steps[-1]]
        if res[0] == 1:
            win1 = side == 0
        elif res[1] == 1:
            win1 = side == 1
        else:
            win1 = None
        return win1
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    seeds = range(201, 211)
    jobs = []
    for s in seeds:
        jobs.append((s, 0))
        jobs.append((s, 1))

    wins = 0
    losses = 0
    draws = 0
    errors = 0

    with mp.Pool(processes=min(8, len(jobs))) as pool:
        for res in pool.imap_unordered(play_one, jobs):
            if isinstance(res, str):
                errors += 1
                print("Error:", res)
            elif res is True:
                wins += 1
            elif res is False:
                losses += 1
            else:
                draws += 1
                
    total = wins + losses
    wr = wins / total if total > 0 else 0.0
    print(f"Results (ML vs Base): {wins} W / {losses} L / {draws} D / {errors} E")
    print(f"Win rate (excl draws/errs): {wr * 100:.1f}%")
