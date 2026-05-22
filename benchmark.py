import os
import sys
import logging
import random
import webbrowser
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ============================================================
# Benchmark Configurations
# ============================================================
benchmarked_agent = "agents/owproto_v3.py"
best_opponent = "agents/owproto_v2.py"

num_games_2p = 15
num_games_4p = 15
best_opponent_prob = 0.7  # 60% chance to pick best opponent


def run_single_game(seed, players, my_slot, record_replay):
    """Self-contained helper executed in child processes to run a single match."""
    # Suppress warnings and logs inside the child process using python interfaces
    import sys
    import os
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    
    # Silence stderr cleanly using file descriptor redirection inside the child process
    stderr_fd = sys.stderr.fileno()
    dup_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    
    try:
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"seed": seed})
        env.run(players)
        
        final_step = env.steps[-1]
        my_reward = final_step[my_slot].reward
        
        html = None
        if record_replay:
            html = env.render(mode="html", width=800, height=600)
            
        return my_reward, html
    except Exception as exc:
        return None, str(exc)
    finally:
        # Restore stderr safely
        os.dup2(dup_stderr, stderr_fd)
        os.close(devnull)
        os.close(dup_stderr)


if __name__ == '__main__':
    # Silence OpenSpiel imports warnings & logs safely on main thread only
    stderr_fd = sys.stderr.fileno()
    dup_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    logging.disable(logging.CRITICAL)
    
    try:
        from kaggle_environments import make
    finally:
        # Restore logging and stderr safely before child processes are spawned
        logging.disable(logging.NOTSET)
        os.dup2(dup_stderr, stderr_fd)
        os.close(devnull)
        os.close(dup_stderr)

    # ============================================================
    # Opponent Pool Discovery
    # ============================================================
    agents_dir = Path("agents")
    all_agents = [str(p).replace("\\", "/") for p in agents_dir.glob("*.py")]

    # Exclude benchmarked agent and any __init__.py files
    opponents_pool = [a for a in all_agents if a != benchmarked_agent and not a.endswith("__init__.py")]
    if not opponents_pool:
        opponents_pool = [benchmarked_agent]

    print("=== Orbit Wars Benchmarking Suite (Parallel) ===")
    print(f"Benchmarked Agent: {benchmarked_agent}")
    print(f"Best Opponent:     {best_opponent}")
    print(f"Opponent Pool:     {opponents_pool}")
    print(f"Matches to Run:    {num_games_2p}x 2-Player, {num_games_4p}x 4-Player\n")

    def get_opponent():
        if best_opponent in opponents_pool and len(opponents_pool) > 1:
            if random.random() < best_opponent_prob:
                return best_opponent
            else:
                other_pool = [o for o in opponents_pool if o != best_opponent]
                return random.choice(other_pool)
        return random.choice(opponents_pool)

    # Seed list to ensure deterministic benchmarks that are different per game
    seeds = [100 + i for i in range(max(num_games_2p, num_games_4p))]

    # ============================================================
    # 2-Player Benchmarking Loop (Parallel)
    # ============================================================
    print("--- Starting 2-Player Benchmark ---")
    wins_2p = 0
    losses_2p = 0
    ties_2p = 0

    replay_2p_idx = random.randint(0, num_games_2p - 1)
    saved_replay_2p_path = None

    with ProcessPoolExecutor() as executor:
        futures_2p = {}
        for i in range(num_games_2p):
            seed = seeds[i]
            opponent = get_opponent()
            
            # Randomize starting slot (player 0 vs player 1)
            my_slot = random.choice([0, 1])
            players = [benchmarked_agent, opponent] if my_slot == 0 else [opponent, benchmarked_agent]
            record_replay = (i == replay_2p_idx)
            
            future = executor.submit(run_single_game, seed, players, my_slot, record_replay)
            futures_2p[future] = (i, seed, opponent)

        for future in as_completed(futures_2p):
            i, seed, opponent = futures_2p[future]
            try:
                res = future.result()
                if isinstance(res, tuple) and res[0] is None:
                    print(f"  Game {i+1:02d} failed inside child process: {res[1]}")
                    continue
                
                my_reward, html = res
                if my_reward > 0:
                    wins_2p += 1
                    result_str = "WIN"
                elif my_reward < 0:
                    losses_2p += 1
                    result_str = "LOSS"
                else:
                    ties_2p += 1
                    result_str = "TIE"
                
                print(f"  Game {i+1:02d}/{num_games_2p:02d} (Seed {seed}, Opponent: {opponent.split('/')[-1]}): {result_str} (reward={my_reward})")
                
                if html is not None:
                    saved_replay_2p_path = Path("replay-2p.html").resolve()
                    saved_replay_2p_path.write_text(html, encoding="utf-8")
                    webbrowser.open(saved_replay_2p_path.as_uri())
            except Exception as exc:
                print(f"  Game {i+1:02d} generated an exception: {exc}")

    # ============================================================
    # 4-Player Benchmarking Loop (Parallel)
    # ============================================================
    print("\n--- Starting 4-Player Benchmark ---")
    wins_4p = 0
    losses_4p = 0
    ties_4p = 0

    replay_4p_idx = random.randint(0, num_games_4p - 1)
    saved_replay_4p_path = None

    with ProcessPoolExecutor() as executor:
        futures_4p = {}
        for i in range(num_games_4p):
            seed = seeds[i]
            
            # Randomize starting slot (player 0, 1, 2, or 3)
            my_slot = random.randint(0, 3)
            
            players = []
            opponent_list = []
            for slot in range(4):
                if slot == my_slot:
                    players.append(benchmarked_agent)
                else:
                    opp = get_opponent()
                    players.append(opp)
                    opponent_list.append(opp.split('/')[-1])
            
            record_replay = (i == replay_4p_idx)
            future = executor.submit(run_single_game, seed, players, my_slot, record_replay)
            futures_4p[future] = (i, seed, opponent_list)

        for future in as_completed(futures_4p):
            i, seed, opponent_list = futures_4p[future]
            try:
                res = future.result()
                if isinstance(res, tuple) and res[0] is None:
                    print(f"  Game {i+1:02d} failed inside child process: {res[1]}")
                    continue
                
                my_reward, html = res
                if my_reward > 0:
                    wins_4p += 1
                    result_str = "WIN"
                elif my_reward < 0:
                    losses_4p += 1
                    result_str = "LOSS"
                else:
                    ties_4p += 1
                    result_str = "TIE"
                
                print(f"  Game {i+1:02d}/{num_games_4p:02d} (Seed {seed}, Opponents: {', '.join(opponent_list)}): {result_str} (reward={my_reward})")
                
                if html is not None:
                    saved_replay_4p_path = Path("replay-4p.html").resolve()
                    saved_replay_4p_path.write_text(html, encoding="utf-8")
                    webbrowser.open(saved_replay_4p_path.as_uri())
            except Exception as exc:
                print(f"  Game {i+1:02d} generated an exception: {exc}")

    # ============================================================
    # Summarize Results
    # ============================================================
    total_2p = wins_2p + losses_2p + ties_2p
    win_rate_2p = (wins_2p / total_2p * 100) if total_2p > 0 else 0.0

    total_4p = wins_4p + losses_4p + ties_4p
    win_rate_4p = (wins_4p / total_4p * 100) if total_4p > 0 else 0.0

    print("\n====================================================")
    print("                    BENCHMARK SUMMARY                ")
    print("====================================================")
    print(f"2-Player Matchups (against randomized pool):")
    print(f"  Total Games: {total_2p}")
    print(f"  Record:      {wins_2p} Wins / {losses_2p} Losses / {ties_2p} Ties")
    print(f"  Win Rate:    {win_rate_2p:.2f}%")
    print()
    print(f"4-Player FFA Matchups (against randomized pool):")
    print(f"  Total Games: {total_4p}")
    print(f"  Record:      {wins_4p} Wins / {losses_4p} Losses / {ties_4p} Ties")
    print(f"  Win Rate:    {win_rate_4p:.2f}%")
    print("====================================================")

    if saved_replay_2p_path:
        print(f"\n2P Replay recorded from game index {replay_2p_idx+1} to: {saved_replay_2p_path}")
    if saved_replay_4p_path:
        print(f"4P Replay recorded from game index {replay_4p_idx+1} to: {saved_replay_4p_path}")
