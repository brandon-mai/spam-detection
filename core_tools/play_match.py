import os
import sys
import webbrowser
import logging

fd_out, fd_err = sys.stdout.fileno(), sys.stderr.fileno()
saved_out, saved_err = os.dup(fd_out), os.dup(fd_err)
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

def main():
    # 1. Declare agents (can modify paths to test other agents)
    AGENTS = [
        "agents/drl/main.py",
        "agents/nearest_planet_sniper.py"
    ]
    OUT_PATH = "play_replay.html"
    SEED = 42
    
    print(f"Running match between {AGENTS} with seed {SEED}...")
    env = make("orbit_wars", configuration={"seed": SEED, "episodeSteps": 500})
    env.run(AGENTS)
    
    # Check outcomes
    final_step = env.steps[-1]
    for idx, agent_path in enumerate(AGENTS):
        print(f"Agent {idx} ({os.path.basename(agent_path)}): Reward = {final_step[idx].reward}")
        
    print(f"Saving render HTML to {OUT_PATH}...")
    html = env.render(mode="html", width=800, height=600)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
        
    # print("Opening replay in web browser...")
    # abs_path = os.path.abspath(OUT_PATH)
    # webbrowser.open(f"file:///{abs_path}")
    # print("Done!")

if __name__ == "__main__":
    main()
