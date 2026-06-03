import os
import sys
import contextlib
import webbrowser

# Silently import make from kaggle_environments
@contextlib.contextmanager
def silence_outputs():
    stdout_fd = 1
    stderr_fd = 2
    try:
        dup_stdout = os.dup(stdout_fd)
        dup_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        os.dup2(dup_stdout, stdout_fd)
        os.dup2(dup_stderr, stderr_fd)
        os.close(devnull)
        os.close(dup_stdout)
        os.close(dup_stderr)

with silence_outputs():
    from kaggle_environments import make

def main():
    # 1. Declare agents (can modify paths to test other agents)
    AGENTS = [
        "agents/hellburner_v2.py",
        "agents/may18.py"
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
        
    print("Opening replay in web browser...")
    abs_path = os.path.abspath(OUT_PATH)
    webbrowser.open(f"file:///{abs_path}")
    print("Done!")

if __name__ == "__main__":
    main()
