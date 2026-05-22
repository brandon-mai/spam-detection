## Useful info on Orbit wars that may not have been mentioned

### Kaggle procedures

```bash
# Submit
uv run kaggle competitions submit orbit-wars -f agents/main.py -m "Nearest planet sniper v1"

# Monitor submissions
uv run kaggle competitions submissions orbit-wars
```

### Color of players

- Player 0 (Blue): Plain solid chevron (no markings).
- Player 1 (Vermillion): 1 center stripe (tip to notch).
- Player 2 (Teal): 2 side stripes (running from the center towards the wings).
- Player 3 (Yellow): 3 stripes (both the center and the two side stripes).

---

## Deep RL Agent (APPO + Entity Transformer)

This repository contains an advanced Asynchronous PPO implementation built with JAX, Flax, and Numba.

### 1. Training on a GPU/CPU Machine
To train the agent, you must run the orchestrator which spawns CPU environments and performs JAX neural network updates.

**GPU Setup:** If moving to a machine with a GPU (e.g., RTX 5090 or T4), install JAX with CUDA support using the official release index to prevent version mismatch warnings:
```bash
uv pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Hyperparameters:**
All PPO training hyperparameters (learning rate, workers, epochs, gamma, etc.) are centralized in a `CONFIG` dictionary at the top of `train.py`.

**Start Training:**
```bash
uv run python train.py
```
*Note: The Numba physics engine uses `@njit(cache=True)`. The very first initialization may take a few seconds to compile LLVM machine code, but it will load instantly on all subsequent runs.*

### 2. Testing Locally
You can render a match between your RL agent and a random baseline to visualize its behavior:
```bash
uv run python -c "from kaggle_environments import make; env = make('orbit_wars'); env.run(['agent_rl.py', 'random']); env.render(mode='html', out_path='replay.html')"
```
Open `replay.html` in your browser to watch the match.

### 3. Kaggle Submission
Kaggle allows multi-file agent submissions for Orbit Wars by bundling your files into a `tar.gz` archive. Because `agent_rl.py` relies on multiple local modules (Numba engine, wrapper, and models), you must compress them:

```bash
# 1. Bundle the agent and its dependencies
tar -czvf submission.tar.gz agent_rl.py rl_env_wrapper.py orbit_wars_env_numba.py models.py

# 2. Submit to Kaggle
uv run kaggle competitions submit orbit-wars -f submission.tar.gz -m "Entity Transformer APPO v1"
```

### Important Architectural Details
- **The Autoregressive Loop**: The RL agent evaluates decisions one at a time within a single game tick. It stops automatically when it selects the `Pass` token or runs out of deployable ships.
- **Safety Soft-Cap**: In `agent_rl.py`, the loop is soft-capped at `MAX_USEFUL_ACTIONS` to mathematically guarantee the agent never exceeds Kaggle's 1.0 second per-turn timeout limit, even if the policy becomes temporarily unstable during early training.
