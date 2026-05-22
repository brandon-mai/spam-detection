## 1. Ground Truth (Deterministic & Calculable)

Everything listed below can be computed with $100\%$ precision directly from the `obs` dictionary on Turn $T$. There is zero stochasticity or "learning" required for these items:

* **Planet Ephemerides:** The exact coordinate $(x, y)$ of any moving planet $i$ at any future tick $T + \Delta t$ using its radius, distance from the sun $(50, 50)$, current position, and the global constant `obs.angular_velocity`.
* **Sun Threat Line:** Whether a linear vector between planet $A$ and planet $B$ intersects the sun's hazard zone (radius $10$ at $50,50$).
* **True Intercept Angles:** The exact launch angle $\theta$ required for a fleet moving at speed $V(\text{ships})$ from planet $A$ to arrive exactly when target planet $B$ occupies that same spatial coordinate.
* **Dynamic Garrison Cost:** The exact minimum fleet size required to successfully capture an enemy planet, calculated as:

$$\text{Garrison}_{\text{arrival}} = \text{Garrison}_{\text{current}} + (\text{ETA} \times \text{Production})$$


* **Arrival Timeline:** The exact turn tick entry of every incoming friendly and enemy fleet currently traveling through space.

---

## 2. Hard Performance Targets (Must Be Accelerated)

To execute deep lookahead searches within Kaggle’s runtime constraints, you must isolate and compile these specific math engines into pure array-based machine code (e.g., using Numba):

* **Ray-Circle Segment Intersections:** A vectorized mathematical function checking if a fleet line segment collides with any circular planet boundary or the sun over a $60$-step forward horizon.
* **Multi-Agent Forward State Simulator:** A minimal, object-free function that accepts flat $2\text{D}$ NumPy arrays of planet data and fleet data, simulates a single state tick forward, handles basic combat math, and updates ownerships. This acts as your deterministic local sandbox.

---

## 3. RL Delegated Sandbox (What is Learned)

The reinforcement learning agent is completely insulated from geometry and physics. It is trained strictly to solve **imperfect information** and **game-theoretic strategies**:

* **Opponent Intent Estimation:** Predicting which planet an opponent is about to target or abandon before their fleets are visibly launched into space.
* **Third-Party Free-For-All Risk:** Evaluating if executing a winning localized attack will leave the agent's home base depleted and vulnerable to an instant counter-attack by a third player.
* **Comet Investment Valuation:** Deciding how many ships are worth sacrificing to contest a fast-moving comet vs. hoarding those forces to fortify secure local production nodes.

---

## 4. Rigorous Problem Formulation

### A. The Structural Pipeline

To keep things lightweight, the game loops through an analytical wrapper that filters state arrays before handing them to the RL core:

```
[Kaggle Obs] ──► [Analytical Engine] ──► [Masked Features] ──► [RL Core] ──► [Selected Intent] ──► [Raw Action]

```

### B. Input (State Features $S$)

The observation space is structured as a fixed-size state matrix, packed tightly for neural network inference:

* **Planet Feature Matrix ($N \times F$):** For each planet: `[owner_one_hot (5), ship_count, production_rate, distance_to_sun, ticks_until_comet_spawn, incoming_friendly_ships, incoming_enemy_ships]`.
* **Global Vector:** `[current_step / 500, my_total_ship_share, opponent_total_ship_share]`.

### C. Output (Action Space $A$)

A discrete choice over a structured **Macro Strategic Intent**:

* **Action Value:** A selection index representing a unique `[Source_Planet_ID, Target_Planet_ID, Allocation_Bin]`.
* **Allocation Bins:** $4$ discrete options representing the size of the offensive force: `[25%, 50%, 75%, 100%]` of the source planet's maximum safe-to-send pool.
* **The Pass Option:** A dedicated index for doing absolutely nothing on that frame to preserve fleet sizes.

### D. Action Masking Criteria

Prior to evaluating the policy network's outputs, the available action choices are passed through a boolean validity filter. An action logit is forced to $-\infty$ if:

1. The source planet is not owned by the agent.
2. The analytical intercept vector intersects the sun or an obstructing planet on its trajectory.
3. The chosen fleet size allocation falls below the environment's hard minimum ship requirement (`MIN_SHIPS_MINE_ATTACK`).

### E. Reward Engineering ($R$)

Executed at terminal step $T_{\text{end}}$ or at every step step $t$ using a reward-shaping formula focused on long-term domination:

$$R_t = \Delta \text{Owned\_Production} + \beta \left( \frac{\text{My\_Ships}_t}{\sum \text{Total\_World\_Ships}_t} \right) - \gamma \cdot \mathbb{I}(\text{Home\_Planet\_Lost})$$

* $\Delta \text{Owned\_Production}$: The net change in the agent's total factory throughput capacity.
* $\mathbb{I}$: An indicator function that applies a severe penalty if a crucial core node is left defenseless and captured by a rival.

### F. Network Architecture

A compact, low-parameter model optimized for lightning-fast initialization and evaluation:

```
                      Input Features (State Matrix S)
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
                 [Dense Layer 128]         [Dense Layer 128]
                        │                         │
                 [Dense Layer 64]          [Dense Layer 64]
                        │                         │
                        ▼                         ▼
               Policy Head (Logits)      Value Head (Scalar V)
              Size: Max Total Actions    Size: 1 (Win Probability)
                        │
             [Apply Boolean Action Mask]

```

* **Policy Head:** Outputs raw priority scores across all available source-target assignments.
* **Value Head:** Estimates the agent's current probability of winning the game from the given state configuration. This scalar is used to score terminal branches during an ongoing MCTS tree search.