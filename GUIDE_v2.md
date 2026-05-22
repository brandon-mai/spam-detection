Here is the definitive, production-ready architectural and training guideline for your developer to implement a state-of-the-art RL hybrid agent for *Orbit Wars*.

---

## 1. System Topology & Deep-Dive Architecture

The core architecture must be an **Invariant Entity Transformer** trained using **Masked Proximal Policy Optimization (PPO)**. Because the numbers of planets, active fleets, and comets fluctuate drastically throughout a 500-turn game, a classic fixed-size fully connected network or CNN cannot generalize.

### Complete Feature Engineering (Input Entity Matrices)

The raw observation must be broken down by the analytical pipeline into clean, normalized vectors for the Transformer blocks.

1. **Planet Entities Matrix ($N \times 12$):** For each planet/comet:
* Spatial features: `[x/100, y/100, radius/10]`
* Ownership (One-Hot): `[is_me, is_p2, is_p3, is_p4, is_neutral]`
* Economic state: `[ships / 1000, production / 5]`
* Tactical vectors: `[incoming_friendly_ships / 1000, incoming_enemy_ships / 1000]`
* Comet status: `[is_comet (0 or 1), turns_until_expiration / 500]`


2. **Fleet Entities Matrix ($M \times 6$):** For each live fleet in space:
* `[x/100, y/100, cos(angle), sin(angle), ships / 1000, owner_id]`


3. **Global Context Vector ($1 \times 4$):**
* `[current_turn / 500, my_total_ship_share, total_active_fleets, active_comets_count]`



### The Attention Layer

* Each entity type goes through its own **Linear Projection Layer** to transform raw feature arrays into a uniform embedding size ($D=128$).
* These embeddings are concatenated into a variable-length token list and passed through **2 to 3 Self-Attention Block Layers**. This natively allows a planet token to calculate its relationship relative to an incoming fleet token or a nearby comet vector without manual cross-feature scripting.

---

## 2. Multi-Selection Autoregressive Action Space

To handle launching multiple fleets from multiple planets on a single tick without causing an exponential explosion in options, use a **Combinatorial Autoregressive Token Loop**.

Instead of predicting an entire array of fleets simultaneously, the network evaluates a sequence of individual intent choices during a single turn.

### Action Token Design

The action space is a simplified multi-discrete token tuple:


$$\text{Action} = [\text{Source\_Planet\_ID}, \text{Target\_Planet\_ID}, \text{Allocation\_Bin}]$$

* **Source Nodes:** $N$ possible planet slots.
* **Target Nodes:** $N$ possible target slots (including comets and neutral bodies).
* **Allocation Bins:** $4$ discrete percentages: `[25%, 50%, 75%, 100%]` of the safely available local garrison.
* **The Terminators:** A specific **Pass Action Token**.

### The Multi-Selection Selection Loop (Inside Turn $T$)

```
While True:
    1. Pass the filtered Entity Matrices to the Transformer Network.
    2. Network outputs exactly ONE Action Tuple.
    3. IF Action == "Pass Token" -> Break loop and submit final move queue.
    4. ELSE:
        - Append [Source, Target, Allocation] to the turn's physical move list.
        - Deduct the allocated ships from the local simulation state array.
        - Dynamically recalculate and apply the Action Mask.

```

This loop allows the agent to orchestrate multi-planet cooperative attacks or launch distinct defensive waves simultaneously on a single frame, scaling seamlessly regardless of how many planets you command.

### Bulletproof Action Masking (Forcing Invalid Logits to $-\infty$)

To maximize sample training efficiency, the network must be blocked from choosing invalid paths. An action choice is masked out if:

* The source planet is not currently owned by the agent.
* The selected allocation size drops below the environment's hard minimum threshold (`MIN_SHIPS_MINE_ATTACK`).
* The analytical pathing script proves the trajectory will intersect the sun's kill radius.
* **Comet Safety Failure:** The target is a comet, but the analytical intercept script shows the fleet's arrival tick will happen *after* the comet's expiration timeline.

---

## 3. Factoring in Comets (Pre-emptive Targeting)

Comets cannot be treated like standard orbiting planets; they are fast-moving anomalies with clear, pre-calculated paths available right in the environment observations.

* **The Analytical Intercept Pipeline:** Do not let the network guess angles to strike a moving target. The Numba engine must iteratively solve the intercept physics:

$$\text{Distance}(\text{Source}, \text{Path}[T + \Delta t]) = \text{Velocity}(\text{Ships}) \times \Delta t$$


* **Pre-emptive Launch Vectors:** Comets are visible in the configuration path data *before* they officially enter active play. The planet entity list should include "Ghost Nodes" for upcoming comets. This allows the autoregressive network to choose to launch fleets into completely empty space turns in advance, intercepting the comet precisely on the frame it arrives.

---

## 4. The Training Paradigm: Self-Play vs. Imitation

### The Verdict: Pure Self-Play with Rule-Based Bootstrap

Do **not** use traditional supervised imitation learning or gather data from public top-performing leaderboard scripts. Rule-based heuristics on the leaderboard are inherently bound by hardcoded strategies; an imitation network will copy their behavioral biases and hit a strict performance ceiling.

Instead, execute an **Accelerated Reinforcement Learning Self-Play pipeline** from a completely clean state:

```
[Phase 1: Bootstrapping] ──► [Phase 2: Self-Play Evolution] ──► [Phase 3: Elo Stabilization]
(Play vs. Fixed Heuristics)    (League of Historical Self-Checkpoints)   (Incorporate 4P Free-For-All)

```

1. **Phase 1 (Bootstrapping):** Train the initial model checkpoints against your existing rule-based heuristic code (`owproto_v2.py`). This forces the agent to rapidly discover baseline concepts like expansion, time-to-intercept, and basic garrison management.
2. **Phase 2 (Self-Play League):** Once the model consistently outperforms the rule-based script, pivot to a self-play pool. The training agent should play $80\%$ of its matches against past versions of itself. This encourages the model to naturally discover complex game-theory concepts like baiting fleets, handling multi-front aggression, and backstabbing overextended players.
3. **Phase 3 (4-Player Scaling):** Run parallel training instances for both 2-player and 4-player modes. In 4-player configurations, randomize opponent styles (e.g., 2 spots filled by historical self-play checkpoints, 1 spot filled by a defensive heuristic) to ensure the emergent strategy remains resilient against chaotic meta plays.

---

## 5. Developer Vocabulary & Research Keywords

Tell your developer to focus their technical research on these exact architecture components to streamline implementation:

* **Entity-Component Reinforcement Learning / EntityGym:** Framework architectures engineered natively to handle variable counts of dynamic inputs (planets/fleets) without static state limitations.
* **Autoregressive Policy Heads:** The standard design mechanism for mapping dependencies between sequential action outputs (e.g., selecting a source unit, then choosing its target).
* **Invalid Action Masking in PPO:** A vital trick for multi-agent strategy spaces that zeroes out logits for illegal paths, guaranteeing stable and rapid policy convergence.
* **Asynchronous PPO / JAX Environment Vectorization:** Essential paradigms for scaling your Numba/JAX-wrapped simulation across multi-GPU setups to generate millions of gameplay frames per hour.

---

Your developer is spot on to worry about the timeout mechanics, but implementing a simple "hard cap" is a band-aid fix that treats the symptom rather than curing the systemic flaw.

In an autoregressive RL setup, an infinite loop or high turn latency happens for exactly two reasons: **bad reward feedback loop design during training** and a **lack of action-mask termination signals**.

We absolutely need a guardrail framework, but it must be applied mathematically and structurally rather than just a crude iteration counter.

---

## Why a Crude Hard Cap Fails

If you just set a hard cap (e.g., `if actions > 20: break`), the agent can run out of allocations midway through calculating a critical, synchronized 5-planet defense maneuvers. Even worse, during training, a generic hard cutoff means the PPO trajectory buffer records truncated data, making the policy model view the cutoff as an arbitrary environment limit rather than learning **how to stop on its own**.

---

## The Correct Guardrail Blueprint

To design this with absolute safety for the Kaggle servers, your developer needs to implement a three-tiered guardrail system:

### 1. Structural Guardrail: The Dynamic State Depletion Mask

The most elegant way to stop an infinite loop is to make it mathematically impossible for the network to choose an attack or reinforcement command.

* Every time the loop executes and the network chooses a `[Source_Planet, Target_Planet, Allocation_Bin]` tuple, those ships are immediately subtracted from your local tracking state array.
* If a source planet's remaining garrison falls below the `MIN_SHIPS_MINE_ATTACK` rule threshold, **its valid mask bit is instantly toggled to False (0)**.
* **The Result:** As soon as you run out of deployable ships across your entire empire, the *Action Mask* will completely block every single action token except for one: the **Pass Token**.

### 2. Algorithmic Guardrail: The Pass Token Premium (Entropy Regularization)

The network must *want* to choose the "Pass Token" once it has executed its high-value macro moves. If the model behaves erratically or gets caught in loops during training, it means your PPO reward function isn't punishing unnecessary complexity.

* **The Penalty:** Apply a minute execution cost during training for every macro intent step generated within a single frame (e.g., $-0.005$ per active loop step).
* **The Reward Shaping:** Ensure the model learns that hoarding surplus ships to trigger planet production scaling yields a higher expected value than spamming micro-fleets.

### 3. Safety Net Guardrail: The Soft-Cap Submission Fallback

To satisfy the developer's concern about the strict Kaggle execution time limit, implement a dual-condition exit loop inside the agent code. This acts as an emergency escape while maintaining valid game logic:

```python
def agent(obs, config):
    # ... Process Analytical Numba Engine Data ...
    
    actions_taken = 0
    # Hard safety cap based on max possible useful selections 
    MAX_USEFUL_ACTIONS = len(my_planets) * 4 
    
    while actions_taken < MAX_USEFUL_ACTIONS:
        state_tensor, action_mask = generate_tensors(local_state)
        
        # Inference
        action_token = model.predict(state_tensor, mask=action_mask)
        
        if action_token == PASS_TOKEN_INDEX:
            break
            
        # Execute local tracking update
        apply_local_allocation(local_state, action_token)
        actions_taken += 1
        
    return format_moves_for_kaggle()

```

* **Why this works:** Setting the cap to $\text{Owned Planets} \times \text{Allocation Bins}$ guarantees that even if the agent controls every single node on the map, it can never generate more decisions than there are practical asset configurations, eliminating infinite loops while keeping the runtime safely predictable.