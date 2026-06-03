This plan maps out the full, finalized production blueprint for the **Single-Atomic GAT + DRL Pipeline** for the Kaggle Orbit Wars environment. It integrates your simple action scheme, rich engineered temporal features, and sequential imitation learning directly using architectural mechanics discovered in the top-tier [orbit-wars-heuristic-bots repository](https://github.com/vkhydras/orbit-wars-heuristic-bots) by `vkhydras`.

---

## 1. System Topology Overview

```
                  ┌────────────────────────────────────────┐
                  │       1. Raw Game JSON / Tuple         │
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │   2. Graph Constructor (V & E Maps)    │
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │ 3. 3-Layer Graph Attention Network     │
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │ 4. Decoupled Categorical Policy Decoder│
                  └────────────────────────────────────────┘
                     /                 |                \
                    /                  |                 \
                   ▼                   ▼                  ▼
             Head A: Source      Head B: Target     Head C: Quota
              (N_max Dim)         (N_max+1 Dim)        (3 Dim)
                   \                   |                  /
                    \                  |                 /
                     ▼                 ▼                ▼
                  ┌────────────────────────────────────────┐
                  │  5. Instant Execution Physics Module  │
                  └────────────────────────────────────────┘

```

---

## 2. Tensor Framework Definitions

### Node Tensors ($V \in \mathbb{R}^{N_{\text{max}} \times 13}$)

Every planet and comet is mapped to a static feature layout. Missing slots or unspawned comets are masked out ($0$).

```
V_i = [
    Radius (Float),
    Production (Int: 1 to 5),
    Garrison (Int),
    Owner_Self (Binary),
    Owner_Enemy_1 (Binary),      # (Your_ID + 1) % 4
    Owner_Enemy_2 (Binary),      # (Your_ID + 2) % 4
    Owner_Enemy_3 (Binary),      # (Your_ID + 3) % 4
    Owner_Neutral (Binary),
    Is_Comet (Binary),
    Ticks_To_Despawn (Int),      # Extracted from obs["comets"]["paths"]
    Velocity_X (Float),          # Pos_X(t) - Pos_X(t-1)
    Velocity_Y (Float),          # Pos_Y(t) - Pos_Y(t-1)
    Net_Garrison_Delta (Int)     # (Prod * 20) + Inbound_Friendly - Inbound_Hostile
]

```

### Edge Tensors ($E \in \mathbb{R}^{N_{\text{max}} \times N_{\text{max}} \times 4}$)

Directional tracking array mapping relational mechanics between Node $i$ and Node $j$.

```
E_ij = [
    Euclidean_Distance (Float),
    Sun_Intersection_Flag (Binary),
    Inbound_Fleet_Mass (Int),
    Threat_Window_ETA (Int)       # Ticks until closest opponent payload impacts
]

```

---

## 3. Network Architecture Specifications

1. **GAT Core:** Linear transformation layers project $V$ and $E$ to $d_{\text{model}} = 128$. Run through 3 sequential Multi-Head Graph Attention Layers ($H=4$). The output is a matrix $X \in \mathbb{R}^{N_{\text{max}} \times 128}$.
2. **Map Context Vector ($C \in \mathbb{R}^{1 \times 128}$):** Generated via Global Average Pooling across the node matrix $X$.
3. **Game Mode Flag Conditioning:** Append a binary format indicator (`0` for 2P, `1` for 4P) directly to $C$. This allows a single model checkpoint to dynamically shift strategic assumptions (aggressive zero-sum vs. passive third-party snipe survival) based on player count.

### Decoupled Decoders

* **Head A: Source Selector ($S$)**
* **Layer:** Pointer attention mapping.
* **Operation:** $\text{logits}_S = \text{Linear}(X) \cdot C^T$.
* **Mask:** Force components to $-\infty$ if $\text{Owner\_Self} \neq 1$.
* **Dimension:** Categorical over $N_{\text{max}}$.


* **Head B: Target Focus ($T$)**
* **Layer:** Pointer attention mapping conditioned on chosen Source embedding ($X_S$).
* **Operation:** $\text{logits}_T = \text{Linear}(X) \cdot [C \mathbin{\Vert} X_S]^T$. Append a single learnable token scalar at index $N_{\text{max}} + 1$ representing `NO_OP`.
* **Mask:** Force components to $-\infty$ if a direct path from Source to Target crosses the sun, or if the target is an unspawned/expired comet.
* **Dimension:** Categorical over $N_{\text{max}} + 1$.


* **Head C: Allocation Quota ($Q$)**
* **Layer:** 2-Layer MLP conditioned on the chosen tuple context.
* **Operation:** $\text{logits}_Q = \text{MLP}([C \mathbin{\Vert} X_S \mathbin{\Vert} X_T])$.
* **Dimension:** Discrete 3-way classification: `[0: 25%, 1: 50%, 2: 100%]`.



---

## 4. Execution Engine Mechanics

When the model selects a valid action (where $S \neq T$ and $T \neq \text{NO\_OP}$), the output payload is launched **instantly on the current tick**. No deferred background execution queues are maintained.

```python
import math

def process_atomic_drl_action(source_id, target_id, quota_index, obs, planets):
    """
    Executes a single fleet deployment instantly.
    quota_index maps to: 0 -> 0.25, 1 -> 0.50, 2 -> 1.00
    """
    quota_map = {0: 0.25, 1: 0.50, 2: 1.00}
    src = planets[source_id]
    tgt = planets[target_id]
    
    # 1. Enforce vkhydras-style safety floor (Never leave home naked)
    safety_floor = max(10, src.production * 3)
    available_ships = src.ships - safety_floor
    if available_ships <= 0:
        return [] # Coerced into a NO_OP

    quota = quota_map[quota_index]
    ship_payload = math.floor(available_ships * quota)
    if ship_payload <= 0:
        return []

    # 2. Compute true physical speed (Logarithmic scaling formula)
    true_speed = 1.0 + 5.0 * (math.log(ship_payload) / math.log(1000)) ** 1.5

    # 3. Fixed-Point Intercept Convergence
    estimated_eta = 20
    for _ in range(6):
        # Read exact comet elliptical matrices or extrapolate circular planet orbits
        future_x, future_y = predict_future_position(tgt, obs["step"] + estimated_eta, obs)
        distance = math.hypot(future_x - src.x, future_y - src.y)
        estimated_eta = math.ceil(distance / true_speed)

    # Final destination coordinate lock
    final_x, final_y = predict_future_position(tgt, obs["step"] + estimated_eta, obs)
    launch_angle = math.atan2(final_y - src.y, final_x - src.x)

    # Returns official Kaggle atomic action format
    return [[int(source_id), float(launch_angle), int(ship_payload)]]

```

---

## 5. Training Strategy & Optimization

### Sequential Imitation Pre-Training (Behavioral Cloning)

To initialize the GAT parameters without wasting compute, generate an offline dataset of $10,000$ games using [vkhydras's open-source 1.1k Elo heuristic script](https://github.com/vkhydras/orbit-wars-heuristic-bots/blob/master/14_main_k_v2_lb1152_LAST_HEURISTIC.py).

Since his code frequently launches multiple fleets simultaneously per step, parse and serialize those outputs into distinct sequential steps inside your dataset:

```python
# Dataset Serialization Protocol
if len(heuristic_multi_launches) > 0:
    for launch in heuristic_multi_launches:
        # Step t: Save frame state, set target labels to this single launch
        dataset.append({
            'node_features': current_v_tensor,
            'edge_features': current_e_matrix,
            'label_source': launch.source_id,
            'label_target': launch.target_id,
            'label_quota': closest_quota_bin(launch.ships / source_available)
        })
else:
    # Save a NO_OP frame
    dataset.append({
        'node_features': current_v_tensor,
        'edge_features': current_e_matrix,
        'label_source': 0, 'label_target': NO_OP_INDEX, 'label_quota': 0
    })

```

Train the core backbone using standard Cross-Entropy Loss until validation accuracy tracking choices matches the baseline script with $>75\%$ structural parity.

### Self-Play Reinforcement Learning Pivot

Once pre-training completes, deploy standard PPO or V-trace Actor-Critic with a centralized asymmetric value function.

* **The Critic Advantage Check:** Feed the Value network the raw, unmasked opponent intents and destination coordinates during training.
* **Reward Structure:** Utilize a purely sparse zero-sum terminal utility metric ($+1$ for a Win, $-1$ for a Loss). To prevent early gradient stagnation, inject an auxiliary reward tracking net production share delta ($\Delta P = \text{My\_Prod\_Share}_t - \text{My\_Prod\_Share}_{t-1}$), decaying this secondary scaling factor linearly to zero by step $50,000$.