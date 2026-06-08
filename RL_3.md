## Behavioral Cloning vs. Pretraining Core Goals

* **Behavioral Cloning (BC)** matches the rule-based expert's precise choices directly.
* **Pretraining** anchors structural weights so your Graph Attention Network understands basic physics before self-play optimization begins.

---

## What the Model Learns During Pretraining

* **Target Priority:** Identifies and targets comets on the exact tick they spawn.
* **Basic Positioning:** Picks valid, owned source planets instead of aiming randomly.
* **Safety Thresholds:** Avoids draining home bases completely to $0$ units, preventing easy counter-snipes.

---

## Why a Heuristic Ceiling Exists

* **Static Logic:** Rule-based scripts cannot dynamically read multi-agent attrition setups in 4-Player matches.
* **Predictable Execution:** Heuristics execute identical, deterministic arrival loops that higher-ranked models can easily read and counter.

---

## Core Milestones to Reach Before Pivoting to Self-Play RL

### 1. Verification Benchmarks

Run your model against the training dataset using a strict cross-entropy loss function. Pivot to reinforcement learning once your categorical accuracy matches these baseline targets:

$$\text{Target Accuracy} \ge 75\%$$

### 2. Validation Metrics Matrix

| Parameter | Validation Metric Target | Action if Target Fails |
| --- | --- | --- |
| **Source Choice Accuracy** | $\ge 85\%$ | Expand GAT hidden dimensions to $256$. |
| **Target Intercept Accuracy** | $\ge 75\%$ | Check and verify raw input feature normalization formulas. |
| **Quota Allocation Bins** | $\ge 70\%$ | Increase the initialization training pool size to $3,000$ games. |

---

## Performance Expectation: From Pretraining to Elite Play

```
   Pretrained Model (BC)                    Elite Self-Play Model (RL)
┌─────────────────────────────┐          ┌─────────────────────────────┐
│ • Baseline Elo: ~1100       │          │ • Target Elo: ~1350+        │
│ • Emulates rule patterns.   │  ──────► │ • Optimizes unit trade efficiency.│
│ • Vulnerable to novel baits.│          │ • Exploits multi-agent attrition. │
└─────────────────────────────┘          └─────────────────────────────┘

```

* **The Pretrained Baseline:** The agent matches the template script cleanly. It moves units efficiently but cannot improvise when an opponent creates an unseen tactical anomaly.
* **The Self-Play Reinforcement Learning Breakout:** Once the policy gradient updates via an omniscient centralized critic, the network sheds rigid rule limits. It learns to bait enemy doom-stacks, starve opponents through resource denial, and time third-party snipes perfectly to top the leaderboard.