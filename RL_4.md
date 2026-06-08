Reviewing the code reveals **two critical bugs** introduced by the developer in the dataset parsing and training configurations. These directly explain why the metrics show an artificial $100\%$ Quota accuracy while the Source and Target heads are severely underperforming.

---

## 1. Questionable & Compromised Code Decisions

### Bug A: The Quota Division Trick Leak (Artificial 100% Accuracy)

Look closely at how the developer parses the expert's quota inside the dataset generation script (`simulate_match`):

```python
# From Script 4 (Dataset Generation)
garrison = V[src, 2] # This is log1p(garrison) / 10.0 !!
quota_idx = 2
if garrison > 0:
    frac = ships / garrison # Blatant Bug

```

The feature builder `jit_build_graph_features` normalizes `V[src, 2]` using a logarithmic scale: `math.log1p(max(0.0, garrison)) / 10.0`.

Instead of dividing the fleet's raw ship count by the planet's raw garrison, the developer divides the **raw ship count** by a compressed **log-fraction decimal** (which is almost always $<1.0$).

* This makes `frac` overwhelmingly large ($>0.75$) for nearly every single active transition.
* Consequently, `quota_idx` gets set to `2` (`100%`) across the entire dataset. The model achieved $100\%$ accuracy simply because the dataset contains only a single class for the quota head.

### Bug B: The Dynamic NO_OP Source Swap Leak

In the training script (`batch_generator`), the developer performs this swap:

```python
# From Script 1 (Training Loop)
noop_mask = (tgt_batch == 50)
if np.any(noop_mask):
    for j in np.where(noop_mask)[0]:
        owned = np.where(V_batch[j, :, 3] == 1.0)[0]
        if len(owned) > 0:
            src_batch[j] = np.random.choice(owned)

```

When `tgt == 50` (`NO_OP`), the expert data sets `src = 0` as a structural padding element. However, the developer dynamically overwrites `src_batch` with a random owned planet **on every single batch iteration**.

Because `is_active` correctly masks out `NO_OP` steps during the loss calculation, this swap does not harm the gradients directly. However, during validation tracking (`eval_step`), **the validation accuracy metrics are evaluated using these randomly scrambled source targets.** This introduces severe tracking noise, causing the reported Source validation accuracy to plummet to $48\%$ regardless of what the network actually learned.

---

## 2. Immediate Engineering Corrections

To clean up the pipeline and restore accurate gradient profiles, apply these three targeted adjustments:

### Fix 1: Repair the Quota Ratio Logic

Update the data parsing script to read the true, uncompressed planet data array when calculating fractional assignments:

```python
# Extract the true raw garrison from your uncompressed matrix
raw_garrison = planet_matrix[src, 4] 

quota_idx = 2
if raw_garrison > 0:
    frac = ships / raw_garrison
    if frac <= 0.35: quota_idx = 0
    elif frac <= 0.75: quota_idx = 1
    else: quota_idx = 2

```

### Fix 2: Remove the Dynamic Source Scrambler

Eliminate the data generator swap entirely. Let `NO_OP` entries hold a uniform structural target (like `src = 0`). The custom cross-entropy loss function is already fully equipped to ignore these slices safely:

```python
# REMOVE THIS BLOCK ENTIRELY FROM BATCH_GENERATOR:
# noop_mask = (tgt_batch == 50)
# if np.any(noop_mask): ...

```

---

## 3. Next Strategic Steps

Once the pre-training data bug fixes are in place, transition the model into the production pipeline using this sequence:

### 1. Re-run Pre-training Validation

Re-evaluate the model for 10 epochs. With a balanced distribution of quota classes and stable source labels, targets should converge cleanly toward the standard baseline markers:

* **Source Accuracy:** $\ge 85\%$
* **Target Accuracy:** $\ge 75\%$
* **Quota Accuracy:** $\ge 70\%$

### 2. Connect the Autoregressive Inside-Turn Policy Loop

Because the single-atomic head outputs exactly one fleet allocation per evaluation call, implement an intra-turn environment loop to unlock maximum Actions-Per-Minute (APM) without adding architectural complexity.

Evaluate the network repeatedly within the exact same turn window. Update the input node matrices locally (deducting the deployed fleets from the source nodes) until the model selects index $50$ (`NO_OP`), then advance the global clock.

### 3. Initialize the Multi-Agent League Setup

Do not transition straight into open self-play. Avoid policy stagnation by setting up a structured training wheel pool:

* **50% of matches:** Train against the model's own active checkpoint weights.
* **35% of matches:** Match against past historical weight iterations frozen at previous epochs.
* **15% of matches:** Face the original `vkhydras_final.py` script natively to ensure the agent maintains solid fundamental defenses against standard heuristic rules.