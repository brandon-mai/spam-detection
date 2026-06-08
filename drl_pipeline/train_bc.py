import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.serialization import to_bytes
from tqdm import tqdm
import functools

from gat_model import OrbitGATModel

def load_data(mode):
    if mode == "2p":
        print("Loading 2P Dataset...")
        data = np.load("drl_pipeline/expert_dataset_2p.npz")
    else:
        print("Loading 4P Dataset...")
        data = np.load("drl_pipeline/expert_dataset_4p.npz")
    return (data["V"], data["E"], data["src"], data["tgt"], data["quota"])

def batch_generator(data, mode_flag, is_val=False, batch_size=256, shuffle=True):
    total_samples = len(data[0])
    val_size = int(total_samples * 0.05)
    train_size = total_samples - val_size
    
    if not is_val:
        indices = np.arange(0, train_size)
    else:
        indices = np.arange(train_size, total_samples)
        
    if shuffle:
        np.random.shuffle(indices)
        
    V_data, E_data, src_data, tgt_data, quota_data = data
    
    for i in range(0, len(indices), batch_size):
        idx = indices[i:i+batch_size]
        
        V_batch = V_data[idx].astype(np.float32)
        E_batch = E_data[idx].astype(np.float32)
        mode_batch = np.full((len(idx), 1), mode_flag, dtype=np.float32)
        src_batch = src_data[idx].astype(np.int32)
        tgt_batch = tgt_data[idx].astype(np.int32)
        quota_batch = quota_data[idx].astype(np.int32)
        
        yield (V_batch, E_batch, mode_batch, src_batch, tgt_batch, quota_batch)



def create_train_state(rng, learning_rate, total_steps):
    model = OrbitGATModel()
    dummy_V = jnp.zeros((1, 50, 13))
    dummy_E = jnp.zeros((1, 50, 50, 4))
    dummy_mode = jnp.zeros((1, 1))
    
    variables = model.init(rng, dummy_V, dummy_E, dummy_mode, true_source=jnp.array([0]), true_target=jnp.array([0]))
    params = variables['params']

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=learning_rate,
        warmup_steps=int(total_steps * 0.05),
        decay_steps=total_steps,
        end_value=1e-6
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    opt_state = tx.init(params)
    
    return params, opt_state, tx, model


@functools.partial(jax.jit, static_argnums=(3, 4))
def train_step(params, opt_state, batch, tx, model):
    V, E, mode, src, tgt, quota = batch

    def loss_fn(p):
        logits_S, logits_T, logits_Q = model.apply({'params': p}, V, E, mode, true_source=src, true_target=tgt)
        
        loss_S_raw = optax.softmax_cross_entropy_with_integer_labels(logits_S, src)
        loss_T_raw = optax.softmax_cross_entropy_with_integer_labels(logits_T, tgt)
        loss_Q_raw = optax.softmax_cross_entropy_with_integer_labels(logits_Q, quota)
        
        is_active = (tgt != 50).astype(jnp.float32)
        
        loss_S = jnp.sum(loss_S_raw * is_active) / jnp.maximum(1.0, jnp.sum(is_active))
        loss_Q = jnp.sum(loss_Q_raw * is_active) / jnp.maximum(1.0, jnp.sum(is_active))
        loss_T = jnp.mean(loss_T_raw)
        
        total_loss = loss_S + loss_T + loss_Q
        
        # Metrics on active frames
        acc_S = jnp.sum((jnp.argmax(logits_S, axis=-1) == src) * is_active) / jnp.maximum(1.0, jnp.sum(is_active))
        acc_T = jnp.sum((jnp.argmax(logits_T, axis=-1) == tgt) * is_active) / jnp.maximum(1.0, jnp.sum(is_active))
        acc_Q = jnp.sum((jnp.argmax(logits_Q, axis=-1) == quota) * is_active) / jnp.maximum(1.0, jnp.sum(is_active))
        
        return total_loss, (loss_S, loss_T, loss_Q, acc_S, acc_T, acc_Q)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, metrics), grads = grad_fn(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, total_loss, metrics


@functools.partial(jax.jit, static_argnums=(2,))
def eval_step(params, batch, model):
    V, E, mode, src, tgt, quota = batch
    logits_S, logits_T, logits_Q = model.apply({'params': params}, V, E, mode, true_source=src, true_target=tgt)
    
    is_active = (tgt != 50).astype(jnp.float32)
    acc_S = jnp.sum((jnp.argmax(logits_S, axis=-1) == src) * is_active) / jnp.maximum(1.0, jnp.sum(is_active))
    acc_T = jnp.sum((jnp.argmax(logits_T, axis=-1) == tgt) * is_active) / jnp.maximum(1.0, jnp.sum(is_active))
    acc_Q = jnp.sum((jnp.argmax(logits_Q, axis=-1) == quota) * is_active) / jnp.maximum(1.0, jnp.sum(is_active))
    
    return acc_S, acc_T, acc_Q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['2p', '4p'], required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', type=str, required=True)
    args = parser.parse_args()

    from flax.serialization import from_bytes
    
    data = load_data(args.mode)
    num_train_samples = int(len(data[0]) * 0.95)
    steps_per_epoch = int(np.ceil(num_train_samples / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    mode_flag = 0.0 if args.mode == '2p' else 1.0

    rng = jax.random.PRNGKey(42)
    params, opt_state, tx, model = create_train_state(rng, args.lr, total_steps)
    
    if args.load:
        print(f"Loading weights from {args.load}...")
        with open(args.load, "rb") as f:
            params = from_bytes(params, f.read())

    os.makedirs('drl_pipeline/checkpoints', exist_ok=True)

    print(f"Starting Training Loop for {args.mode.upper()}...")
    for epoch in range(args.epochs):
        train_gen = batch_generator(data, mode_flag, is_val=False, batch_size=args.batch_size, shuffle=True)
        
        epoch_losses = []
        epoch_acc_S = []
        epoch_acc_T = []
        epoch_acc_Q = []
        
        pbar = tqdm(train_gen, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = tuple(jnp.array(x) for x in batch)
            params, opt_state, total_loss, metrics = train_step(params, opt_state, batch, tx, model)
            
            _, _, _, acc_S, acc_T, acc_Q = metrics
            epoch_losses.append(total_loss)
            epoch_acc_S.append(acc_S)
            epoch_acc_T.append(acc_T)
            epoch_acc_Q.append(acc_Q)
            
            pbar.set_postfix({"Loss": f"{total_loss:.4f}", "S_Acc": f"{acc_S:.2f}"})

        # Validation
        val_gen = batch_generator(data, mode_flag, is_val=True, batch_size=args.batch_size, shuffle=False)
        val_acc_S, val_acc_T, val_acc_Q = [], [], []
        for batch in val_gen:
            batch = tuple(jnp.array(x) for x in batch)
            v_S, v_T, v_Q = eval_step(params, batch, model)
            val_acc_S.append(v_S)
            val_acc_T.append(v_T)
            val_acc_Q.append(v_Q)

        avg_loss = np.mean(epoch_losses)
        avg_train_S = np.mean(epoch_acc_S)
        avg_train_T = np.mean(epoch_acc_T)
        avg_train_Q = np.mean(epoch_acc_Q)
        
        avg_val_S = np.mean(val_acc_S)
        avg_val_T = np.mean(val_acc_T)
        avg_val_Q = np.mean(val_acc_Q)

        print(f"\\n--- Epoch {epoch+1} Results ---")
        print(f"Train Loss: {avg_loss:.4f} | Train Acc: S={avg_train_S:.2%}, T={avg_train_T:.2%}, Q={avg_train_Q:.2%}")
        print(f"Val Acc:   S={avg_val_S:.2%}, T={avg_val_T:.2%}, Q={avg_val_Q:.2%}")
        
    # Save Final Checkpoint
    ckpt_path = f"drl_pipeline/checkpoints/{args.save}"
    with open(ckpt_path, "wb") as f:
        f.write(to_bytes(params))
    print(f"Final Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    main()
