import os
# Prevent JAX from instantly pre-allocating 90% of the GPU VRAM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import optax
import multiprocessing as mp
import time
import numpy as np

from models import EntityTransformer
from ppo_trainer import compute_gae, ppo_update_step
from rollout_worker import worker_process
from flax.training import train_state
from flax import serialization

# Hyperparameters Configuration
CONFIG = {
    "num_workers": 30,            # Number of CPU rollout processes
    "num_iterations": 1000,       # Total training updates
    "episodes_per_worker": 1,     # Rollouts generated per worker per iter
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "c_vf": 0.5,
    "c_ent": 0.05,
}

def pad_and_stack(list_of_arrays):
    return np.stack(list_of_arrays)

def main():
    num_workers = CONFIG["num_workers"]
    num_iterations = CONFIG["num_iterations"]
    
    # Initialize Model and Optimizer
    rng = jax.random.PRNGKey(0)
    model = EntityTransformer()
    
    # Dummy inputs for init
    dummy_p = jnp.zeros((1, 60, 14))
    dummy_f = jnp.zeros((1, 1000, 9))
    dummy_g = jnp.zeros((1, 4))
    
    variables = model.init(rng, dummy_p, dummy_f, dummy_g)
    
    # Optax Warmup Cosine Decay Schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=CONFIG["learning_rate"],
        warmup_steps=100,
        decay_steps=CONFIG["num_iterations"] * 4, # roughly epochs
        end_value=1e-6
    )
    tx = optax.adam(learning_rate=lr_schedule)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    weights_queue = mp.Queue()
    traj_queue = mp.Queue()
    
    # Eval queues
    eval_weights_queue = mp.Queue()
    eval_traj_queue = mp.Queue()
    
    workers = []
    # Leave 1 worker for background evaluation
    train_workers = num_workers - 1
    for i in range(train_workers):
        p = mp.Process(target=worker_process, args=(i, weights_queue, traj_queue, CONFIG["episodes_per_worker"]))
        p.start()
        workers.append(p)
        
    # Start Eval Worker (runs 1 greedy episode per request)
    p_eval = mp.Process(target=worker_process, args=(-1, eval_weights_queue, eval_traj_queue, 1))
    p_eval.start()
    workers.append(p_eval)
    
    from flax import serialization
    
    start_iter = 0
    if os.path.exists("train_state.msgpack"):
        with open("train_state.msgpack", "rb") as f:
            state = serialization.from_bytes(state, f.read())
        print("Restored TrainState from train_state.msgpack")
        
        if os.path.exists("training_log.csv"):
            import csv
            with open("training_log.csv", "r") as f:
                reader = csv.reader(f)
                lines = list(reader)
                if len(lines) > 1:
                    try:
                        start_iter = int(lines[-1][0])
                    except:
                        pass
        print(f"Resuming from iteration {start_iter}")
    else:
        # Save Initial Weights for Pure Self-Play only if starting fresh
        if not os.path.exists("champion.msgpack"):
            with open("champion.msgpack", "wb") as f:
                f.write(serialization.to_bytes(state.params))
    
    # Start pure self-play
    current_opponent = "agents/self_play_agent.py"
    champion_version = 0
    recent_wins = 0
    eval_match_count = 0
    
    for _ in range(train_workers):
        weights_queue.put({"weights": state.params, "opponent": current_opponent})
    
    print("Starting APPO Training Loop...")
    
    batch_size_trajectories = train_workers # 15
    epochs = 4
    
    for it in range(start_iter, num_iterations):
        print(f"Waiting for {batch_size_trajectories} trajectories for Iteration {it+1}...")
        trajs = []
        for _ in range(batch_size_trajectories):
            trajs.append(traj_queue.get())
            
        all_p_mats, all_f_mats, all_g_vecs, all_masks = [], [], [], []
        all_actions, all_logprobs, all_advantages, all_returns = [], [], [], []
        
        sum_rewards = 0.0
        
        for traj in trajs:
            p_mats = pad_and_stack(traj['planet_matrices'])
            f_mats = pad_and_stack(traj['fleet_matrices'])
            g_vecs = pad_and_stack(traj['global_vecs'])
            masks = pad_and_stack(traj['masks'])
            actions = np.array(traj['actions'])
            rewards = np.array(traj['rewards'])
            values = np.array(traj['values'])
            logprobs = np.array(traj['logprobs'])
            dones = np.array(traj['dones'])
            
            sum_rewards += rewards.sum()
            
            next_values = np.append(values[1:], 0.0)
            advantages, returns = compute_gae(
                rewards[:, None], values[:, None], next_values[:, None], dones[:, None],
                gamma=CONFIG["gamma"], gae_lambda=CONFIG["gae_lambda"]
            )
            
            all_p_mats.append(p_mats)
            all_f_mats.append(f_mats)
            all_g_vecs.append(g_vecs)
            all_masks.append(masks)
            all_actions.append(actions)
            all_logprobs.append(logprobs)
            all_advantages.append(advantages.squeeze(1))
            all_returns.append(returns.squeeze(1))
            
        batch = {
            'planet_matrix': jnp.array(np.concatenate(all_p_mats)),
            'fleet_matrix': jnp.array(np.concatenate(all_f_mats)),
            'global_vec': jnp.array(np.concatenate(all_g_vecs)),
            'masks': jnp.array(np.concatenate(all_masks)),
            'actions': jnp.array(np.concatenate(all_actions)),
            'values': jnp.array(np.concatenate([np.array(t['values']) for t in trajs])),
            'logprobs': jnp.array(np.concatenate(all_logprobs)),
            'advantages': jnp.array(np.concatenate(all_advantages)),
            'returns': jnp.array(np.concatenate(all_returns))
        }
        
        # PPO Update in multiple epochs
        batch_size = 256
        T_total = batch['actions'].shape[0]
        num_chunks = int(np.ceil(T_total / batch_size))
        
        if num_chunks == 0:
            print("Trajectory entirely empty, skipping PPO update.")
            loss = pg_loss = v_loss = ent_loss = clip_frac = 0.0
        else:
            total_loss, total_pg, total_v, total_ent, total_clip = 0.0, 0.0, 0.0, 0.0, 0.0
            updates = 0
            
            for epoch in range(epochs):
                perms = np.random.permutation(T_total)
                for i in range(num_chunks):
                    start = i * batch_size
                    end = min(start + batch_size, T_total)
                    idx = perms[start:end]
                    
                    chunk_batch = {k: v[idx] for k, v in batch.items()}
                    
                    # Pad the minibatch if it's smaller than batch_size (for JIT stability)
                    current_len = end - start
                    if current_len < batch_size:
                        pad_len = batch_size - current_len
                        padded_chunk = {}
                        for k, v in chunk_batch.items():
                            if k == 'masks':
                                padded_chunk[k] = jnp.pad(v, ((0, pad_len), (0, 0)))
                            else:
                                padded_chunk[k] = jnp.pad(v, ((0, pad_len),) + ((0, 0),) * (v.ndim - 1))
                        padded_chunk['pad_mask'] = jnp.array([True] * current_len + [False] * pad_len)
                        chunk_batch = padded_chunk
                    else:
                        chunk_batch['pad_mask'] = jnp.ones((batch_size,), dtype=jnp.bool_)
                        
                    state, l, aux = ppo_update_step(state, chunk_batch, clip_eps=0.2, c_vf=0.5, c_ent=0.01)
                    l.block_until_ready()
                    
                    total_loss += l
                    total_pg += aux[0]
                    total_v += aux[1]
                    total_ent += aux[2]
                    total_clip += aux[3]
                    updates += 1
                    
            loss = total_loss / updates
            pg_loss = total_pg / updates
            v_loss = total_v / updates
            ent_loss = total_ent / updates
            clip_frac = total_clip / updates
            
        print(f"Iter {it+1}: Loss: {loss:.4f} (PG: {pg_loss:.4f}, V: {v_loss:.4f}, Ent: {ent_loss:.4f}, Clip: {clip_frac:.4f}), Reward sum: {sum_rewards:.1f}")
        
        # CSV Logging
        log_file = "training_log.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a") as f:
            if not file_exists:
                f.write("Iter,Loss,PG_Loss,V_Loss,Ent_Loss,Clip_Frac,Reward_Sum\n")
            f.write(f"{it+1},{loss:.4f},{pg_loss:.4f},{v_loss:.4f},{ent_loss:.4f},{clip_frac:.4f},{sum_rewards:.1f}\n")
            
        # Non-blocking Eval Queue Check
        while not eval_traj_queue.empty():
            eval_traj = eval_traj_queue.get()
            eval_match_count += 1
            eval_reward = sum(eval_traj['rewards'])
            print(f">>> EVAL Match {eval_match_count} vs owproto_v2: Reward = {eval_reward}")
            with open("eval_log.csv", "a") as f:
                f.write(f"{it+1},{eval_reward}\n")
        
        # Dispatch background eval request every 5 iterations
        if (it + 1) % 5 == 0:
            eval_weights_queue.put({"weights": state.params, "opponent": "agents/owproto_v2.py", "greedy": True})
        
        # Track wins (each trajectory can be 1.0)
        if sum_rewards > 0:
            recent_wins += 1
            
        # Send new weights
        weights_queue.put({"weights": state.params, "opponent": current_opponent})
        
        # Evaluate Curriculum
        if (it + 1) % 50 == 0:
            win_rate = recent_wins / 50.0
            print(f"--- Last 50 Episodes Win Rate against {current_opponent}: {win_rate*100:.1f}% ---")
            
            if win_rate >= 0.55:
                champion_version += 1
                print(f"New Champion! Promoting agent to version {champion_version}.")
                
                # Save new champion weights
                with open("champion.msgpack", "wb") as f:
                    f.write(serialization.to_bytes(state.params))
                with open(f"checkpoint_{champion_version}.msgpack", "wb") as f:
                    f.write(serialization.to_bytes(state.params))
            
            recent_wins = 0
            
        # Save periodic training checkpoint
        if (it + 1) % 10 == 0:
            with open("model_weights.msgpack", "wb") as f:
                f.write(serialization.to_bytes(state.params))
            with open("train_state.msgpack", "wb") as f:
                f.write(serialization.to_bytes(state))
            print(f"Periodic training weights and train_state saved at iteration {it+1}.")
            
    # Shut down workers
    for _ in range(train_workers):
        weights_queue.put(None)
    eval_weights_queue.put(None)
        
    for p in workers:
        p.join()
        
    print("Training complete.")

if __name__ == "__main__":
    # Required for Windows multiprocessing compatibility, and strictly required 
    # for JAX on Linux to prevent fork() deadlocks.
    mp.set_start_method('spawn', force=True)
    main()
