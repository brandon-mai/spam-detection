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
    "num_workers": 16,            # Number of CPU rollout processes
    "num_iterations": 1000,       # Total training updates
    "episodes_per_worker": 5,     # Rollouts generated per worker per iter
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "c_vf": 0.5,
    "c_ent": 0.01,
}

def pad_and_stack(list_of_arrays):
    return jnp.stack(list_of_arrays)

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
    tx = optax.adam(learning_rate=CONFIG["learning_rate"])
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    weights_queue = mp.Queue()
    traj_queue = mp.Queue()
    
    workers = []
    for i in range(num_workers):
        p = mp.Process(target=worker_process, args=(i, weights_queue, traj_queue, CONFIG["episodes_per_worker"]))
        p.start()
        workers.append(p)
        
    # Send initial weights
    # Convert frozen dict to standard dict for pickling if necessary, or just send params
    weights_queue.put(state.params)
    
    print("Starting APPO Training Loop...")
    
    for it in range(num_iterations * num_workers):
        print(f"Waiting for trajectory {it+1}...")
        traj = traj_queue.get()
        
        # Unpack
        p_mats = pad_and_stack(traj['planet_matrices'])
        f_mats = pad_and_stack(traj['fleet_matrices'])
        g_vecs = pad_and_stack(traj['global_vecs'])
        masks = pad_and_stack(traj['masks'])
        actions = jnp.array(traj['actions'])
        rewards = jnp.array(traj['rewards'])
        values = jnp.array(traj['values'])
        logprobs = jnp.array(traj['logprobs'])
        dones = jnp.array(traj['dones'])
        
        # Next value for GAE is 0 since the episode is done
        next_values = jnp.append(values[1:], 0.0)
        
        # Add a dummy batch dimension for GAE (T, B)
        advantages, returns = compute_gae(
            rewards[:, None], values[:, None], next_values[:, None], dones[:, None],
            gamma=CONFIG["gamma"], gae_lambda=CONFIG["gae_lambda"]
        )
        
        # Remove dummy batch dim
        advantages = advantages.squeeze(1)
        returns = returns.squeeze(1)
        
        batch = {
            'planet_matrix': p_mats,
            'fleet_matrix': f_mats,
            'global_vec': g_vecs,
            'masks': masks,
            'actions': actions,
            'logprobs': logprobs,
            'advantages': advantages,
            'returns': returns
        }
        
        # PPO Update
        state, loss, aux = ppo_update_step(state, batch, 
            clip_eps=CONFIG["clip_eps"], c_vf=CONFIG["c_vf"], c_ent=CONFIG["c_ent"]
        )
        pg_loss, v_loss, ent_loss = aux
        print(f"Iter {it+1}: Loss: {loss:.4f} (PG: {pg_loss:.4f}, V: {v_loss:.4f}, Ent: {ent_loss:.4f}), Reward sum: {rewards.sum():.1f}")
        
        # Send new weights
        weights_queue.put(state.params)
        
        # Save Checkpoint
        if (it + 1) % 50 == 0 or (it + 1) == (num_iterations * num_workers):
            with open("model_weights.msgpack", "wb") as f:
                f.write(serialization.to_bytes(state.params))
            print(f"Checkpoint saved at iteration {it+1}.")
            
    for p in workers:
        p.join()
        
    print("Training complete.")

if __name__ == "__main__":
    # Required for Windows multiprocessing compatibility, and strictly required 
    # for JAX on Linux to prevent fork() deadlocks.
    mp.set_start_method('spawn', force=True)
    main()
