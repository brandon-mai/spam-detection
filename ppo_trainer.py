import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

import numpy as np

def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Computes Generalized Advantage Estimation.
    """
    T, B = rewards.shape
    advantages = np.zeros((T, B), dtype=np.float32)
    lastgaelam = np.zeros(B, dtype=np.float32)
    
    # We must loop backwards over time
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - dones[t]
        nextvalues = next_values[t]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
        
    returns = advantages + values
    return advantages, returns

@jax.jit
def ppo_update_step(state, batch, clip_eps=0.2, c_vf=0.5, c_ent=0.01):
    """
    Executes a single PPO update step.
    batch: dict containing 'planet_matrix', 'fleet_matrix', 'global_vec', 
           'actions', 'logprobs', 'advantages', 'returns', 'masks'
    state: TrainState containing params, apply_fn, tx
    """
    
    def loss_fn(params):
        logits, values = state.apply_fn(
            {'params': params}, 
            batch['planet_matrix'], 
            batch['fleet_matrix'], 
            batch['global_vec']
        )
        
        # Mask out invalid actions
        invalid_masks = ~batch['masks']
        logits = jnp.where(invalid_masks, -jnp.inf, logits)
        
        log_probs = jax.nn.log_softmax(logits)
        action_log_probs = jnp.take_along_axis(log_probs, batch['actions'][:, None], axis=1).squeeze(1)
        
        # Policy Loss
        ratio = jnp.exp(action_log_probs - batch['logprobs'])
        adv = batch['advantages']
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2)
        
        # Value Loss with Clipping
        v_clipped = batch['values'] + jnp.clip(values - batch['values'], -clip_eps, clip_eps)
        v_loss1 = (values - batch['returns']) ** 2
        v_loss2 = (v_clipped - batch['returns']) ** 2
        v_loss = 0.5 * jnp.maximum(v_loss1, v_loss2)
        
        # Clip Fraction
        clip_frac_mask = jnp.abs(ratio - 1.0) > clip_eps
        
        # Entropy Loss
        probs = jax.nn.softmax(logits)
        # Prevent 0 * -inf = NaN which poisons gradients in JAX even if masked later
        safe_log_probs = jnp.where(invalid_masks, 0.0, log_probs)
        entropy = -jnp.sum(probs * safe_log_probs, axis=1)
        
        # Apply Pad Mask
        pad_mask = batch.get('pad_mask', jnp.ones_like(pg_loss, dtype=jnp.bool_))
        
        pg_loss = jnp.where(pad_mask, pg_loss, 0.0)
        v_loss = jnp.where(pad_mask, v_loss, 0.0)
        entropy = jnp.where(pad_mask, entropy, 0.0)
        clip_frac_mask = jnp.where(pad_mask, clip_frac_mask, 0.0)
        
        # Mean over non-padded elements
        valid_count = jnp.sum(pad_mask)
        valid_count = jnp.maximum(valid_count, 1.0) # avoid division by zero
        
        pg_loss = jnp.sum(pg_loss) / valid_count
        v_loss = jnp.sum(v_loss) / valid_count
        ent_loss = -jnp.sum(entropy) / valid_count
        clip_frac = jnp.sum(clip_frac_mask) / valid_count
        
        total_loss = pg_loss + c_vf * v_loss + c_ent * ent_loss
        return total_loss, (pg_loss, v_loss, ent_loss, clip_frac)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss, aux
