import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Computes Generalized Advantage Estimation.
    Args:
        rewards: (T, B)
        values: (T, B)
        next_values: (T, B)
        dones: (T, B)
    """
    T, B = rewards.shape
    advantages = jnp.zeros((T, B))
    lastgaelam = jnp.zeros(B)
    
    # We must loop backwards over time
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - dones[t]
        nextvalues = next_values[t]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages = advantages.at[t].set(lastgaelam)
        
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
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        
        # Value Loss
        v_loss = 0.5 * jnp.mean((values - batch['returns']) ** 2)
        
        # Entropy Loss
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs, axis=1)
        # NaN safe entropy since log_probs can be -inf for masked
        entropy = jnp.where(jnp.isnan(entropy), 0.0, entropy)
        ent_loss = -jnp.mean(entropy)
        
        total_loss = pg_loss + c_vf * v_loss + c_ent * ent_loss
        return total_loss, (pg_loss, v_loss, ent_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss, aux
