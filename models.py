import jax
import jax.numpy as jnp
import flax.linen as nn

MAX_PLANETS = 60
MAX_FLEETS = 1000

class EntityTransformer(nn.Module):
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    
    @nn.compact
    def __call__(self, planet_matrix, fleet_matrix, global_vec):
        """
        planet_matrix: (B, MAX_PLANETS, 14)
        fleet_matrix: (B, MAX_FLEETS, 9)
        global_vec: (B, 4)
        """
        B = planet_matrix.shape[0]
        
        # 1. Project Entities
        planet_emb = nn.Dense(self.d_model, name="planet_proj")(planet_matrix) # (B, P, D)
        fleet_emb = nn.Dense(self.d_model, name="fleet_proj")(fleet_matrix)    # (B, F, D)
        
        # Expand global vec to sequence length 1
        global_emb = nn.Dense(self.d_model, name="global_proj")(global_vec)    # (B, D)
        global_emb = jnp.expand_dims(global_emb, 1)                            # (B, 1, D)
        
        # 2. Add Type Embeddings (so transformer knows what is what)
        type_emb = self.param('type_emb', nn.initializers.normal(stddev=0.02), (3, self.d_model))
        planet_emb = planet_emb + type_emb[0]
        fleet_emb = fleet_emb + type_emb[1]
        global_emb = global_emb + type_emb[2]
        
        # 3. Concatenate into sequence
        # Sequence length = MAX_PLANETS + MAX_FLEETS + 1
        x = jnp.concatenate([global_emb, planet_emb, fleet_emb], axis=1)
        
        # 4. Self-Attention Blocks
        for i in range(self.num_layers):
            # Pre-LN
            x_norm = nn.LayerNorm()(x)
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads, 
                qkv_features=self.d_model,
                name=f"attn_{i}"
            )(x_norm, x_norm)
            x = x + attn_out
            
            # FFN
            x_norm = nn.LayerNorm()(x)
            ff_out = nn.Dense(self.d_model * 2)(x_norm)
            ff_out = nn.relu(ff_out)
            ff_out = nn.Dense(self.d_model)(ff_out)
            x = x + ff_out
            
        # 5. Pooling (Take the global token which has attended to everything)
        # global token is at index 0
        pooled = x[:, 0, :] # (B, D)
        
        # 6. Actor Head
        num_actions = MAX_PLANETS * MAX_PLANETS * 4 + 1
        actor_logits = nn.Dense(num_actions, name="actor_dense")(pooled)
        
        # 7. Critic Head
        critic_value = nn.Dense(1, name="critic_dense")(pooled)
        critic_value = jnp.squeeze(critic_value, axis=-1)
        
        return actor_logits, critic_value

