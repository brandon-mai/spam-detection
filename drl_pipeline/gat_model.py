import jax
import jax.numpy as jnp
import flax.linen as nn

class GATLayer(nn.Module):
    features: int
    num_heads: int

    @nn.compact
    def __call__(self, h, edge_features):
        # h: (B, N, F_in)
        # edge_features: (B, N, N, E_in)
        B, N, _ = h.shape
        
        # Project node features
        h_proj = nn.Dense(self.features * self.num_heads)(h)
        h_proj = h_proj.reshape((B, N, self.num_heads, self.features))
        
        # Project edge features
        e_proj = nn.Dense(self.features * self.num_heads)(edge_features)
        e_proj = e_proj.reshape((B, N, N, self.num_heads, self.features))
        
        # Attention mechanism
        # For simplicity, dot product attention using both nodes and edges
        q = h_proj
        k = h_proj
        # (B, N_q, num_heads, features) @ (B, N_k, num_heads, features) -> (B, num_heads, N_q, N_k)
        attn_scores = jnp.einsum('bqhd,bkhd->bhqk', q, k)
        
        # Add edge features to attention
        # A simple way is to project edge features to scalar scores
        e_scores = nn.Dense(1)(e_proj).squeeze(-1) # (B, N, N, num_heads)
        e_scores = jnp.transpose(e_scores, (0, 3, 1, 2)) # (B, num_heads, N, N)
        
        attn_scores = (attn_scores + e_scores) / jnp.sqrt(self.features)
        
        attn_weights = nn.softmax(attn_scores, axis=-1)
        
        # Aggregate (B, num_heads, N, N) @ (B, N, num_heads, features) -> (B, N, num_heads, features)
        v = h_proj
        out = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        
        out = out.reshape((B, N, self.features * self.num_heads))
        return nn.leaky_relu(out)

class OrbitGATModel(nn.Module):
    d_model: int = 128
    num_heads: int = 4
    max_nodes: int = 50

    @nn.compact
    def __call__(self, node_features, edge_features, game_mode_flag):
        # node_features: (B, N, 13)
        # edge_features: (B, N, N, 4)
        # game_mode_flag: (B, 1) -> 0 for 2P, 1 for 4P
        
        B, N, _ = node_features.shape
        
        # Input projection
        h = nn.Dense(self.d_model)(node_features)
        e = nn.Dense(self.d_model)(edge_features)
        
        # 3-Layer GAT
        for _ in range(3):
            h = GATLayer(features=self.d_model // self.num_heads, num_heads=self.num_heads)(h, e)
            
        # h is now X matrix: (B, N, 128)
        X = h
        
        # Context Vector C: Global Average Pooling (B, 128)
        C = jnp.mean(X, axis=1)
        
        # Append game mode flag (B, 129)
        C = jnp.concatenate([C, game_mode_flag], axis=-1)
        
        # --- Decoupled Decoders ---
        
        # Head A: Source Selector (S)
        # Pointer attention: logits_S = Linear(X) @ C^T
        X_S_proj = nn.Dense(C.shape[-1])(X) # (B, N, 129)
        # (B, N, 129) @ (B, 129, 1) -> (B, N)
        logits_S = jnp.einsum('bnd,bd->bn', X_S_proj, C)
        
        # We don't apply mask here inside the model, it can be applied externally or passed as an argument.
        # But for architecture completeness, we return raw logits.
        
        # Assume for generation we take argmax, but during training we output all targets.
        # To compute Head B and C, we need the chosen Source embedding.
        # We'll use a dummy argmax for structural testing if source_idx isn't provided.
        source_idx = jnp.argmax(logits_S, axis=-1) # (B,)
        
        # Gather X_S: (B, 128)
        batch_indices = jnp.arange(B)
        X_S = X[batch_indices, source_idx, :]
        
        # Head B: Target Focus (T)
        # Conditioned on Source embedding
        C_T = jnp.concatenate([C, X_S], axis=-1) # (B, 129 + 128)
        X_T_proj = nn.Dense(C_T.shape[-1])(X) # (B, N, 257)
        logits_T_nodes = jnp.einsum('bnd,bd->bn', X_T_proj, C_T) # (B, N)
        
        # Learnable NO_OP token
        no_op_token = self.param('no_op_token', nn.initializers.zeros, (1, C_T.shape[-1]))
        # Broadcast NO_OP token to batch
        no_op_token_batch = jnp.tile(no_op_token, (B, 1)) # (B, 257)
        logits_T_noop = jnp.einsum('bd,bd->b', no_op_token_batch, C_T) # (B,)
        logits_T_noop = logits_T_noop[:, None] # (B, 1)
        
        logits_T = jnp.concatenate([logits_T_nodes, logits_T_noop], axis=-1) # (B, N+1)
        
        # Head C: Allocation Quota (Q)
        target_idx = jnp.argmax(logits_T, axis=-1) # (B,)
        
        # If target is NO_OP, X_T is zeros or a specific embedding.
        # For simplicity, if target == N, use zeros.
        X_T = jnp.where(
            (target_idx == N)[:, None], 
            jnp.zeros_like(X[:, 0, :]), 
            X[batch_indices, jnp.clip(target_idx, 0, N-1), :]
        )
        
        # MLP([C || X_S || X_T]) -> (B, 3)
        C_Q = jnp.concatenate([C, X_S, X_T], axis=-1)
        q_hidden = nn.Dense(128)(C_Q)
        q_hidden = nn.relu(q_hidden)
        logits_Q = nn.Dense(3)(q_hidden)
        
        return logits_S, logits_T, logits_Q
