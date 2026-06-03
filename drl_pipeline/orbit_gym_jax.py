import jax
import jax.numpy as jnp
import functools
from flax import struct

MAX_NODES = 50
MAX_FLEETS = 100

@struct.dataclass
class JaxStateObject:
    planet_positions: jnp.ndarray # shape: (MAX_NODES, 2)
    planet_radius: jnp.ndarray    # shape: (MAX_NODES,)
    planet_prod: jnp.ndarray      # shape: (MAX_NODES,)
    planet_owners: jnp.ndarray    # shape: (MAX_NODES,)
    planet_garrisons: jnp.ndarray # shape: (MAX_NODES,)
    fleet_data: jnp.ndarray       # shape: (MAX_FLEETS, 4) -> x, y, heading, ships
    fleet_owners: jnp.ndarray     # shape: (MAX_FLEETS,)
    step_count: jnp.ndarray       # shape: ()

class JaxOrbitWarsEnv:
    def __init__(self, max_steps=500):
        self.max_steps = max_steps

    def reset(self, rng) -> JaxStateObject:
        """
        Vectorized generation of random, 4-fold symmetric planet graph topologies from baseline seeds.
        Output is statically padded arrays.
        """
        # A simple stub initialization for 50 nodes. 
        # In a full implementation, you'd generate the 4-fold symmetric map here.
        rng, subrng = jax.random.split(rng)
        planet_positions = jax.random.uniform(subrng, (MAX_NODES, 2)) * 100.0
        planet_radius = jnp.ones((MAX_NODES,)) * 5.0
        planet_prod = jnp.ones((MAX_NODES,), dtype=jnp.int32) * 2
        planet_owners = jnp.ones((MAX_NODES,), dtype=jnp.int32) * 4 # 4=neutral
        planet_garrisons = jnp.zeros((MAX_NODES,), dtype=jnp.int32)
        
        # P1, P2, P3, P4 homes
        planet_owners = planet_owners.at[0].set(0)
        planet_owners = planet_owners.at[1].set(1)
        planet_owners = planet_owners.at[2].set(2)
        planet_owners = planet_owners.at[3].set(3)
        
        planet_garrisons = planet_garrisons.at[0:4].set(100)
        
        fleet_data = jnp.zeros((MAX_FLEETS, 4))
        fleet_owners = jnp.ones((MAX_FLEETS,), dtype=jnp.int32) * -1
        step_count = jnp.array(0, dtype=jnp.int32)

        return JaxStateObject(
            planet_positions=planet_positions,
            planet_radius=planet_radius,
            planet_prod=planet_prod,
            planet_owners=planet_owners,
            planet_garrisons=planet_garrisons,
            fleet_data=fleet_data,
            fleet_owners=fleet_owners,
            step_count=step_count
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: JaxStateObject, actions_matrix: jnp.ndarray):
        """
        Progresses the world clock. Advances comet path indices, processes step selections, 
        applies production compounding, and updates geometry matrices.
        actions_matrix shape: [3] -> [source, target, quota]
        """
        # Simplified step: increment clock, update production for owned planets.
        new_step_count = state.step_count + 1
        
        # Add production to non-neutral
        is_owned = state.planet_owners < 4
        new_garrisons = state.planet_garrisons + jnp.where(is_owned, state.planet_prod, 0)
        
        # In a real environment, you'd apply the numba physics here using jax.pure_callback 
        # or implement pure JAX version of physics.
        
        new_state = state.replace(
            step_count=new_step_count,
            planet_garrisons=new_garrisons
        )
        
        # Simple terminal condition
        done = new_step_count >= self.max_steps
        rewards = jnp.zeros(4, dtype=jnp.float32) # Stub sparse rewards
        
        return new_state, rewards, done

    @functools.partial(jax.jit, static_argnums=(0,))
    def batch_generate_graph_features(self, state: JaxStateObject):
        """
        Transforms raw continuous state parameters into structured embeddings.
        Returns Node Tensor (50x13) and Edge Tensor (50x50x4).
        """
        # Node features: 13 dims
        node_features = jnp.zeros((MAX_NODES, 13), dtype=jnp.float32)
        # Edge features: 4 dims
        edge_features = jnp.zeros((MAX_NODES, MAX_NODES, 4), dtype=jnp.float32)
        
        # Calculate distance matrix for edges
        pos = state.planet_positions
        diff = pos[:, None, :] - pos[None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1)
        edge_features = edge_features.at[:, :, 0].set(dist)
        
        return node_features, edge_features

