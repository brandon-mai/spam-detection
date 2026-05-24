import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

# Constants matching orbit_wars.py
BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0

class EnvState(NamedTuple):
    planets: jnp.ndarray        # (MAX_PLANETS, 7): [owner, x, y, radius, ships, production, init_r]
    fleets: jnp.ndarray         # (MAX_FLEETS, 7): [owner, x, y, target, ships, dx, dy]
    step_num: jnp.ndarray       # int32 scalar
    angular_velocity: jnp.ndarray # float32 scalar

def swept_pair_hit(A, B, P0, P1, r):
    d0 = A - P0
    dv = (B - A) - (P1 - P0)
    a = jnp.dot(dv, dv)
    b = 2.0 * jnp.dot(d0, dv)
    c = jnp.dot(d0, d0) - r * r
    
    cond_a_small = a < 1e-12
    hit_small = c <= 0.0
    
    disc = b * b - 4.0 * a * c
    cond_disc = disc >= 0.0
    
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    # Add epsilon to prevent division by zero NaN propagation
    a_safe = jnp.where(cond_a_small, 1.0, a)
    t1 = (-b - sq) / (2.0 * a_safe)
    t2 = (-b + sq) / (2.0 * a_safe)
    
    hit_normal = (t2 >= 0.0) & (t1 <= 1.0)
    return jnp.where(cond_a_small, hit_small, jnp.where(cond_disc, hit_normal, False))

@jax.jit
def step(state: EnvState, actions: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    """
    actions: (MAX_PLANETS, 3) -> [valid_mask, angle, ships]
    """
    pass
