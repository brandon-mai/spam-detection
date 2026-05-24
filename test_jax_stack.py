import jax.numpy as jnp
import numpy as np
import time

lists = [np.random.rand(60, 14) for _ in range(1000)]
start = time.time()
res1 = np.stack(lists)
print("np.stack:", time.time() - start)

start = time.time()
res2 = jnp.stack(lists)
print("jnp.stack:", time.time() - start)
