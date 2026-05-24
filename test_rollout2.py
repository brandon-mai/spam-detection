import multiprocessing as mp
import time
def test():
    from rollout_worker import worker_process
    q1 = mp.Queue()
    q2 = mp.Queue()
    p = mp.Process(target=worker_process, args=(0, q1, q2, 1))
    p.start()
    
    # Push dummy weights
    from models import EntityTransformer
    import jax
    rng = jax.random.PRNGKey(0)
    model = EntityTransformer()
    import jax.numpy as jnp
    dummy_p = jnp.zeros((1, 60, 14))
    dummy_f = jnp.zeros((1, 1000, 9))
    dummy_g = jnp.zeros((1, 4))
    variables = model.init(rng, dummy_p, dummy_f, dummy_g)
    
    q1.put(variables['params'])
    print("Pushed weights, waiting for trajectory...")
    start = time.time()
    traj = q2.get()
    print("Received trajectory with reward:", sum(traj['rewards']), "Time:", time.time() - start)
    q1.put(None)
    p.join()

if __name__ == '__main__':
    test()
