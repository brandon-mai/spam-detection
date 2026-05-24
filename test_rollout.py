import multiprocessing as mp
import time

def test():
    from rollout_worker import worker_process
    q1 = mp.Queue()
    q2 = mp.Queue()
    print("Starting worker process...")
    p = mp.Process(target=worker_process, args=(0, q1, q2, 1))
    p.start()
    time.sleep(2)
    p.terminate()
    print("Terminated.")

if __name__ == '__main__':
    test()
