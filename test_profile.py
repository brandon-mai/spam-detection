import time
from kaggle_environments import make
from agents.self_play_agent import agent
import jax

env = make('orbit_wars', debug=False)
trainer = env.train([None, "agents/self_play_agent.py"])
start = time.time()
obs = trainer.reset()
reset_time = time.time() - start

steps = 0
total_infer_time = 0
while not env.done:
    s = time.time()
    # just run the agent logic
    actions = agent(obs, None)
    total_infer_time += time.time() - s
    
    s = time.time()
    obs, _, done, _ = trainer.step(actions)
    steps += 1
    if done:
        break

total_time = time.time() - start
print(f"Total time: {total_time:.2f}s")
print(f"Reset time: {reset_time:.2f}s")
print(f"Agent infer time: {total_infer_time:.2f}s")
print(f"Env step time: {total_time - reset_time - total_infer_time:.2f}s")
print(f"Steps: {steps}")
