from kaggle_environments import make
env = make('orbit_wars', configuration={"episodeSteps": 10})
trainer = env.train([None, "random"])
obs = trainer.reset()
steps = 0
done = False
while not done:
    obs, reward, done, info = trainer.step([])
    steps += 1
print("Done in steps:", steps)
