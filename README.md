## Useful info on Orbit wars that may not have been mentioned

### Kaggle procedures

```bash
# Submit
uv run kaggle competitions submit orbit-wars -f agents/main.py -m "Nearest planet sniper v1"

# Monitor submissions
uv run kaggle competitions submissions orbit-wars
```

### Color of players

- Player 0 (Blue): Plain solid chevron (no markings).
- Player 1 (Vermillion): 1 center stripe (tip to notch).
- Player 2 (Teal): 2 side stripes (running from the center towards the wings).
- Player 3 (Yellow): 3 stripes (both the center and the two side stripes).

## Rule-Based Agents

This repository focuses on heuristic, rule-based agents located in the `agents/` directory.

### Local Testing

To test a rule-based agent (e.g., `pilkwang_ppo.py`) against another agent:

```bash
uv run python -c "from kaggle_environments import make; env = make('orbit_wars'); env.run(['agents/pilkwang_ppo.py', 'random']); env.render(mode='html', out_path='replay.html')"
```

Open `replay.html` in your browser to watch the match.

### Kaggle Submission

You can submit any of the rule-based agents from the `agents/` directory. For example, to submit `pilkwang_ppo.py`:

```bash
uv run kaggle competitions submit orbit-wars -f agents/pilkwang_ppo.py -m "Pilkwang PPO Rule-Based Agent v1"
```

