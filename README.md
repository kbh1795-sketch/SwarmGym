# SwarmGym

SwarmGym is a Python toolkit for multi-agent reinforcement learning environments focused on cooperative robot-swarm exploration. The initial release provides a PettingZoo-compatible 2D grid-world environment where multiple robot agents search for targets while avoiding obstacles and agent-agent collisions.

The project is intentionally environment-first: it does not include deep learning training code yet. The goal is to provide a stable, inspectable benchmark surface that researchers and engineers can plug into their own MARL stacks.

## Features

- PettingZoo `ParallelEnv` API via `CooperativeSearchEnv`
- Gymnasium observation and action spaces
- Configurable `grid_size`, `num_agents`, `num_targets`, `num_obstacles`, and `max_steps`
- Five discrete actions: stay, up, down, left, right
- Observations containing agent position, target positions, obstacle map, and visited map
- Reward signals for target discovery, coverage, movement cost, obstacle collisions, and agent collisions
- ANSI and RGB-array rendering
- Random and greedy nearest-target baseline policies
- Episode metrics: `total_reward`, `targets_found`, `collision_count`, and `coverage_ratio`
- Configurable experiment runner for repeated simulations and CSV/JSON results
- Browser-based SwarmGym Studio for interactive 2D simulation control

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/kbh1795-sketch/SwarmGym.git
cd swarmgym
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest
```

## Quick Start

```python
from swarmgym import CooperativeSearchEnv

env = CooperativeSearchEnv(
    grid_size=10,
    num_agents=3,
    num_targets=5,
    num_obstacles=10,
    max_steps=100,
)

observations, infos = env.reset(seed=42)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

print(env.render(mode="ansi"))
print(env.get_metrics())
```

## SwarmGym Studio

SwarmGym Studio is an interactive 2D simulator for quickly exploring scenarios in the browser. It is not a full physics simulator like Gazebo, but it gives you a Gazebo-like control surface for this grid-world domain: live world rendering, adjustable environment variables, policy selection, step/run controls, metrics, agent table, event log, and CSV export.

Open:

```text
studio/index.html
```

Controls available in the Studio:

- World variables: `grid_size`, `num_agents`, `num_targets`, `num_obstacles`, `max_steps`
- Experiment variables: `policy`, `seed`, simulation speed
- Policies: `random`, `greedy_nearest_target`, `frontier`, and custom JavaScript policy
- Execution: reset, single step, continuous run, full episode
- Output: total reward, targets found, collision count, coverage, reward/coverage chart, CSV export

## Environment

`CooperativeSearchEnv` implements the PettingZoo ParallelEnv interface:

```python
observations, infos = env.reset(seed=0)
observations, rewards, terminations, truncations, infos = env.step(actions)
```

Each agent has a discrete action space:

| Action | Meaning |
| --- | --- |
| `0` | stay |
| `1` | up |
| `2` | down |
| `3` | left |
| `4` | right |

Each observation is a Gymnasium `Dict` space with:

- `agent_position`: the observing agent's `(row, col)` location
- `target_positions`: target coordinates, with found targets marked as `[-1, -1]`
- `obstacle_map`: binary grid marking blocked cells
- `visited_map`: binary grid marking cells visited by any agent

Rewards:

| Event | Reward |
| --- | ---: |
| Target found | `+10` |
| New cell visited | `+1` |
| Movement cost | `-0.1` |
| Obstacle or boundary collision | `-5` |
| Agent-agent collision | `-5` |

## Baseline Examples

Run the random policy:

```bash
python examples/run_random.py
```

Run the greedy nearest-target policy:

```bash
python examples/run_greedy.py
```

The greedy baseline moves each agent toward the nearest visible target using Manhattan distance. It deliberately ignores obstacles and other agents, making it a simple reference rather than a planner.

## Configurable Experiments

SwarmGym includes a small experiment runner so you can change environment variables, choose a policy, run multiple episodes, and save results without writing a training loop from scratch.

Run from a JSON config:

```bash
swarmgym-run --config examples/experiment_config.json --output-dir runs/random_demo
```

Or override values directly from the command line:

```bash
swarmgym-run \
  --policy greedy_nearest_target \
  --episodes 20 \
  --seed 123 \
  --grid-size 12 \
  --num-agents 4 \
  --num-targets 8 \
  --num-obstacles 18 \
  --max-steps 150 \
  --output-dir runs/greedy_grid12
```

The runner writes:

- `<name>.episodes.csv`: one row per episode
- `<name>.summary.json`: config, aggregate summary, and episode metrics
- `<name>.episode_000.ansi.txt`: final grid render for each episode, unless `--no-render` is used

Example config:

```json
{
  "name": "cooperative_search_random_demo",
  "policy": "random",
  "episodes": 5,
  "seed": 42,
  "save_final_render": true,
  "env": {
    "grid_size": 10,
    "num_agents": 3,
    "num_targets": 5,
    "num_obstacles": 10,
    "max_steps": 100
  }
}
```

Available policies:

- `random`
- `greedy_nearest_target`

Custom learning policies can be added by implementing an object with an `act(observations) -> dict[str, int]` method and registering it in `swarmgym.experiments.policies`.

Typical experiment knobs:

- Environment size: `grid_size`
- Swarm size: `num_agents`
- Task difficulty: `num_targets`, `num_obstacles`, `max_steps`
- Algorithm or policy: `policy`
- Statistical repeatability: `episodes`, `seed`

This makes it straightforward to compare questions such as "Does a larger swarm improve coverage?", "How does obstacle density affect collisions?", or "Which policy finds more targets under the same scenario?"

## Metrics

Episode metrics are available from `env.get_metrics()` and are also included under `infos[agent]["metrics"]`:

- `total_reward`: sum of rewards across all agents and steps
- `targets_found`: number of targets discovered
- `collision_count`: obstacle, boundary, and agent collision events
- `coverage_ratio`: visited traversable cells divided by total non-obstacle cells

## Project Goals

SwarmGym aims to become a practical benchmark suite for cooperative search, coverage, and coordination in robot swarms. The first release focuses on clarity and correctness in a small 2D grid world. Future releases will expand environment variety while preserving simple installation and clean MARL interoperability.

## Roadmap

- Add partial-observability and communication variants
- Add obstacle-aware heuristic baselines
- Add curriculum and scenario generation utilities
- Add wrappers for common MARL training libraries
- Add visualization tools for trajectories and coverage
- Add larger benchmark suites for search, rescue, and persistent monitoring

## License

SwarmGym is released under the MIT License. See [LICENSE](LICENSE) for details.
