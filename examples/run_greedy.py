"""Run the greedy nearest-target policy baseline in CooperativeSearchEnv."""

from __future__ import annotations

from swarmgym.baselines import GreedyNearestTargetPolicy
from swarmgym.envs import CooperativeSearchEnv


def main() -> None:
    """Execute one greedy-policy episode and print metrics."""
    env = CooperativeSearchEnv(grid_size=10, num_agents=3, num_targets=5, max_steps=100)
    policy = GreedyNearestTargetPolicy()
    observations, _ = env.reset(seed=7)

    while env.agents:
        actions = policy.act(observations)
        observations, _, _, _, infos = env.step(actions)

    first_info = next(iter(infos.values()), {"metrics": env.get_metrics()})
    print(env.render(mode="ansi"))
    print(first_info["metrics"])


if __name__ == "__main__":
    main()
