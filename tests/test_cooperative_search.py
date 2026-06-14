"""Unit tests for the CooperativeSearchEnv."""

from __future__ import annotations

import numpy as np
from pettingzoo.test import parallel_api_test

from swarmgym.envs import CooperativeSearchEnv


def test_reset_returns_valid_parallel_env_payload() -> None:
    env = CooperativeSearchEnv(
        grid_size=6,
        num_agents=2,
        num_targets=2,
        num_obstacles=3,
        max_steps=20,
    )

    observations, infos = env.reset(seed=123)

    assert set(observations) == {"agent_0", "agent_1"}
    assert set(infos) == {"agent_0", "agent_1"}
    for agent, observation in observations.items():
        assert env.observation_space(agent).contains(observation)
        assert observation["agent_position"].shape == (2,)
        assert observation["target_positions"].shape == (2, 2)
        assert observation["obstacle_map"].shape == (6, 6)
        assert observation["visited_map"].shape == (6, 6)


def test_step_tracks_metrics_and_terminates_at_max_steps() -> None:
    env = CooperativeSearchEnv(
        grid_size=5,
        num_agents=2,
        num_targets=1,
        num_obstacles=0,
        max_steps=2,
    )
    observations, _ = env.reset(seed=1)

    for _ in range(2):
        actions = {agent: 0 for agent in observations}
        observations, rewards, terminations, truncations, infos = env.step(actions)

    assert not any(terminations.values())
    assert all(truncations.values())
    assert env.agents == []
    assert isinstance(sum(rewards.values()), float)
    metrics = next(iter(infos.values()))["metrics"]
    assert set(metrics) == {
        "total_reward",
        "targets_found",
        "collision_count",
        "coverage_ratio",
    }


def test_obstacle_collision_penalty_and_count() -> None:
    env = CooperativeSearchEnv(
        grid_size=4,
        num_agents=1,
        num_targets=0,
        num_obstacles=0,
        max_steps=5,
    )
    env.reset(seed=5)
    env.agent_positions["agent_0"] = (0, 0)

    _, rewards, _, _, infos = env.step({"agent_0": 1})

    assert rewards["agent_0"] == -5.1
    assert infos["agent_0"]["metrics"]["collision_count"] == 1
    assert env.agent_positions["agent_0"] == (0, 0)


def test_render_modes() -> None:
    env = CooperativeSearchEnv(grid_size=5, num_agents=2, num_targets=1, num_obstacles=1)
    env.reset(seed=10)

    ansi = env.render(mode="ansi")
    rgb = env.render(mode="rgb_array")

    assert isinstance(ansi, str)
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (120, 120, 3)
    assert rgb.dtype == np.uint8


def test_pettingzoo_parallel_api_compliance() -> None:
    env = CooperativeSearchEnv(
        grid_size=5,
        num_agents=2,
        num_targets=1,
        num_obstacles=1,
        max_steps=5,
    )

    parallel_api_test(env, num_cycles=10)
