"""PettingZoo ParallelEnv implementation for cooperative multi-robot search."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pettingzoo import ParallelEnv

from swarmgym.core.grid import (
    GridConfig,
    Position,
    in_bounds,
    move_position,
    positions_to_map,
    sample_unique_positions,
)
from swarmgym.core.metrics import EpisodeMetrics
from swarmgym.core.renderer import render_ansi, render_rgb_array


class CooperativeSearchEnv(ParallelEnv):
    """Cooperative 2D grid-world search environment.

    Multiple robot agents move simultaneously through a square grid, avoid
    obstacles, discover targets, and receive shared observations with local
    per-agent position information.
    """

    metadata = {
        "name": "swarmgym/cooperative_search_v0",
        "render_modes": ["ansi", "rgb_array"],
    }

    def __init__(
        self,
        grid_size: int = 10,
        num_agents: int = 3,
        num_targets: int = 5,
        num_obstacles: int = 10,
        max_steps: int = 100,
        render_mode: str | None = None,
    ) -> None:
        self.config = GridConfig(
            grid_size=grid_size,
            num_agents=num_agents,
            num_targets=num_targets,
            num_obstacles=num_obstacles,
        )
        self.config.validate()
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than 0.")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.grid_size = grid_size
        self._num_agents = num_agents
        self._num_targets = num_targets
        self._num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{index}" for index in range(self._num_agents)]
        self.agents: list[str] = []
        self.steps = 0

        self._rng = np.random.default_rng()
        self.agent_positions: dict[str, Position] = {}
        self.target_positions: list[Position] = []
        self.found_targets: set[Position] = set()
        self.obstacle_map = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.visited_map = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.metrics = EpisodeMetrics(traversable_cells=grid_size * grid_size)

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        """Return the observation space for an agent."""
        self._validate_agent(agent)
        return spaces.Dict(
            {
                "agent_position": spaces.Box(
                    low=0,
                    high=self.grid_size - 1,
                    shape=(2,),
                    dtype=np.int32,
                ),
                "target_positions": spaces.Box(
                    low=-1,
                    high=self.grid_size - 1,
                    shape=(self._num_targets, 2),
                    dtype=np.int32,
                ),
                "obstacle_map": spaces.MultiBinary((self.grid_size, self.grid_size)),
                "visited_map": spaces.MultiBinary((self.grid_size, self.grid_size)),
            }
        )

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        """Return the action space for an agent."""
        self._validate_agent(agent)
        return spaces.Discrete(5)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, NDArray[np.int32] | NDArray[np.int8]]], dict[str, dict[str, Any]]]:
        """Reset the environment and return initial observations and infos."""
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self.steps = 0
        self.found_targets = set()

        excluded: set[Position] = set()
        agent_positions = sample_unique_positions(
            self._rng, self.grid_size, self._num_agents, excluded
        )
        excluded.update(agent_positions)
        self.agent_positions = dict(zip(self.possible_agents, agent_positions, strict=True))

        self.target_positions = sample_unique_positions(
            self._rng, self.grid_size, self._num_targets, excluded
        )
        excluded.update(self.target_positions)

        obstacle_positions = sample_unique_positions(
            self._rng, self.grid_size, self._num_obstacles, excluded
        )
        self.obstacle_map = positions_to_map(obstacle_positions, self.grid_size)
        self.visited_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for position in self.agent_positions.values():
            row, col = position
            self.visited_map[row, col] = 1

        traversable_cells = self.grid_size * self.grid_size - self._num_obstacles
        self.metrics = EpisodeMetrics(
            visited_cells=int(self.visited_map.sum()),
            traversable_cells=traversable_cells,
        )

        observations = self._observations()
        infos = {agent: {"metrics": self.metrics.as_dict()} for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, dict[str, NDArray[np.int32] | NDArray[np.int8]]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Apply simultaneous actions and advance the environment by one step."""
        if not self.agents:
            return {}, {}, {}, {}, {}

        self.steps += 1
        rewards = {agent: -0.1 for agent in self.agents}
        current_positions = self.agent_positions.copy()
        proposed_positions: dict[str, Position] = {}

        for agent in self.agents:
            action = int(actions.get(agent, 0))
            proposed = move_position(current_positions[agent], action)
            if not self._is_free_cell(proposed):
                proposed_positions[agent] = current_positions[agent]
                rewards[agent] -= 5.0
                self.metrics.collision_count += 1
            else:
                proposed_positions[agent] = proposed

        colliding_agents = self._detect_agent_collisions(current_positions, proposed_positions)
        for agent in colliding_agents:
            proposed_positions[agent] = current_positions[agent]
            rewards[agent] -= 5.0
            self.metrics.collision_count += 1

        self.agent_positions = proposed_positions

        for agent, position in self.agent_positions.items():
            row, col = position
            moved_to_new_cell = (
                position != current_positions[agent] and not self.visited_map[row, col]
            )
            if moved_to_new_cell:
                rewards[agent] += 1.0
                self.visited_map[row, col] = 1

            if position in self.target_positions and position not in self.found_targets:
                rewards[agent] += 10.0
                self.found_targets.add(position)
                self.metrics.targets_found += 1

        self.metrics.visited_cells = int(self.visited_map.sum())
        self.metrics.total_reward += float(sum(rewards.values()))

        all_targets_found = len(self.found_targets) == self._num_targets
        reached_step_limit = self.steps >= self.max_steps
        terminations = {agent: all_targets_found for agent in self.agents}
        truncations = {agent: reached_step_limit and not all_targets_found for agent in self.agents}

        infos = {agent: {"metrics": self.metrics.as_dict()} for agent in self.agents}
        observations = self._observations()

        if all_targets_found or reached_step_limit:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self, mode: str | None = None) -> str | NDArray[np.uint8]:
        """Render the current environment state."""
        mode = mode or self.render_mode or "ansi"
        if mode == "ansi":
            return render_ansi(
                self.grid_size,
                self.agent_positions,
                self.target_positions,
                self.found_targets,
                self.obstacle_map,
                self.visited_map,
            )
        if mode == "rgb_array":
            return render_rgb_array(
                self.grid_size,
                self.agent_positions,
                self.target_positions,
                self.found_targets,
                self.obstacle_map,
                self.visited_map,
            )
        raise ValueError(f"Unsupported render mode: {mode}")

    def observe(self, agent: str) -> dict[str, NDArray[np.int32] | NDArray[np.int8]]:
        """Return the latest observation for a single agent."""
        self._validate_agent(agent)
        return self._observation(agent)

    def close(self) -> None:
        """Release environment resources."""

    def get_metrics(self) -> dict[str, float | int]:
        """Return current episode metrics."""
        return self.metrics.as_dict()

    def _observations(self) -> dict[str, dict[str, NDArray[np.int32] | NDArray[np.int8]]]:
        return {agent: self._observation(agent) for agent in self.agents}

    def _observation(self, agent: str) -> dict[str, NDArray[np.int32] | NDArray[np.int8]]:
        target_positions = np.full((self._num_targets, 2), -1, dtype=np.int32)
        for index, position in enumerate(self.target_positions):
            if position not in self.found_targets:
                target_positions[index] = np.array(position, dtype=np.int32)

        return {
            "agent_position": np.array(self.agent_positions[agent], dtype=np.int32),
            "target_positions": target_positions,
            "obstacle_map": self.obstacle_map.copy(),
            "visited_map": self.visited_map.copy(),
        }

    def _is_free_cell(self, position: Position) -> bool:
        if not in_bounds(position, self.grid_size):
            return False
        row, col = position
        return not bool(self.obstacle_map[row, col])

    def _detect_agent_collisions(
        self,
        current_positions: dict[str, Position],
        proposed_positions: dict[str, Position],
    ) -> set[str]:
        colliding_agents: set[str] = set()
        counts = Counter(proposed_positions.values())
        for agent, position in proposed_positions.items():
            if counts[position] > 1:
                colliding_agents.add(agent)

        agent_names = list(proposed_positions)
        for i, first in enumerate(agent_names):
            for second in agent_names[i + 1 :]:
                first_swaps = (
                    proposed_positions[first] == current_positions[second]
                    and proposed_positions[second] == current_positions[first]
                )
                if first_swaps:
                    colliding_agents.update({first, second})

        return colliding_agents

    def _validate_agent(self, agent: str) -> None:
        if agent not in self.possible_agents:
            raise KeyError(f"Unknown agent: {agent}")
