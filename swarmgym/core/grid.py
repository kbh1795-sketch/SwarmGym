"""Grid utilities for cooperative search environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

Position = tuple[int, int]


@dataclass(frozen=True)
class GridConfig:
    """Configuration for a square 2D grid."""

    grid_size: int
    num_agents: int
    num_targets: int
    num_obstacles: int

    def validate(self) -> None:
        """Validate that the grid can hold all requested entities."""
        if self.grid_size <= 1:
            raise ValueError("grid_size must be greater than 1.")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be greater than 0.")
        if self.num_targets < 0:
            raise ValueError("num_targets must be non-negative.")
        if self.num_obstacles < 0:
            raise ValueError("num_obstacles must be non-negative.")

        capacity = self.grid_size * self.grid_size
        requested = self.num_agents + self.num_targets + self.num_obstacles
        if requested > capacity:
            raise ValueError(
                "grid_size is too small for the requested agents, targets, and obstacles."
            )


ACTION_TO_DELTA: dict[int, Position] = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


def move_position(position: Position, action: int) -> Position:
    """Return a new position after applying an action."""
    dr, dc = ACTION_TO_DELTA.get(int(action), ACTION_TO_DELTA[0])
    row, col = position
    return row + dr, col + dc


def in_bounds(position: Position, grid_size: int) -> bool:
    """Return whether a position lies inside the square grid."""
    row, col = position
    return 0 <= row < grid_size and 0 <= col < grid_size


def positions_to_map(positions: Iterable[Position], grid_size: int) -> NDArray[np.int8]:
    """Convert positions to a binary occupancy map."""
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    for row, col in positions:
        if in_bounds((row, col), grid_size):
            grid[row, col] = 1
    return grid


def sample_unique_positions(
    rng: np.random.Generator,
    grid_size: int,
    count: int,
    excluded: set[Position] | None = None,
) -> list[Position]:
    """Sample unique grid positions while avoiding excluded cells."""
    excluded = set() if excluded is None else set(excluded)
    all_positions = [
        (row, col)
        for row in range(grid_size)
        for col in range(grid_size)
        if (row, col) not in excluded
    ]
    if count > len(all_positions):
        raise ValueError("Not enough free cells to sample the requested positions.")

    indices = rng.choice(len(all_positions), size=count, replace=False)
    return [all_positions[int(index)] for index in indices]
