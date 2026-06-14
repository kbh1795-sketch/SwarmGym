"""Renderers for 2D cooperative search grids."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from swarmgym.core.grid import Position


def render_ansi(
    grid_size: int,
    agent_positions: Mapping[str, Position],
    target_positions: list[Position],
    found_targets: set[Position],
    obstacle_map: NDArray[np.int8],
    visited_map: NDArray[np.int8],
) -> str:
    """Render the environment as an ANSI-friendly string."""
    cells = [["." for _ in range(grid_size)] for _ in range(grid_size)]

    for row in range(grid_size):
        for col in range(grid_size):
            if visited_map[row, col]:
                cells[row][col] = "v"
            if obstacle_map[row, col]:
                cells[row][col] = "#"

    for row, col in target_positions:
        if (row, col) not in found_targets:
            cells[row][col] = "T"

    for index, (_, (row, col)) in enumerate(agent_positions.items()):
        cells[row][col] = str(index % 10)

    return "\n".join(" ".join(row) for row in cells)


def render_rgb_array(
    grid_size: int,
    agent_positions: Mapping[str, Position],
    target_positions: list[Position],
    found_targets: set[Position],
    obstacle_map: NDArray[np.int8],
    visited_map: NDArray[np.int8],
    cell_size: int = 24,
) -> NDArray[np.uint8]:
    """Render the environment as an RGB image array."""
    image = np.full((grid_size, grid_size, 3), 245, dtype=np.uint8)
    image[visited_map.astype(bool)] = np.array([214, 239, 255], dtype=np.uint8)
    image[obstacle_map.astype(bool)] = np.array([45, 52, 54], dtype=np.uint8)

    for row, col in target_positions:
        if (row, col) not in found_targets:
            image[row, col] = np.array([46, 204, 113], dtype=np.uint8)

    agent_colors = [
        np.array([52, 152, 219], dtype=np.uint8),
        np.array([231, 76, 60], dtype=np.uint8),
        np.array([155, 89, 182], dtype=np.uint8),
        np.array([241, 196, 15], dtype=np.uint8),
        np.array([26, 188, 156], dtype=np.uint8),
    ]
    for index, (_, (row, col)) in enumerate(agent_positions.items()):
        image[row, col] = agent_colors[index % len(agent_colors)]

    upscaled = np.repeat(np.repeat(image, cell_size, axis=0), cell_size, axis=1)
    upscaled[::cell_size, :, :] = 190
    upscaled[:, ::cell_size, :] = 190
    return upscaled
