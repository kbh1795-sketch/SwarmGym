"""Greedy nearest-target baseline policy for SwarmGym."""

from __future__ import annotations

from typing import Any

import numpy as np


class GreedyNearestTargetPolicy:
    """Move each agent toward its nearest currently visible target.

    This simple baseline ignores other agents and obstacles. It is intended as
    a transparent reference policy, not as an optimal planner.
    """

    def act(self, observations: dict[str, dict[str, Any]]) -> dict[str, int]:
        """Return greedy actions for all active agents."""
        return {agent: self._act_one(observation) for agent, observation in observations.items()}

    def _act_one(self, observation: dict[str, Any]) -> int:
        agent_position = np.asarray(observation["agent_position"], dtype=np.int32)
        target_positions = np.asarray(observation["target_positions"], dtype=np.int32)
        visible_targets = target_positions[target_positions[:, 0] >= 0]

        if len(visible_targets) == 0:
            return 0

        distances = np.abs(visible_targets - agent_position).sum(axis=1)
        target = visible_targets[int(np.argmin(distances))]
        row_delta = int(target[0] - agent_position[0])
        col_delta = int(target[1] - agent_position[1])

        if abs(row_delta) >= abs(col_delta) and row_delta != 0:
            return 2 if row_delta > 0 else 1
        if col_delta != 0:
            return 4 if col_delta > 0 else 3
        return 0
