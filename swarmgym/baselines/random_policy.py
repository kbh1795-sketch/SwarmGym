"""Random baseline policy for SwarmGym environments."""

from __future__ import annotations

from typing import Any

import numpy as np


class RandomPolicy:
    """Sample uniformly random actions for all active agents."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observations: dict[str, dict[str, Any]]) -> dict[str, int]:
        """Return one random action per observed agent."""
        return {agent: int(self.rng.integers(0, 5)) for agent in observations}
