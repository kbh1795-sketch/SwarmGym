"""Episode metrics for SwarmGym environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    """Mutable metrics tracked during one environment episode."""

    total_reward: float = 0.0
    targets_found: int = 0
    collision_count: int = 0
    visited_cells: int = 0
    traversable_cells: int = 1

    @property
    def coverage_ratio(self) -> float:
        """Return the fraction of non-obstacle cells visited by any agent."""
        if self.traversable_cells <= 0:
            return 0.0
        return self.visited_cells / self.traversable_cells

    def as_dict(self) -> dict[str, float | int]:
        """Return metrics in a stable dictionary format."""
        return {
            "total_reward": float(self.total_reward),
            "targets_found": int(self.targets_found),
            "collision_count": int(self.collision_count),
            "coverage_ratio": float(self.coverage_ratio),
        }
