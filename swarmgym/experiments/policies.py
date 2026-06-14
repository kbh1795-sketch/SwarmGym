"""Policy registry used by experiment runners."""

from __future__ import annotations

from typing import Any, Protocol

from swarmgym.baselines import GreedyNearestTargetPolicy, RandomPolicy


class Policy(Protocol):
    """Minimal interface expected by the experiment runner."""

    def act(self, observations: dict[str, dict[str, Any]]) -> dict[str, int]:
        """Return one action per active agent."""


def make_policy(name: str, seed: int | None = None) -> Policy:
    """Create a policy by registry name."""
    normalized = name.strip().lower()
    if normalized == "random":
        return RandomPolicy(seed=seed)
    if normalized in {"greedy", "greedy_nearest_target", "nearest_target"}:
        return GreedyNearestTargetPolicy()
    raise ValueError(
        f"Unknown policy '{name}'. Available policies: random, greedy_nearest_target."
    )
