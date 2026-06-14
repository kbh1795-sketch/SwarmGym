"""Configuration models for SwarmGym experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for a CooperativeSearchEnv instance."""

    grid_size: int = 10
    num_agents: int = 3
    num_targets: int = 5
    num_obstacles: int = 10
    max_steps: int = 100

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EnvConfig":
        """Build environment config from a dictionary."""
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown environment config keys: {sorted(unknown)}")
        return cls(**data)

    def as_kwargs(self) -> dict[str, int]:
        """Return keyword arguments accepted by CooperativeSearchEnv."""
        return {
            "grid_size": self.grid_size,
            "num_agents": self.num_agents,
            "num_targets": self.num_targets,
            "num_obstacles": self.num_obstacles,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration for repeated policy evaluation."""

    name: str = "swarmgym_experiment"
    policy: str = "random"
    episodes: int = 10
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    save_final_render: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Build experiment config from a dictionary."""
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown experiment config keys: {sorted(unknown)}")

        env_data = data.get("env", {})
        if not isinstance(env_data, dict):
            raise ValueError("env must be a dictionary.")

        values = dict(data)
        values["env"] = EnvConfig.from_mapping(env_data)
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ExperimentConfig":
        """Load experiment config from a JSON file."""
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("Experiment config JSON must contain an object.")
        return cls.from_mapping(data)

    def validate(self) -> None:
        """Validate experiment-level settings."""
        if not self.name:
            raise ValueError("Experiment name must not be empty.")
        if self.episodes <= 0:
            raise ValueError("episodes must be greater than 0.")
        if not self.policy:
            raise ValueError("policy must not be empty.")
