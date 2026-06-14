"""Run configurable SwarmGym experiments and save results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import csv
import json

from swarmgym.envs import CooperativeSearchEnv
from swarmgym.experiments.config import ExperimentConfig
from swarmgym.experiments.policies import make_policy


@dataclass(frozen=True)
class EpisodeResult:
    """Metrics and metadata from one episode."""

    episode: int
    seed: int
    policy: str
    steps: int
    total_reward: float
    targets_found: int
    collision_count: int
    coverage_ratio: float


@dataclass(frozen=True)
class ExperimentResult:
    """Complete output from an experiment run."""

    config: ExperimentConfig
    episodes: list[EpisodeResult]
    summary: dict[str, float]
    output_dir: Path | None = None


def run_experiment(
    config: ExperimentConfig,
    output_dir: str | Path | None = None,
) -> ExperimentResult:
    """Run a configured experiment and optionally write result files."""
    episode_results: list[EpisodeResult] = []

    for episode in range(config.episodes):
        episode_seed = config.seed + episode
        policy = make_policy(config.policy, seed=episode_seed)
        env = CooperativeSearchEnv(**config.env.as_kwargs())
        observations, _ = env.reset(seed=episode_seed)

        while env.agents:
            actions = policy.act(observations)
            observations, _, _, _, _ = env.step(actions)

        metrics = env.get_metrics()
        episode_results.append(
            EpisodeResult(
                episode=episode,
                seed=episode_seed,
                policy=config.policy,
                steps=env.steps,
                total_reward=float(metrics["total_reward"]),
                targets_found=int(metrics["targets_found"]),
                collision_count=int(metrics["collision_count"]),
                coverage_ratio=float(metrics["coverage_ratio"]),
            )
        )

        if output_dir is not None and config.save_final_render:
            _write_final_render(Path(output_dir), config.name, episode, env)

    summary = summarize_episodes(episode_results)
    result = ExperimentResult(
        config=config,
        episodes=episode_results,
        summary=summary,
        output_dir=Path(output_dir) if output_dir is not None else None,
    )

    if output_dir is not None:
        write_result_files(result, Path(output_dir))

    return result


def summarize_episodes(episodes: list[EpisodeResult]) -> dict[str, float]:
    """Aggregate episode results into means and totals."""
    if not episodes:
        return {}

    count = len(episodes)
    return {
        "episodes": float(count),
        "mean_total_reward": sum(item.total_reward for item in episodes) / count,
        "mean_targets_found": sum(item.targets_found for item in episodes) / count,
        "mean_collision_count": sum(item.collision_count for item in episodes) / count,
        "mean_coverage_ratio": sum(item.coverage_ratio for item in episodes) / count,
        "max_coverage_ratio": max(item.coverage_ratio for item in episodes),
        "min_coverage_ratio": min(item.coverage_ratio for item in episodes),
    }


def write_result_files(result: ExperimentResult, output_dir: Path) -> None:
    """Write experiment config, episode table, and summary to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_name(result.config.name)
    base = output_dir / safe_name

    with (base.with_suffix(".episodes.csv")).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(result.episodes[0]).keys()))
        writer.writeheader()
        for episode in result.episodes:
            writer.writerow(asdict(episode))

    payload: dict[str, Any] = {
        "config": {
            **asdict(result.config),
            "env": asdict(result.config.env),
        },
        "summary": result.summary,
        "episodes": [asdict(episode) for episode in result.episodes],
    }
    with (base.with_suffix(".summary.json")).open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _write_final_render(
    output_dir: Path,
    experiment_name: str,
    episode: int,
    env: CooperativeSearchEnv,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_name(experiment_name)
    path = output_dir / f"{safe_name}.episode_{episode:03d}.ansi.txt"
    path.write_text(str(env.render(mode="ansi")), encoding="utf-8")


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
