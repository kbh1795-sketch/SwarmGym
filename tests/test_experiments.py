"""Tests for configurable SwarmGym experiment runs."""

from __future__ import annotations

import json
from pathlib import Path

from swarmgym.experiments import EnvConfig, ExperimentConfig, run_experiment
from swarmgym.experiments.cli import main


def test_run_experiment_writes_results(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="tiny_random",
        policy="random",
        episodes=2,
        seed=10,
        env=EnvConfig(
            grid_size=5,
            num_agents=2,
            num_targets=1,
            num_obstacles=1,
            max_steps=8,
        ),
    )

    result = run_experiment(config, output_dir=tmp_path)

    assert len(result.episodes) == 2
    assert result.summary["episodes"] == 2.0
    assert (tmp_path / "tiny_random.episodes.csv").exists()
    assert (tmp_path / "tiny_random.summary.json").exists()
    assert (tmp_path / "tiny_random.episode_000.ansi.txt").exists()


def test_config_loads_from_json(tmp_path: Path) -> None:
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "name": "json_demo",
                "policy": "greedy_nearest_target",
                "episodes": 1,
                "seed": 3,
                "env": {
                    "grid_size": 6,
                    "num_agents": 2,
                    "num_targets": 2,
                    "num_obstacles": 2,
                    "max_steps": 10,
                },
            }
        ),
        encoding="utf-8",
    )

    config = ExperimentConfig.from_json_file(path)

    assert config.name == "json_demo"
    assert config.policy == "greedy_nearest_target"
    assert config.env.grid_size == 6


def test_cli_runs_with_overrides(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--policy",
            "random",
            "--episodes",
            "1",
            "--grid-size",
            "5",
            "--num-agents",
            "2",
            "--num-targets",
            "1",
            "--num-obstacles",
            "1",
            "--max-steps",
            "5",
            "--output-dir",
            str(tmp_path),
            "--no-render",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "swarmgym_experiment.episodes.csv").exists()
    assert not (tmp_path / "swarmgym_experiment.episode_000.ansi.txt").exists()
