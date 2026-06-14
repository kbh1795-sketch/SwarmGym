"""Command line interface for running SwarmGym experiments."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

import json

from swarmgym.experiments.config import ExperimentConfig
from swarmgym.experiments.runner import run_experiment


def build_parser() -> ArgumentParser:
    """Create the experiment CLI argument parser."""
    parser = ArgumentParser(description="Run configurable SwarmGym experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON experiment config file.",
    )
    parser.add_argument("--policy", choices=["random", "greedy_nearest_target"], help="Policy name.")
    parser.add_argument("--episodes", type=int, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, help="Base random seed.")
    parser.add_argument("--grid-size", type=int, help="Grid width and height.")
    parser.add_argument("--num-agents", type=int, help="Number of robot agents.")
    parser.add_argument("--num-targets", type=int, help="Number of targets.")
    parser.add_argument("--num-obstacles", type=int, help="Number of obstacles.")
    parser.add_argument("--max-steps", type=int, help="Maximum steps per episode.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where CSV, JSON, and render files are written.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Do not save final ANSI render files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit status code."""
    args = build_parser().parse_args(argv)
    config = _load_config(args)
    result = run_experiment(config, output_dir=args.output_dir)

    print(json.dumps(result.summary, indent=2))
    print(f"Results written to: {args.output_dir.resolve()}")
    return 0


def _load_config(args: Namespace) -> ExperimentConfig:
    if args.config is not None:
        data = ExperimentConfig.from_json_file(args.config)
    else:
        data = ExperimentConfig()

    raw = {
        "name": data.name,
        "policy": args.policy or data.policy,
        "episodes": args.episodes if args.episodes is not None else data.episodes,
        "seed": args.seed if args.seed is not None else data.seed,
        "save_final_render": not args.no_render and data.save_final_render,
        "env": {
            "grid_size": args.grid_size if args.grid_size is not None else data.env.grid_size,
            "num_agents": args.num_agents if args.num_agents is not None else data.env.num_agents,
            "num_targets": args.num_targets if args.num_targets is not None else data.env.num_targets,
            "num_obstacles": (
                args.num_obstacles if args.num_obstacles is not None else data.env.num_obstacles
            ),
            "max_steps": args.max_steps if args.max_steps is not None else data.env.max_steps,
        },
    }
    return ExperimentConfig.from_mapping(raw)


if __name__ == "__main__":
    raise SystemExit(main())
