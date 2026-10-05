"""Command-line interface for configuration and future simulation commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import ConfigError, SimulationConfig
from .simulation import Simulation


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without causing side effects."""

    parser = argparse.ArgumentParser(
        prog="wormpimc",
        description="Continuous-space Worm PIMC reference implementation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check-config",
        help="validate and resolve a TOML file without running a simulation",
    )
    check.add_argument("path", type=Path, help="path to the TOML configuration")
    check.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit a machine-readable JSON document",
    )
    check.set_defaults(handler=_handle_check_config)

    run = subparsers.add_parser(
        "run",
        help="run a new simulation",
    )
    run.add_argument("path", type=Path, help="path to the TOML configuration")
    run.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="override output.root without changing the physical config hash",
    )
    run.add_argument(
        "--verify-local-delta",
        action="store_true",
        help="compare every local target delta with a full recomputation",
    )
    run.set_defaults(handler=_handle_run)

    resume = subparsers.add_parser(
        "resume",
        help="continue the run owning a checkpoint",
    )
    resume.add_argument("path", type=Path, help="checkpoint .npz path")
    resume.add_argument(
        "--verify-local-delta",
        action="store_true",
        help="compare every local target delta with a full recomputation",
    )
    resume.set_defaults(handler=_handle_resume)

    summarize = subparsers.add_parser(
        "summarize",
        help="read an existing run directory without modifying it",
    )
    summarize.add_argument("path", type=Path, help="run directory")
    summarize.add_argument("--json", action="store_true", dest="as_json")
    summarize.set_defaults(handler=_handle_summarize)
    return parser


def _handle_check_config(args: argparse.Namespace) -> int:
    try:
        config = SimulationConfig.from_toml(args.path)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        payload = {
            "status": "valid",
            "source": str(args.path.resolve()),
            "config_hash": config.config_hash,
            "resolved_config": config.resolved_dict(),
        }
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print("Configuration valid")
    print(f"source: {args.path.resolve()}")
    print(f"schema_version: {config.SCHEMA_VERSION}")
    print(f"tau: {config.tau:.17g}")
    print(f"config_hash: {config.config_hash}")
    print("move_probabilities:")
    for name, probability in config.moves.weights.normalized().items():
        print(f"  {name}: {probability:.17g}")
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    try:
        config = SimulationConfig.from_toml(args.path)
        simulation = Simulation(
            config, verify_local_delta=args.verify_local_delta
        )
        result = simulation.run(output_root=args.output_root)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"simulation error: {exc}", file=sys.stderr)
        return 1
    print(f"status: {result.status}")
    print(f"run_directory: {result.run_directory}")
    print(f"measurements: {result.measurement_count}")
    return 0 if result.status == "completed" else 130


def _handle_resume(args: argparse.Namespace) -> int:
    checkpoint_path = args.path.resolve()
    try:
        simulation = Simulation.from_checkpoint(
            checkpoint_path,
            verify_local_delta=args.verify_local_delta,
        )
        run_directory = checkpoint_path.parent.parent
        result = simulation.run(
            run_directory=run_directory,
            checkpoint_source=checkpoint_path,
        )
    except Exception as exc:
        print(f"resume error: {exc}", file=sys.stderr)
        return 1
    print(f"status: {result.status}")
    print(f"run_directory: {result.run_directory}")
    print(f"measurements: {result.measurement_count}")
    return 0 if result.status == "completed" else 130


def _handle_summarize(args: argparse.Namespace) -> int:
    run_directory = args.path.resolve()
    status_path = run_directory / "status.json"
    metadata_path = run_directory / "metadata.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"summary error: {exc}", file=sys.stderr)
        return 1
    payload = {
        "run_directory": str(run_directory),
        "status": status,
        "metadata": metadata,
    }
    if args.as_json:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"run_directory: {run_directory}")
        print(f"phase: {status.get('phase')}")
        print(
            "sweeps: "
            f"{status.get('current_sweep')}/{status.get('target_sweep')}"
        )
        print(f"config_hash: {metadata.get('config_hash')}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
