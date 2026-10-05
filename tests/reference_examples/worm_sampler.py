"""Run the interacting extended-ensemble Worm PIMC reference sampler."""

# %% Imports and command-line options
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from wormpimc import Simulation, SimulationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "worm_sampler.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--verify-local-delta",
        action="store_true",
        help="enable the slow full-recomputation oracle on every proposal",
    )
    return parser


# %% Run lifecycle
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SimulationConfig.from_toml(args.config)
    simulation = Simulation(
        config,
        verify_local_delta=args.verify_local_delta,
    )
    result = simulation.run(output_root=args.output_root)

    print("Interacting full Worm PIMC reference run")
    print(f"status: {result.status}")
    print(f"run_directory: {result.run_directory}")
    print(f"diagonal_measurements: {result.measurement_count}")
    print(f"sector_visits: {simulation.sampler.sector_visits}")
    for move_name, counter in simulation.sampler.counters.items():
        print(
            f"{move_name}: evaluated={counter.action_evaluated} "
            f"accepted={counter.accepted} "
            f"acceptance={counter.acceptance_given_evaluated:.6f}"
        )
    passed = (
        result.status == "completed"
        and all(simulation.sampler.sector_visits.values())
        and simulation.sampler.counters["swap"].accepted > 0
    )
    print(f"verification: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
