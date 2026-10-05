"""Sample and verify the exact one-particle free closed-path reference model."""

# %% Imports and command-line options
from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from wormpimc import (
    SimulationConfig,
    sample_free_closed_worldline,
    winding_distribution,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "closed_worldline.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override run.seed from the TOML configuration",
    )
    return parser


# %% Independent closed-path experiment
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 100:
        raise ValueError("--samples must be at least 100")
    config = SimulationConfig.from_toml(args.config)
    if config.potential.pair != "none" or config.potential.external != "none":
        raise ValueError("this reference experiment requires zero potential")

    seed = config.run.seed if args.seed is None else args.seed
    rng = np.random.default_rng(seed)
    winding_reference = winding_distribution(
        config.system.box_length,
        config.system.lambda_kin,
        config.system.beta,
    )
    midpoint = config.discretization.n_slices // 2
    winding_squared = np.empty(args.samples, dtype=np.float64)
    bridge_residual = np.empty(args.samples, dtype=np.float64)

    for sample_index in range(args.samples):
        state = sample_free_closed_worldline(config, rng)
        state.validate()
        unwrapped = state.unwrapped_cycle()
        winding = int(state.cycle_winding()[0])
        winding_squared[sample_index] = winding * winding
        linear_midpoint = unwrapped[0, 0] + (
            midpoint
            / config.discretization.n_slices
            * (unwrapped[-1, 0] - unwrapped[0, 0])
        )
        bridge_residual[sample_index] = (
            unwrapped[midpoint, 0] - linear_midpoint
        )

    observed_winding_square = float(np.mean(winding_squared))
    expected_winding_square = winding_reference.mean_square
    winding_variance = (
        winding_reference.fourth_moment - expected_winding_square**2
    )
    winding_standard_error = math.sqrt(winding_variance / args.samples)
    winding_z = (
        abs(observed_winding_square - expected_winding_square)
        / winding_standard_error
        if winding_standard_error > 0.0
        else 0.0
    )

    observed_bridge_variance = float(np.var(bridge_residual, ddof=1))
    expected_bridge_variance = (
        2.0
        * config.system.lambda_kin
        * config.tau
        * midpoint
        * (config.discretization.n_slices - midpoint)
        / config.discretization.n_slices
    )
    bridge_standard_error = expected_bridge_variance * math.sqrt(
        2.0 / (args.samples - 1)
    )
    bridge_z = (
        abs(observed_bridge_variance - expected_bridge_variance)
        / bridge_standard_error
    )
    passed = winding_z <= 5.0 and bridge_z <= 5.0

    print("Exact free closed-worldline experiment")
    print(f"config: {args.config.resolve()}")
    print(f"config_hash: {config.config_hash}")
    print(f"samples: {args.samples}")
    print(f"seed: {seed}")
    print(
        "winding_support: "
        f"[{winding_reference.values[0]}, {winding_reference.values[-1]}]"
    )
    print(
        "mean_winding_squared: "
        f"observed={observed_winding_square:.8f} "
        f"expected={expected_winding_square:.8f} z={winding_z:.3f}"
    )
    print(
        "midpoint_bridge_variance: "
        f"observed={observed_bridge_variance:.8f} "
        f"expected={expected_bridge_variance:.8f} z={bridge_z:.3f}"
    )
    print(f"verification: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
