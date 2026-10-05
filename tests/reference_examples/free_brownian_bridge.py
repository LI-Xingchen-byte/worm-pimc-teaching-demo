"""Verify one marginal of the free Brownian-bridge sampler."""

# %% Imports and command-line options
from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import numpy as np

from wormpimc import brownian_bridge_moments, sample_brownian_bridge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=13579)
    parser.add_argument("--links", type=int, default=16)
    return parser


# %% Experiment
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 100:
        raise ValueError("--samples must be at least 100")
    if args.links < 2:
        raise ValueError("--links must be at least 2")

    lambda_kin = 0.5
    tau = 0.125
    start = np.array([0.25])
    end = np.array([1.25])
    midpoint = args.links // 2
    rng = np.random.default_rng(args.seed)

    observations = np.empty(args.samples, dtype=np.float64)
    for sample_index in range(args.samples):
        path = sample_brownian_bridge(
            start,
            end,
            args.links,
            lambda_kin,
            tau,
            rng,
        )
        observations[sample_index] = path[midpoint, 0]

    means, variances = brownian_bridge_moments(
        start,
        end,
        args.links,
        lambda_kin,
        tau,
    )
    expected_mean = means[midpoint, 0]
    expected_variance = variances[midpoint]
    observed_mean = float(np.mean(observations))
    observed_variance = float(np.var(observations, ddof=1))
    mean_standard_error = math.sqrt(expected_variance / args.samples)
    variance_standard_error = expected_variance * math.sqrt(
        2.0 / (args.samples - 1)
    )
    mean_z = abs(observed_mean - expected_mean) / mean_standard_error
    variance_z = (
        abs(observed_variance - expected_variance) / variance_standard_error
    )
    passed = mean_z <= 5.0 and variance_z <= 5.0

    print("Free Brownian-bridge marginal check")
    print(f"samples: {args.samples}")
    print(f"seed: {args.seed}")
    print(f"midpoint_mean: observed={observed_mean:.8f} expected={expected_mean:.8f}")
    print(
        "midpoint_variance: "
        f"observed={observed_variance:.8f} expected={expected_variance:.8f}"
    )
    print(f"z_scores: mean={mean_z:.3f} variance={variance_z:.3f}")
    print(f"verification: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
