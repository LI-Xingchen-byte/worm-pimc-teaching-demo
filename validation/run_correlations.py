"""Bounded exploratory g2/S(k) run using the optional public collector.

Run from repository root: python -m validation.run_correlations --output ...
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wormpimc import Simulation, SimulationConfig, __version__  # noqa: E402
from wormpimc.estimators import EqualTimeCorrelations  # noqa: E402
from .correlation_reference import ideal_correlations  # noqa: E402


def run(args):
    root = Path(args.output)
    config = SimulationConfig.small_system(
        n_slices=8,
        chemical_potential=-0.7,
        seed=args.seed,
        warmup_sweeps=args.warmup,
        measurement_sweeps=args.sweeps,
        steps_per_sweep=8,
    )
    config = replace(
        config, moves=replace(config.moves, worm_sector_weight=0.5, max_segment_links=3)
    )
    root.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(
            [*Path("src/wormpimc").rglob("*.py"), *Path("validation").glob("*.py")]
        )
    }
    (root / "config.json").write_text(
        json.dumps(config.resolved_dict(), indent=2), encoding="utf-8"
    )
    simulation = Simulation(config)
    correlations = EqualTimeCorrelations(config, spatial_bins=8, modes=(1, 2, 3))
    start = time.perf_counter()

    def observe(sim):
        if (
            sim.phase == "measurement"
            and sim.measurement_completed % config.run.measurement_stride == 0
        ):
            correlations.measure(sim.state)
        if sim.completed_sweeps % 2048 == 0:
            print(
                f"seed={args.seed} sweep={sim.completed_sweeps} elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )
        if time.perf_counter() - start > args.timeout:
            raise TimeoutError("exploratory run wall-clock budget exhausted")

    status = "COMPLETED"
    try:
        simulation.advance(after_sweep=observe)
    except TimeoutError:
        status = "TIMEOUT"
    elapsed = time.perf_counter() - start
    samples = correlations.samples
    np.save(root / "samples.npy", samples)
    (root / "correlations.json").write_text(
        json.dumps(correlations.snapshot(), allow_nan=False), encoding="utf-8"
    )
    simulation.checkpoint(root / "simulation.npz")
    result = correlations.results()
    reference = ideal_correlations()
    for name in ("g2", "structure_factor"):
        for row, exact in zip(result[name], reference[name]):
            row["ideal_reference"] = exact
            se = row["standard_error"]
            row["reference_residual_SE"] = (
                (row["mean"] - exact) / se if se and row["mean"] is not None else None
            )
    report = {
        "status": status,
        "purpose": "exploratory_trend_and_scale",
        "seed": args.seed,
        "config_hash": config.config_hash,
        "package_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_seconds": elapsed,
        "attempts_per_second": simulation.sampler.atomic_steps / elapsed,
        "correlations": result,
        "ideal_reference": reference,
        "samples_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
        "state_digest": simulation.state.state_digest(),
        "rng_state": simulation.rng.bit_generator.state,
        "summary": simulation.summary(),
        "moves": simulation.sampler.counters_snapshot(),
        "source_sha256": source_hashes,
    }
    (root / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(root),
                "status": status,
                "seconds": elapsed,
                "mean_N": result["mean_N"],
                "g2": [(row["mean"], row["standard_error"]) for row in result["g2"]],
                "S": [
                    (row["mean"], row["standard_error"])
                    for row in result["structure_factor"]
                ],
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--warmup", type=int, default=2048)
    parser.add_argument("--sweeps", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    run(args)


if __name__ == "__main__":
    main()
