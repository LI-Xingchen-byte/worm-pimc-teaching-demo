"""Bounded, reproducible full-chain P1 experiment (run from repository root).

python -m validation.run_p1 --output runs/validation/ideal/pilot --pilot
python -m validation.run_p1 --output runs/validation/ideal/base_2718 --seed 2718
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wormpimc import Simulation, SimulationConfig  # noqa: E402
from wormpimc.estimators.energy import thermodynamic_energy  # noqa: E402
from wormpimc.statistics import BlockingRatioAccumulator  # noqa: E402
from .ideal_reference import ideal_bose_reference  # noqa: E402


NAMES = ("N", "N2", "E", "exchange_particles", "P0", "P1", "P2", "P3", "P4", "Pge5")


def analyze(series: np.ndarray, reference: dict) -> dict:
    """All-opportunity ratios, including G zeros and rejected-state repeats."""
    rows = {}
    for column, name in enumerate(NAMES, 1):
        accumulator = BlockingRatioAccumulator()
        for row in series:
            accumulator.add(float(row[column]), float(row[0]))
        levels = accumulator.estimates()
        if not levels:
            rows[name] = {"status": "INCONCLUSIVE", "reason": "no Z measurements"}
            continue
        summary = accumulator.summary(
            independent_count=len(series),
            independent_standard_error=levels[0].standard_error,
        )
        estimate = levels[0].estimate
        se = summary.standard_error
        z_score = (estimate - reference[name]) / se if se and se > 0 else None
        # Constant/sparse event streams do not demonstrate convergence.
        enough_events = (
            np.count_nonzero(series[:, column]) >= 32
            and np.count_nonzero(series[:, column] - estimate * series[:, 0]) >= 32
        )
        rows[name] = {
            "estimate": estimate,
            "reference": reference[name],
            "blocking": asdict(summary),
            "z_score": z_score,
            "status": (
                "INCONCLUSIVE"
                if z_score is None or not enough_events
                else "CONSISTENT" if abs(z_score) <= 4 else "DISCREPANCY"
            ),
            "levels": [asdict(level) for level in levels],
        }
    return rows


def run(args: argparse.Namespace) -> dict:
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    config = SimulationConfig.small_system(
        chemical_potential=-0.7,
        n_slices=8,
        seed=args.seed,
        warmup_sweeps=512 if args.pilot else args.warmup,
        measurement_sweeps=2048 if args.pilot else args.sweeps,
        steps_per_sweep=8,
    )
    weights = config.moves.weights
    if args.asymmetric:
        weights = replace(weights, open=2.0, remove=2.0, advance=2.0)
    config = replace(
        config,
        moves=replace(
            config.moves,
            max_segment_links=3,
            worm_sector_weight=args.c0,
            weights=weights,
        ),
    )
    (directory / "config.json").write_text(
        json.dumps(config.resolved_dict(), indent=2), encoding="utf-8"
    )
    reference = ideal_bose_reference()
    simulation = Simulation(config)
    data = []
    started = time.perf_counter()
    timed_out = False

    def observe(sim):
        if sim.phase == "measurement":
            state = sim.state
            if state.sector.value == "Z":
                n = state.number_of_beads / state.n_slices
                # Independent graph readout; cycles of l*M beads contain l particles.
                exchange = sum(
                    len(state.cycle_ids(start)) / state.n_slices
                    for start in state.cycle_starts()
                    if len(state.cycle_ids(start)) > state.n_slices
                )
                energy = thermodynamic_energy(state, sim.target).total
                data.append(
                    [
                        1,
                        n,
                        n * n,
                        energy,
                        exchange,
                        *(int(n == i) for i in range(5)),
                        int(n >= 5),
                    ]
                )
            else:
                data.append([0] * (len(NAMES) + 1))
        if sim.completed_sweeps % 2048 == 0:
            print(
                f"seed={args.seed} C0={args.c0} sweep={sim.completed_sweeps} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
        if time.perf_counter() - started > args.timeout:
            raise TimeoutError("predeclared per-run wall-clock budget exhausted")

    try:
        simulation.advance(after_sweep=observe)
    except TimeoutError:
        timed_out = True
    elapsed = time.perf_counter() - started
    series = np.asarray(data, dtype=np.float64).reshape(-1, len(NAMES) + 1)
    np.save(directory / "series.npy", series)
    simulation.checkpoint(directory / "final.npz")
    result = {
        "status": "TIMEOUT" if timed_out else "COMPLETED",
        "role": "pilot" if args.pilot else "confirmation",
        "seed": args.seed,
        "c0": args.c0,
        "asymmetric": args.asymmetric,
        "config_hash": config.config_hash,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_seconds": elapsed,
        "attempts_per_second": simulation.sampler.atomic_steps / elapsed,
        "columns": ["Z", *NAMES],
        "opportunities": len(series),
        "Z_fraction": float(series[:, 0].mean()) if len(series) else None,
        "reference": reference,
        "comparisons": analyze(series, reference["values"]),
        "summary": simulation.summary(),
        "moves": simulation.sampler.counters_snapshot(),
        "series_sha256": hashlib.sha256(series.tobytes()).hexdigest(),
        "state_digest": simulation.state.state_digest(),
        "rng_state": simulation.rng.bit_generator.state,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [*Path("src/wormpimc").rglob("*.py"), *Path("validation").glob("*.py")]
            )
        },
    }
    (directory / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(directory),
                "status": result["status"],
                "seconds": elapsed,
                "Z_fraction": result["Z_fraction"],
                "comparisons": {
                    name: {
                        key: row.get(key)
                        for key in ("estimate", "reference", "z_score", "status")
                    }
                    for name, row in result["comparisons"].items()
                },
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--c0", type=float, default=1.0)
    parser.add_argument("--asymmetric", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--warmup", type=int, default=2048)
    parser.add_argument("--sweeps", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.timeout <= 0 or not np.isfinite(args.timeout):
        parser.error("--timeout must be finite and positive")
    run(args)


if __name__ == "__main__":
    main()
