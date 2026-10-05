"""Run-directory creation and deterministic JSON/CSV result writers."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulationConfig
from .estimators import EstimatorManager
from .sampler import MoveCounter
from .statistics import (
    BLOCKING_MIN_BLOCKS,
    BLOCKING_MIN_NUMERATOR_EVENTS,
    BLOCKING_PLATEAU_LEVELS,
    BLOCKING_PLATEAU_SIGMA,
)
from .version import __version__


OUTPUT_SCHEMA_VERSION = "3.0"

MOVE_SECTORS = {
    "wiggle": ("Z", "Z"),
    "displace": ("Z", "Z"),
    "open": ("Z", "G"),
    "close": ("G", "Z"),
    "insert": ("Z", "G"),
    "remove": ("G", "Z"),
    "advance": ("G", "G"),
    "recede": ("G", "G"),
    "swap": ("G", "G"),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def create_run_directory(
    config: SimulationConfig,
    *,
    output_root: str | Path | None = None,
) -> Path:
    root = Path(config.output.root if output_root is None else output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = f"{timestamp}_{config.output.label}_{config.config_hash[:8]}"
    run_directory = root / run_name
    run_directory.mkdir(exist_ok=False)
    (run_directory / "checkpoints").mkdir()
    return run_directory


def initial_metadata(config: SimulationConfig) -> dict[str, Any]:
    enabled_moves = [
        name
        for name, weight in config.moves.weights.as_dict().items()
        if weight > 0.0
    ]
    topology_enabled = any(
        name in enabled_moves
        for name in ("open", "insert", "advance", "swap")
    )
    return {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "package_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "config_hash": config.config_hash,
        "rng_algorithm": "PCG64",
        "initial_seed": config.run.seed,
        "action": config.discretization.action,
        "pair_potential": config.potential.pair,
        "external_potential": config.potential.external,
        "potential_parameters": config.resolved_dict()["potential"],
        "move_families": enabled_moves,
        "sampling_scope": (
            "grand_canonical_extended_ZG_ensemble"
            if topology_enabled
            else "fixed_particle_number_fixed_connectivity_fixed_winding_sector"
        ),
        "worm_sector_weight": config.moves.worm_sector_weight,
        "worm_measure_coefficient": config.worm_measure_coefficient,
        "uncertainty_method": "dyadic_blocking_ratio_delta",
        "blocking_min_blocks": BLOCKING_MIN_BLOCKS,
        "blocking_min_numerator_events": BLOCKING_MIN_NUMERATOR_EVENTS,
        "blocking_plateau_levels": BLOCKING_PLATEAU_LEVELS,
        "blocking_plateau_sigma": BLOCKING_PLATEAU_SIGMA,
        "debug_invariants": config.run.invariant_check_interval_steps > 0,
        "start_time": utc_now(),
        "end_time": None,
        "run_status": "initializing",
        "checkpoint_source": None,
        "command_python": sys.executable,
    }


def write_scalar_observables(
    path: str | Path,
    estimators: EstimatorManager,
) -> None:
    rows = estimators.rows()
    fieldnames = [
        "observable",
        "mean",
        "standard_error",
        "uncorrected_standard_error",
        "n_measurements",
        "n_effective",
        "blocking_level",
        "block_size_measurements",
        "n_blocks",
        "uncertainty_status",
        "estimator_variant",
        "sector",
        "units",
        "normalization_status",
        "warning",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_green_histogram(
    path: str | Path,
    estimators: EstimatorManager,
) -> None:
    rows = estimators.green_rows()
    fieldnames = [
        "time_index",
        "imaginary_time",
        "left_edge",
        "right_edge",
        "count",
        "n_measurement_opportunities",
        "n_z_measurements",
        "z_particle_sum",
        "endpoint_density",
        "green_function",
        "g1",
        "standard_error",
        "n_effective",
        "blocking_level",
        "block_size_measurements",
        "n_blocks",
        "uncertainty_status",
        "g1_standard_error",
        "g1_n_effective",
        "g1_uncertainty_status",
        "normalization_status",
        "estimator_variant",
        "warning",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_blocking_diagnostics(
    path: str | Path,
    estimators: EstimatorManager,
) -> None:
    """Write complete dyadic curves behind published uncertainty fields."""

    fieldnames = [
        "observable",
        "kind",
        "time_index",
        "bin_index",
        "block_level",
        "block_size_measurements",
        "n_blocks",
        "used_measurements",
        "numerator_sum",
        "denominator_sum",
        "estimate",
        "standard_error",
        "variance_relative_uncertainty",
        "eligible_for_plateau",
        "blocking_min_blocks",
        "blocking_min_numerator_events",
        "blocking_plateau_levels",
        "blocking_plateau_sigma",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(estimators.blocking_rows())


def write_move_statistics(
    path: str | Path,
    counters: dict[str, MoveCounter],
) -> None:
    component_names = (
        "kinetic",
        "potential",
        "chemical",
        "sector_measure",
        "proposal_selection",
    )
    fieldnames = [
        "move",
        "sector_before",
        "sector_after",
        "selected",
        "not_applicable",
        "structurally_invalid",
        "action_evaluated",
        "accepted",
        "acceptance_given_evaluated",
        *[f"mean_{name}" for name in component_names],
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for move_name, counter in counters.items():
            denominator = counter.action_evaluated
            sector_before, sector_after = MOVE_SECTORS[move_name]
            row: dict[str, str | int | float] = {
                "move": move_name,
                "sector_before": sector_before,
                "sector_after": sector_after,
                "selected": counter.selected,
                "not_applicable": counter.not_applicable,
                "structurally_invalid": counter.structurally_invalid,
                "action_evaluated": denominator,
                "accepted": counter.accepted,
                "acceptance_given_evaluated": (
                    counter.acceptance_given_evaluated
                    if denominator
                    else ""
                ),
            }
            for name in component_names:
                total = float(getattr(counter, f"sum_{name}"))
                row[f"mean_{name}"] = total / denominator if denominator else ""
            writer.writerow(row)


def write_sector_residence(
    path: str | Path,
    visits: dict[str, int],
) -> None:
    """Write raw Markov-chain residence without Green normalization claims."""

    if set(visits) != {"Z", "G"}:
        raise ValueError("sector residence requires exactly Z and G")
    total = sum(visits.values())
    fieldnames = ("sector", "visits", "fraction", "normalization_status")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sector in ("Z", "G"):
            writer.writerow(
                {
                    "sector": sector,
                    "visits": visits[sector],
                    "fraction": visits[sector] / total if total else "",
                    "normalization_status": "raw_extended_ensemble_residence",
                }
            )
