"""Optional spatial density with time-slice averaging and opportunity blocking."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..config import SimulationConfig
from ..configuration import Configuration
from ..statistics import BLOCKING_MIN_NUMERATOR_EVENTS, BlockingRatioAccumulator
from ..types import Sector


class DensityProfile:
    """Measure rho(x), with integral <N>, without changing the sampling RNG.

    Attach after_sweep to Simulation.advance. Save snapshot() alongside the
    simulation checkpoint to continue this optional measurement history.
    Memory is O(spatial_bins * log(measurement_opportunities)).
    """

    SCHEMA_VERSION = 1

    def __init__(self, config: SimulationConfig, *, spatial_bins: int = 16) -> None:
        if config.system.ndim != 1:
            raise ValueError("density profile requires one dimension")
        if type(spatial_bins) is not int or spatial_bins < 1:
            raise ValueError("spatial_bins must be a positive integer")
        self.config = config
        self.spatial_bins = spatial_bins
        self._blocking = [BlockingRatioAccumulator() for _ in range(spatial_bins)]
        self._nonzero_opportunities = np.zeros(spatial_bins, dtype=np.int64)

    @property
    def bin_width(self) -> float:
        return self.config.system.box_length / self.spatial_bins

    @property
    def bin_edges(self) -> np.ndarray:
        return np.linspace(0.0, self.config.system.box_length, self.spatial_bins + 1)

    @property
    def measurement_count(self) -> int:
        return self._blocking[0].measurement_count

    @property
    def z_measurements(self) -> int:
        return int(self._blocking[0].denominator_sum)

    def measure(self, state: Configuration) -> None:
        """One opportunity, including G zeros and repeated rejected states."""
        if (
            state.ndim != 1
            or state.n_slices != self.config.discretization.n_slices
            or state.box_length != self.config.system.box_length
        ):
            raise ValueError("state and density profile use different geometry")
        counts = np.zeros(self.spatial_bins)
        is_z = state.sector is Sector.Z
        if is_z:
            x = state.positions[state.active_ids, 0] % state.box_length
            bins = np.minimum((x / self.bin_width).astype(int), self.spatial_bins - 1)
            counts = np.bincount(bins, minlength=self.spatial_bins) / state.n_slices
            self._nonzero_opportunities += counts > 0
        for count, blocker in zip(counts, self._blocking):
            blocker.add(float(count), float(is_z))

    def after_sweep(self, simulation) -> None:
        """Observer matching the simulation's scheduled measurement points."""
        if simulation.config.config_hash != self.config.config_hash:
            raise ValueError(
                "simulation and density profile use different configurations"
            )
        if (
            simulation.phase == "measurement"
            and simulation.measurement_completed % self.config.run.measurement_stride
            == 0
        ):
            self.measure(simulation.state)

    def results(self) -> dict[str, Any]:
        edges = self.bin_edges
        rows = []
        for index, blocker in enumerate(self._blocking):
            mean = (
                blocker.numerator_sum / (blocker.denominator_sum * self.bin_width)
                if blocker.denominator_sum
                else None
            )
            events = int(self._nonzero_opportunities[index])
            estimates = blocker.estimates(scale=1.0 / self.bin_width)
            summary = blocker.summary(
                scale=1.0 / self.bin_width,
                independent_count=self.measurement_count,
                independent_standard_error=(
                    estimates[0].standard_error if estimates else float("nan")
                ),
            )
            status = summary.status
            if self.z_measurements == 0:
                status = "unavailable_no_z_reference"
            elif events < BLOCKING_MIN_NUMERATOR_EVENTS:
                status = "insufficient_numerator_events"
            publish = status == "blocking_plateau"
            rows.append(
                {
                    "left_edge": float(edges[index]),
                    "right_edge": float(edges[index + 1]),
                    "mean": mean,
                    "standard_error": summary.standard_error if publish else None,
                    "uncertainty_status": status,
                    "nonzero_opportunities": events,
                    "slice_average_count_sum": blocker.numerator_sum,
                    "blocking_level": summary.level if publish else None,
                    "block_size_measurements": summary.block_size if publish else None,
                    "n_blocks": summary.n_blocks if publish else None,
                }
            )
        integral = (
            math.fsum(blocker.numerator_sum for blocker in self._blocking)
            / self.z_measurements
            if self.z_measurements
            else None
        )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "config_hash": self.config.config_hash,
            "n_measurement_opportunities": self.measurement_count,
            "n_z_measurements": self.z_measurements,
            "bin_width": self.bin_width,
            "integrated_density": integral,
            "normalization": "particle_number_density_z_conditional",
            "estimator_variant": "slice_average_all_opportunity_ratio_blocking",
            "rows": rows,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "config_hash": self.config.config_hash,
            "spatial_bins": self.spatial_bins,
            "nonzero_opportunities": self._nonzero_opportunities.tolist(),
            "blocking": [blocker.snapshot() for blocker in self._blocking],
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Validate a saved auxiliary history before replacing current state."""
        if set(snapshot) != set(self.snapshot()) or any(
            snapshot[key] != expected
            for key, expected in (
                ("schema_version", self.SCHEMA_VERSION),
                ("config_hash", self.config.config_hash),
                ("spatial_bins", self.spatial_bins),
            )
        ):
            raise ValueError("incompatible density snapshot")
        events = snapshot["nonzero_opportunities"]
        if (
            not isinstance(events, list)
            or len(events) != self.spatial_bins
            or any(type(value) is not int or value < 0 for value in events)
        ):
            raise ValueError("invalid density event counts")
        raw = snapshot["blocking"]
        if not isinstance(raw, list) or len(raw) != self.spatial_bins:
            raise ValueError("invalid density blocking state")
        blockers = [BlockingRatioAccumulator() for _ in raw]
        for blocker, data in zip(blockers, raw):
            blocker.restore(data)
        opportunities = blockers[0].measurement_count
        z = blockers[0].denominator_sum
        if z < 0 or z > opportunities or z != int(z):
            raise ValueError("invalid density Z residence")
        for blocker, event_count, data in zip(blockers, events, raw):
            if (
                blocker.measurement_count != opportunities
                or blocker.denominator_sum != z
                or event_count > z
                or (event_count == 0) != (blocker.numerator_sum == 0)
                or any(
                    level[key] < 0
                    for level in data["levels"]
                    for key in (
                        "sum_numerator",
                        "sum_numerator_square",
                        "sum_denominator",
                        "sum_denominator_square",
                        "sum_cross",
                    )
                )
            ):
                raise ValueError("inconsistent density blocking totals")
        self._blocking = blockers
        self._nonzero_opportunities = np.array(events, dtype=np.int64)
