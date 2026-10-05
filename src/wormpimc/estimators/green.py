"""Finite-bin Green-function residence accumulator for a one-dimensional ring."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import SimulationConfig
from ..configuration import Configuration
from ..geometry import centered_displacement
from ..types import Sector


class GreenHistogramAccumulator:
    """Checkpointable implementation of ``eq-green-histogram-estimator``."""

    SCHEMA_VERSION = 1

    def __init__(self, config: SimulationConfig) -> None:
        if config.system.ndim != 1:
            raise ValueError("Green histogram currently requires ndim = 1")
        self.n_slices = config.discretization.n_slices
        self.n_bins = config.estimators.green_spatial_bins
        self.box_length = config.system.box_length
        self.tau = config.tau
        self.worm_measure_coefficient = config.worm_measure_coefficient
        weights = config.moves.weights
        self.topology_enabled = weights.open > 0.0 or weights.insert > 0.0
        self.counts = np.zeros(
            (self.n_slices, self.n_bins),
            dtype=np.int64,
        )
        self.total_measurements = 0
        self.z_measurements = 0
        self.z_particle_sum = 0.0

    @property
    def bin_width(self) -> float:
        return self.box_length / self.n_bins

    @property
    def bin_edges(self) -> NDArray[np.float64]:
        edges = np.linspace(
            -0.5 * self.box_length,
            0.5 * self.box_length,
            self.n_bins + 1,
            dtype=np.float64,
        )
        edges.setflags(write=False)
        return edges

    def accumulate(self, state: Configuration) -> int | None:
        """Record one opportunity and return its flat G bin, if present."""

        if (
            state.n_slices != self.n_slices
            or state.ndim != 1
            or state.box_length != self.box_length
        ):
            raise ValueError("state and Green histogram use different geometry")
        self.total_measurements += 1
        if state.sector is Sector.Z:
            self.z_measurements += 1
            self.z_particle_sum += state.n_particles
            return None
        time_index = (
            int(state.slice_of[state.worm_head])
            - int(state.slice_of[state.worm_tail])
        ) % state.n_slices
        displacement = float(
            np.asarray(
                centered_displacement(
                    state.positions[state.worm_head],
                    state.positions[state.worm_tail],
                    state.box_length,
                )
            )[0]
        )
        bin_index = int(
            np.floor(
                (displacement + 0.5 * self.box_length) / self.bin_width
            )
        )
        bin_index = min(max(bin_index, 0), self.n_bins - 1)
        self.counts[time_index, bin_index] += 1
        return time_index * self.n_bins + bin_index

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "n_slices": self.n_slices,
            "n_bins": self.n_bins,
            "box_length": self.box_length,
            "tau": self.tau,
            "worm_measure_coefficient": self.worm_measure_coefficient,
            "topology_enabled": self.topology_enabled,
            "counts": self.counts.tolist(),
            "total_measurements": self.total_measurements,
            "z_measurements": self.z_measurements,
            "z_particle_sum": self.z_particle_sum,
        }

    def restore(self, data: dict[str, Any]) -> None:
        expected = {
            "schema_version",
            "n_slices",
            "n_bins",
            "box_length",
            "tau",
            "worm_measure_coefficient",
            "topology_enabled",
            "counts",
            "total_measurements",
            "z_measurements",
            "z_particle_sum",
        }
        if set(data) != expected:
            raise ValueError("invalid Green histogram checkpoint fields")
        for name, expected_value in (
            ("schema_version", self.SCHEMA_VERSION),
            ("n_slices", self.n_slices),
            ("n_bins", self.n_bins),
            ("box_length", self.box_length),
            ("tau", self.tau),
            (
                "worm_measure_coefficient",
                self.worm_measure_coefficient,
            ),
            ("topology_enabled", self.topology_enabled),
        ):
            if data[name] != expected_value:
                raise ValueError(f"Green histogram checkpoint mismatch: {name}")
        counts = np.asarray(data["counts"], dtype=np.int64)
        if counts.shape != (self.n_slices, self.n_bins) or np.any(counts < 0):
            raise ValueError("invalid Green histogram counts")
        total = int(data["total_measurements"])
        diagonal = int(data["z_measurements"])
        particle_sum = float(data["z_particle_sum"])
        if (
            total < 0
            or diagonal < 0
            or diagonal > total
            or int(np.sum(counts)) != total - diagonal
            or not np.isfinite(particle_sum)
            or particle_sum < 0.0
            or not particle_sum.is_integer()
        ):
            raise ValueError("invalid Green histogram residence totals")
        self.counts = np.array(counts, copy=True)
        self.total_measurements = total
        self.z_measurements = diagonal
        self.z_particle_sum = particle_sum

    def rows(self) -> list[dict[str, str | int | float]]:
        """Finalize auditable bin averages before uncertainty augmentation."""

        edges = self.bin_edges
        rows: list[dict[str, str | int | float]] = []
        density = (
            self.z_particle_sum / (self.z_measurements * self.box_length)
            if self.z_measurements > 0
            else float("nan")
        )
        for time_index in range(self.n_slices):
            for bin_index in range(self.n_bins):
                count = int(self.counts[time_index, bin_index])
                endpoint_density: str | float = ""
                green: str | float = ""
                g1: str | float = ""
                if self.total_measurements > 0:
                    endpoint_density = count / (
                        self.total_measurements * self.bin_width
                    )
                if not self.topology_enabled:
                    status = "not_sampled_topology_disabled"
                elif self.total_measurements == 0:
                    status = "unavailable_no_measurements"
                elif self.z_measurements == 0:
                    status = "unavailable_no_z_reference"
                else:
                    green = count / (
                        self.z_measurements
                        * self.bin_width
                        * self.worm_measure_coefficient
                        * self.n_slices
                        * self.box_length
                    )
                    status = "normalized_green"
                    if time_index == self.n_slices - 1:
                        if density > 0.0:
                            g1 = green / density
                            status += ";beta_minus_finite_tau"
                        else:
                            status += ";g1_unavailable_zero_density"
                    else:
                        status += ";g1_not_applicable"
                rows.append(
                    {
                        "time_index": time_index,
                        "imaginary_time": time_index * self.tau,
                        "left_edge": float(edges[bin_index]),
                        "right_edge": float(edges[bin_index + 1]),
                        "count": count,
                        "n_measurement_opportunities": self.total_measurements,
                        "n_z_measurements": self.z_measurements,
                        "z_particle_sum": self.z_particle_sum,
                        "endpoint_density": endpoint_density,
                        "green_function": green,
                        "g1": g1,
                        "standard_error": "",
                        "normalization_status": status,
                        "estimator_variant": (
                            "direct_sector_residence_centered_bins"
                        ),
                        "warning": "autocorrelation_and_ratio_error_not_estimated",
                    }
                )
        return rows
