"""Readable equal-time g2 and nonzero S(k) for small-system runs.

This optional collector retains raw per-opportunity vectors in memory. It is
explicitly attached with an after_sweep callback; it is not silently included
in Simulation.results() or the main simulation checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from ..config import SimulationConfig
from ..configuration import Configuration
from ..statistics import BlockingLevelEstimate, select_blocking_plateau
from ..types import Sector


def _validate_grid(spatial_bins, modes):
    if type(spatial_bins) is not int or spatial_bins < 1:
        raise ValueError("spatial_bins must be a positive integer")
    modes = tuple(modes)
    if not modes or any(type(mode) is not int or mode <= 0 for mode in modes):
        raise ValueError("modes must be positive nonzero integers")
    if len(set(modes)) != len(modes):
        raise ValueError("modes must be distinct")
    return modes


def equal_time_sample(
    state: Configuration,
    *,
    spatial_bins: int = 8,
    modes: tuple[int, ...] = (1, 2, 3),
) -> np.ndarray:
    """Return [I_Z, I_Z*N, I_Z*N(N-1), pair bins..., |rho_k|²...].

    Pair counts are ordered, exclude self pairs, use (x_j-x_i) mod L in
    [0,L), and are averaged over slices. Density powers include self terms.
    G returns a zero vector; empty Z has I_Z=1 and all other entries zero.
    Time slices within a configuration are NOT independent MC observations.
    """
    modes = _validate_grid(spatial_bins, modes)
    if state.ndim != 1:
        raise ValueError("equal-time correlations currently require one dimension")
    result = np.zeros(3 + spatial_bins + len(modes))
    if state.sector is Sector.G:
        return result
    n = state.n_particles
    result[:3] = (1, n, n * (n - 1))
    if n == 0:
        return result
    wave_numbers = 2 * np.pi * np.array(modes) / state.box_length
    for time_slice in range(state.n_slices):
        positions = state.positions_on_slice(time_slice)[:, 0]
        separation = (positions[None, :] - positions[:, None]) % state.box_length
        pair_bins = np.floor(
            separation[~np.eye(n, dtype=bool)] * spatial_bins / state.box_length
        ).astype(int)
        result[3 : 3 + spatial_bins] += np.bincount(
            np.minimum(pair_bins, spatial_bins - 1), minlength=spatial_bins
        )
        rho = np.exp(1j * wave_numbers[:, None] * positions).sum(axis=1)
        result[3 + spatial_bins :] += np.abs(rho) ** 2
    result[3:] /= state.n_slices
    return result


class EqualTimeCorrelations:
    """Optional raw-sample collector with covariance-aware dyadic blocking.

    Memory grows with measurement count; intended for teaching and pilots.
    Save snapshot() alongside the simulation checkpoint when resuming both.
    """

    def __init__(
        self, config: SimulationConfig, *, spatial_bins: int = 8, modes=(1, 2, 3)
    ):
        self.modes = _validate_grid(spatial_bins, modes)
        self.spatial_bins = spatial_bins
        self.config = config
        self._samples: list[np.ndarray] = []

    @property
    def samples(self) -> np.ndarray:
        """Detached raw vectors, one per measurement opportunity including G."""
        return np.array(self._samples, dtype=float).reshape(
            -1, 3 + self.spatial_bins + len(self.modes)
        )

    def measure(self, state: Configuration) -> None:
        if (
            state.box_length != self.config.system.box_length
            or state.n_slices != self.config.discretization.n_slices
        ):
            raise ValueError("state and correlation collector use different geometry")
        self._samples.append(
            equal_time_sample(state, spatial_bins=self.spatial_bins, modes=self.modes)
        )

    def snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "config_hash": self.config.config_hash,
            "spatial_bins": self.spatial_bins,
            "modes": list(self.modes),
            "samples": self.samples.tolist(),
        }

    def restore(self, snapshot: dict) -> None:
        expected = self.snapshot()
        if set(snapshot) != set(expected) or any(
            snapshot[key] != expected[key] for key in expected if key != "samples"
        ):
            raise ValueError("incompatible correlation snapshot")
        data = np.asarray(snapshot["samples"], dtype=float)
        if data.size == 0 and data.shape == (0,):
            data = data.reshape(0, 3 + self.spatial_bins + len(self.modes))
        if (
            data.ndim != 2
            or data.shape[1] != 3 + self.spatial_bins + len(self.modes)
            or not np.all(np.isfinite(data))
            or np.any(data < 0)
        ):
            raise ValueError("invalid correlation samples")
        z, n = data[:, 0], data[:, 1]
        if (
            np.any((z != 0) & (z != 1))
            or np.any(n != np.floor(n))
            or np.any(data[z == 0] != 0)
            or not np.allclose(data[:, 2], n * (n - 1), rtol=0, atol=1e-10)
            or not np.allclose(
                data[:, 3 : 3 + self.spatial_bins].sum(axis=1),
                data[:, 2],
                rtol=1e-12,
                atol=1e-10,
            )
        ):
            raise ValueError("inconsistent correlation counts")
        self._samples = [row.copy() for row in data]

    def _value_gradient(self, mean, column, pair):
        z, n = mean[:2]
        gradient = np.zeros_like(mean)
        if n <= 0:
            return None, gradient
        if pair:
            value = self.spatial_bins * mean[column] * z / n**2
            gradient[0] = self.spatial_bins * mean[column] / n**2
            gradient[1] = -2 * value / n
            gradient[column] = self.spatial_bins * z / n**2
        else:
            value = mean[column] / n
            gradient[1] = -value / n
            gradient[column] = 1 / n
        return float(value), gradient

    def results(self) -> dict:
        data = self.samples
        mean = data.mean(axis=0) if len(data) else np.zeros(data.shape[1])
        # Each covariance is across block MEANS, retaining Z/N/count covariance.
        blocked = []
        for level in range(max(0, len(data).bit_length() - 1)):
            size = 1 << level
            count = len(data) // size
            blocks = (
                data[: count * size].reshape(count, size, data.shape[1]).mean(axis=1)
            )
            blocked.append(
                (
                    level,
                    count,
                    blocks.mean(axis=0),
                    np.cov(blocks, rowvar=False) / count,
                )
            )
        rows = []
        for column in range(3, data.shape[1]):
            pair = column < 3 + self.spatial_bins
            value, _ = self._value_gradient(mean, column, pair)
            estimates = []
            for level, count, block_mean, covariance in blocked:
                estimate, gradient = self._value_gradient(block_mean, column, pair)
                if estimate is None:
                    continue
                se = math.sqrt(max(0.0, float(gradient @ covariance @ gradient)))
                estimates.append(
                    BlockingLevelEstimate(
                        level,
                        1 << level,
                        count,
                        count * (1 << level),
                        float(block_mean[column] * count * (1 << level)),
                        float(block_mean[1] * count * (1 << level)),
                        estimate,
                        se,
                        math.sqrt(2 / (count - 1)),
                    )
                )
            summary = select_blocking_plateau(
                estimates,
                independent_count=len(data),
                independent_standard_error=(
                    estimates[0].standard_error if estimates else float("nan")
                ),
            )
            events = int(np.count_nonzero(data[:, column]))
            status, error = summary.status, summary.standard_error
            if value is None:
                status, error = "no_particle_normalization", None
            elif events < 32 or np.count_nonzero(data[:, 1]) < 32:
                status, error = "insufficient_nonzero_opportunities", None
            row = {
                "mean": value,
                "standard_error": error,
                "uncertainty_status": status,
                "nonzero_opportunities": events,
                "block_size": summary.block_size,
                "n_blocks": summary.n_blocks,
                "blocking_levels": [asdict(estimate) for estimate in estimates],
            }
            if pair:
                index = column - 3
                width = self.config.system.box_length / self.spatial_bins
                row.update(
                    left_edge=index * width,
                    right_edge=(index + 1) * width,
                    slice_averaged_pair_count_sum=float(data[:, column].sum()),
                )
            else:
                mode = self.modes[column - 3 - self.spatial_bins]
                row.update(
                    mode=mode,
                    wave_number=2 * math.pi * mode / self.config.system.box_length,
                    density_power_sum=float(data[:, column].sum()),
                )
            rows.append(row)
        z, n, factorial = mean[:3]
        return {
            "measurement_opportunities": len(data),
            "z_measurements": int(data[:, 0].sum()),
            "particle_sum": float(data[:, 1].sum()),
            "mean_N": float(n / z) if z else None,
            "mean_N_Nminus1": float(factorial / z) if z else None,
            "g2_spatial_average_sum_rule": float(factorial * z / n**2) if n else None,
            "uncertainty_method": "all_opportunity_multivariate_delta_blocking",
            "g2": rows[: self.spatial_bins],
            "structure_factor": rows[self.spatial_bins :],
        }
