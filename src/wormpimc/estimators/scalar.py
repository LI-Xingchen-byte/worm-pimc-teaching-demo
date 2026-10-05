"""Transparent scalar accumulation without hidden plotting or I/O state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import SimulationConfig
from ..configuration import Configuration
from ..measure import PrimitiveTargetMeasure
from ..statistics import (
    BLOCKING_MIN_BLOCKS,
    BLOCKING_MIN_NUMERATOR_EVENTS,
    BLOCKING_PLATEAU_LEVELS,
    BLOCKING_PLATEAU_SIGMA,
    BlockingRatioAccumulator,
    BlockingSummary,
    HistogramBlockingAccumulator,
)
from ..types import Sector
from .energy import thermodynamic_energy
from .green import GreenHistogramAccumulator


@dataclass(slots=True)
class ScalarAccumulator:
    """Count, sum, and sum of squares for deterministic checkpointing."""

    count: int = 0
    total: float = 0.0
    total_square: float = 0.0

    def add(self, value: float) -> None:
        observation = float(value)
        if not math.isfinite(observation):
            raise ValueError("observable must be finite")
        self.count += 1
        self.total += observation
        self.total_square += observation * observation

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    @property
    def sample_variance(self) -> float:
        if self.count < 2:
            return float("nan")
        numerator = self.total_square - self.total * self.total / self.count
        return max(0.0, numerator / (self.count - 1))

    @property
    def uncorrected_standard_error(self) -> float:
        if self.count < 2:
            return float("nan")
        return math.sqrt(self.sample_variance / self.count)

    def snapshot(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "total": self.total,
            "total_square": self.total_square,
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "ScalarAccumulator":
        if set(data) != {"count", "total", "total_square"}:
            raise ValueError("invalid scalar accumulator checkpoint")
        count = int(data["count"])
        if count < 0:
            raise ValueError("accumulator count may not be negative")
        return cls(
            count=count,
            total=float(data["total"]),
            total_square=float(data["total_square"]),
        )


class EstimatorManager:
    """Accumulate diagonal scalars and extended-ensemble Green residence."""

    SCHEMA_VERSION = 3
    NAMES = (
        "particle_number",
        "potential_energy",
        "kinetic_energy",
        "total_energy",
        "winding_squared",
    )

    def __init__(self, config: SimulationConfig) -> None:
        self.accumulators = {
            name: ScalarAccumulator() for name in self.NAMES
        }
        self.green = GreenHistogramAccumulator(config)
        self.scalar_blocking = {
            name: BlockingRatioAccumulator() for name in self.NAMES
        }
        self.histogram_blocking = HistogramBlockingAccumulator(
            self.green.n_slices * self.green.n_bins
        )
        self.blocking_history_complete = True

    @property
    def measurement_count(self) -> int:
        return self.accumulators["particle_number"].count

    @property
    def total_measurement_count(self) -> int:
        return self.green.total_measurements

    def measure(
        self,
        state: Configuration,
        target: PrimitiveTargetMeasure,
    ) -> None:
        flat_bin = self.green.accumulate(state)
        if state.sector is not Sector.Z:
            for blocker in self.scalar_blocking.values():
                blocker.add(0.0, 0.0)
            assert flat_bin is not None
            self.histogram_blocking.add(flat_bin)
            return
        energy = thermodynamic_energy(state, target)
        winding = state.total_winding().astype(np.float64)
        values = {
            "particle_number": float(state.n_particles),
            "potential_energy": energy.potential,
            "kinetic_energy": energy.kinetic,
            "total_energy": energy.total,
            "winding_squared": float(np.dot(winding, winding)),
        }
        for name, value in values.items():
            self.accumulators[name].add(value)
            self.scalar_blocking[name].add(value, 1.0)
        assert flat_bin is None
        self.histogram_blocking.add(
            None,
            z_particle_number=values["particle_number"],
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "scalar_accumulators": {
                name: accumulator.snapshot()
                for name, accumulator in self.accumulators.items()
            },
            "green_histogram": self.green.snapshot(),
            "scalar_blocking": {
                name: blocker.snapshot()
                for name, blocker in self.scalar_blocking.items()
            },
            "histogram_blocking": self.histogram_blocking.snapshot(),
            "blocking_history_complete": self.blocking_history_complete,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        schema_version = snapshot.get("schema_version")
        legacy_fields = {
            "schema_version",
            "scalar_accumulators",
            "green_histogram",
        }
        current_fields = {
            *legacy_fields,
            "scalar_blocking",
            "histogram_blocking",
            "blocking_history_complete",
        }
        if (
            schema_version == 2
            and set(snapshot) != legacy_fields
            or schema_version == self.SCHEMA_VERSION
            and set(snapshot) != current_fields
            or schema_version not in {2, self.SCHEMA_VERSION}
        ):
            raise ValueError("checkpoint estimator schema does not match")
        scalar_snapshot = snapshot["scalar_accumulators"]
        if not isinstance(scalar_snapshot, dict) or set(scalar_snapshot) != set(
            self.NAMES
        ):
            raise ValueError("checkpoint estimator names do not match")
        self.accumulators = {
            name: ScalarAccumulator.from_snapshot(dict(scalar_snapshot[name]))
            for name in self.NAMES
        }
        green_snapshot = snapshot["green_histogram"]
        if not isinstance(green_snapshot, dict):
            raise ValueError("checkpoint Green histogram is invalid")
        self.green.restore(green_snapshot)
        self.scalar_blocking = {
            name: BlockingRatioAccumulator() for name in self.NAMES
        }
        self.histogram_blocking = HistogramBlockingAccumulator(
            self.green.n_slices * self.green.n_bins
        )
        if schema_version == 2:
            self.blocking_history_complete = self.green.total_measurements == 0
            return
        complete = snapshot["blocking_history_complete"]
        if type(complete) is not bool:
            raise ValueError("blocking history completeness must be boolean")
        blocking_snapshot = snapshot["scalar_blocking"]
        if not isinstance(blocking_snapshot, dict) or set(
            blocking_snapshot
        ) != set(self.NAMES):
            raise ValueError("checkpoint scalar-blocking names do not match")
        for name, blocker in self.scalar_blocking.items():
            data = blocking_snapshot[name]
            if not isinstance(data, dict):
                raise ValueError("checkpoint scalar-blocking state is invalid")
            blocker.restore(data)
        histogram_snapshot = snapshot["histogram_blocking"]
        if not isinstance(histogram_snapshot, dict):
            raise ValueError("checkpoint histogram-blocking state is invalid")
        self.histogram_blocking.restore(histogram_snapshot)
        self.blocking_history_complete = complete
        if complete:
            expected_total = self.green.total_measurements
            for name, blocker in self.scalar_blocking.items():
                accumulator = self.accumulators[name]
                if (
                    blocker.measurement_count != expected_total
                    or not math.isclose(
                        blocker.denominator_sum,
                        self.green.z_measurements,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    or not math.isclose(
                        blocker.numerator_sum,
                        accumulator.total,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                ):
                    raise ValueError(
                        "checkpoint scalar-blocking totals do not match accumulators"
                    )
            if self.histogram_blocking.measurement_count != expected_total:
                raise ValueError(
                    "checkpoint histogram-blocking count does not match histogram"
                )

    def rows(self) -> list[dict[str, str | int | float]]:
        metadata = {
            "particle_number": (
                "link_occupation",
                "1",
                "physical_z_conditional",
                "",
            ),
            "potential_energy": (
                "primitive_slice_average",
                "energy",
                "physical_z_conditional",
                "",
            ),
            "kinetic_energy": (
                "primitive_thermodynamic_kinetic",
                "energy",
                "physical_z_conditional",
                "high_variance_with_n_slices",
            ),
            "total_energy": (
                "primitive_thermodynamic_total",
                "energy",
                "physical_z_conditional",
                "high_variance_with_n_slices",
            ),
            "winding_squared": (
                "topological_winding",
                "1",
                "physical_z_conditional",
                "",
            ),
        }
        rows: list[dict[str, str | int | float]] = []
        for name in self.NAMES:
            accumulator = self.accumulators[name]
            variant, units, normalization_status, warning = metadata[name]
            if self.blocking_history_complete:
                summary = self.scalar_blocking[name].summary(
                    independent_count=accumulator.count,
                    independent_standard_error=(
                        accumulator.uncorrected_standard_error
                    ),
                )
            else:
                summary = BlockingSummary(
                    status="incomplete_history_after_legacy_resume",
                    standard_error=None,
                    level=None,
                    block_size=None,
                    n_blocks=None,
                    used_measurements=None,
                    n_effective=None,
                )
            warnings = [value for value in (warning,) if value]
            if summary.status != "blocking_plateau":
                warnings.append(summary.status)
            rows.append(
                {
                    "observable": name,
                    "mean": accumulator.mean,
                    "standard_error": (
                        summary.standard_error
                        if summary.standard_error is not None
                        else ""
                    ),
                    "uncorrected_standard_error": (
                        accumulator.uncorrected_standard_error
                    ),
                    "n_measurements": accumulator.count,
                    "n_effective": (
                        summary.n_effective
                        if summary.n_effective is not None
                        else ""
                    ),
                    "blocking_level": (
                        summary.level if summary.level is not None else ""
                    ),
                    "block_size_measurements": (
                        summary.block_size
                        if summary.block_size is not None
                        else ""
                    ),
                    "n_blocks": (
                        summary.n_blocks if summary.n_blocks is not None else ""
                    ),
                    "uncertainty_status": summary.status,
                    "estimator_variant": variant,
                    "sector": "Z",
                    "units": units,
                    "normalization_status": normalization_status,
                    "warning": ";".join(warnings),
                }
            )
        return rows

    def green_rows(self) -> list[dict[str, str | int | float]]:
        rows = self.green.rows()
        green_scale = 1.0 / (
            self.green.bin_width
            * self.green.worm_measure_coefficient
            * self.green.n_slices
            * self.green.box_length
        )
        g1_scale = 1.0 / (
            self.green.bin_width
            * self.green.worm_measure_coefficient
            * self.green.n_slices
        )
        for component, row in enumerate(rows):
            if self.blocking_history_complete and row["green_function"] != "":
                green_summary = self.histogram_blocking.summary(
                    component,
                    denominator="z",
                    scale=green_scale,
                    independent_count=self.green.total_measurements,
                )
            else:
                green_summary = BlockingSummary(
                    status=(
                        "incomplete_history_after_legacy_resume"
                        if not self.blocking_history_complete
                        else "uncertainty_not_applicable"
                    ),
                    standard_error=None,
                    level=None,
                    block_size=None,
                    n_blocks=None,
                    used_measurements=None,
                    n_effective=None,
                )
            if self.blocking_history_complete and row["g1"] != "":
                g1_summary = self.histogram_blocking.summary(
                    component,
                    denominator="particle",
                    scale=g1_scale,
                    independent_count=self.green.total_measurements,
                )
            else:
                g1_summary = BlockingSummary(
                    status=(
                        "incomplete_history_after_legacy_resume"
                        if not self.blocking_history_complete
                        else "uncertainty_not_applicable"
                    ),
                    standard_error=None,
                    level=None,
                    block_size=None,
                    n_blocks=None,
                    used_measurements=None,
                    n_effective=None,
                )
            row.update(
                {
                    "standard_error": (
                        green_summary.standard_error
                        if green_summary.standard_error is not None
                        else ""
                    ),
                    "n_effective": (
                        green_summary.n_effective
                        if green_summary.n_effective is not None
                        else ""
                    ),
                    "blocking_level": (
                        green_summary.level
                        if green_summary.level is not None
                        else ""
                    ),
                    "block_size_measurements": (
                        green_summary.block_size
                        if green_summary.block_size is not None
                        else ""
                    ),
                    "n_blocks": (
                        green_summary.n_blocks
                        if green_summary.n_blocks is not None
                        else ""
                    ),
                    "uncertainty_status": green_summary.status,
                    "g1_standard_error": (
                        g1_summary.standard_error
                        if g1_summary.standard_error is not None
                        else ""
                    ),
                    "g1_n_effective": (
                        g1_summary.n_effective
                        if g1_summary.n_effective is not None
                        else ""
                    ),
                    "g1_uncertainty_status": g1_summary.status,
                    "warning": (
                        ""
                        if green_summary.status == "blocking_plateau"
                        else green_summary.status
                    ),
                }
            )
        return rows

    def blocking_rows(self) -> list[dict[str, str | int | float]]:
        """Return every auditable blocking level used by result summaries."""

        rows: list[dict[str, str | int | float]] = []
        for name in self.NAMES:
            for estimate in self.scalar_blocking[name].estimates():
                rows.append(self._blocking_row(name, "scalar", estimate))
        green_scale = 1.0 / (
            self.green.bin_width
            * self.green.worm_measure_coefficient
            * self.green.n_slices
            * self.green.box_length
        )
        g1_scale = 1.0 / (
            self.green.bin_width
            * self.green.worm_measure_coefficient
            * self.green.n_slices
        )
        for time_index in range(self.green.n_slices):
            for bin_index in range(self.green.n_bins):
                if self.green.counts[time_index, bin_index] == 0:
                    continue
                component = time_index * self.green.n_bins + bin_index
                for estimate in self.histogram_blocking.estimates(
                    component,
                    denominator="z",
                    scale=green_scale,
                ):
                    row = self._blocking_row(
                        "green_function", "histogram_ratio", estimate
                    )
                    row["time_index"] = time_index
                    row["bin_index"] = bin_index
                    row["eligible_for_plateau"] = (
                        row["eligible_for_plateau"]
                        and int(self.green.counts[time_index, bin_index])
                        >= BLOCKING_MIN_NUMERATOR_EVENTS
                    )
                    rows.append(row)
                if time_index == self.green.n_slices - 1:
                    for estimate in self.histogram_blocking.estimates(
                        component,
                        denominator="particle",
                        scale=g1_scale,
                    ):
                        row = self._blocking_row(
                            "g1", "histogram_ratio", estimate
                        )
                        row["time_index"] = time_index
                        row["bin_index"] = bin_index
                        row["eligible_for_plateau"] = (
                            row["eligible_for_plateau"]
                            and int(self.green.counts[time_index, bin_index])
                            >= BLOCKING_MIN_NUMERATOR_EVENTS
                        )
                        rows.append(row)
        return rows

    @staticmethod
    def _blocking_row(
        observable: str,
        kind: str,
        estimate: Any,
    ) -> dict[str, str | int | float]:
        return {
            "observable": observable,
            "kind": kind,
            "time_index": "",
            "bin_index": "",
            "block_level": estimate.level,
            "block_size_measurements": estimate.block_size,
            "n_blocks": estimate.n_blocks,
            "used_measurements": estimate.used_measurements,
            "numerator_sum": estimate.numerator_sum,
            "denominator_sum": estimate.denominator_sum,
            "estimate": estimate.estimate,
            "standard_error": estimate.standard_error,
            "variance_relative_uncertainty": (
                estimate.variance_relative_uncertainty
            ),
            "eligible_for_plateau": (
                estimate.n_blocks >= BLOCKING_MIN_BLOCKS
            ),
            "blocking_min_blocks": BLOCKING_MIN_BLOCKS,
            "blocking_min_numerator_events": (
                BLOCKING_MIN_NUMERATOR_EVENTS
            ),
            "blocking_plateau_levels": BLOCKING_PLATEAU_LEVELS,
            "blocking_plateau_sigma": BLOCKING_PLATEAU_SIGMA,
        }
