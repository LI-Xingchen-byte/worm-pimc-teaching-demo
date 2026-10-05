"""Checkpointable dyadic blocking for correlated ratio estimators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


BLOCKING_MIN_BLOCKS = 32
BLOCKING_MIN_NUMERATOR_EVENTS = 32
BLOCKING_PLATEAU_LEVELS = 3
BLOCKING_PLATEAU_SIGMA = 2.0


@dataclass(frozen=True, slots=True)
class BlockingLevelEstimate:
    """Ratio estimate and uncertainty at one non-overlapping block size."""

    level: int
    block_size: int
    n_blocks: int
    used_measurements: int
    numerator_sum: float
    denominator_sum: float
    estimate: float
    standard_error: float
    variance_relative_uncertainty: float


@dataclass(frozen=True, slots=True)
class BlockingSummary:
    """Conservative automatic selection from a dyadic blocking curve."""

    status: str
    standard_error: float | None
    level: int | None
    block_size: int | None
    n_blocks: int | None
    used_measurements: int | None
    n_effective: float | None


@dataclass(slots=True)
class _RatioLevelState:
    n_blocks: int = 0
    sum_numerator: float = 0.0
    sum_numerator_square: float = 0.0
    sum_denominator: float = 0.0
    sum_denominator_square: float = 0.0
    sum_cross: float = 0.0
    pending_numerator: float | None = None
    pending_denominator: float | None = None


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _estimate_ratio_level(
    *,
    level: int,
    n_blocks: int,
    sum_numerator: float,
    sum_numerator_square: float,
    sum_denominator: float,
    sum_denominator_square: float,
    sum_cross: float,
    scale: float,
) -> BlockingLevelEstimate | None:
    if n_blocks < 2 or sum_denominator == 0.0:
        return None
    ratio = sum_numerator / sum_denominator
    mean_denominator = sum_denominator / n_blocks
    residual_sum_square = (
        sum_numerator_square
        - 2.0 * ratio * sum_cross
        + ratio * ratio * sum_denominator_square
    )
    residual_sum_square = max(0.0, residual_sum_square)
    residual_variance = residual_sum_square / (n_blocks - 1)
    standard_error = (
        abs(scale)
        * math.sqrt(residual_variance / n_blocks)
        / abs(mean_denominator)
    )
    return BlockingLevelEstimate(
        level=level,
        block_size=1 << level,
        n_blocks=n_blocks,
        used_measurements=n_blocks * (1 << level),
        numerator_sum=sum_numerator * (1 << level),
        denominator_sum=sum_denominator * (1 << level),
        estimate=scale * ratio,
        standard_error=standard_error,
        variance_relative_uncertainty=math.sqrt(2.0 / (n_blocks - 1)),
    )


def select_blocking_plateau(
    estimates: list[BlockingLevelEstimate],
    *,
    independent_count: int,
    independent_standard_error: float,
    min_blocks: int = BLOCKING_MIN_BLOCKS,
    plateau_levels: int = BLOCKING_PLATEAU_LEVELS,
    plateau_sigma: float = BLOCKING_PLATEAU_SIGMA,
    zero_numerator: bool = False,
) -> BlockingSummary:
    """Select a plateau only when adjacent variance intervals overlap."""

    if zero_numerator:
        return BlockingSummary(
            status="zero_numerator_no_uncertainty_estimate",
            standard_error=None,
            level=None,
            block_size=None,
            n_blocks=None,
            used_measurements=None,
            n_effective=None,
        )
    eligible = [
        estimate
        for estimate in estimates
        if estimate.n_blocks >= min_blocks
        and math.isfinite(estimate.standard_error)
    ]
    if len(eligible) < plateau_levels:
        return BlockingSummary(
            status="insufficient_blocking_levels",
            standard_error=None,
            level=None,
            block_size=None,
            n_blocks=None,
            used_measurements=None,
            n_effective=None,
        )
    candidates = eligible[-plateau_levels:]
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for estimate in candidates:
        variance = estimate.standard_error**2
        relative = plateau_sigma * estimate.variance_relative_uncertainty
        lower_bounds.append(max(0.0, variance * (1.0 - relative)))
        upper_bounds.append(variance * (1.0 + relative))
    if max(lower_bounds) > min(upper_bounds):
        return BlockingSummary(
            status="blocking_plateau_not_reached",
            standard_error=None,
            level=None,
            block_size=None,
            n_blocks=None,
            used_measurements=None,
            n_effective=None,
        )
    selected = max(candidates, key=lambda estimate: estimate.standard_error)
    if selected.standard_error == 0.0:
        n_effective = float(independent_count)
    elif (
        independent_count > 0
        and math.isfinite(independent_standard_error)
        and independent_standard_error >= 0.0
    ):
        n_effective = min(
            float(independent_count),
            independent_count
            * (independent_standard_error / selected.standard_error) ** 2,
        )
    else:
        n_effective = None
    return BlockingSummary(
        status="blocking_plateau",
        standard_error=selected.standard_error,
        level=selected.level,
        block_size=selected.block_size,
        n_blocks=selected.n_blocks,
        used_measurements=selected.used_measurements,
        n_effective=n_effective,
    )


class BlockingRatioAccumulator:
    """Online dyadic blocking for a ratio of two correlated sample means."""

    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self._levels: list[_RatioLevelState] = []

    @property
    def measurement_count(self) -> int:
        return self._levels[0].n_blocks if self._levels else 0

    @property
    def numerator_sum(self) -> float:
        return self._levels[0].sum_numerator if self._levels else 0.0

    @property
    def denominator_sum(self) -> float:
        return self._levels[0].sum_denominator if self._levels else 0.0

    def add(self, numerator: float, denominator: float) -> None:
        self._push(
            0,
            _finite(numerator, "numerator"),
            _finite(denominator, "denominator"),
        )

    def _push(
        self,
        level: int,
        numerator_sum: float,
        denominator_sum: float,
    ) -> None:
        if level == len(self._levels):
            self._levels.append(_RatioLevelState())
        state = self._levels[level]
        block_size = 1 << level
        numerator = numerator_sum / block_size
        denominator = denominator_sum / block_size
        state.n_blocks += 1
        state.sum_numerator += numerator
        state.sum_numerator_square += numerator * numerator
        state.sum_denominator += denominator
        state.sum_denominator_square += denominator * denominator
        state.sum_cross += numerator * denominator
        if state.pending_numerator is None:
            state.pending_numerator = numerator_sum
            state.pending_denominator = denominator_sum
            return
        parent_numerator = state.pending_numerator + numerator_sum
        assert state.pending_denominator is not None
        parent_denominator = state.pending_denominator + denominator_sum
        state.pending_numerator = None
        state.pending_denominator = None
        self._push(level + 1, parent_numerator, parent_denominator)

    def estimates(self, *, scale: float = 1.0) -> list[BlockingLevelEstimate]:
        factor = _finite(scale, "scale")
        result: list[BlockingLevelEstimate] = []
        for level, state in enumerate(self._levels):
            estimate = _estimate_ratio_level(
                level=level,
                n_blocks=state.n_blocks,
                sum_numerator=state.sum_numerator,
                sum_numerator_square=state.sum_numerator_square,
                sum_denominator=state.sum_denominator,
                sum_denominator_square=state.sum_denominator_square,
                sum_cross=state.sum_cross,
                scale=factor,
            )
            if estimate is not None:
                result.append(estimate)
        return result

    def summary(
        self,
        *,
        independent_count: int,
        independent_standard_error: float,
        scale: float = 1.0,
    ) -> BlockingSummary:
        return select_blocking_plateau(
            self.estimates(scale=scale),
            independent_count=independent_count,
            independent_standard_error=independent_standard_error,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "levels": [
                {
                    "n_blocks": state.n_blocks,
                    "sum_numerator": state.sum_numerator,
                    "sum_numerator_square": state.sum_numerator_square,
                    "sum_denominator": state.sum_denominator,
                    "sum_denominator_square": state.sum_denominator_square,
                    "sum_cross": state.sum_cross,
                    "pending_numerator": state.pending_numerator,
                    "pending_denominator": state.pending_denominator,
                }
                for state in self._levels
            ],
        }

    def restore(self, data: dict[str, Any]) -> None:
        if set(data) != {"schema_version", "levels"} or data[
            "schema_version"
        ] != self.SCHEMA_VERSION:
            raise ValueError("invalid ratio-blocking checkpoint")
        raw_levels = data["levels"]
        if not isinstance(raw_levels, list):
            raise ValueError("ratio-blocking levels must be a list")
        expected = {
            "n_blocks",
            "sum_numerator",
            "sum_numerator_square",
            "sum_denominator",
            "sum_denominator_square",
            "sum_cross",
            "pending_numerator",
            "pending_denominator",
        }
        levels: list[_RatioLevelState] = []
        total: int | None = None
        for level, raw in enumerate(raw_levels):
            if not isinstance(raw, dict) or set(raw) != expected:
                raise ValueError("invalid ratio-blocking level fields")
            n_blocks = int(raw["n_blocks"])
            if total is None:
                total = n_blocks
            if n_blocks < 1 or n_blocks != total // (1 << level):
                raise ValueError("invalid ratio-blocking block count")
            pending_numerator = raw["pending_numerator"]
            pending_denominator = raw["pending_denominator"]
            if (pending_numerator is None) != (pending_denominator is None):
                raise ValueError("incomplete ratio-blocking pending pair")
            if (n_blocks % 2 == 1) != (pending_numerator is not None):
                raise ValueError("ratio-blocking pending parity mismatch")
            values = [
                _finite(raw[name], name)
                for name in (
                    "sum_numerator",
                    "sum_numerator_square",
                    "sum_denominator",
                    "sum_denominator_square",
                    "sum_cross",
                )
            ]
            levels.append(
                _RatioLevelState(
                    n_blocks=n_blocks,
                    sum_numerator=values[0],
                    sum_numerator_square=values[1],
                    sum_denominator=values[2],
                    sum_denominator_square=values[3],
                    sum_cross=values[4],
                    pending_numerator=(
                        None
                        if pending_numerator is None
                        else _finite(pending_numerator, "pending_numerator")
                    ),
                    pending_denominator=(
                        None
                        if pending_denominator is None
                        else _finite(pending_denominator, "pending_denominator")
                    ),
                )
            )
        self._levels = levels


@dataclass(slots=True)
class _HistogramLevelState:
    n_blocks: int
    sum_numerator: NDArray[np.float64]
    sum_numerator_square: NDArray[np.float64]
    sum_cross_z: NDArray[np.float64]
    sum_cross_particle: NDArray[np.float64]
    sum_z: float
    sum_z_square: float
    sum_particle: float
    sum_particle_square: float
    pending_counts: dict[int, int] | None
    pending_z: int | None
    pending_particle: float | None


class HistogramBlockingAccumulator:
    """Sparse-update blocking moments for all Green histogram bins."""

    SCHEMA_VERSION = 1

    def __init__(self, n_components: int) -> None:
        if type(n_components) is not int or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        self.n_components = n_components
        self._levels: list[_HistogramLevelState] = []

    @property
    def measurement_count(self) -> int:
        return self._levels[0].n_blocks if self._levels else 0

    def _new_level(self) -> _HistogramLevelState:
        zeros = lambda: np.zeros(self.n_components, dtype=np.float64)
        return _HistogramLevelState(
            n_blocks=0,
            sum_numerator=zeros(),
            sum_numerator_square=zeros(),
            sum_cross_z=zeros(),
            sum_cross_particle=zeros(),
            sum_z=0.0,
            sum_z_square=0.0,
            sum_particle=0.0,
            sum_particle_square=0.0,
            pending_counts=None,
            pending_z=None,
            pending_particle=None,
        )

    def add(
        self,
        component: int | None,
        *,
        z_particle_number: float = 0.0,
    ) -> None:
        particle = _finite(z_particle_number, "z_particle_number")
        if particle < 0.0:
            raise ValueError("z_particle_number may not be negative")
        if component is None:
            counts: dict[int, int] = {}
            z_count = 1
        else:
            if type(component) is not int or not 0 <= component < self.n_components:
                raise ValueError("histogram component lies outside configured range")
            if particle != 0.0:
                raise ValueError("G-sector histogram event may not carry Z particles")
            counts = {component: 1}
            z_count = 0
        self._push(0, counts, z_count, particle)

    def _push(
        self,
        level: int,
        counts: dict[int, int],
        z_count: int,
        particle_sum: float,
    ) -> None:
        if level == len(self._levels):
            self._levels.append(self._new_level())
        state = self._levels[level]
        block_size = 1 << level
        z_value = z_count / block_size
        particle_value = particle_sum / block_size
        state.n_blocks += 1
        state.sum_z += z_value
        state.sum_z_square += z_value * z_value
        state.sum_particle += particle_value
        state.sum_particle_square += particle_value * particle_value
        for component, count in counts.items():
            value = count / block_size
            state.sum_numerator[component] += value
            state.sum_numerator_square[component] += value * value
            state.sum_cross_z[component] += value * z_value
            state.sum_cross_particle[component] += value * particle_value
        if state.pending_counts is None:
            state.pending_counts = dict(counts)
            state.pending_z = z_count
            state.pending_particle = particle_sum
            return
        parent_counts = dict(state.pending_counts)
        for component, count in counts.items():
            parent_counts[component] = parent_counts.get(component, 0) + count
        assert state.pending_z is not None
        assert state.pending_particle is not None
        parent_z = state.pending_z + z_count
        parent_particle = state.pending_particle + particle_sum
        state.pending_counts = None
        state.pending_z = None
        state.pending_particle = None
        self._push(level + 1, parent_counts, parent_z, parent_particle)

    def estimates(
        self,
        component: int,
        *,
        denominator: Literal["z", "particle"],
        scale: float = 1.0,
    ) -> list[BlockingLevelEstimate]:
        if type(component) is not int or not 0 <= component < self.n_components:
            raise ValueError("histogram component lies outside configured range")
        if denominator not in {"z", "particle"}:
            raise ValueError("unknown histogram ratio denominator")
        factor = _finite(scale, "scale")
        result: list[BlockingLevelEstimate] = []
        for level, state in enumerate(self._levels):
            if denominator == "z":
                sum_denominator = state.sum_z
                sum_denominator_square = state.sum_z_square
                sum_cross = state.sum_cross_z[component]
            else:
                sum_denominator = state.sum_particle
                sum_denominator_square = state.sum_particle_square
                sum_cross = state.sum_cross_particle[component]
            estimate = _estimate_ratio_level(
                level=level,
                n_blocks=state.n_blocks,
                sum_numerator=float(state.sum_numerator[component]),
                sum_numerator_square=float(
                    state.sum_numerator_square[component]
                ),
                sum_denominator=sum_denominator,
                sum_denominator_square=sum_denominator_square,
                sum_cross=float(sum_cross),
                scale=factor,
            )
            if estimate is not None:
                result.append(estimate)
        return result

    def summary(
        self,
        component: int,
        *,
        denominator: Literal["z", "particle"],
        scale: float,
        independent_count: int,
    ) -> BlockingSummary:
        estimates = self.estimates(
            component,
            denominator=denominator,
            scale=scale,
        )
        independent_error = (
            estimates[0].standard_error if estimates else float("nan")
        )
        numerator_events = (
            0.0
            if not self._levels
            else float(self._levels[0].sum_numerator[component])
        )
        zero_numerator = numerator_events == 0.0
        if 0.0 < numerator_events < BLOCKING_MIN_NUMERATOR_EVENTS:
            return BlockingSummary(
                status="insufficient_numerator_events",
                standard_error=None,
                level=None,
                block_size=None,
                n_blocks=None,
                used_measurements=None,
                n_effective=None,
            )
        return select_blocking_plateau(
            estimates,
            independent_count=independent_count,
            independent_standard_error=independent_error,
            zero_numerator=bool(zero_numerator),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "n_components": self.n_components,
            "levels": [
                {
                    "n_blocks": state.n_blocks,
                    "sum_numerator": state.sum_numerator.tolist(),
                    "sum_numerator_square": (
                        state.sum_numerator_square.tolist()
                    ),
                    "sum_cross_z": state.sum_cross_z.tolist(),
                    "sum_cross_particle": state.sum_cross_particle.tolist(),
                    "sum_z": state.sum_z,
                    "sum_z_square": state.sum_z_square,
                    "sum_particle": state.sum_particle,
                    "sum_particle_square": state.sum_particle_square,
                    "pending_counts": (
                        None
                        if state.pending_counts is None
                        else [
                            [component, count]
                            for component, count in sorted(
                                state.pending_counts.items()
                            )
                        ]
                    ),
                    "pending_z": state.pending_z,
                    "pending_particle": state.pending_particle,
                }
                for state in self._levels
            ],
        }

    def restore(self, data: dict[str, Any]) -> None:
        if set(data) != {"schema_version", "n_components", "levels"}:
            raise ValueError("invalid histogram-blocking checkpoint fields")
        if data["schema_version"] != self.SCHEMA_VERSION:
            raise ValueError("histogram-blocking schema mismatch")
        if data["n_components"] != self.n_components:
            raise ValueError("histogram-blocking component count mismatch")
        raw_levels = data["levels"]
        if not isinstance(raw_levels, list):
            raise ValueError("histogram-blocking levels must be a list")
        expected = {
            "n_blocks",
            "sum_numerator",
            "sum_numerator_square",
            "sum_cross_z",
            "sum_cross_particle",
            "sum_z",
            "sum_z_square",
            "sum_particle",
            "sum_particle_square",
            "pending_counts",
            "pending_z",
            "pending_particle",
        }
        levels: list[_HistogramLevelState] = []
        total: int | None = None
        for level, raw in enumerate(raw_levels):
            if not isinstance(raw, dict) or set(raw) != expected:
                raise ValueError("invalid histogram-blocking level fields")
            n_blocks = int(raw["n_blocks"])
            if total is None:
                total = n_blocks
            if n_blocks < 1 or n_blocks != total // (1 << level):
                raise ValueError("histogram-blocking level sequence is inconsistent")
            arrays: list[NDArray[np.float64]] = []
            for name in (
                "sum_numerator",
                "sum_numerator_square",
                "sum_cross_z",
                "sum_cross_particle",
            ):
                array = np.asarray(raw[name], dtype=np.float64)
                if array.shape != (self.n_components,) or not np.all(
                    np.isfinite(array)
                ):
                    raise ValueError(f"invalid histogram-blocking array: {name}")
                arrays.append(np.array(array, copy=True))
            pending_raw = raw["pending_counts"]
            if pending_raw is None:
                pending_counts = None
            elif isinstance(pending_raw, list):
                pending_counts = {}
                for pair in pending_raw:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ValueError("invalid pending histogram count")
                    component, count = int(pair[0]), int(pair[1])
                    if (
                        not 0 <= component < self.n_components
                        or count < 1
                        or component in pending_counts
                    ):
                        raise ValueError("invalid pending histogram component")
                    pending_counts[component] = count
            else:
                raise ValueError("invalid pending histogram counts")
            pending_z = raw["pending_z"]
            pending_particle = raw["pending_particle"]
            pending_present = pending_counts is not None
            if pending_present != (pending_z is not None) or pending_present != (
                pending_particle is not None
            ):
                raise ValueError("incomplete histogram-blocking pending block")
            if (n_blocks % 2 == 1) != pending_present:
                raise ValueError("histogram-blocking pending parity mismatch")
            levels.append(
                _HistogramLevelState(
                    n_blocks=n_blocks,
                    sum_numerator=arrays[0],
                    sum_numerator_square=arrays[1],
                    sum_cross_z=arrays[2],
                    sum_cross_particle=arrays[3],
                    sum_z=_finite(raw["sum_z"], "sum_z"),
                    sum_z_square=_finite(raw["sum_z_square"], "sum_z_square"),
                    sum_particle=_finite(raw["sum_particle"], "sum_particle"),
                    sum_particle_square=_finite(
                        raw["sum_particle_square"], "sum_particle_square"
                    ),
                    pending_counts=pending_counts,
                    pending_z=(None if pending_z is None else int(pending_z)),
                    pending_particle=(
                        None
                        if pending_particle is None
                        else _finite(pending_particle, "pending_particle")
                    ),
                )
            )
        self._levels = levels
