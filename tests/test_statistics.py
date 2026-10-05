from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from wormpimc import Simulation, SimulationConfig
from wormpimc.estimators import EstimatorManager
from wormpimc.statistics import (
    BlockingRatioAccumulator,
    HistogramBlockingAccumulator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ratio_blocking_level_matches_manual_delta_method() -> None:
    numerators = (1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    denominators = (1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    accumulator = BlockingRatioAccumulator()
    for numerator, denominator in zip(numerators, denominators, strict=True):
        accumulator.add(numerator, denominator)

    level = accumulator.estimates()[1]

    assert level.block_size == 2
    assert level.n_blocks == 4
    assert level.estimate == pytest.approx(1.0)
    assert level.standard_error == pytest.approx(math.sqrt(0.5))


def test_iid_blocking_plateau_reproduces_independent_standard_error() -> None:
    values = np.random.default_rng(20260901).normal(size=131_072)
    accumulator = BlockingRatioAccumulator()
    for value in values:
        accumulator.add(float(value), 1.0)
    independent_error = float(np.std(values, ddof=1) / np.sqrt(values.size))

    summary = accumulator.summary(
        independent_count=values.size,
        independent_standard_error=independent_error,
    )

    assert summary.status == "blocking_plateau"
    assert summary.standard_error == pytest.approx(independent_error, rel=0.25)
    assert summary.n_effective is not None
    assert 0.5 * values.size <= summary.n_effective <= values.size


def test_ar1_blocking_detects_correlation_inflation() -> None:
    phi = 0.8
    size = 131_072
    innovations = np.random.default_rng(314159).normal(size=size)
    values = np.empty(size, dtype=np.float64)
    values[0] = innovations[0] / math.sqrt(1.0 - phi * phi)
    for index in range(1, size):
        values[index] = phi * values[index - 1] + innovations[index]
    accumulator = BlockingRatioAccumulator()
    for value in values:
        accumulator.add(float(value), 1.0)
    independent_error = float(np.std(values, ddof=1) / np.sqrt(size))
    theoretical_error = independent_error * math.sqrt((1.0 + phi) / (1.0 - phi))

    summary = accumulator.summary(
        independent_count=size,
        independent_standard_error=independent_error,
    )

    assert summary.status == "blocking_plateau"
    # A single blocking curve is itself noisy; the implementation selects the
    # largest compatible plateau value to avoid optimistic error bars.
    assert summary.standard_error == pytest.approx(theoretical_error, rel=0.5)
    assert summary.n_effective is not None
    assert summary.n_effective < 0.2 * size


def test_histogram_blocking_matches_scalar_ratio_curves() -> None:
    rng = np.random.default_rng(271828)
    histogram = HistogramBlockingAccumulator(4)
    z_ratio = BlockingRatioAccumulator()
    particle_ratio = BlockingRatioAccumulator()
    for _ in range(4096):
        event = int(rng.integers(0, 6))
        if event < 2:
            particle_number = float(event + 1)
            histogram.add(None, z_particle_number=particle_number)
            numerator = 0.0
            z = 1.0
        else:
            component = event - 2
            histogram.add(component)
            numerator = float(component == 1)
            z = 0.0
            particle_number = 0.0
        z_ratio.add(numerator, z)
        particle_ratio.add(numerator, particle_number)

    actual_z = histogram.estimates(1, denominator="z", scale=2.5)
    expected_z = z_ratio.estimates(scale=2.5)
    actual_particle = histogram.estimates(
        1, denominator="particle", scale=0.75
    )
    expected_particle = particle_ratio.estimates(scale=0.75)

    assert actual_z == expected_z
    assert actual_particle == expected_particle


def test_blocking_checkpoint_round_trip_is_exact() -> None:
    ratio = BlockingRatioAccumulator()
    histogram = HistogramBlockingAccumulator(5)
    for index in range(257):
        ratio.add(float(index % 7), float(index % 3 == 0))
        if index % 4 == 0:
            histogram.add(None, z_particle_number=float(index % 3 + 1))
        else:
            histogram.add(index % 5)

    restored_ratio = BlockingRatioAccumulator()
    restored_ratio.restore(ratio.snapshot())
    restored_histogram = HistogramBlockingAccumulator(5)
    restored_histogram.restore(histogram.snapshot())

    assert restored_ratio.snapshot() == ratio.snapshot()
    assert restored_histogram.snapshot() == histogram.snapshot()


def test_sparse_histogram_does_not_publish_asymptotic_error() -> None:
    histogram = HistogramBlockingAccumulator(2)
    for index in range(4096):
        if index == 0:
            histogram.add(1)
        else:
            histogram.add(None, z_particle_number=1.0)

    summary = histogram.summary(
        1,
        denominator="z",
        scale=1.0,
        independent_count=4096,
    )

    assert summary.status == "insufficient_numerator_events"
    assert summary.standard_error is None


def test_legacy_estimator_checkpoint_restores_without_false_error_bar() -> None:
    config = SimulationConfig.from_toml(
        PROJECT_ROOT / "configs" / "worm_sampler.toml"
    )
    source = EstimatorManager(config)
    simulation = Simulation(config)
    source.measure(simulation.state, simulation.target)
    legacy = {
        "schema_version": 2,
        "scalar_accumulators": {
            name: accumulator.snapshot()
            for name, accumulator in source.accumulators.items()
        },
        "green_histogram": source.green.snapshot(),
    }
    restored = EstimatorManager(config)

    restored.restore(legacy)

    assert not restored.blocking_history_complete
    assert {
        row["uncertainty_status"] for row in restored.rows()
    } == {"incomplete_history_after_legacy_resume"}
    assert {row["standard_error"] for row in restored.rows()} == {""}
