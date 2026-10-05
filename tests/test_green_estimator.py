from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wormpimc.config import SimulationConfig
from wormpimc.configuration import Configuration, initialize_configuration
from wormpimc.estimators import GreenHistogramAccumulator
from wormpimc.moves import OpenMove
from wormpimc.types import MoveStatus


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _one_link_open_state(config: SimulationConfig) -> Configuration:
    state = initialize_configuration(config)
    move = OpenMove(config, 1.0, 1.0)
    for seed in range(100):
        patch = move.propose(state, np.random.default_rng(seed))
        if (
            patch.status is MoveStatus.PROPOSED
            and patch.metadata_dict()["segment_links"] == 1
        ):
            state.apply(patch)
            return state
    raise AssertionError("could not construct a one-link Open proposal")


def test_green_histogram_hand_normalization_and_beta_minus_g1() -> None:
    config = SimulationConfig.from_toml(CONFIG_DIR / "worm_sampler.toml")
    diagonal = initialize_configuration(config)
    open_state = _one_link_open_state(config)
    accumulator = GreenHistogramAccumulator(config)

    accumulator.accumulate(diagonal)
    accumulator.accumulate(open_state)
    populated = [row for row in accumulator.rows() if row["count"] == 1]

    assert len(populated) == 1
    row = populated[0]
    expected_green = 1.0 / (
        accumulator.bin_width
        * config.worm_measure_coefficient
        * config.discretization.n_slices
        * config.system.box_length
    )
    density = diagonal.n_particles / config.system.box_length
    assert row["time_index"] == config.discretization.n_slices - 1
    assert row["endpoint_density"] == pytest.approx(
        1.0 / (2.0 * accumulator.bin_width)
    )
    assert row["n_measurement_opportunities"] == 2
    assert row["n_z_measurements"] == 1
    assert row["z_particle_sum"] == diagonal.n_particles
    assert row["green_function"] == pytest.approx(expected_green)
    assert row["g1"] == pytest.approx(expected_green / density)
    assert row["standard_error"] == ""
    assert row["normalization_status"] == (
        "normalized_green;beta_minus_finite_tau"
    )
    assert row["estimator_variant"] == (
        "direct_sector_residence_centered_bins"
    )


def test_green_histogram_checkpoint_round_trip_is_exact() -> None:
    config = SimulationConfig.from_toml(CONFIG_DIR / "worm_sampler.toml")
    accumulator = GreenHistogramAccumulator(config)
    accumulator.accumulate(initialize_configuration(config))
    accumulator.accumulate(_one_link_open_state(config))

    restored = GreenHistogramAccumulator(config)
    restored.restore(accumulator.snapshot())

    assert restored.snapshot() == accumulator.snapshot()
    assert restored.rows() == accumulator.rows()


def test_green_rows_do_not_claim_g1_away_from_beta_minus() -> None:
    config = SimulationConfig.from_toml(CONFIG_DIR / "worm_sampler.toml")
    accumulator = GreenHistogramAccumulator(config)
    accumulator.accumulate(initialize_configuration(config))

    rows = accumulator.rows()

    assert all(
        row["g1"] == ""
        for row in rows
        if row["time_index"] != config.discretization.n_slices - 1
    )
    assert all(
        "g1_not_applicable" in str(row["normalization_status"])
        for row in rows
        if row["time_index"] != config.discretization.n_slices - 1
    )


def test_disabled_topology_is_explicit_in_output_status() -> None:
    config = SimulationConfig.from_toml(CONFIG_DIR / "closed_sampler.toml")
    accumulator = GreenHistogramAccumulator(config)
    accumulator.accumulate(initialize_configuration(config))

    rows = accumulator.rows()

    assert {row["normalization_status"] for row in rows} == {
        "not_sampled_topology_disabled"
    }
    assert all(row["green_function"] == "" for row in rows)
