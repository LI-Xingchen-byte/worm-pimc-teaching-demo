from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wormpimc import Configuration, SimulationConfig, sample_free_closed_worldline
from wormpimc.types import Sector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = PROJECT_ROOT / "configs" / "closed_worldline.toml"


def test_fixed_winding_closed_path_round_trips_through_wrapped_storage() -> None:
    config = SimulationConfig.from_toml(REFERENCE_CONFIG)
    state = sample_free_closed_worldline(
        config, np.random.default_rng(123), winding=-2
    )
    unwrapped = state.unwrapped_cycle()

    assert state.sector is Sector.Z
    assert state.number_of_beads == config.discretization.n_slices
    assert state.cycle_winding().tolist() == [-2]
    assert unwrapped[-1, 0] - unwrapped[0, 0] == pytest.approx(
        -2 * config.system.box_length
    )
    assert np.sum(state.image_to_next[:, 0]) == -2
    state.validate()


def test_configuration_arrays_are_read_only() -> None:
    config = SimulationConfig.from_toml(REFERENCE_CONFIG)
    state = sample_free_closed_worldline(config, np.random.default_rng(7))

    with pytest.raises(ValueError, match="read-only"):
        state.positions[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        state.next_of[0] = 2
    with pytest.raises(AttributeError):
        state.revision = 99  # type: ignore[misc]


def test_closed_path_endpoint_must_agree_modulo_box() -> None:
    path = np.array([[0.1], [0.4], [1.2]])

    with pytest.raises(ValueError, match="agree modulo"):
        Configuration.from_closed_worldline(path, 2.0)


def test_closed_worldline_sampling_is_reproducible() -> None:
    config = SimulationConfig.from_toml(REFERENCE_CONFIG)
    first = sample_free_closed_worldline(config, np.random.default_rng(9876))
    second = sample_free_closed_worldline(config, np.random.default_rng(9876))

    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.image_to_next, second.image_to_next)
    assert np.array_equal(first.unwrapped_cycle(), second.unwrapped_cycle())


def test_interacting_config_is_rejected_by_free_reference_sampler() -> None:
    interacting = SimulationConfig.from_toml(
        PROJECT_ROOT / "configs" / "gaussian_repulsion.toml"
    )

    with pytest.raises(ValueError, match="requires zero potential"):
        sample_free_closed_worldline(interacting, np.random.default_rng(1))
