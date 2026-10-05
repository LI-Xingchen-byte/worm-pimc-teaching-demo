from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wormpimc import (
    SimulationConfig,
    sample_free_closed_worldline,
    winding_distribution,
)
from wormpimc.configuration import Configuration, initialize_configuration
from wormpimc.estimators import thermodynamic_energy
from wormpimc.measure import PrimitiveTargetMeasure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _with_beta(config: SimulationConfig, beta: float) -> SimulationConfig:
    data = config.input_dict()
    data["system"]["beta"] = beta
    return SimulationConfig.from_mapping(data)


@pytest.mark.parametrize(
    "config_name,winding",
    [("closed_worldline.toml", -2), ("worm_sampler.toml", 0)],
)
def test_thermodynamic_energy_matches_beta_derivative(
    config_name: str,
    winding: int,
) -> None:
    """Check the estimator against the complete normalized path weight."""

    config = SimulationConfig.from_toml(CONFIG_DIR / config_name)
    if config.potential.pair == "none":
        state = sample_free_closed_worldline(
            config,
            np.random.default_rng(20260901),
            winding=winding,
        )
    else:
        state = initialize_configuration(config)
    step = 1.0e-6 * config.system.beta
    plus = _with_beta(config, config.system.beta + step)
    minus = _with_beta(config, config.system.beta - step)
    log_plus = PrimitiveTargetMeasure(plus).full_components(state).total
    log_minus = PrimitiveTargetMeasure(minus).full_components(state).total
    derivative = (log_plus - log_minus) / (2.0 * step)
    expected_total = -derivative + config.system.chemical_potential * state.n_particles

    actual = thermodynamic_energy(state, PrimitiveTargetMeasure(config))

    assert actual.total == pytest.approx(expected_total, rel=2.0e-8, abs=2.0e-8)
    assert actual.total == pytest.approx(actual.kinetic + actual.potential)


def test_thermodynamic_energy_uses_explicit_winding_images() -> None:
    config = SimulationConfig.from_toml(CONFIG_DIR / "closed_worldline.toml")
    state = sample_free_closed_worldline(
        config,
        np.random.default_rng(37),
        winding=-2,
    )
    path = state.unwrapped_cycle()
    squared_links = float(np.sum(np.diff(path[:, 0]) ** 2))
    expected = (
        state.n_slices / (2.0 * config.system.beta)
        - squared_links
        * state.n_slices
        / (
            4.0
            * config.system.lambda_kin
            * config.system.beta
            * config.system.beta
        )
    )

    actual = thermodynamic_energy(state, PrimitiveTargetMeasure(config))

    assert state.cycle_winding().tolist() == [-2]
    assert actual.kinetic == pytest.approx(expected)
    assert actual.potential == 0.0


def test_independent_free_paths_reproduce_exact_ring_energy() -> None:
    config = SimulationConfig.from_toml(CONFIG_DIR / "closed_worldline.toml")
    rng = np.random.default_rng(314159)
    target = PrimitiveTargetMeasure(config)
    sample = np.array(
        [
            thermodynamic_energy(
                sample_free_closed_worldline(config, rng),
                target,
            ).total
            for _ in range(3000)
        ]
    )
    winding = winding_distribution(
        config.system.box_length,
        config.system.lambda_kin,
        config.system.beta,
    )
    exact = (
        1.0 / (2.0 * config.system.beta)
        - config.system.box_length**2
        * winding.mean_square
        / (
            4.0
            * config.system.lambda_kin
            * config.system.beta**2
        )
    )
    six_standard_errors = 6.0 * float(np.std(sample, ddof=1)) / np.sqrt(sample.size)

    assert float(np.mean(sample)) == pytest.approx(
        exact,
        abs=six_standard_errors,
    )


def test_energy_rejects_open_sector() -> None:
    config = SimulationConfig.from_toml(CONFIG_DIR / "closed_worldline.toml")
    state = Configuration.from_straight_worldlines(
        particle_count=1,
        n_slices=config.discretization.n_slices,
        ndim=1,
        box_length=config.system.box_length,
    )
    snapshot = state.snapshot()
    snapshot["sector"] = "G"
    snapshot["worm_head"] = 0
    snapshot["worm_tail"] = 1
    snapshot["next_of"][0] = -1
    snapshot["prev_of"][1] = -1
    snapshot["image_to_next"][0] = 0
    open_state = Configuration.from_snapshot(snapshot)

    with pytest.raises(ValueError, match="requires sector Z"):
        thermodynamic_energy(open_state, PrimitiveTargetMeasure(config))
