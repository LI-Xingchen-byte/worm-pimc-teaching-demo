"""Independent one-body action checks, model provenance and continuation."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tomllib

import numpy as np
import pytest

from wormpimc import ConfigError, Configuration, Simulation, SimulationConfig
from wormpimc.config import MOVE_NAMES
from wormpimc.estimators import thermodynamic_energy
from wormpimc.measure import PrimitiveTargetMeasure, assert_local_matches_full
from wormpimc.output import initial_metadata
from wormpimc.potentials import FourierExternalPotential, potential_from_config


def field_config(**kwargs):
    return SimulationConfig.small_system(
        n_slices=8,
        external="fourier",
        external_offset=0.5,
        external_cosine=(-0.5, 0.1),
        external_sine=(0.2,),
        **kwargs,
    )


def test_fourier_phase_harmonics_periodicity_empty_slice_and_pair_composition():
    field = FourierExternalPotential(
        box_length=4, offset=0.7, cosine=(0.3, -0.2), sine=(0.6,)
    )
    x = np.array([-1.0, 0.0, 0.5, 2.0, 4.0, 9.0])
    expected = (
        0.7
        + 0.3 * np.cos(np.pi * x / 2)
        - 0.2 * np.cos(np.pi * x)
        + 0.6 * np.sin(np.pi * x / 2)
    )
    assert field.energy(x) == pytest.approx(expected)
    assert field.energy(x + 12) == pytest.approx(expected)
    assert field.energy(0.5) == pytest.approx(expected[2])
    assert field.total_energy(np.empty((0, 1))) == 0
    config = field_config(pair="periodized_gaussian", epsilon=1, sigma=0.5)
    combined = potential_from_config(config)
    points = np.array([[0.1], [1.5], [3.9]])
    pairs = sum(
        combined.pair_energy(points[i] - points[j])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    assert combined.total_energy(points) == pytest.approx(
        pairs + combined.external.total_energy(points)
    )
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            field.energy(bad)
    with pytest.raises(ValueError, match="shape"):
        field.total_energy(np.zeros((2, 2)))


@pytest.mark.parametrize(
    "changes",
    [
        {"external_cosine": "cos(x)"},
        {"external_sine": [True]},
        {"external_cosine": [[1.0]]},
        {"external_offset": float("nan")},
        {"external_cosine": [float("inf")]},
        {"external": "callable"},
        {"external_offset": 1e308, "external_cosine": [1e308]},
    ],
)
def test_bad_external_parameters_are_rejected(changes):
    config = field_config()
    data = config.input_dict()
    data["potential"].update(changes)
    with pytest.raises(ConfigError):
        SimulationConfig.from_mapping(data)


def test_immutable_coefficients_round_trip_hash_and_metadata():
    cosine = [-0.3, 0.2]
    config = SimulationConfig.small_system(external="fourier", external_cosine=cosine)
    cosine[0] = 99
    assert config.potential.external_cosine == (-0.3, 0.2)
    assert SimulationConfig.from_mapping(tomllib.loads(config.to_toml())) == config
    assert (
        replace(
            config, potential=replace(config.potential, external_sine=(0.1,))
        ).config_hash
        != config.config_hash
    )
    assert (
        initial_metadata(config)["potential_parameters"]
        == config.resolved_dict()["potential"]
    )
    assert initial_metadata(config)["external_potential"] == "fourier"
    with pytest.raises(ConfigError, match="not allowed"):
        SimulationConfig.small_system(external_cosine=[1.0])
    # Historical no-field hashes are part of checkpoint compatibility.
    root = Path(__file__).resolve().parents[1]
    legacy = SimulationConfig.from_toml(root / "configs/ideal_bose_gas.toml")
    assert (
        legacy.config_hash
        == "7462de91a42dda8d6a9b25ab144336348e00dfab71bfdd581a2cdbf9cc833bb6"
    )


def test_conservative_stability_includes_offset_and_both_harmonic_phases():
    # c0 - hypot(3,4) = -3, rather than c0 - max(a,b) or c0 alone.
    with pytest.raises(ConfigError, match="conservative Fourier lower bound"):
        SimulationConfig.small_system(
            external="fourier",
            external_offset=2,
            external_cosine=[3],
            external_sine=[4],
            chemical_potential=-3,
        )
    SimulationConfig.small_system(
        external="fourier",
        external_offset=2,
        external_cosine=[3],
        external_sine=[4],
        chemical_potential=-3.1,
    )
    # A positive constant raises the allowed chemical potential; a negative one lowers it.
    SimulationConfig.small_system(
        external="fourier", external_offset=2, chemical_potential=1
    )
    with pytest.raises(ConfigError, match="lower bound"):
        SimulationConfig.small_system(
            external="fourier", external_offset=-2, chemical_potential=-1
        )


def independent_external_action(state, config):
    """Sum one-body half weights over links, without calling potential/measure helpers."""

    def value(x):
        phase = 2 * math.pi * float(x) / state.box_length
        return (
            config.potential.external_offset
            + math.fsum(
                a * math.cos(n * phase)
                for n, a in enumerate(config.potential.external_cosine, 1)
            )
            + math.fsum(
                b * math.sin(n * phase)
                for n, b in enumerate(config.potential.external_sine, 1)
            )
        )

    return (
        config.tau
        / 2
        * math.fsum(
            value(state.positions[i, 0]) + value(state.positions[state.next_of[i], 0])
            for i in state.active_ids
            if state.next_of[i] >= 0
        )
    )


def test_all_move_actions_and_reverse_patches_include_external_endpoint_halves():
    config = field_config(
        pair="periodized_gaussian",
        epsilon=1,
        sigma=0.5,
        chemical_potential=-0.7,
        seed=2718,
    )
    data = config.input_dict()
    data["potential"] = {
        key: value
        for key, value in data["potential"].items()
        if not key.startswith("external")
    }
    pair_target = PrimitiveTargetMeasure(SimulationConfig.from_mapping(data))
    sim = Simulation(config, verify_local_delta=True)
    proposed = set()
    for _ in range(1200):
        event = sim.step(trace=True)
        candidate = event.candidate()
        if candidate is None:
            continue
        proposed.add(event.move_name)
        before = Configuration.from_snapshot(event.before)
        for state in (before, candidate):
            assert sim.target.full_components(
                state
            ).potential - pair_target.full_components(state).potential == pytest.approx(
                -independent_external_action(state, config), abs=1e-11
            )
        local = sim.target.delta_local(before, event.patch)
        assert_local_matches_full(local, sim.target.delta_full(before, event.patch))
        reverse = event.patch.reversed(base_revision=candidate.revision)
        assert local.total + sim.target.delta_local(
            candidate, reverse
        ).total == pytest.approx(0, abs=1e-10)
    assert proposed == set(MOVE_NAMES)


def test_constant_field_mu_shift_and_energy_derivative_in_both_sectors():
    base = SimulationConfig.small_system(n_slices=8, chemical_potential=-1, seed=9)
    constant = replace(
        base,
        potential=replace(
            base.potential,
            external="fourier",
            external_offset=0.4,
            external_cosine=(),
            external_sine=(),
        ),
    )
    shifted = replace(base, system=replace(base.system, chemical_potential=-1.4))
    sim = Simulation(constant)
    shifted_target = PrimitiveTargetMeasure(shifted)
    visited = set()
    for _ in range(100):
        sim.step()
        visited.add(sim.state.sector.value)
        assert sim.target.full_components(sim.state).total == pytest.approx(
            shifted_target.full_components(sim.state).total
        )
    assert visited == {"Z", "G"}
    closed = Simulation(base).state
    original_energy = thermodynamic_energy(closed, PrimitiveTargetMeasure(base))
    constant_energy = thermodynamic_energy(closed, PrimitiveTargetMeasure(constant))
    assert constant_energy.total - original_energy.total == pytest.approx(
        0.4 * closed.n_particles
    )
    assert constant_energy.kinetic == original_energy.kinetic
    state = Simulation(field_config()).state
    target = PrimitiveTargetMeasure(field_config())
    beta = target.config.system.beta
    plus = replace(
        target.config, system=replace(target.config.system, beta=beta + 1e-6)
    )
    minus = replace(
        target.config, system=replace(target.config.system, beta=beta - 1e-6)
    )
    derivative = (
        PrimitiveTargetMeasure(plus).full_components(state).total
        - PrimitiveTargetMeasure(minus).full_components(state).total
    ) / 2e-6
    energy = thermodynamic_energy(state, target)
    assert energy.total == pytest.approx(
        -derivative + target.config.system.chemical_potential * state.n_particles,
        abs=1e-8,
    )


def test_zero_fourier_preserves_original_trajectory():
    config = SimulationConfig.small_system(n_slices=8, seed=77)
    zero = replace(config, potential=replace(config.potential, external="fourier"))
    original, with_zero = Simulation(config), Simulation(zero)
    for _ in range(200):
        assert original.step() == with_zero.step()
    assert original.rng.bit_generator.state == with_zero.rng.bit_generator.state
    assert original.state.state_digest(
        include_revision=True
    ) == with_zero.state.state_digest(include_revision=True)


def test_external_run_provenance_and_checkpoint_continuation(tmp_path):
    config = field_config(warmup_sweeps=2, measurement_sweeps=12, steps_per_sweep=8)
    complete = Simulation(config)
    complete.advance()
    partial = Simulation(config)
    partial.advance(max_sweeps=5)
    checkpoint = partial.checkpoint(tmp_path / "partial.npz")
    restored = Simulation.from_checkpoint(checkpoint, verify_local_delta=True)
    restored.advance()
    assert complete.state.state_digest(
        include_revision=True
    ) == restored.state.state_digest(include_revision=True)
    assert complete.rng.bit_generator.state == restored.rng.bit_generator.state
    assert complete.estimators.snapshot() == restored.estimators.snapshot()
    run = Simulation(config).run(output_root=tmp_path / "runs")
    metadata = json.loads(
        (run.run_directory / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["external_potential"] == "fourier"
    assert SimulationConfig.from_toml(run.run_directory / "input.toml") == config
