"""Density normalization, correlated opportunities and auxiliary continuation."""

from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from wormpimc import Configuration, Simulation, SimulationConfig
from wormpimc.estimators import DensityProfile


def fixed_state(positions):
    return Configuration.from_closed_worldlines(
        [np.full((9, 1), x) for x in positions], 4.0
    )


def empty_z(config):
    sim = Simulation(config)
    for _ in range(2000):
        sim.step()
        if sim.state.sector.value == "Z" and sim.state.n_particles == 0:
            return Configuration.from_snapshot(sim.state.snapshot())
    raise AssertionError("test did not find empty Z")


def open_state(config):
    sim = Simulation(config)
    for _ in range(200):
        sim.step()
        if sim.state.sector.value == "G":
            return Configuration.from_snapshot(sim.state.snapshot())
    raise AssertionError("test did not find G")


def test_slice_average_periodic_bins_empty_z_and_g_reference():
    config = SimulationConfig.small_system(n_slices=8)
    density = DensityProfile(config, spatial_bins=4)
    # Half the slices in bin 0 and half in bin 1; exactly one MC observation.
    state = Configuration.from_closed_worldline(
        np.array([0.25, 1.25] * 4 + [0.25])[:, None], 4
    )
    density.measure(state)
    density.measure(fixed_state([0.0, 3.999]))
    density.measure(empty_z(config))
    density.measure(open_state(config))
    result = density.results()
    assert result["n_measurement_opportunities"] == 4
    assert result["n_z_measurements"] == 3
    assert [row["mean"] for row in result["rows"]] == pytest.approx(
        [0.5, 1 / 6, 0, 1 / 3]
    )
    assert result["integrated_density"] == pytest.approx(1)
    assert all(row["standard_error"] is None for row in result["rows"])
    assert density.snapshot()["blocking"][0]["levels"][0]["n_blocks"] == 4
    density = DensityProfile(config)
    density.measure(open_state(config))
    assert density.results()["integrated_density"] is None


def test_observer_stride_passive_measurement_and_paired_resume(tmp_path):
    config = SimulationConfig.small_system(
        n_slices=8,
        external="fourier",
        external_offset=0.5,
        external_cosine=[-0.5],
        warmup_sweeps=3,
        measurement_sweeps=101,
        steps_per_sweep=8,
    )
    config = replace(config, run=replace(config.run, measurement_stride=3))
    observed, plain = Simulation(config), Simulation(config)
    full = DensityProfile(config, spatial_bins=8)
    observed.advance(after_sweep=full.after_sweep)
    plain.advance()
    assert full.measurement_count == 33
    assert observed.state.state_digest(
        include_revision=True
    ) == plain.state.state_digest(include_revision=True)
    assert observed.rng.bit_generator.state == plain.rng.bit_generator.state
    assert observed.results() == plain.results()
    assert full.results()["integrated_density"] == pytest.approx(
        observed.results()["scalars"]["particle_number"]["mean"]
    )
    partial = Simulation(config)
    density = DensityProfile(config, spatial_bins=8)
    partial.advance(max_sweeps=23, after_sweep=density.after_sweep)
    checkpoint = partial.checkpoint(tmp_path / "simulation.npz")
    restored = Simulation.from_checkpoint(checkpoint)
    resumed = DensityProfile(restored.config, spatial_bins=8)
    resumed.restore(json.loads(json.dumps(density.snapshot())))
    restored.advance(after_sweep=resumed.after_sweep)
    assert resumed.snapshot() == full.snapshot()
    assert resumed.results() == full.results()


def test_density_blocking_retains_repeats_and_sparse_bins():
    config = SimulationConfig.small_system(n_slices=8)
    iid, repeated = (
        DensityProfile(config, spatial_bins=4),
        DensityProfile(config, spatial_bins=4),
    )
    rng = np.random.default_rng(74)
    raw = rng.integers(0, 2, size=1024)
    for value in raw:
        state = fixed_state([0.25 + value])
        iid.measure(state)
        for _ in range(16):
            repeated.measure(state)
    a, b = iid.results()["rows"][0], repeated.results()["rows"][0]
    assert a["mean"] == b["mean"]
    assert b["standard_error"] is not None
    # 16x more recorded repeats must not give a 4x smaller error.
    assert b["standard_error"] > 0.7 * a["standard_error"]
    assert iid.results()["rows"][3]["standard_error"] is None
    assert (
        iid.results()["rows"][3]["uncertainty_status"]
        == "insufficient_numerator_events"
    )


def test_density_ratio_covariance_matches_independent_residual_formula():
    config = SimulationConfig.small_system(n_slices=8)
    density = DensityProfile(config, spatial_bins=4)
    states = [fixed_state([0.25, 0.5]), fixed_state([1.25]), open_state(config)]
    choices = np.random.default_rng(48).integers(0, 3, size=1024)
    for i in choices:
        density.measure(states[i])
    y = np.where(choices == 0, 2.0, 0.0)
    z = (choices != 2).astype(float)
    ratio = y.mean() / z.mean()
    expected = np.std(y - ratio * z, ddof=1) / (np.sqrt(len(z)) * z.mean())
    assert density._blocking[0].estimates()[0].standard_error == pytest.approx(expected)


def test_restore_rejects_different_models_geometry_and_corrupt_counts():
    config = SimulationConfig.small_system(n_slices=8)
    density = DensityProfile(config, spatial_bins=4)
    density.measure(fixed_state([0.2]))
    other = replace(config, system=replace(config.system, chemical_potential=-3))
    with pytest.raises(ValueError, match="incompatible"):
        DensityProfile(other, spatial_bins=4).restore(density.snapshot())
    bad = copy.deepcopy(density.snapshot())
    bad["nonzero_opportunities"][0] = 2
    with pytest.raises(ValueError, match="inconsistent"):
        density.restore(bad)
    with pytest.raises(ValueError, match="geometry"):
        density.measure(Configuration.from_closed_worldline(np.full((5, 1), 0.2), 4))
    for bins in (0, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            DensityProfile(config, spatial_bins=bins)
