from dataclasses import replace

import numpy as np
import pytest

from wormpimc import Configuration, Simulation, SimulationConfig
from wormpimc.estimators import EqualTimeCorrelations, equal_time_sample
from validation.correlation_reference import ideal_correlations


def fixed_state(positions):
    return Configuration.from_closed_worldlines(
        [np.full((9, 1), x) for x in positions], 4.0
    )


def test_ordered_pairs_periodic_wrap_and_fourier_self_terms():
    state = fixed_state([0.25, 3.75])
    row = equal_time_sample(state)
    assert row[:3].tolist() == [1, 2, 2]
    assert row[3:11].tolist() == [0, 1, 0, 0, 0, 0, 0, 1]
    k = 2 * np.pi * np.array([1, 2, 3]) / 4
    assert row[11:] == pytest.approx(2 + 2 * np.cos(k * 0.5))
    shifted = equal_time_sample(fixed_state([0.75, 0.25]))
    assert shifted == pytest.approx(row)
    assert equal_time_sample(fixed_state([0.2]))[11:] == pytest.approx([1, 1, 1])
    # Coincident distinct particles remain pairs; only the identity diagonal is removed.
    assert equal_time_sample(fixed_state([0.2, 0.2]))[3] == 2


def test_slices_are_averaged_as_one_observation():
    a = np.zeros((9, 1))
    b = np.array([0.5, 1.5] * 4 + [0.5])[:, None]
    row = equal_time_sample(Configuration.from_closed_worldlines([a, b], 4.0))
    expected = (
        equal_time_sample(fixed_state([0, 0.5]))
        + equal_time_sample(fixed_state([0, 1.5]))
    ) / 2
    assert row == pytest.approx(expected)


def test_denominator_fluctuations_sum_rule_and_block_covariance():
    config = SimulationConfig.small_system(n_slices=8)
    collector = EqualTimeCorrelations(config)
    rng = np.random.default_rng(72)
    # Artificial independent mixture, with G opportunities interspersed.
    vectors = [
        equal_time_sample(fixed_state([0.1, 0.6])),
        equal_time_sample(fixed_state([0.7])),
        np.zeros(14),
    ]
    raw = np.array([vectors[i] for i in rng.integers(0, 3, size=4096)])
    snapshot = collector.snapshot()
    snapshot["samples"] = raw.tolist()
    collector.restore(snapshot)
    result = collector.results()
    z, n = raw[:, 0].mean(), raw[:, 1].mean()
    assert result["g2"][1]["mean"] == pytest.approx(8 * raw[:, 4].mean() * z / n**2)
    assert np.mean([row["mean"] for row in result["g2"]]) == pytest.approx(
        result["g2_spatial_average_sum_rule"]
    )
    # Independent finite-difference gradient, no copied analytic derivatives.
    mean = raw.mean(axis=0)

    def observable(v):
        return 8 * v[4] * v[0] / v[1] ** 2

    gradient = []
    for i in range(raw.shape[1]):
        shift = np.zeros(raw.shape[1])
        shift[i] = 1e-5
        gradient.append((observable(mean + shift) - observable(mean - shift)) / 2e-5)
    se = np.sqrt(np.array(gradient) @ np.cov(raw, rowvar=False) @ gradient / len(raw))
    assert result["g2"][1]["blocking_levels"][0]["standard_error"] == pytest.approx(
        se, rel=1e-8
    )
    # Exact whole-bin normalization differs from per-state normalization.
    assert result["g2"][1]["mean"] != pytest.approx(
        np.mean([8 * row[4] / row[1] ** 2 for row in raw if row[1] > 0]), abs=0.05
    )


def test_observation_and_snapshot_do_not_change_rng_or_sampling(tmp_path):
    config = SimulationConfig.small_system(
        n_slices=8, warmup_sweeps=3, measurement_sweeps=100, seed=27
    )
    simulation, baseline = Simulation(config), Simulation(config)
    collector = EqualTimeCorrelations(config)

    def observe(sim):
        if (
            sim.phase == "measurement"
            and sim.measurement_completed % config.run.measurement_stride == 0
        ):
            collector.measure(sim.state)

    simulation.advance(max_sweeps=41, after_sweep=observe)
    snapshot = collector.snapshot()
    simulation.checkpoint(tmp_path / "main.npz")
    resumed = Simulation.from_checkpoint(tmp_path / "main.npz")
    restored = EqualTimeCorrelations(config)
    restored.restore(snapshot)
    resumed.advance(after_sweep=lambda sim: restored.measure(sim.state))
    simulation.advance(after_sweep=observe)
    baseline.advance()
    assert restored.snapshot() == collector.snapshot()
    assert len(collector.samples) == 100
    assert simulation.state.state_digest() == baseline.state.state_digest()
    assert simulation.rng.bit_generator.state == baseline.rng.bit_generator.state
    assert simulation.results() == baseline.results()
    assert np.any(np.all(collector.samples == 0, axis=1))  # G retained.
    assert np.any(
        (collector.samples[:, 0] == 1) & (collector.samples[:, 1] == 0)
    )  # Empty Z retained.
    snapshot["samples"][0][0] = 2
    with pytest.raises(ValueError):
        restored.restore(snapshot)


def test_no_data_and_sparse_pairs_do_not_publish_errors():
    collector = EqualTimeCorrelations(SimulationConfig.small_system(n_slices=8))
    assert collector.results()["g2"][0]["mean"] is None
    for _ in range(256):
        collector.measure(fixed_state([0.1]))
    assert all(
        row["mean"] == 0 and row["standard_error"] is None
        for row in collector.results()["g2"]
    )
    assert all(
        row["mean"] == pytest.approx(1)
        for row in collector.results()["structure_factor"]
    )


def test_reference_limits_bin_average_and_sum_rule():
    result = ideal_correlations()
    reference = result["reference_truncation"]["values"]
    assert np.mean(result["g2"]) == pytest.approx(
        (reference["N2"] - reference["N"]) / reference["N"] ** 2
    )
    assert result["g2"] == pytest.approx(result["g2"][::-1])
    assert result["g2"][0] < 2  # Finite bin != g2(0).
    assert (
        result["structure_factor"][0]
        > result["structure_factor"][1]
        > result["structure_factor"][2]
        > 1
    )
    assert ideal_correlations(mode_cutoff=32)["g2"] == pytest.approx(result["g2"])
    single = ideal_correlations(box_length=0.1)
    assert single["g2"] == pytest.approx([2] * 8)
    assert single["structure_factor"] == pytest.approx([1] * 3)


def test_blocking_accounts_for_repeated_correlated_configurations():
    config = SimulationConfig.small_system(n_slices=8)
    collector = EqualTimeCorrelations(config)
    choices = [
        equal_time_sample(fixed_state([0.1, 0.6])),
        equal_time_sample(fixed_state([0.7])),
        np.zeros(14),
    ]
    rng = np.random.default_rng(221)
    independent = np.array([choices[i] for i in rng.integers(0, 3, 2048)])
    snapshot = collector.snapshot()
    snapshot["samples"] = np.repeat(independent, 16, axis=0).tolist()
    collector.restore(snapshot)
    row = collector.results()["g2"][1]
    assert row["uncertainty_status"] == "blocking_plateau"
    assert row["standard_error"] > 2.5 * row["blocking_levels"][0]["standard_error"]


@pytest.mark.parametrize(
    "arguments",
    [dict(spatial_bins=0), dict(modes=(0,)), dict(modes=(-1,)), dict(modes=(1, 1))],
)
def test_invalid_grid(arguments):
    with pytest.raises(ValueError):
        EqualTimeCorrelations(SimulationConfig.small_system(), **arguments)


def test_incompatible_geometry_or_snapshot():
    config = SimulationConfig.small_system(n_slices=16)
    collector = EqualTimeCorrelations(config)
    with pytest.raises(ValueError):
        collector.measure(fixed_state([0.1]))
    other = EqualTimeCorrelations(
        replace(config, system=replace(config.system, beta=2.0))
    )
    with pytest.raises(ValueError):
        other.restore(collector.snapshot())
