"""Independent reference and actual-candidate proposal checks for P1."""

from dataclasses import replace
import math

import numpy as np
import pytest

from validation.ideal_reference import ideal_bose_reference
from validation.proposal_audit import REVERSE, densities, ideal_log_weight, swap_table
from validation.run_p1 import analyze
from wormpimc import Simulation, SimulationConfig
from wormpimc.configuration import Configuration


def test_reference_ground_mode_and_probability_tail():
    result = ideal_bose_reference(
        box_length=0.1, chemical_potential=-math.log(2), number_cutoff=12
    )
    assert result["values"]["N"] == pytest.approx(1)
    assert result["values"]["N2"] == pytest.approx(3)
    assert result["values"]["E"] == 0
    assert result["values"]["exchange_particles"] == pytest.approx(0.5)
    assert result["number_probabilities"] == pytest.approx(
        [0.5 ** (i + 1) for i in range(13)]
    )
    assert result["number_tail_probability"] == pytest.approx(0.5**13)


def test_reference_first_canonical_coefficients_and_mode_tail_bound():
    short = ideal_bose_reference(mode_cutoff=1)
    long = ideal_bose_reference(mode_cutoff=16)
    q = np.exp(-0.7 - 0.5 * (2 * np.pi * np.arange(-1, 2) / 4) ** 2)
    p0 = np.prod(1 - q)
    assert short["number_probabilities"][:3] == pytest.approx(
        [p0, p0 * q.sum(), p0 * (q.sum() ** 2 + (q * q).sum()) / 2]
    )
    for name, bound in (
        ("N", "omitted_mean_N_upper_bound"),
        ("E", "omitted_energy_upper_bound"),
    ):
        difference = long["values"][name] - short["values"][name]
        assert 0 < difference <= short[bound]
    assert ideal_bose_reference(mode_cutoff=32)["values"] == pytest.approx(
        long["values"]
    )


@pytest.mark.parametrize(
    "parameters",
    [
        dict(beta=0),
        dict(chemical_potential=0),
        dict(box_length=float("nan")),
        dict(mode_cutoff=0),
        dict(number_cutoff=3),
    ],
)
def test_reference_rejects_invalid_domain(parameters):
    with pytest.raises(ValueError):
        ideal_bose_reference(**parameters)


def test_sparse_or_absent_measurements_cannot_pass():
    references = ideal_bose_reference()["values"]
    for series in (
        np.zeros((0, 11)),
        np.zeros((512, 11)),
        np.tile([1] + [0] * 10, (512, 1)),
    ):
        assert all(
            row["status"] == "INCONCLUSIVE"
            for row in analyze(series, references).values()
        )


@pytest.mark.parametrize("smax", [2, 3])
def test_real_candidates_have_independently_reconstructed_densities(smax):
    config = SimulationConfig.small_system(
        seed=1649, n_slices=8, chemical_potential=-0.7
    )
    config = replace(
        config,
        moves=replace(
            config.moves,
            max_segment_links=smax,
            weights=replace(config.moves.weights, open=2.0, remove=2.0, advance=2.0),
        ),
    )
    simulation = Simulation(config)
    counts = dict.fromkeys(REVERSE, 0)
    time_wrap = False
    changed_head = False
    empty_z = False
    unit_open = False
    spatial_wrap = False
    for _ in range(6000):
        result = simulation.step(trace=True)
        if result.move_name not in REVERSE or result.proposal_ratio is None:
            continue
        before = Configuration.from_snapshot(result.before)
        empty_z |= before.sector.value == "Z" and before.number_of_beads == 0
        after = result.candidate()
        assert after is not None
        qf, qr = densities(result.move_name, before, after, config)
        actual = result.proposal_ratio
        assert actual.log_q_forward == pytest.approx(qf, abs=2e-10)
        assert actual.log_q_reverse == pytest.approx(qr, abs=2e-10)
        weights = config.moves.weights.as_dict()
        selection = math.log(
            weights[REVERSE[result.move_name]] / weights[result.move_name]
        )
        expected = (
            ideal_log_weight(after, config)
            - ideal_log_weight(before, config)
            + qr
            - qf
            + selection
        )
        assert result.breakdown.log_ratio == pytest.approx(expected, abs=2e-10)
        counts[result.move_name] += 1
        if result.move_name == "open":
            unit_open |= before.number_of_beads == after.number_of_beads
        if result.move_name == "swap":
            table = simulation.sampler._moves["swap"]._candidates(before)
            assert set(table[0]) == set(swap_table(before, config))
            changed_head |= before.worm_head != after.worm_head
        for change in result.patch.link_changes:
            if change.after_next >= 0 and after.slice_of[change.bead_id] == 7:
                time_wrap = True
            spatial_wrap |= any(change.after_image)
    assert min(counts.values()) >= 3, counts
    assert time_wrap and spatial_wrap and changed_head and empty_z and unit_open
