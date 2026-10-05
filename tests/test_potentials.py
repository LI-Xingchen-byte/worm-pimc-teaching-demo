from __future__ import annotations

import math

import numpy as np
import pytest

from wormpimc.potentials import PeriodizedGaussianPotential, ZeroPotential


def test_zero_potential_accepts_general_dimension() -> None:
    potential = ZeroPotential()

    assert potential.total_energy(np.zeros((3, 2))) == 0.0
    assert potential.pair_energy([100.0, -7.0]) == 0.0


def test_periodized_gaussian_matches_long_image_sum() -> None:
    potential = PeriodizedGaussianPotential(
        epsilon=1.7,
        sigma=0.8,
        box_length=4.0,
        image_tolerance=1.0e-14,
    )
    displacement = 1.3
    expected = 1.7 * math.fsum(
        math.exp(-((displacement + image * 4.0) ** 2) / (2.0 * 0.8**2))
        for image in range(-20, 21)
    )

    assert potential.pair_energy(displacement) == pytest.approx(
        expected, rel=2.0e-14
    )


def test_periodized_gaussian_is_periodic_and_reports_tail_bound() -> None:
    potential = PeriodizedGaussianPotential(
        epsilon=1.0,
        sigma=0.5,
        box_length=3.0,
        image_tolerance=1.0e-12,
    )

    assert potential.pair_energy(0.7) == pytest.approx(
        potential.pair_energy(3.7), rel=1.0e-14
    )
    _, relative_bound = potential.image_diagnostics(0.7)
    assert relative_bound <= 1.0e-12


def test_total_energy_counts_each_unordered_pair_once() -> None:
    potential = PeriodizedGaussianPotential(
        epsilon=1.0,
        sigma=0.4,
        box_length=5.0,
        image_tolerance=1.0e-13,
    )
    positions = np.array([[0.0], [1.0], [3.0]])
    expected = (
        potential.pair_energy(-1.0)
        + potential.pair_energy(-3.0)
        + potential.pair_energy(-2.0)
    )

    assert potential.total_energy(positions) == pytest.approx(expected)
