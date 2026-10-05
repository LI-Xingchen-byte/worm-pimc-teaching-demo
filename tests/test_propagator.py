from __future__ import annotations

import math

import numpy as np
import pytest

from wormpimc.propagator import (
    brownian_bridge_log_density,
    brownian_bridge_moments,
    free_log_density,
    image_resolved_log_density,
    periodic_log_density,
    sample_brownian_bridge,
    sample_free_walk,
    sample_periodic_bridge,
    winding_distribution,
)


def test_free_log_density_matches_analytic_formula() -> None:
    displacement = np.array([0.75])
    lambda_kin = 0.5
    imaginary_time = 0.25
    expected = -0.5 * math.log(
        4.0 * math.pi * lambda_kin * imaginary_time
    ) - displacement[0] ** 2 / (4.0 * lambda_kin * imaginary_time)

    assert free_log_density(
        displacement, lambda_kin, imaginary_time
    ) == pytest.approx(expected)


def test_image_resolved_log_density_uses_requested_periodic_image() -> None:
    short = image_resolved_log_density(9.0, 1.0, 1, 10.0, 0.5, 0.25)
    long = image_resolved_log_density(9.0, 1.0, 0, 10.0, 0.5, 0.25)

    assert short > long
    assert short == pytest.approx(free_log_density([2.0], 0.5, 0.25))


def test_brownian_bridge_is_reproducible_and_keeps_endpoints() -> None:
    first = sample_brownian_bridge(
        [0.25], [2.25], 8, 0.5, 0.125, np.random.default_rng(314)
    )
    second = sample_brownian_bridge(
        [0.25], [2.25], 8, 0.5, 0.125, np.random.default_rng(314)
    )

    assert np.array_equal(first, second)
    assert first[0] == pytest.approx([0.25])
    assert first[-1] == pytest.approx([2.25])


def test_bridge_log_density_equals_product_of_links_over_endpoint_density() -> None:
    lambda_kin = 0.5
    tau = 0.125
    path = sample_brownian_bridge(
        [-0.5], [1.0], 7, lambda_kin, tau, np.random.default_rng(19)
    )
    link_log_density = math.fsum(
        free_log_density(path[index + 1] - path[index], lambda_kin, tau)
        for index in range(path.shape[0] - 1)
    )
    endpoint_log_density = free_log_density(
        path[-1] - path[0], lambda_kin, 7 * tau
    )

    assert brownian_bridge_log_density(
        path, lambda_kin, tau
    ) == pytest.approx(link_log_density - endpoint_log_density)


def test_brownian_bridge_midpoint_statistics_match_analytic_moments() -> None:
    sample_count = 6_000
    n_links = 8
    index = 3
    lambda_kin = 0.5
    tau = 0.2
    start = np.array([-0.25])
    end = np.array([1.75])
    rng = np.random.default_rng(20260830)
    observations = np.empty(sample_count)
    for sample_index in range(sample_count):
        observations[sample_index] = sample_brownian_bridge(
            start, end, n_links, lambda_kin, tau, rng
        )[index, 0]

    means, variances = brownian_bridge_moments(
        start, end, n_links, lambda_kin, tau
    )
    mean_error = math.sqrt(variances[index] / sample_count)
    variance_error = variances[index] * math.sqrt(2.0 / (sample_count - 1))

    assert abs(np.mean(observations) - means[index, 0]) < 5.0 * mean_error
    assert (
        abs(np.var(observations, ddof=1) - variances[index])
        < 5.0 * variance_error
    )


def test_winding_distribution_is_symmetric_normalized_and_controlled() -> None:
    distribution = winding_distribution(2.0, 0.5, 2.0)

    assert math.fsum(distribution.probabilities) == pytest.approx(1.0)
    assert distribution.probabilities == pytest.approx(
        distribution.probabilities[::-1]
    )
    zero = int(np.flatnonzero(distribution.values == 0)[0])
    one = int(np.flatnonzero(distribution.values == 1)[0])
    assert (
        distribution.probabilities[one] / distribution.probabilities[zero]
    ) == pytest.approx(math.exp(-1.0))
    assert distribution.omitted_probability_bound <= 1.0e-14


def test_one_link_bridge_has_no_random_intermediate_points() -> None:
    path = sample_brownian_bridge(
        [0.0], [1.0], 1, 0.5, 0.25, np.random.default_rng(5)
    )

    assert np.array_equal(path, np.array([[0.0], [1.0]]))
    assert brownian_bridge_log_density(path, 0.5, 0.25) == 0.0


def test_periodic_density_matches_direct_image_sum() -> None:
    start = np.array([1.7])
    end = np.array([0.2])
    length = 3.0
    lam = 0.5
    time = 0.8
    direct = math.fsum(
        math.exp(
            image_resolved_log_density(
                start,
                end,
                [image],
                length,
                lam,
                time,
            )
        )
        for image in range(-20, 21)
    )

    assert periodic_log_density(start, end, length, lam, time) == pytest.approx(
        math.log(direct), abs=1.0e-13
    )


def test_periodic_bridge_density_equals_links_over_periodic_endpoint() -> None:
    sample = sample_periodic_bridge(
        [2.8],
        [0.3],
        5,
        3.0,
        0.5,
        0.2,
        np.random.default_rng(2026),
    )
    link_sum = math.fsum(
        image_resolved_log_density(
            sample.wrapped_path[index],
            sample.wrapped_path[index + 1],
            sample.link_images[index],
            3.0,
            0.5,
            0.2,
        )
        for index in range(5)
    )
    endpoint = periodic_log_density([2.8], [0.3], 3.0, 0.5, 1.0)

    assert sample.log_proposal_density == pytest.approx(link_sum - endpoint)
    assert np.sum(sample.link_images, axis=0) == pytest.approx(
        sample.total_image
    )


def test_free_walk_records_its_normalized_link_density() -> None:
    sample = sample_free_walk(
        [1.9], 6, 2.0, 0.5, 0.1, np.random.default_rng(77)
    )
    direct = math.fsum(
        image_resolved_log_density(
            sample.wrapped_path[index],
            sample.wrapped_path[index + 1],
            sample.link_images[index],
            2.0,
            0.5,
            0.1,
        )
        for index in range(6)
    )

    assert sample.log_proposal_density == pytest.approx(direct)
    assert np.all(sample.wrapped_path >= 0.0)
    assert np.all(sample.wrapped_path < 2.0)
