from __future__ import annotations

import math

import pytest

from wormpimc.worm_measure import (
    WormMeasureConvention,
    advance_ratio,
    close_ratio,
    insert_ratio,
    open_ratio,
    recede_ratio,
    remove_ratio,
    swap_ratio,
)


def test_worm_measure_coefficient_uses_dimensionless_tuning_weight() -> None:
    convention = WormMeasureConvention(
        volume=7.5,
        n_slices=32,
        max_segment_links=8,
        worm_sector_weight=1.25,
    )

    assert convention.coefficient == pytest.approx(1.25 / (7.5 * 32 * 8))
    assert convention.log_coefficient == pytest.approx(
        math.log(convention.coefficient)
    )


@pytest.mark.parametrize("seed_offset", range(12))
def test_open_close_reference_ratios_are_exact_inverses(
    seed_offset: int,
) -> None:
    segment_log = -2.7 + 0.13 * seed_offset
    endpoint_log = -1.4 - 0.07 * seed_offset
    delta_action = -0.8 + 0.11 * seed_offset
    mu = -0.9 + 0.03 * seed_offset
    coefficient = 0.0025
    p_open = 0.17
    p_close = 0.11

    forward = open_ratio(
        segment_log_kinetic=segment_log,
        endpoint_log_free_density=endpoint_log,
        delta_potential_action=delta_action,
        chemical_potential=mu,
        tau=0.0625,
        segment_links=5,
        diagonal_bead_count=96,
        max_segment_links=8,
        worm_measure_coefficient=coefficient,
        move_probability=p_open,
        reverse_move_probability=p_close,
    )
    reverse = close_ratio(
        segment_log_kinetic=segment_log,
        endpoint_log_free_density=endpoint_log,
        delta_potential_action=-delta_action,
        chemical_potential=mu,
        tau=0.0625,
        segment_links=5,
        final_diagonal_bead_count=96,
        max_segment_links=8,
        worm_measure_coefficient=coefficient,
        move_probability=p_close,
        reverse_move_probability=p_open,
    )

    assert forward.log_ratio + reverse.log_ratio == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert forward.target.total + reverse.target.total == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert forward.proposal.total + reverse.proposal.total == pytest.approx(
        0.0, abs=2.0e-14
    )


@pytest.mark.parametrize("seed_offset", range(12))
def test_insert_remove_reference_ratios_are_exact_inverses(
    seed_offset: int,
) -> None:
    segment_log = -3.1 + 0.09 * seed_offset
    delta_action = 0.5 - 0.08 * seed_offset
    mu = -1.2 + 0.04 * seed_offset
    convention = WormMeasureConvention(
        volume=4.0,
        n_slices=16,
        max_segment_links=6,
        worm_sector_weight=0.75,
    )
    p_insert = 0.08
    p_remove = 0.14

    forward = insert_ratio(
        segment_log_kinetic=segment_log,
        delta_potential_action=delta_action,
        chemical_potential=mu,
        tau=0.0625,
        segment_links=4,
        volume=convention.volume,
        n_slices=convention.n_slices,
        max_segment_links=convention.max_segment_links,
        worm_measure_coefficient=convention.coefficient,
        move_probability=p_insert,
        reverse_move_probability=p_remove,
    )
    reverse = remove_ratio(
        segment_log_kinetic=segment_log,
        delta_potential_action=-delta_action,
        chemical_potential=mu,
        tau=0.0625,
        segment_links=4,
        volume=convention.volume,
        n_slices=convention.n_slices,
        max_segment_links=convention.max_segment_links,
        worm_measure_coefficient=convention.coefficient,
        move_probability=p_remove,
        reverse_move_probability=p_insert,
    )

    assert forward.log_ratio + reverse.log_ratio == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert forward.target.total + reverse.target.total == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert forward.proposal.total + reverse.proposal.total == pytest.approx(
        0.0, abs=2.0e-14
    )


def test_insert_ratio_reduces_to_dimensionless_sector_weight() -> None:
    convention = WormMeasureConvention(
        volume=5.0,
        n_slices=20,
        max_segment_links=4,
        worm_sector_weight=1.7,
    )
    ratio = insert_ratio(
        segment_log_kinetic=-2.3,
        delta_potential_action=0.4,
        chemical_potential=-0.6,
        tau=0.05,
        segment_links=3,
        volume=convention.volume,
        n_slices=convention.n_slices,
        max_segment_links=convention.max_segment_links,
        worm_measure_coefficient=convention.coefficient,
        move_probability=0.2,
        reverse_move_probability=0.2,
    )

    expected = math.log(1.7) - 0.4 + (-0.6) * 0.05 * 3
    assert ratio.log_ratio == pytest.approx(expected, abs=1.0e-14)


def test_open_ratio_matches_documented_simplified_expression() -> None:
    coefficient = 0.003
    segment_log = -5.2
    endpoint_log = -1.8
    delta_action = 0.37
    mu = -0.45
    tau = 0.025
    links = 6
    bead_count = 128
    maximum = 9
    p_open = 0.13
    p_close = 0.21

    audit = open_ratio(
        segment_log_kinetic=segment_log,
        endpoint_log_free_density=endpoint_log,
        delta_potential_action=delta_action,
        chemical_potential=mu,
        tau=tau,
        segment_links=links,
        diagonal_bead_count=bead_count,
        max_segment_links=maximum,
        worm_measure_coefficient=coefficient,
        move_probability=p_open,
        reverse_move_probability=p_close,
    )

    expected = (
        math.log(coefficient * bead_count * maximum)
        - endpoint_log
        - delta_action
        - mu * tau * links
        + math.log(p_close / p_open)
    )
    assert audit.log_ratio == pytest.approx(expected, abs=1.0e-14)
    assert audit.target.kinetic == -segment_log
    assert audit.proposal.log_q_reverse == segment_log - endpoint_log


@pytest.mark.parametrize("offset", range(10))
def test_advance_recede_reference_ratios_are_exact_inverses(
    offset: int,
) -> None:
    segment_log = -4.0 + 0.17 * offset
    delta_action = -0.6 + 0.09 * offset
    mu = -0.7 + 0.02 * offset
    p_advance = 0.12
    p_recede = 0.08
    forward = advance_ratio(
        segment_log_kinetic=segment_log,
        delta_potential_action=delta_action,
        chemical_potential=mu,
        tau=0.04,
        segment_links=3,
        max_segment_links=7,
        move_probability=p_advance,
        reverse_move_probability=p_recede,
    )
    reverse = recede_ratio(
        segment_log_kinetic=segment_log,
        delta_potential_action=-delta_action,
        chemical_potential=mu,
        tau=0.04,
        segment_links=3,
        max_segment_links=7,
        move_probability=p_recede,
        reverse_move_probability=p_advance,
    )

    assert forward.log_ratio + reverse.log_ratio == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert forward.target.total + reverse.target.total == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert forward.proposal.total + reverse.proposal.total == pytest.approx(
        0.0, abs=2.0e-14
    )


@pytest.mark.parametrize("offset", range(10))
def test_swap_reference_ratio_is_self_inverse(offset: int) -> None:
    old_segment = -5.0 + 0.11 * offset
    new_segment = -4.3 - 0.07 * offset
    log_before = 0.2 + 0.05 * offset
    log_after = -0.1 + 0.03 * offset
    delta_action = 0.9 - 0.08 * offset
    forward = swap_ratio(
        old_segment_log_kinetic=old_segment,
        new_segment_log_kinetic=new_segment,
        log_candidate_normalizer_before=log_before,
        log_candidate_normalizer_after=log_after,
        delta_potential_action=delta_action,
        move_probability=0.15,
        reverse_move_probability=0.15,
    )
    reverse = swap_ratio(
        old_segment_log_kinetic=new_segment,
        new_segment_log_kinetic=old_segment,
        log_candidate_normalizer_before=log_after,
        log_candidate_normalizer_after=log_before,
        delta_potential_action=-delta_action,
        move_probability=0.15,
        reverse_move_probability=0.15,
    )

    expected = -delta_action + log_before - log_after
    assert forward.log_ratio == pytest.approx(expected, abs=1.0e-14)
    assert forward.log_ratio + reverse.log_ratio == pytest.approx(
        0.0, abs=2.0e-14
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"volume": 0.0},
        {"n_slices": 0},
        {"max_segment_links": 0},
        {"n_slices": 6, "max_segment_links": 6},
        {"worm_sector_weight": -1.0},
    ],
)
def test_worm_measure_convention_rejects_invalid_inputs(
    kwargs: dict[str, float | int],
) -> None:
    values: dict[str, float | int] = {
        "volume": 4.0,
        "n_slices": 16,
        "max_segment_links": 6,
        "worm_sector_weight": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        WormMeasureConvention(**values)  # type: ignore[arg-type]
