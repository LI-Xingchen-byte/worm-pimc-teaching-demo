"""Gate-E reference algebra for topology-changing Worm proposals.

This module deliberately contains no graph mutation.  It turns the measure and
proposal factors fixed in ``derivations.md`` into auditable log-ratio objects so
that the future topology moves have an independent mathematical oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import AcceptanceBreakdown, ProposalRatio, TargetDelta


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class WormMeasureConvention:
    """Relation between dimensionless tuning weight and measure coefficient.

    ``worm_sector_weight`` is the dimensionless :math:`C_0`.  The coefficient
    multiplying the endpoint-integrated Green numerator is
    :math:`C_G=C_0/(V M s_{max})` and has units of inverse volume.
    """

    volume: float
    n_slices: int
    max_segment_links: int
    worm_sector_weight: float

    def __post_init__(self) -> None:
        _positive_finite(self.volume, "volume")
        slices = _positive_integer(self.n_slices, "n_slices")
        maximum = _positive_integer(
            self.max_segment_links,
            "max_segment_links",
        )
        if maximum >= slices:
            raise ValueError("max_segment_links must be smaller than n_slices")
        _positive_finite(self.worm_sector_weight, "worm_sector_weight")

    @property
    def coefficient(self) -> float:
        """Return the dimensionful endpoint-measure coefficient ``C_G``."""

        return self.worm_sector_weight / (
            self.volume * self.n_slices * self.max_segment_links
        )

    @property
    def log_coefficient(self) -> float:
        return (
            math.log(self.worm_sector_weight)
            - math.log(self.volume)
            - math.log(self.n_slices)
            - math.log(self.max_segment_links)
        )


@dataclass(frozen=True, slots=True)
class TopologyRatioAudit:
    """Target and proposal decomposition for one prospective Worm move."""

    target: TargetDelta
    proposal: ProposalRatio

    @property
    def acceptance(self) -> AcceptanceBreakdown:
        return AcceptanceBreakdown(
            kinetic=self.target.kinetic,
            potential=self.target.potential,
            chemical=self.target.chemical,
            sector_measure=self.target.sector_measure,
            proposal_selection=self.proposal.total,
        )

    @property
    def log_ratio(self) -> float:
        return self.acceptance.log_ratio


def _proposal_ratio(
    *,
    log_q_forward: float,
    log_q_reverse: float,
    move_probability: float,
    reverse_move_probability: float,
) -> ProposalRatio:
    return ProposalRatio(
        log_q_forward=_finite(log_q_forward, "log_q_forward"),
        log_q_reverse=_finite(log_q_reverse, "log_q_reverse"),
        log_move_selection_forward=math.log(
            _positive_finite(move_probability, "move_probability")
        ),
        log_move_selection_reverse=math.log(
            _positive_finite(
                reverse_move_probability,
                "reverse_move_probability",
            )
        ),
    )


def _segment_inputs(
    *,
    segment_links: int,
    max_segment_links: int,
    tau: float,
) -> tuple[int, int, float]:
    links = _positive_integer(segment_links, "segment_links")
    maximum = _positive_integer(max_segment_links, "max_segment_links")
    if links > maximum:
        raise ValueError("segment_links may not exceed max_segment_links")
    return links, maximum, _positive_finite(tau, "tau")


def open_ratio(
    *,
    segment_log_kinetic: float,
    endpoint_log_free_density: float,
    delta_potential_action: float,
    chemical_potential: float,
    tau: float,
    segment_links: int,
    diagonal_bead_count: int,
    max_segment_links: int,
    worm_measure_coefficient: float,
    move_probability: float,
    reverse_move_probability: float,
) -> TopologyRatioAudit:
    """Reference Open ratio from ``eq-open-log-ratio``.

    The endpoint density is the *periodic* free density at ``segment_links*tau``.
    ``delta_potential_action`` always means ``S_after - S_before``.
    """

    links, maximum, time_step = _segment_inputs(
        segment_links=segment_links,
        max_segment_links=max_segment_links,
        tau=tau,
    )
    bead_count = _positive_integer(diagonal_bead_count, "diagonal_bead_count")
    log_segment = _finite(segment_log_kinetic, "segment_log_kinetic")
    log_endpoint = _finite(
        endpoint_log_free_density,
        "endpoint_log_free_density",
    )
    delta_action = _finite(delta_potential_action, "delta_potential_action")
    mu = _finite(chemical_potential, "chemical_potential")
    coefficient = _positive_finite(
        worm_measure_coefficient,
        "worm_measure_coefficient",
    )
    return TopologyRatioAudit(
        target=TargetDelta(
            kinetic=-log_segment,
            potential=-delta_action,
            chemical=-mu * time_step * links,
            sector_measure=math.log(coefficient),
        ),
        proposal=_proposal_ratio(
            log_q_forward=-math.log(bead_count) - math.log(maximum),
            log_q_reverse=log_segment - log_endpoint,
            move_probability=move_probability,
            reverse_move_probability=reverse_move_probability,
        ),
    )


def close_ratio(
    *,
    segment_log_kinetic: float,
    endpoint_log_free_density: float,
    delta_potential_action: float,
    chemical_potential: float,
    tau: float,
    segment_links: int,
    final_diagonal_bead_count: int,
    max_segment_links: int,
    worm_measure_coefficient: float,
    move_probability: float,
    reverse_move_probability: float,
) -> TopologyRatioAudit:
    """Reference Close ratio from ``eq-close-log-ratio``."""

    links, maximum, time_step = _segment_inputs(
        segment_links=segment_links,
        max_segment_links=max_segment_links,
        tau=tau,
    )
    bead_count = _positive_integer(
        final_diagonal_bead_count,
        "final_diagonal_bead_count",
    )
    log_segment = _finite(segment_log_kinetic, "segment_log_kinetic")
    log_endpoint = _finite(
        endpoint_log_free_density,
        "endpoint_log_free_density",
    )
    delta_action = _finite(delta_potential_action, "delta_potential_action")
    mu = _finite(chemical_potential, "chemical_potential")
    coefficient = _positive_finite(
        worm_measure_coefficient,
        "worm_measure_coefficient",
    )
    return TopologyRatioAudit(
        target=TargetDelta(
            kinetic=log_segment,
            potential=-delta_action,
            chemical=mu * time_step * links,
            sector_measure=-math.log(coefficient),
        ),
        proposal=_proposal_ratio(
            log_q_forward=log_segment - log_endpoint,
            log_q_reverse=-math.log(bead_count) - math.log(maximum),
            move_probability=move_probability,
            reverse_move_probability=reverse_move_probability,
        ),
    )


def insert_ratio(
    *,
    segment_log_kinetic: float,
    delta_potential_action: float,
    chemical_potential: float,
    tau: float,
    segment_links: int,
    volume: float,
    n_slices: int,
    max_segment_links: int,
    worm_measure_coefficient: float,
    move_probability: float,
    reverse_move_probability: float,
) -> TopologyRatioAudit:
    """Reference Insert ratio from ``eq-insert-log-ratio``."""

    links, maximum, time_step = _segment_inputs(
        segment_links=segment_links,
        max_segment_links=max_segment_links,
        tau=tau,
    )
    physical_volume = _positive_finite(volume, "volume")
    slices = _positive_integer(n_slices, "n_slices")
    if maximum >= slices:
        raise ValueError("max_segment_links must be smaller than n_slices")
    log_segment = _finite(segment_log_kinetic, "segment_log_kinetic")
    delta_action = _finite(delta_potential_action, "delta_potential_action")
    mu = _finite(chemical_potential, "chemical_potential")
    coefficient = _positive_finite(
        worm_measure_coefficient,
        "worm_measure_coefficient",
    )
    return TopologyRatioAudit(
        target=TargetDelta(
            kinetic=log_segment,
            potential=-delta_action,
            chemical=mu * time_step * links,
            sector_measure=math.log(coefficient),
        ),
        proposal=_proposal_ratio(
            log_q_forward=(
                log_segment
                - math.log(slices)
                - math.log(physical_volume)
                - math.log(maximum)
            ),
            log_q_reverse=0.0,
            move_probability=move_probability,
            reverse_move_probability=reverse_move_probability,
        ),
    )


def remove_ratio(
    *,
    segment_log_kinetic: float,
    delta_potential_action: float,
    chemical_potential: float,
    tau: float,
    segment_links: int,
    volume: float,
    n_slices: int,
    max_segment_links: int,
    worm_measure_coefficient: float,
    move_probability: float,
    reverse_move_probability: float,
) -> TopologyRatioAudit:
    """Reference Remove ratio from ``eq-remove-log-ratio``."""

    links, maximum, time_step = _segment_inputs(
        segment_links=segment_links,
        max_segment_links=max_segment_links,
        tau=tau,
    )
    physical_volume = _positive_finite(volume, "volume")
    slices = _positive_integer(n_slices, "n_slices")
    if maximum >= slices:
        raise ValueError("max_segment_links must be smaller than n_slices")
    log_segment = _finite(segment_log_kinetic, "segment_log_kinetic")
    delta_action = _finite(delta_potential_action, "delta_potential_action")
    mu = _finite(chemical_potential, "chemical_potential")
    coefficient = _positive_finite(
        worm_measure_coefficient,
        "worm_measure_coefficient",
    )
    return TopologyRatioAudit(
        target=TargetDelta(
            kinetic=-log_segment,
            potential=-delta_action,
            chemical=-mu * time_step * links,
            sector_measure=-math.log(coefficient),
        ),
        proposal=_proposal_ratio(
            log_q_forward=0.0,
            log_q_reverse=(
                log_segment
                - math.log(slices)
                - math.log(physical_volume)
                - math.log(maximum)
            ),
            move_probability=move_probability,
            reverse_move_probability=reverse_move_probability,
        ),
    )


def advance_ratio(
    *,
    segment_log_kinetic: float,
    delta_potential_action: float,
    chemical_potential: float,
    tau: float,
    segment_links: int,
    max_segment_links: int,
    move_probability: float,
    reverse_move_probability: float,
) -> TopologyRatioAudit:
    """Reference Advance ratio from ``eq-advance-log-ratio``."""

    links, maximum, time_step = _segment_inputs(
        segment_links=segment_links,
        max_segment_links=max_segment_links,
        tau=tau,
    )
    log_segment = _finite(segment_log_kinetic, "segment_log_kinetic")
    delta_action = _finite(delta_potential_action, "delta_potential_action")
    mu = _finite(chemical_potential, "chemical_potential")
    return TopologyRatioAudit(
        target=TargetDelta(
            kinetic=log_segment,
            potential=-delta_action,
            chemical=mu * time_step * links,
            sector_measure=0.0,
        ),
        proposal=_proposal_ratio(
            log_q_forward=log_segment - math.log(maximum),
            log_q_reverse=-math.log(maximum),
            move_probability=move_probability,
            reverse_move_probability=reverse_move_probability,
        ),
    )


def recede_ratio(
    *,
    segment_log_kinetic: float,
    delta_potential_action: float,
    chemical_potential: float,
    tau: float,
    segment_links: int,
    max_segment_links: int,
    move_probability: float,
    reverse_move_probability: float,
) -> TopologyRatioAudit:
    """Reference Recede ratio from ``eq-recede-log-ratio``."""

    links, maximum, time_step = _segment_inputs(
        segment_links=segment_links,
        max_segment_links=max_segment_links,
        tau=tau,
    )
    log_segment = _finite(segment_log_kinetic, "segment_log_kinetic")
    delta_action = _finite(delta_potential_action, "delta_potential_action")
    mu = _finite(chemical_potential, "chemical_potential")
    return TopologyRatioAudit(
        target=TargetDelta(
            kinetic=-log_segment,
            potential=-delta_action,
            chemical=-mu * time_step * links,
            sector_measure=0.0,
        ),
        proposal=_proposal_ratio(
            log_q_forward=-math.log(maximum),
            log_q_reverse=log_segment - math.log(maximum),
            move_probability=move_probability,
            reverse_move_probability=reverse_move_probability,
        ),
    )


def swap_ratio(
    *,
    old_segment_log_kinetic: float,
    new_segment_log_kinetic: float,
    log_candidate_normalizer_before: float,
    log_candidate_normalizer_after: float,
    delta_potential_action: float,
    move_probability: float,
    reverse_move_probability: float,
) -> TopologyRatioAudit:
    """Reference Swap ratio from ``eq-swap-log-ratio``."""

    old_segment = _finite(
        old_segment_log_kinetic,
        "old_segment_log_kinetic",
    )
    new_segment = _finite(
        new_segment_log_kinetic,
        "new_segment_log_kinetic",
    )
    log_before = _finite(
        log_candidate_normalizer_before,
        "log_candidate_normalizer_before",
    )
    log_after = _finite(
        log_candidate_normalizer_after,
        "log_candidate_normalizer_after",
    )
    delta_action = _finite(delta_potential_action, "delta_potential_action")
    return TopologyRatioAudit(
        target=TargetDelta(
            kinetic=new_segment - old_segment,
            potential=-delta_action,
            chemical=0.0,
            sector_measure=0.0,
        ),
        proposal=_proposal_ratio(
            log_q_forward=new_segment - log_before,
            log_q_reverse=old_segment - log_after,
            move_probability=move_probability,
            reverse_move_probability=reverse_move_probability,
        ),
    )
