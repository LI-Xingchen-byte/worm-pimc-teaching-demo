"""Primitive Z/G target measure with local and full reference paths."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .configuration import Configuration
from .potentials import PotentialModel, potential_from_config
from .propagator import image_resolved_log_density
from .types import ProposalPatch, Sector, TargetDelta
from .worm_measure import WormMeasureConvention


@dataclass(frozen=True, slots=True)
class TargetComponents:
    """Absolute log-weight components for one configuration."""

    kinetic: float
    potential: float
    chemical: float
    sector_measure: float

    @property
    def total(self) -> float:
        return self.kinetic + self.potential + self.chemical + self.sector_measure


class PrimitiveTargetMeasure:
    """Symmetric primitive measure from derivations.md ``eq-z-log-weight``."""

    def __init__(
        self,
        config: SimulationConfig,
        potential: PotentialModel | None = None,
    ) -> None:
        if config.discretization.action != "primitive":
            raise ValueError("PrimitiveTargetMeasure requires action='primitive'")
        self.config = config
        self.potential = potential or potential_from_config(config)

    def link_log_weight(self, state: Configuration, bead_id: int) -> float:
        """Return the complete normalized free weight of one outgoing link."""

        next_id = int(state.next_of[bead_id])
        if next_id < 0:
            raise ValueError("bead has no outgoing link")
        return image_resolved_log_density(
            state.positions[bead_id],
            state.positions[next_id],
            state.image_to_next[bead_id],
            state.box_length,
            self.config.system.lambda_kin,
            self.config.tau,
        )

    def slice_potential_energy(
        self,
        state: Configuration,
        slice_id: int,
    ) -> float:
        return self.potential.total_energy(state.positions_on_slice(slice_id))

    def slice_endpoint_potential_energy(
        self,
        state: Configuration,
        slice_id: int,
    ) -> tuple[float, float]:
        """Return incoming/outgoing endpoint energies for the primitive action."""

        incoming = self.potential.total_energy(
            state.incoming_positions_on_slice(slice_id)
        )
        outgoing = self.potential.total_energy(
            state.outgoing_positions_on_slice(slice_id)
        )
        return incoming, outgoing

    def full_components(self, state: Configuration) -> TargetComponents:
        """Recompute every target component without using local caches."""

        state.validate()
        kinetic = math.fsum(
            self.link_log_weight(state, int(bead_id))
            for bead_id in state.active_ids
            if int(state.next_of[bead_id]) >= 0
        )
        potential_action = 0.5 * self.config.tau * math.fsum(
            math.fsum(self.slice_endpoint_potential_energy(state, slice_id))
            for slice_id in range(state.n_slices)
        )
        chemical = (
            self.config.system.chemical_potential
            * self.config.tau
            * state.total_occupied_links
        )
        sector_measure = 0.0
        if state.sector is Sector.G:
            convention = WormMeasureConvention(
                volume=state.box_length**state.ndim,
                n_slices=state.n_slices,
                max_segment_links=self.config.moves.max_segment_links,
                worm_sector_weight=self.config.moves.worm_sector_weight,
            )
            sector_measure = convention.log_coefficient
        return TargetComponents(
            kinetic=kinetic,
            potential=-potential_action,
            chemical=chemical,
            sector_measure=sector_measure,
        )

    def delta_full(
        self,
        state: Configuration,
        patch: ProposalPatch,
    ) -> TargetDelta:
        """Reference delta obtained from two complete recomputations."""

        before = self.full_components(state)
        after = self.full_components(state.preview(patch))
        return TargetDelta(
            kinetic=after.kinetic - before.kinetic,
            potential=after.potential - before.potential,
            chemical=after.chemical - before.chemical,
            sector_measure=after.sector_measure - before.sector_measure,
        )

    def delta_local(
        self,
        state: Configuration,
        patch: ProposalPatch,
    ) -> TargetDelta:
        """Recompute only links and slices touched by a topology-preserving patch."""

        if (
            patch.added_beads
            or patch.removed_beads
            or patch.sector_before is not patch.sector_after
            or state.sector is Sector.G
        ):
            return self.delta_full(state, patch)

        candidate = state.preview(patch)
        link_ids = {change.bead_id for change in patch.link_changes}
        for change in patch.position_changes:
            link_ids.add(change.bead_id)
            link_ids.add(int(state.prev_of[change.bead_id]))
            link_ids.add(int(candidate.prev_of[change.bead_id]))
        link_ids = {
            bead_id
            for bead_id in link_ids
            if 0 <= bead_id < state.capacity and state.active[bead_id]
        }
        kinetic = math.fsum(
            self.link_log_weight(candidate, bead_id)
            - self.link_log_weight(state, bead_id)
            for bead_id in sorted(link_ids)
        )

        affected_slices = set(int(value) for value in patch.affected_slices)
        affected_slices.update(
            int(state.slice_of[change.bead_id])
            for change in patch.position_changes
        )
        potential = -self.config.tau * math.fsum(
            self.slice_potential_energy(candidate, slice_id)
            - self.slice_potential_energy(state, slice_id)
            for slice_id in sorted(affected_slices)
        )
        chemical = 0.0
        sector_measure = 0.0
        return TargetDelta(
            kinetic=kinetic,
            potential=potential,
            chemical=chemical,
            sector_measure=sector_measure,
        )


def assert_local_matches_full(
    local: TargetDelta,
    full: TargetDelta,
    *,
    absolute_tolerance: float = 1.0e-11,
) -> None:
    """Raise when a local target delta disagrees with the reference path."""

    for name in ("kinetic", "potential", "chemical", "sector_measure", "total"):
        local_value = float(getattr(local, name))
        full_value = float(getattr(full, name))
        if not math.isclose(
            local_value,
            full_value,
            rel_tol=1.0e-11,
            abs_tol=absolute_tolerance,
        ):
            raise AssertionError(
                f"local/full mismatch for {name}: {local_value} != {full_value}"
            )
