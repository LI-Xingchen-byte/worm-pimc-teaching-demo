"""Symmetric rigid translation of one complete closed component."""

from __future__ import annotations

import math

import numpy as np

from ..config import SimulationConfig
from ..configuration import Configuration
from ..geometry import link_image_from_unwrapped, wrap
from ..types import (
    EndpointState,
    LinkChange,
    MoveStatus,
    PositionChange,
    ProposalPatch,
    ProposalRatio,
    Sector,
)
from .base import ratio_from_metadata
from .topology import not_applicable


class DisplaceMove:
    """Self-inverse component displacement from derivations.md Section 12.1."""

    name = "displace"

    def __init__(
        self,
        config: SimulationConfig,
        selection_probability: float,
    ) -> None:
        self.config = config
        self.selection_probability = float(selection_probability)
        self.scale = config.moves.displacement_scale

    def propose(
        self,
        state: Configuration,
        rng: np.random.Generator,
    ) -> ProposalPatch:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if state.sector is not Sector.Z or not state.cycle_starts():
            return not_applicable(
                state,
                self.name,
                "Displace currently requires a closed component in sector Z",
            )
        cycle_starts = state.cycle_starts()
        cycle_index = int(rng.integers(len(cycle_starts)))
        start = cycle_starts[cycle_index]
        cycle_ids = state.cycle_ids(start)
        old_path = state.unwrapped_cycle(start)
        displacement = rng.uniform(-self.scale, self.scale, size=state.ndim)
        new_path = old_path + displacement

        position_changes = tuple(
            PositionChange(
                bead_id=bead_id,
                before=tuple(float(x) for x in state.positions[bead_id]),
                after=tuple(
                    float(x)
                    for x in np.asarray(wrap(new_path[index], state.box_length))
                ),
            )
            for index, bead_id in enumerate(cycle_ids)
        )
        link_changes: list[LinkChange] = []
        for index, bead_id in enumerate(cycle_ids):
            new_image = np.asarray(
                link_image_from_unwrapped(
                    new_path[index], new_path[index + 1], state.box_length
                ),
                dtype=np.int64,
            )
            link_changes.append(
                LinkChange(
                    bead_id=bead_id,
                    before_next=int(state.next_of[bead_id]),
                    after_next=int(state.next_of[bead_id]),
                    before_prev=int(state.prev_of[bead_id]),
                    after_prev=int(state.prev_of[bead_id]),
                    before_image=tuple(
                        int(x) for x in state.image_to_next[bead_id]
                    ),
                    after_image=tuple(int(x) for x in new_image),
                )
            )

        log_q = -math.log(len(cycle_starts)) - state.ndim * math.log(
            2.0 * self.scale
        )
        endpoint_state = EndpointState(state.worm_head, state.worm_tail)
        return ProposalPatch(
            move_name=self.name,
            base_revision=state.revision,
            status=MoveStatus.PROPOSED,
            position_changes=position_changes,
            link_changes=tuple(link_changes),
            sector_before=state.sector,
            sector_after=state.sector,
            endpoints_before=endpoint_state,
            endpoints_after=endpoint_state,
            affected_slices=tuple(
                sorted({int(state.slice_of[bead_id]) for bead_id in cycle_ids})
            ),
            metadata=(
                ("cycle_start", start),
                ("cycle_ids", tuple(int(value) for value in cycle_ids)),
                ("displacement", tuple(float(value) for value in displacement)),
                ("log_q_forward", log_q),
                ("log_q_reverse", log_q),
            ),
        )

    def proposal_ratio(self, patch: ProposalPatch) -> ProposalRatio:
        return ratio_from_metadata(patch, self.selection_probability)
