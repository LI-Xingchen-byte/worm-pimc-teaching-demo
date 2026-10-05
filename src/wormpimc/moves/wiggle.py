"""Brownian-bridge resampling of a fixed-topology worldline segment."""

from __future__ import annotations

import math

import numpy as np

from ..config import SimulationConfig
from ..configuration import Configuration
from ..geometry import link_image_from_unwrapped, wrap
from ..propagator import brownian_bridge_log_density, sample_brownian_bridge
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


class WiggleMove:
    """Self-inverse free-bridge proposal from derivations.md Section 12.1."""

    name = "wiggle"

    def __init__(
        self,
        config: SimulationConfig,
        selection_probability: float,
    ) -> None:
        self.config = config
        self.selection_probability = float(selection_probability)
        self.max_segment_links = config.moves.max_segment_links
        if self.max_segment_links < 2:
            raise ValueError("Wiggle requires max_segment_links >= 2")

    def propose(
        self,
        state: Configuration,
        rng: np.random.Generator,
    ) -> ProposalPatch:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if state.sector is not Sector.Z or state.number_of_beads == 0:
            return not_applicable(
                state,
                self.name,
                "Wiggle currently requires a nonempty Z sector",
            )
        active_ids = state.active_ids
        start = int(active_ids[int(rng.integers(active_ids.size))])
        n_links = int(rng.integers(2, self.max_segment_links + 1))
        segment_ids, old_path = state.unwrapped_segment(start, n_links)
        new_path = sample_brownian_bridge(
            old_path[0],
            old_path[-1],
            n_links,
            self.config.system.lambda_kin,
            self.config.tau,
            rng,
        )

        position_changes = tuple(
            PositionChange(
                bead_id=segment_ids[index],
                before=tuple(float(x) for x in state.positions[segment_ids[index]]),
                after=tuple(
                    float(x)
                    for x in np.asarray(
                        wrap(new_path[index], state.box_length)
                    )
                ),
            )
            for index in range(1, n_links)
        )
        link_changes: list[LinkChange] = []
        for index in range(n_links):
            bead_id = segment_ids[index]
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

        discrete_log_probability = -math.log(state.number_of_beads) - math.log(
            self.max_segment_links - 1
        )
        log_q_forward = discrete_log_probability + brownian_bridge_log_density(
            new_path,
            self.config.system.lambda_kin,
            self.config.tau,
        )
        log_q_reverse = discrete_log_probability + brownian_bridge_log_density(
            old_path,
            self.config.system.lambda_kin,
            self.config.tau,
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
                sorted(
                    {
                        int(state.slice_of[segment_ids[index]])
                        for index in range(1, n_links)
                    }
                )
            ),
            metadata=(
                ("start_bead", start),
                ("n_links", n_links),
                ("segment_ids", tuple(int(value) for value in segment_ids)),
                ("log_q_forward", log_q_forward),
                ("log_q_reverse", log_q_reverse),
            ),
        )

    def proposal_ratio(self, patch: ProposalPatch) -> ProposalRatio:
        return ratio_from_metadata(patch, self.selection_probability)
