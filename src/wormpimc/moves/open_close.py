"""Mutually inverse Open and Close Worm updates."""

from __future__ import annotations

import math

import numpy as np

from ..config import SimulationConfig
from ..configuration import Configuration, NONE
from ..propagator import (
    image_resolved_log_density,
    periodic_log_density,
    sample_periodic_bridge,
)
from ..types import EndpointState, MoveStatus, ProposalPatch, ProposalRatio, Sector
from .base import ratio_from_metadata
from .topology import ZERO_ENDPOINTS, active_record, changed_link, endpoint_state, not_applicable


class OpenMove:
    name = "open"

    def __init__(self, config: SimulationConfig, selection_probability: float, reverse_selection_probability: float) -> None:
        self.config = config
        self.selection_probability = float(selection_probability)
        self.reverse_selection_probability = float(reverse_selection_probability)

    def propose(self, state: Configuration, rng: np.random.Generator) -> ProposalPatch:
        if state.sector is not Sector.Z or state.number_of_beads == 0:
            return not_applicable(state, self.name, "Open requires a nonempty Z sector")
        maximum = self.config.moves.max_segment_links
        head = int(rng.choice(state.active_ids))
        links = int(rng.integers(1, maximum + 1))
        ids, _ = state.unwrapped_segment(head, links)
        tail = ids[-1]
        removed = tuple(state.bead_record(bead_id) for bead_id in ids[1:-1])
        log_segment = math.fsum(
            image_resolved_log_density(
                state.positions[bead_id],
                state.positions[int(state.next_of[bead_id])],
                state.image_to_next[bead_id],
                state.box_length,
                self.config.system.lambda_kin,
                self.config.tau,
            )
            for bead_id in ids[:-1]
        )
        log_endpoint = periodic_log_density(
            state.positions[head],
            state.positions[tail],
            state.box_length,
            self.config.system.lambda_kin,
            links * self.config.tau,
        )
        link_changes = (
            changed_link(
                state,
                head,
                next_id=NONE,
                image=np.zeros(state.ndim, dtype=np.int64),
            ),
            changed_link(state, tail, prev_id=NONE),
        )
        return ProposalPatch(
            move_name=self.name,
            base_revision=state.revision,
            status=MoveStatus.PROPOSED,
            removed_beads=removed,
            link_changes=link_changes,
            sector_before=Sector.Z,
            sector_after=Sector.G,
            endpoints_before=ZERO_ENDPOINTS,
            endpoints_after=EndpointState(head, tail),
            affected_slices=tuple(sorted({int(state.slice_of[value]) for value in ids})),
            metadata=(
                ("segment_links", links),
                ("segment_ids", tuple(int(value) for value in ids)),
                ("segment_log_kinetic", log_segment),
                ("endpoint_log_free_density", log_endpoint),
                ("log_q_forward", -math.log(state.number_of_beads) - math.log(maximum)),
                ("log_q_reverse", log_segment - log_endpoint),
            ),
        )

    def proposal_ratio(self, patch: ProposalPatch) -> ProposalRatio:
        return ratio_from_metadata(patch, self.selection_probability, self.reverse_selection_probability)


class CloseMove:
    name = "close"

    def __init__(self, config: SimulationConfig, selection_probability: float, reverse_selection_probability: float) -> None:
        self.config = config
        self.selection_probability = float(selection_probability)
        self.reverse_selection_probability = float(reverse_selection_probability)

    def propose(self, state: Configuration, rng: np.random.Generator) -> ProposalPatch:
        if state.sector is not Sector.G:
            return not_applicable(state, self.name, "Close requires sector G")
        head = state.worm_head
        tail = state.worm_tail
        links = (int(state.slice_of[tail]) - int(state.slice_of[head])) % state.n_slices
        maximum = self.config.moves.max_segment_links
        if links < 1 or links > maximum:
            return not_applicable(state, self.name, "endpoint gap lies outside Close support")
        bridge = sample_periodic_bridge(
            state.positions[head],
            state.positions[tail],
            links,
            state.box_length,
            self.config.system.lambda_kin,
            self.config.tau,
            rng,
        )
        added_ids = state.preview_bead_ids(links - 1)
        chain = (head, *added_ids, tail)
        added = tuple(
            active_record(
                bead_id=bead_id,
                position=bridge.wrapped_path[index],
                slice_id=(int(state.slice_of[head]) + index) % state.n_slices,
                next_id=chain[index + 1],
                prev_id=chain[index - 1],
                image_to_next=bridge.link_images[index],
            )
            for index, bead_id in enumerate(added_ids, start=1)
        )
        link_changes = (
            changed_link(state, head, next_id=chain[1], image=bridge.link_images[0]),
            changed_link(state, tail, prev_id=chain[-2]),
        )
        final_beads = state.number_of_beads + len(added)
        log_endpoint = periodic_log_density(
            state.positions[head],
            state.positions[tail],
            state.box_length,
            self.config.system.lambda_kin,
            links * self.config.tau,
        )
        return ProposalPatch(
            move_name=self.name,
            base_revision=state.revision,
            status=MoveStatus.PROPOSED,
            added_beads=added,
            link_changes=link_changes,
            sector_before=Sector.G,
            sector_after=Sector.Z,
            endpoints_before=endpoint_state(state),
            endpoints_after=ZERO_ENDPOINTS,
            affected_slices=tuple((int(state.slice_of[head]) + index) % state.n_slices for index in range(links + 1)),
            metadata=(
                ("segment_links", links),
                ("segment_ids", tuple(int(value) for value in chain)),
                ("segment_log_kinetic", bridge.log_proposal_density + log_endpoint),
                ("endpoint_log_free_density", log_endpoint),
                ("log_q_forward", bridge.log_proposal_density),
                ("log_q_reverse", -math.log(final_beads) - math.log(maximum)),
            ),
        )

    def proposal_ratio(self, patch: ProposalPatch) -> ProposalRatio:
        return ratio_from_metadata(patch, self.selection_probability, self.reverse_selection_probability)
