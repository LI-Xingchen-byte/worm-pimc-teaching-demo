"""Mutually inverse Advance and Recede Worm updates."""

from __future__ import annotations

import math

import numpy as np

from ..config import SimulationConfig
from ..configuration import Configuration, NONE
from ..propagator import image_resolved_log_density, sample_free_walk
from ..types import EndpointState, MoveStatus, ProposalPatch, ProposalRatio, Sector
from .base import ratio_from_metadata
from .topology import active_record, changed_link, endpoint_state, not_applicable


class AdvanceMove:
    name = "advance"

    def __init__(self, config: SimulationConfig, selection_probability: float, reverse_selection_probability: float) -> None:
        self.config = config
        self.selection_probability = float(selection_probability)
        self.reverse_selection_probability = float(reverse_selection_probability)

    def propose(self, state: Configuration, rng: np.random.Generator) -> ProposalPatch:
        if state.sector is not Sector.G:
            return not_applicable(state, self.name, "Advance requires sector G")
        maximum = self.config.moves.max_segment_links
        links = int(rng.integers(1, maximum + 1))
        old_head = state.worm_head
        walk = sample_free_walk(
            state.positions[old_head],
            links,
            state.box_length,
            self.config.system.lambda_kin,
            self.config.tau,
            rng,
        )
        ids = state.preview_bead_ids(links)
        chain = (old_head, *ids)
        records = tuple(
            active_record(
                bead_id=bead_id,
                position=walk.wrapped_path[index],
                slice_id=(int(state.slice_of[old_head]) + index) % state.n_slices,
                next_id=(chain[index + 1] if index < links else NONE),
                prev_id=chain[index - 1],
                image_to_next=(
                    walk.link_images[index]
                    if index < links
                    else np.zeros(state.ndim, dtype=np.int64)
                ),
            )
            for index, bead_id in enumerate(ids, start=1)
        )
        return ProposalPatch(
            move_name=self.name,
            base_revision=state.revision,
            status=MoveStatus.PROPOSED,
            added_beads=records,
            link_changes=(changed_link(state, old_head, next_id=ids[0], image=walk.link_images[0]),),
            sector_before=Sector.G,
            sector_after=Sector.G,
            endpoints_before=endpoint_state(state),
            endpoints_after=EndpointState(ids[-1], state.worm_tail),
            affected_slices=tuple((int(state.slice_of[old_head]) + index) % state.n_slices for index in range(links + 1)),
            metadata=(
                ("segment_links", links),
                ("segment_ids", tuple(int(value) for value in chain)),
                ("segment_log_kinetic", walk.log_proposal_density),
                ("log_q_forward", walk.log_proposal_density - math.log(maximum)),
                ("log_q_reverse", -math.log(maximum)),
            ),
        )

    def proposal_ratio(self, patch: ProposalPatch) -> ProposalRatio:
        return ratio_from_metadata(patch, self.selection_probability, self.reverse_selection_probability)


class RecedeMove:
    name = "recede"

    def __init__(self, config: SimulationConfig, selection_probability: float, reverse_selection_probability: float) -> None:
        self.config = config
        self.selection_probability = float(selection_probability)
        self.reverse_selection_probability = float(reverse_selection_probability)

    def propose(self, state: Configuration, rng: np.random.Generator) -> ProposalPatch:
        if state.sector is not Sector.G:
            return not_applicable(state, self.name, "Recede requires sector G")
        maximum = self.config.moves.max_segment_links
        links = int(rng.integers(1, maximum + 1))
        if state.open_chain_links <= links:
            return not_applicable(state, self.name, "Recede may not remove the last open link")
        chain = state.open_chain_ids()
        new_head = chain[-links - 1]
        removed_ids = chain[-links:]
        segment_sources = (new_head, *removed_ids[:-1])
        log_segment = math.fsum(
            image_resolved_log_density(
                state.positions[bead_id],
                state.positions[int(state.next_of[bead_id])],
                state.image_to_next[bead_id],
                state.box_length,
                self.config.system.lambda_kin,
                self.config.tau,
            )
            for bead_id in segment_sources
        )
        return ProposalPatch(
            move_name=self.name,
            base_revision=state.revision,
            status=MoveStatus.PROPOSED,
            removed_beads=tuple(state.bead_record(bead_id) for bead_id in removed_ids),
            link_changes=(changed_link(state, new_head, next_id=NONE, image=np.zeros(state.ndim, dtype=np.int64)),),
            sector_before=Sector.G,
            sector_after=Sector.G,
            endpoints_before=endpoint_state(state),
            endpoints_after=EndpointState(new_head, state.worm_tail),
            affected_slices=tuple(sorted({int(state.slice_of[value]) for value in (new_head, *removed_ids)})),
            metadata=(
                ("segment_links", links),
                ("segment_ids", tuple(int(value) for value in (new_head, *removed_ids))),
                ("segment_log_kinetic", log_segment),
                ("log_q_forward", -math.log(maximum)),
                ("log_q_reverse", log_segment - math.log(maximum)),
            ),
        )

    def proposal_ratio(self, patch: ProposalPatch) -> ProposalRatio:
        return ratio_from_metadata(patch, self.selection_probability, self.reverse_selection_probability)
