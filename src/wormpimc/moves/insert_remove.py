"""Mutually inverse Insert and Remove Worm updates."""

from __future__ import annotations

import math

import numpy as np

from ..config import SimulationConfig
from ..configuration import Configuration, NONE
from ..propagator import image_resolved_log_density, sample_free_walk
from ..types import EndpointState, MoveStatus, ProposalPatch, ProposalRatio, Sector
from .base import ratio_from_metadata
from .topology import ZERO_ENDPOINTS, active_record, endpoint_state, not_applicable


class InsertMove:
    name = "insert"

    def __init__(self, config: SimulationConfig, selection_probability: float, reverse_selection_probability: float) -> None:
        self.config = config
        self.selection_probability = float(selection_probability)
        self.reverse_selection_probability = float(reverse_selection_probability)

    def propose(self, state: Configuration, rng: np.random.Generator) -> ProposalPatch:
        if state.sector is not Sector.Z:
            return not_applicable(state, self.name, "Insert requires sector Z")
        maximum = self.config.moves.max_segment_links
        links = int(rng.integers(1, maximum + 1))
        tail_slice = int(rng.integers(state.n_slices))
        volume = state.box_length**state.ndim
        tail_position = rng.uniform(0.0, state.box_length, size=state.ndim)
        walk = sample_free_walk(
            tail_position,
            links,
            state.box_length,
            self.config.system.lambda_kin,
            self.config.tau,
            rng,
        )
        ids = state.preview_bead_ids(links + 1)
        records = tuple(
            active_record(
                bead_id=bead_id,
                position=walk.wrapped_path[index],
                slice_id=(tail_slice + index) % state.n_slices,
                next_id=(ids[index + 1] if index < links else NONE),
                prev_id=(ids[index - 1] if index > 0 else NONE),
                image_to_next=(
                    walk.link_images[index]
                    if index < links
                    else np.zeros(state.ndim, dtype=np.int64)
                ),
            )
            for index, bead_id in enumerate(ids)
        )
        log_q_forward = (
            walk.log_proposal_density
            - math.log(state.n_slices)
            - math.log(volume)
            - math.log(maximum)
        )
        return ProposalPatch(
            move_name=self.name,
            base_revision=state.revision,
            status=MoveStatus.PROPOSED,
            added_beads=records,
            sector_before=Sector.Z,
            sector_after=Sector.G,
            endpoints_before=ZERO_ENDPOINTS,
            endpoints_after=EndpointState(ids[-1], ids[0]),
            affected_slices=tuple((tail_slice + index) % state.n_slices for index in range(links + 1)),
            metadata=(
                ("segment_links", links),
                ("segment_ids", tuple(int(value) for value in ids)),
                ("segment_log_kinetic", walk.log_proposal_density),
                ("log_q_forward", log_q_forward),
                ("log_q_reverse", 0.0),
            ),
        )

    def proposal_ratio(self, patch: ProposalPatch) -> ProposalRatio:
        return ratio_from_metadata(patch, self.selection_probability, self.reverse_selection_probability)


class RemoveMove:
    name = "remove"

    def __init__(self, config: SimulationConfig, selection_probability: float, reverse_selection_probability: float) -> None:
        self.config = config
        self.selection_probability = float(selection_probability)
        self.reverse_selection_probability = float(reverse_selection_probability)

    def propose(self, state: Configuration, rng: np.random.Generator) -> ProposalPatch:
        del rng
        if state.sector is not Sector.G:
            return not_applicable(state, self.name, "Remove requires sector G")
        links = state.open_chain_links
        maximum = self.config.moves.max_segment_links
        if links < 1 or links > maximum:
            return not_applicable(state, self.name, "open component lies outside Remove support")
        ids = state.open_chain_ids()
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
        volume = state.box_length**state.ndim
        log_q_reverse = (
            log_segment
            - math.log(state.n_slices)
            - math.log(volume)
            - math.log(maximum)
        )
        return ProposalPatch(
            move_name=self.name,
            base_revision=state.revision,
            status=MoveStatus.PROPOSED,
            removed_beads=tuple(state.bead_record(bead_id) for bead_id in ids),
            sector_before=Sector.G,
            sector_after=Sector.Z,
            endpoints_before=endpoint_state(state),
            endpoints_after=ZERO_ENDPOINTS,
            affected_slices=tuple(sorted({int(state.slice_of[value]) for value in ids})),
            metadata=(
                ("segment_links", links),
                ("segment_ids", tuple(int(value) for value in ids)),
                ("segment_log_kinetic", log_segment),
                ("log_q_forward", 0.0),
                ("log_q_reverse", log_q_reverse),
            ),
        )

    def proposal_ratio(self, patch: ProposalPatch) -> ProposalRatio:
        return ratio_from_metadata(patch, self.selection_probability, self.reverse_selection_probability)
