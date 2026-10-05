"""Permutation-sampling Swap update for the open Worm component."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..config import SimulationConfig
from ..configuration import Configuration, NONE
from ..propagator import (
    image_resolved_log_density,
    periodic_log_density,
    sample_periodic_bridge,
)
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
from .topology import endpoint_state, not_applicable


def _logsumexp(values: tuple[float, ...]) -> float:
    maximum = max(values)
    return maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in values)
    )


class SwapMove:
    """Reconnect the head through an exactly normalized candidate table."""

    name = "swap"

    def __init__(
        self,
        config: SimulationConfig,
        selection_probability: float,
    ) -> None:
        self.config = config
        self.selection_probability = float(selection_probability)
        self.segment_links = config.moves.max_segment_links

    def _candidates(
        self,
        state: Configuration,
    ) -> tuple[tuple[int, ...], tuple[float, ...], float] | None:
        target_slice = (
            int(state.slice_of[state.worm_head]) + self.segment_links
        ) % state.n_slices
        candidate_ids: list[int] = []
        log_weights: list[float] = []
        for value in state.bead_ids_on_slice(target_slice):
            candidate = int(value)
            try:
                predecessors = state.predecessor_ids(
                    candidate,
                    self.segment_links,
                )
            except ValueError:
                continue
            if predecessors[-1] == state.worm_tail:
                continue
            candidate_ids.append(candidate)
            log_weights.append(
                periodic_log_density(
                    state.positions[state.worm_head],
                    state.positions[candidate],
                    state.box_length,
                    self.config.system.lambda_kin,
                    self.segment_links * self.config.tau,
                )
            )
        if not candidate_ids:
            return None
        logs = tuple(log_weights)
        return tuple(candidate_ids), logs, _logsumexp(logs)

    def propose(
        self,
        state: Configuration,
        rng: np.random.Generator,
    ) -> ProposalPatch:
        if state.sector is not Sector.G:
            return not_applicable(state, self.name, "Swap requires sector G")
        table = self._candidates(state)
        if table is None:
            return not_applicable(
                state,
                self.name,
                "Swap has no valid target candidate",
            )
        candidate_ids, log_weights, log_normalizer_before = table
        probabilities = np.exp(
            np.asarray(log_weights, dtype=np.float64) - log_normalizer_before
        )
        alpha = int(rng.choice(candidate_ids, p=probabilities))
        backward = state.predecessor_ids(alpha, self.segment_links)
        old_chain = tuple(reversed(backward))
        zeta = old_chain[0]
        old_head = state.worm_head
        bridge = sample_periodic_bridge(
            state.positions[old_head],
            state.positions[alpha],
            self.segment_links,
            state.box_length,
            self.config.system.lambda_kin,
            self.config.tau,
            rng,
        )
        old_log_kinetic = math.fsum(
            image_resolved_log_density(
                state.positions[bead_id],
                state.positions[int(state.next_of[bead_id])],
                state.image_to_next[bead_id],
                state.box_length,
                self.config.system.lambda_kin,
                self.config.tau,
            )
            for bead_id in old_chain[:-1]
        )
        endpoint_log_density = periodic_log_density(
            state.positions[old_head],
            state.positions[alpha],
            state.box_length,
            self.config.system.lambda_kin,
            self.segment_links * self.config.tau,
        )
        new_log_kinetic = bridge.log_proposal_density + endpoint_log_density

        position_changes = tuple(
            PositionChange(
                bead_id=old_chain[index],
                before=tuple(
                    float(value)
                    for value in state.positions[old_chain[index]]
                ),
                after=tuple(
                    float(value) for value in bridge.wrapped_path[index]
                ),
            )
            for index in range(1, self.segment_links)
        )

        affected = {old_head, zeta, alpha, *old_chain[1:-1]}
        after_next = {
            bead_id: int(state.next_of[bead_id]) for bead_id in affected
        }
        after_prev = {
            bead_id: int(state.prev_of[bead_id]) for bead_id in affected
        }
        after_image = {
            bead_id: np.array(state.image_to_next[bead_id], copy=True)
            for bead_id in affected
        }
        after_next[old_head] = old_chain[1]
        after_image[old_head] = np.array(bridge.link_images[0], copy=True)
        after_next[zeta] = NONE
        after_image[zeta] = np.zeros(state.ndim, dtype=np.int64)
        after_prev[old_chain[1]] = old_head
        for index in range(1, self.segment_links):
            after_image[old_chain[index]] = np.array(
                bridge.link_images[index], copy=True
            )

        link_changes = tuple(
            LinkChange(
                bead_id=bead_id,
                before_next=int(state.next_of[bead_id]),
                after_next=after_next[bead_id],
                before_prev=int(state.prev_of[bead_id]),
                after_prev=after_prev[bead_id],
                before_image=tuple(
                    int(value) for value in state.image_to_next[bead_id]
                ),
                after_image=tuple(
                    int(value) for value in after_image[bead_id]
                ),
            )
            for bead_id in sorted(affected)
        )
        placeholder = ProposalPatch(
            move_name=self.name,
            base_revision=state.revision,
            status=MoveStatus.PROPOSED,
            position_changes=position_changes,
            link_changes=link_changes,
            sector_before=Sector.G,
            sector_after=Sector.G,
            endpoints_before=endpoint_state(state),
            endpoints_after=EndpointState(zeta, state.worm_tail),
            affected_slices=tuple(
                sorted({int(state.slice_of[value]) for value in affected})
            ),
            metadata=(("log_q_forward", 0.0), ("log_q_reverse", 0.0)),
        )
        candidate_state = state.preview(placeholder)
        reverse_table = self._candidates(candidate_state)
        if reverse_table is None or alpha not in reverse_table[0]:
            return not_applicable(
                state,
                self.name,
                "Swap reverse candidate is absent",
            )
        log_normalizer_after = reverse_table[2]
        return replace(
            placeholder,
            metadata=(
                ("segment_links", self.segment_links),
                ("candidate_id", alpha),
                ("old_head", old_head),
                ("new_head", zeta),
                ("old_segment_log_kinetic", old_log_kinetic),
                ("new_segment_log_kinetic", new_log_kinetic),
                (
                    "log_candidate_normalizer_before",
                    log_normalizer_before,
                ),
                ("log_candidate_normalizer_after", log_normalizer_after),
                (
                    "log_q_forward",
                    new_log_kinetic - log_normalizer_before,
                ),
                (
                    "log_q_reverse",
                    old_log_kinetic - log_normalizer_after,
                ),
            ),
        )

    def proposal_ratio(self, patch: ProposalPatch) -> ProposalRatio:
        return ratio_from_metadata(patch, self.selection_probability)
