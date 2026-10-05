"""Shared constructors for topology-changing Worm proposals."""

from __future__ import annotations

import numpy as np

from ..configuration import Configuration, NONE
from ..types import (
    BeadRecord,
    EndpointState,
    LinkChange,
    MoveStatus,
    ProposalPatch,
)


def endpoint_state(state: Configuration) -> EndpointState:
    return EndpointState(state.worm_head, state.worm_tail)


def not_applicable(state: Configuration, move_name: str, reason: str) -> ProposalPatch:
    endpoints = endpoint_state(state)
    return ProposalPatch(
        move_name=move_name,
        base_revision=state.revision,
        status=MoveStatus.NOT_APPLICABLE,
        invalid_reason=reason,
        sector_before=state.sector,
        sector_after=state.sector,
        endpoints_before=endpoints,
        endpoints_after=endpoints,
    )


def active_record(
    *,
    bead_id: int,
    position: np.ndarray,
    slice_id: int,
    next_id: int,
    prev_id: int,
    image_to_next: np.ndarray,
) -> BeadRecord:
    return BeadRecord(
        bead_id=int(bead_id),
        position=tuple(float(value) for value in position),
        slice_id=int(slice_id),
        next_id=int(next_id),
        prev_id=int(prev_id),
        image_to_next=tuple(int(value) for value in image_to_next),
        active=True,
    )


def changed_link(
    state: Configuration,
    bead_id: int,
    *,
    next_id: int | None = None,
    prev_id: int | None = None,
    image: np.ndarray | None = None,
) -> LinkChange:
    before_image = tuple(int(value) for value in state.image_to_next[bead_id])
    return LinkChange(
        bead_id=int(bead_id),
        before_next=int(state.next_of[bead_id]),
        after_next=(int(state.next_of[bead_id]) if next_id is None else int(next_id)),
        before_prev=int(state.prev_of[bead_id]),
        after_prev=(int(state.prev_of[bead_id]) if prev_id is None else int(prev_id)),
        before_image=before_image,
        after_image=(
            before_image
            if image is None
            else tuple(int(value) for value in np.asarray(image, dtype=np.int64))
        ),
    )


ZERO_ENDPOINTS = EndpointState(NONE, NONE)
