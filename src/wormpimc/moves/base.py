"""Shared move protocol and proposal-ratio reconstruction."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from ..configuration import Configuration
from ..types import ProposalPatch, ProposalRatio


class Move(Protocol):
    """Interface owned by the sampler for one proposal family."""

    name: str
    selection_probability: float

    def propose(
        self,
        state: Configuration,
        rng: np.random.Generator,
    ) -> ProposalPatch:
        """Create an immutable patch without changing ``state``."""

    def proposal_ratio(self, patch: ProposalPatch) -> ProposalRatio:
        """Return the complete reverse-over-forward proposal ratio."""


def ratio_from_metadata(
    patch: ProposalPatch,
    selection_probability: float,
    reverse_selection_probability: float | None = None,
) -> ProposalRatio:
    """Build the audited ratio saved by a topology-preserving proposal."""

    if not math.isfinite(selection_probability) or selection_probability <= 0.0:
        raise ValueError("selection_probability must be positive and finite")
    metadata = patch.metadata_dict()
    try:
        log_q_forward = float(metadata["log_q_forward"])
        log_q_reverse = float(metadata["log_q_reverse"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("proposal metadata lacks finite q densities") from exc
    if not math.isfinite(log_q_forward) or not math.isfinite(log_q_reverse):
        raise ValueError("proposal log densities must be finite")
    reverse_probability = (
        selection_probability
        if reverse_selection_probability is None
        else reverse_selection_probability
    )
    if not math.isfinite(reverse_probability) or reverse_probability <= 0.0:
        raise ValueError("reverse_selection_probability must be positive and finite")
    return ProposalRatio(
        log_q_forward=log_q_forward,
        log_q_reverse=log_q_reverse,
        log_move_selection_forward=math.log(selection_probability),
        log_move_selection_reverse=math.log(reverse_probability),
    )
