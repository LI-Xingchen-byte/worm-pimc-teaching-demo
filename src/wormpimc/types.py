"""Small shared data types for proposals, target deltas, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TypeAlias


MetadataValue: TypeAlias = (
    bool | int | float | str | tuple[int, ...] | tuple[float, ...]
)


class Sector(str, Enum):
    """Configuration sector names fixed by derivations.md."""

    Z = "Z"
    G = "G"


class MoveStatus(str, Enum):
    """High-level outcome of a move proposal before Metropolis acceptance."""

    PROPOSED = "proposed"
    NOT_APPLICABLE = "not_applicable"
    STRUCTURALLY_INVALID = "structurally_invalid"


@dataclass(frozen=True, slots=True)
class BeadRecord:
    """Complete persisted value of one bead for future topology patches."""

    bead_id: int
    position: tuple[float, ...]
    slice_id: int
    next_id: int
    prev_id: int
    image_to_next: tuple[int, ...]
    active: bool


@dataclass(frozen=True, slots=True)
class PositionChange:
    """Before/after coordinates for one existing bead."""

    bead_id: int
    before: tuple[float, ...]
    after: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LinkChange:
    """Before/after connectivity and image for one outgoing link."""

    bead_id: int
    before_next: int
    after_next: int
    before_prev: int
    after_prev: int
    before_image: tuple[int, ...]
    after_image: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EndpointState:
    """Worm endpoint IDs; both are ``-1`` in the diagonal sector."""

    worm_head: int
    worm_tail: int


@dataclass(frozen=True, slots=True)
class ProposalPatch:
    """Immutable, reversible state difference submitted atomically."""

    move_name: str
    base_revision: int
    status: MoveStatus
    invalid_reason: str | None = None
    added_beads: tuple[BeadRecord, ...] = ()
    removed_beads: tuple[BeadRecord, ...] = ()
    position_changes: tuple[PositionChange, ...] = ()
    link_changes: tuple[LinkChange, ...] = ()
    sector_before: Sector = Sector.Z
    sector_after: Sector = Sector.Z
    endpoints_before: EndpointState = EndpointState(-1, -1)
    endpoints_after: EndpointState = EndpointState(-1, -1)
    affected_slices: tuple[int, ...] = ()
    metadata: tuple[tuple[str, MetadataValue], ...] = ()

    def metadata_dict(self) -> dict[str, MetadataValue]:
        """Return a fresh dictionary of immutable proposal metadata."""

        return dict(self.metadata)

    def reversed(self, *, base_revision: int) -> "ProposalPatch":
        """Construct the exact inverse patch for round-trip tests."""

        if self.status is not MoveStatus.PROPOSED:
            raise ValueError("only a proposed patch has a reversible state delta")
        positions = tuple(
            PositionChange(change.bead_id, change.after, change.before)
            for change in self.position_changes
        )
        links = tuple(
            LinkChange(
                bead_id=change.bead_id,
                before_next=change.after_next,
                after_next=change.before_next,
                before_prev=change.after_prev,
                after_prev=change.before_prev,
                before_image=change.after_image,
                after_image=change.before_image,
            )
            for change in self.link_changes
        )
        metadata = self.metadata_dict()
        forward = metadata.get("log_q_forward")
        reverse = metadata.get("log_q_reverse")
        if isinstance(forward, (int, float)) and isinstance(
            reverse, (int, float)
        ):
            metadata["log_q_forward"] = float(reverse)
            metadata["log_q_reverse"] = float(forward)
        displacement = metadata.get("displacement")
        if isinstance(displacement, tuple):
            metadata["displacement"] = tuple(-float(x) for x in displacement)
        return replace(
            self,
            base_revision=base_revision,
            added_beads=self.removed_beads,
            removed_beads=self.added_beads,
            position_changes=positions,
            link_changes=links,
            sector_before=self.sector_after,
            sector_after=self.sector_before,
            endpoints_before=self.endpoints_after,
            endpoints_after=self.endpoints_before,
            metadata=tuple(metadata.items()),
        )


@dataclass(frozen=True, slots=True)
class TargetDelta:
    """Auditable change in each target-measure component."""

    kinetic: float
    potential: float
    chemical: float
    sector_measure: float

    @property
    def total(self) -> float:
        return self.kinetic + self.potential + self.chemical + self.sector_measure


@dataclass(frozen=True, slots=True)
class ProposalRatio:
    """Complete reverse-over-forward proposal probability decomposition."""

    log_q_forward: float
    log_q_reverse: float
    log_move_selection_forward: float
    log_move_selection_reverse: float

    @property
    def total(self) -> float:
        return (
            self.log_q_reverse - self.log_q_forward
        ) + (
            self.log_move_selection_reverse
            - self.log_move_selection_forward
        )


@dataclass(frozen=True, slots=True)
class AcceptanceBreakdown:
    """Complete log-space Metropolis--Hastings decision record."""

    kinetic: float
    potential: float
    chemical: float
    sector_measure: float
    proposal_selection: float
    log_uniform: float = float("nan")
    accepted: bool = False

    @property
    def log_ratio(self) -> float:
        """Return the sum fixed by derivations.md ``eq-metropolis-log-ratio``."""

        return (
            self.kinetic
            + self.potential
            + self.chemical
            + self.sector_measure
            + self.proposal_selection
        )
