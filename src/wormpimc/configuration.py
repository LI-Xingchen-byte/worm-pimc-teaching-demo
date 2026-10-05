"""Validated structure-of-arrays storage for closed worldline configurations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import ConfigError, SimulationConfig
from .geometry import (
    image_resolved_displacement,
    link_image_from_unwrapped,
    winding_from_images,
    wrap,
)
from .propagator import sample_brownian_bridge, winding_distribution
from .types import BeadRecord, MoveStatus, ProposalPatch, Sector


NONE = -1


class InvariantViolation(RuntimeError):
    """Raised when a live worldline violates a mandatory graph invariant."""


class StalePatchError(RuntimeError):
    """Raised when a patch no longer matches the live configuration revision."""


def _readonly_copy(value: ArrayLike, dtype: np.dtype[np.generic]) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _readonly_view(value: NDArray) -> NDArray:
    result = value.view()
    result.setflags(write=False)
    return result


class Configuration:
    """The live Z/G worldline graph with one controlled mutation entry point."""

    def __init__(
        self,
        *,
        positions: ArrayLike,
        slice_of: ArrayLike,
        next_of: ArrayLike,
        prev_of: ArrayLike,
        image_to_next: ArrayLike,
        active: ArrayLike,
        box_length: float,
        n_slices: int,
        sector: Sector = Sector.Z,
        worm_head: int = NONE,
        worm_tail: int = NONE,
        revision: int = 0,
    ) -> None:
        self._positions = _readonly_copy(positions, np.dtype(np.float64))
        self._slice_of = _readonly_copy(slice_of, np.dtype(np.int64))
        self._next_of = _readonly_copy(next_of, np.dtype(np.int64))
        self._prev_of = _readonly_copy(prev_of, np.dtype(np.int64))
        self._image_to_next = _readonly_copy(
            image_to_next, np.dtype(np.int64)
        )
        self._active = _readonly_copy(active, np.dtype(np.bool_))
        self._box_length = float(box_length)
        self._n_slices = int(n_slices)
        self._sector = Sector(sector)
        self._worm_head = int(worm_head)
        self._worm_tail = int(worm_tail)
        self._revision = int(revision)
        self.validate()

    @classmethod
    def from_closed_worldline(
        cls,
        unwrapped_path: ArrayLike,
        box_length: float,
    ) -> "Configuration":
        """Build one closed cycle from an unwrapped path with both endpoints."""

        return cls.from_closed_worldlines([unwrapped_path], box_length)

    @classmethod
    def from_closed_worldlines(
        cls,
        unwrapped_paths: Sequence[ArrayLike],
        box_length: float,
    ) -> "Configuration":
        """Build independent closed cycles with a common slice grid."""

        if not unwrapped_paths:
            raise ValueError("at least one closed worldline is required")
        length = float(box_length)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("box_length must be positive and finite")

        paths: list[NDArray[np.float64]] = []
        expected_shape: tuple[int, int] | None = None
        for source in unwrapped_paths:
            path = np.asarray(source, dtype=np.float64)
            if path.ndim == 1:
                path = path[:, np.newaxis]
            if path.ndim != 2 or path.shape[0] < 3 or path.shape[1] < 1:
                raise ValueError(
                    "each path must contain at least two links and one dimension"
                )
            if not np.all(np.isfinite(path)):
                raise ValueError("unwrapped paths must contain only finite values")
            if expected_shape is None:
                expected_shape = path.shape
            elif path.shape != expected_shape:
                raise ValueError("all closed worldlines must use the same grid")
            if not np.allclose(
                np.asarray(wrap(path[-1], length)),
                np.asarray(wrap(path[0], length)),
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise ValueError(
                    "closed path endpoints must agree modulo the box length"
                )
            paths.append(path)

        assert expected_shape is not None
        n_slices = expected_shape[0] - 1
        ndim = expected_shape[1]
        capacity = len(paths) * n_slices
        positions = np.empty((capacity, ndim), dtype=np.float64)
        slice_of = np.empty(capacity, dtype=np.int64)
        next_of = np.empty(capacity, dtype=np.int64)
        prev_of = np.empty(capacity, dtype=np.int64)
        images = np.empty((capacity, ndim), dtype=np.int64)

        for cycle_index, path in enumerate(paths):
            offset = cycle_index * n_slices
            ids = offset + np.arange(n_slices, dtype=np.int64)
            positions[ids] = np.asarray(wrap(path[:-1], length))
            slice_of[ids] = np.arange(n_slices, dtype=np.int64)
            next_of[ids] = np.roll(ids, -1)
            prev_of[ids] = np.roll(ids, 1)
            for local_index, bead_id in enumerate(ids):
                images[bead_id] = link_image_from_unwrapped(
                    path[local_index], path[local_index + 1], length
                )

        return cls(
            positions=positions,
            slice_of=slice_of,
            next_of=next_of,
            prev_of=prev_of,
            image_to_next=images,
            active=np.ones(capacity, dtype=np.bool_),
            box_length=length,
            n_slices=n_slices,
        )

    @classmethod
    def from_straight_worldlines(
        cls,
        *,
        particle_count: int,
        n_slices: int,
        ndim: int,
        box_length: float,
    ) -> "Configuration":
        """Create separated zero-winding cycles without consuming RNG state."""

        if type(particle_count) is not int or particle_count < 1:
            raise ValueError("particle_count must be a positive integer")
        if type(n_slices) is not int or n_slices < 2:
            raise ValueError("n_slices must be an integer of at least two")
        if type(ndim) is not int or ndim < 1:
            raise ValueError("ndim must be a positive integer")
        length = float(box_length)
        paths: list[NDArray[np.float64]] = []
        for particle_index in range(particle_count):
            point = np.full(ndim, 0.5 * length, dtype=np.float64)
            point[0] = (particle_index + 0.5) * length / particle_count
            paths.append(np.repeat(point[np.newaxis, :], n_slices + 1, axis=0))
        return cls.from_closed_worldlines(paths, length)

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "Configuration":
        """Restore a validated configuration from checkpoint fields."""

        required = {
            "positions",
            "slice_of",
            "next_of",
            "prev_of",
            "image_to_next",
            "active",
            "box_length",
            "n_slices",
            "sector",
            "worm_head",
            "worm_tail",
            "revision",
        }
        missing = sorted(required - set(snapshot))
        if missing:
            raise ValueError("configuration snapshot missing: " + ", ".join(missing))
        return cls(
            positions=snapshot["positions"],
            slice_of=snapshot["slice_of"],
            next_of=snapshot["next_of"],
            prev_of=snapshot["prev_of"],
            image_to_next=snapshot["image_to_next"],
            active=snapshot["active"],
            box_length=float(snapshot["box_length"]),
            n_slices=int(snapshot["n_slices"]),
            sector=Sector(str(snapshot["sector"])),
            worm_head=int(snapshot["worm_head"]),
            worm_tail=int(snapshot["worm_tail"]),
            revision=int(snapshot["revision"]),
        )

    @property
    def positions(self) -> NDArray[np.float64]:
        return _readonly_view(self._positions)

    @property
    def box_length(self) -> float:
        return self._box_length

    @property
    def n_slices(self) -> int:
        return self._n_slices

    @property
    def sector(self) -> Sector:
        return self._sector

    @property
    def worm_head(self) -> int:
        return self._worm_head

    @property
    def worm_tail(self) -> int:
        return self._worm_tail

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def slice_of(self) -> NDArray[np.int64]:
        return _readonly_view(self._slice_of)

    @property
    def next_of(self) -> NDArray[np.int64]:
        return _readonly_view(self._next_of)

    @property
    def prev_of(self) -> NDArray[np.int64]:
        return _readonly_view(self._prev_of)

    @property
    def image_to_next(self) -> NDArray[np.int64]:
        return _readonly_view(self._image_to_next)

    @property
    def active(self) -> NDArray[np.bool_]:
        return _readonly_view(self._active)

    @property
    def active_ids(self) -> NDArray[np.int64]:
        result = np.flatnonzero(self._active).astype(np.int64, copy=False)
        result.setflags(write=False)
        return result

    @property
    def capacity(self) -> int:
        return int(self._active.size)

    @property
    def ndim(self) -> int:
        return int(self._positions.shape[1])

    @property
    def number_of_beads(self) -> int:
        return int(np.count_nonzero(self._active))

    @property
    def n_particles(self) -> int:
        """Return the constant number of occupied links per time interval."""

        if self.sector is not Sector.Z:
            raise ValueError("n_particles is defined only in sector Z")
        return int(self.link_occupation_by_interval()[0])

    def link_occupation_by_interval(self) -> NDArray[np.int64]:
        """Return ``N_j`` from outgoing link counts on every slice."""

        occupied = self._active & (self._next_of != NONE)
        result = np.bincount(
            self._slice_of[occupied],
            minlength=self.n_slices,
        ).astype(np.int64, copy=False)
        result.setflags(write=False)
        return result

    @property
    def total_occupied_links(self) -> int:
        return int(np.count_nonzero(self._active & (self._next_of != NONE)))

    def bead_ids_on_slice(self, slice_id: int) -> NDArray[np.int64]:
        if type(slice_id) is not int or not 0 <= slice_id < self.n_slices:
            raise ValueError("slice_id lies outside [0, n_slices)")
        result = np.flatnonzero(self._active & (self._slice_of == slice_id))
        result = result.astype(np.int64, copy=False)
        result.setflags(write=False)
        return result

    def positions_on_slice(self, slice_id: int) -> NDArray[np.float64]:
        result = np.array(
            self._positions[self.bead_ids_on_slice(slice_id)], copy=True
        )
        result.setflags(write=False)
        return result

    def incoming_positions_on_slice(
        self,
        slice_id: int,
    ) -> NDArray[np.float64]:
        ids = self.bead_ids_on_slice(slice_id)
        result = np.array(
            self._positions[ids[self._prev_of[ids] != NONE]],
            copy=True,
        )
        result.setflags(write=False)
        return result

    def outgoing_positions_on_slice(
        self,
        slice_id: int,
    ) -> NDArray[np.float64]:
        ids = self.bead_ids_on_slice(slice_id)
        result = np.array(
            self._positions[ids[self._next_of[ids] != NONE]],
            copy=True,
        )
        result.setflags(write=False)
        return result

    def bead_record(self, bead_id: int) -> BeadRecord:
        """Return a complete immutable record for one allocated bead."""

        if bead_id < 0 or bead_id >= self.capacity:
            raise ValueError("bead_id lies outside the allocated capacity")
        return BeadRecord(
            bead_id=bead_id,
            position=tuple(float(x) for x in self._positions[bead_id]),
            slice_id=int(self._slice_of[bead_id]),
            next_id=int(self._next_of[bead_id]),
            prev_id=int(self._prev_of[bead_id]),
            image_to_next=tuple(
                int(x) for x in self._image_to_next[bead_id]
            ),
            active=bool(self._active[bead_id]),
        )

    def preview_bead_ids(self, count: int) -> tuple[int, ...]:
        """Return deterministic free/append IDs without changing live state."""

        if type(count) is not int or count < 0:
            raise ValueError("count must be a nonnegative integer")
        free = [
            int(bead_id)
            for bead_id in np.flatnonzero(~self._active)[:count]
        ]
        missing = count - len(free)
        free.extend(range(self.capacity, self.capacity + missing))
        return tuple(free)

    def _cycle_ids(self, start: int) -> Iterator[int]:
        if start < 0 or start >= self.capacity or not self._active[start]:
            raise ValueError("start must identify an active bead")
        current = start
        for _ in range(self.number_of_beads + 1):
            yield current
            current = int(self._next_of[current])
            if current == NONE:
                raise ValueError("start belongs to the open component")
            if current == start:
                return
        raise InvariantViolation("cycle traversal did not return to its start")

    def cycle_ids(self, start: int = 0) -> tuple[int, ...]:
        """Return one directed cycle in traversal order."""

        return tuple(self._cycle_ids(start))

    def cycle_starts(self) -> tuple[int, ...]:
        """Return the smallest bead ID in each cycle, sorted by ID."""

        starts: list[int] = []
        unvisited = set(int(bead_id) for bead_id in self.active_ids)
        if self.sector is Sector.G:
            unvisited.difference_update(self.open_chain_ids())
        while unvisited:
            candidate = min(unvisited)
            ids = self.cycle_ids(candidate)
            starts.append(min(ids))
            unvisited.difference_update(ids)
        return tuple(sorted(starts))

    def open_chain_ids(self) -> tuple[int, ...]:
        """Return tail-to-head IDs for the unique open component."""

        if self.sector is not Sector.G:
            raise ValueError("open_chain_ids requires sector G")
        ids: list[int] = []
        current = self.worm_tail
        for _ in range(self.number_of_beads + 1):
            if current == NONE:
                raise InvariantViolation("open chain ended before reaching head")
            ids.append(current)
            if current == self.worm_head:
                return tuple(ids)
            current = int(self._next_of[current])
        raise InvariantViolation("open-chain traversal exceeded active bead count")

    @property
    def open_chain_links(self) -> int:
        return len(self.open_chain_ids()) - 1

    def predecessor_ids(self, end: int, n_links: int) -> tuple[int, ...]:
        """Return ``end, prev(end), ...`` for exactly ``n_links`` links."""

        if type(n_links) is not int or n_links < 0:
            raise ValueError("n_links must be a nonnegative integer")
        if end < 0 or end >= self.capacity or not self._active[end]:
            raise ValueError("end must identify an active bead")
        ids = [end]
        current = end
        for _ in range(n_links):
            current = int(self._prev_of[current])
            if current == NONE:
                raise ValueError("predecessor chain is shorter than n_links")
            ids.append(current)
        return tuple(ids)

    def unwrapped_cycle(self, start: int = 0) -> NDArray[np.float64]:
        """Reconstruct a cycle as coordinates including its terminal image."""

        ids = self.cycle_ids(start)
        path = np.empty((len(ids) + 1, self.ndim), dtype=np.float64)
        path[0] = self._positions[start]
        for index, bead_id in enumerate(ids, start=1):
            next_id = int(self._next_of[bead_id])
            delta = image_resolved_displacement(
                self._positions[bead_id],
                self._positions[next_id],
                self._image_to_next[bead_id],
                self.box_length,
            )
            path[index] = path[index - 1] + np.asarray(delta)
        return path

    def unwrapped_segment(
        self,
        start: int,
        n_links: int,
    ) -> tuple[tuple[int, ...], NDArray[np.float64]]:
        """Return bead IDs and unwrapped coordinates for a directed segment."""

        if type(n_links) is not int or n_links < 1:
            raise ValueError("n_links must be a positive integer")
        if start < 0 or start >= self.capacity or not self._active[start]:
            raise ValueError("start must identify an active bead")
        ids = [start]
        path = np.empty((n_links + 1, self.ndim), dtype=np.float64)
        path[0] = self._positions[start]
        current = start
        for index in range(n_links):
            next_id = int(self._next_of[current])
            if next_id == NONE:
                raise ValueError("directed segment reaches the worm head")
            delta = image_resolved_displacement(
                self._positions[current],
                self._positions[next_id],
                self._image_to_next[current],
                self.box_length,
            )
            path[index + 1] = path[index] + np.asarray(delta)
            ids.append(next_id)
            current = next_id
        return tuple(ids), path

    def cycle_winding(self, start: int = 0) -> NDArray[np.int64]:
        """Return the integer winding vector of one directed cycle."""

        ids = self.cycle_ids(start)
        result = winding_from_images(self._image_to_next[list(ids)])
        return np.asarray(result, dtype=np.int64)

    def total_winding(self) -> NDArray[np.int64]:
        """Return the sum of winding vectors over all closed components."""

        result = np.zeros(self.ndim, dtype=np.int64)
        for start in self.cycle_starts():
            result += self.cycle_winding(start)
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return a complete defensive-copy snapshot for checkpointing."""

        return {
            "positions": np.array(self._positions, copy=True),
            "slice_of": np.array(self._slice_of, copy=True),
            "next_of": np.array(self._next_of, copy=True),
            "prev_of": np.array(self._prev_of, copy=True),
            "image_to_next": np.array(self._image_to_next, copy=True),
            "active": np.array(self._active, copy=True),
            "box_length": self.box_length,
            "n_slices": self.n_slices,
            "sector": self.sector.value,
            "worm_head": self.worm_head,
            "worm_tail": self.worm_tail,
            "revision": self.revision,
        }

    def validate_against(self, config: SimulationConfig) -> None:
        """Check geometry against the model, without constraining live N."""

        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig")
        for name, expected in (
            ("ndim", config.system.ndim),
            ("box_length", config.system.box_length),
            ("n_slices", config.discretization.n_slices),
        ):
            actual = getattr(self, name)
            if actual != expected:
                raise ConfigError(f"state.{name}={actual} does not match config {expected}")
        self.validate()

    def state_digest(self, *, include_revision: bool = False) -> str:
        """Return a deterministic physical-state digest.

        Revision is excluded by default because applying a forward and reverse
        patch restores the physical state while correctly advancing revision.
        Checkpoint tests can request it explicitly.
        """

        digest = hashlib.sha256()
        active_ids = self.active_ids
        digest.update(active_ids.tobytes(order="C"))
        for array in (
            self._positions[active_ids],
            self._slice_of[active_ids],
            self._next_of[active_ids],
            self._prev_of[active_ids],
            self._image_to_next[active_ids],
        ):
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        digest.update(
            repr(
                (
                    self.box_length,
                    self.n_slices,
                    self.sector.value,
                    self.worm_head,
                    self.worm_tail,
                )
            ).encode("ascii")
        )
        if include_revision:
            digest.update(str(self.revision).encode("ascii"))
        return digest.hexdigest()

    def preview(self, patch: ProposalPatch) -> "Configuration":
        """Validate a patch against this state and return an uncommitted copy."""

        if patch.status is not MoveStatus.PROPOSED:
            raise ValueError("only a proposed patch can be previewed")
        if patch.base_revision != self.revision:
            raise StalePatchError(
                f"patch revision {patch.base_revision} does not match "
                f"configuration revision {self.revision}"
            )
        if patch.sector_before is not self.sector:
            raise InvariantViolation("patch sector_before does not match live state")
        if (
            patch.endpoints_before.worm_head != self.worm_head
            or patch.endpoints_before.worm_tail != self.worm_tail
        ):
            raise InvariantViolation("patch endpoint state does not match live state")

        added_ids = {record.bead_id for record in patch.added_beads}
        removed_ids = {record.bead_id for record in patch.removed_beads}
        if len(added_ids) != len(patch.added_beads):
            raise InvariantViolation("patch adds one bead more than once")
        if len(removed_ids) != len(patch.removed_beads):
            raise InvariantViolation("patch removes one bead more than once")
        if added_ids & removed_ids:
            raise InvariantViolation("a patch may not add and remove the same bead")
        maximum_id = max(added_ids, default=self.capacity - 1)
        capacity = max(self.capacity, maximum_id + 1)
        positions = np.zeros((capacity, self.ndim), dtype=np.float64)
        slice_of = np.zeros(capacity, dtype=np.int64)
        next_of = np.full(capacity, NONE, dtype=np.int64)
        prev_of = np.full(capacity, NONE, dtype=np.int64)
        images = np.zeros((capacity, self.ndim), dtype=np.int64)
        active = np.zeros(capacity, dtype=np.bool_)
        positions[: self.capacity] = self._positions
        slice_of[: self.capacity] = self._slice_of
        next_of[: self.capacity] = self._next_of
        prev_of[: self.capacity] = self._prev_of
        images[: self.capacity] = self._image_to_next
        active[: self.capacity] = self._active

        for record in patch.removed_beads:
            bead_id = int(record.bead_id)
            if bead_id < 0 or bead_id >= self.capacity or not self._active[bead_id]:
                raise InvariantViolation("removed bead is not active")
            if record != self.bead_record(bead_id):
                raise StalePatchError("removed bead record does not match live state")

        for record in patch.added_beads:
            bead_id = int(record.bead_id)
            if bead_id < 0:
                raise InvariantViolation("added bead ID must be nonnegative")
            if bead_id < self.capacity and self._active[bead_id]:
                raise InvariantViolation("added bead ID is already active")
            if not record.active:
                raise InvariantViolation("an added bead record must be active")
            if len(record.position) != self.ndim:
                raise InvariantViolation("added bead position has wrong dimension")
            if len(record.image_to_next) != self.ndim:
                raise InvariantViolation("added bead image has wrong dimension")
            positions[bead_id] = np.asarray(record.position, dtype=np.float64)
            slice_of[bead_id] = int(record.slice_id)
            next_of[bead_id] = int(record.next_id)
            prev_of[bead_id] = int(record.prev_id)
            images[bead_id] = np.asarray(record.image_to_next, dtype=np.int64)
            active[bead_id] = True

        changed_position_ids: set[int] = set()
        for change in patch.position_changes:
            bead_id = int(change.bead_id)
            if bead_id in changed_position_ids:
                raise InvariantViolation("patch changes one position more than once")
            changed_position_ids.add(bead_id)
            if (
                bead_id < 0
                or bead_id >= self.capacity
                or not self._active[bead_id]
                or bead_id in removed_ids
            ):
                raise InvariantViolation(
                    "position change must reference a retained active bead"
                )
            if not np.array_equal(positions[bead_id], np.asarray(change.before)):
                raise StalePatchError("position before-value does not match live state")
            positions[bead_id] = np.asarray(change.after, dtype=np.float64)

        changed_link_ids: set[int] = set()
        for change in patch.link_changes:
            bead_id = int(change.bead_id)
            if bead_id in changed_link_ids:
                raise InvariantViolation("patch changes one link more than once")
            changed_link_ids.add(bead_id)
            if (
                bead_id < 0
                or bead_id >= self.capacity
                or not self._active[bead_id]
                or bead_id in removed_ids
            ):
                raise InvariantViolation(
                    "link change must reference a retained active bead"
                )
            if (
                int(next_of[bead_id]) != change.before_next
                or int(prev_of[bead_id]) != change.before_prev
                or not np.array_equal(images[bead_id], change.before_image)
            ):
                raise StalePatchError("link before-value does not match live state")
            next_of[bead_id] = change.after_next
            prev_of[bead_id] = change.after_prev
            images[bead_id] = np.asarray(change.after_image, dtype=np.int64)

        for bead_id in removed_ids:
            positions[bead_id] = 0.0
            slice_of[bead_id] = 0
            next_of[bead_id] = NONE
            prev_of[bead_id] = NONE
            images[bead_id] = 0
            active[bead_id] = False

        return Configuration(
            positions=positions,
            slice_of=slice_of,
            next_of=next_of,
            prev_of=prev_of,
            image_to_next=images,
            active=active,
            box_length=self.box_length,
            n_slices=self.n_slices,
            sector=patch.sector_after,
            worm_head=patch.endpoints_after.worm_head,
            worm_tail=patch.endpoints_after.worm_tail,
            revision=self.revision + 1,
        )

    def apply(self, patch: ProposalPatch) -> None:
        """Atomically commit a validated patch and increment the revision."""

        candidate = self.preview(patch)
        self._positions = candidate._positions
        self._slice_of = candidate._slice_of
        self._next_of = candidate._next_of
        self._prev_of = candidate._prev_of
        self._image_to_next = candidate._image_to_next
        self._active = candidate._active
        self._sector = candidate.sector
        self._worm_head = candidate.worm_head
        self._worm_tail = candidate.worm_tail
        self._revision = candidate.revision

    def validate(self) -> None:
        """Check the mandatory Z/G graph invariants from derivations.md."""

        if not np.isfinite(self.box_length) or self.box_length <= 0.0:
            raise InvariantViolation("box_length must be positive and finite")
        if self.n_slices < 1:
            raise InvariantViolation("n_slices must be positive")
        if self._positions.ndim != 2:
            raise InvariantViolation("positions must have shape (capacity, ndim)")
        capacity, ndim = self._positions.shape
        if capacity == 0 or ndim == 0:
            raise InvariantViolation("configuration arrays may not be empty")
        if self._image_to_next.shape != (capacity, ndim):
            raise InvariantViolation("image_to_next has inconsistent shape")
        for name, array in (
            ("slice_of", self._slice_of),
            ("next_of", self._next_of),
            ("prev_of", self._prev_of),
            ("active", self._active),
        ):
            if array.shape != (capacity,):
                raise InvariantViolation(f"{name} has inconsistent shape")
        if self.revision < 0:
            raise InvariantViolation("revision may not be negative")

        active_ids = np.flatnonzero(self._active)
        positions = self._positions[active_ids]
        if not np.all(np.isfinite(positions)):
            raise InvariantViolation("active positions must be finite")
        if np.any(positions < 0.0) or np.any(positions >= self.box_length):
            raise InvariantViolation("wrapped coordinates must lie in [0, L)")

        if self.sector is Sector.Z:
            if self.worm_head != NONE or self.worm_tail != NONE:
                raise InvariantViolation("Z-sector configurations have no endpoints")
        else:
            if self.worm_head == self.worm_tail:
                raise InvariantViolation("G-sector endpoints must be distinct")
            for name, endpoint in (
                ("worm_head", self.worm_head),
                ("worm_tail", self.worm_tail),
            ):
                if (
                    endpoint < 0
                    or endpoint >= capacity
                    or not self._active[endpoint]
                ):
                    raise InvariantViolation(f"{name} must be an active bead")
            if int(self._next_of[self.worm_head]) != NONE:
                raise InvariantViolation("worm head must have no outgoing link")
            if int(self._prev_of[self.worm_tail]) != NONE:
                raise InvariantViolation("worm tail must have no incoming link")
            if int(self._prev_of[self.worm_head]) == NONE:
                raise InvariantViolation("worm head must have an incoming link")
            if int(self._next_of[self.worm_tail]) == NONE:
                raise InvariantViolation("worm tail must have an outgoing link")

        occupied = np.zeros(self.n_slices, dtype=np.int64)
        for bead_id_value in active_ids:
            bead_id = int(bead_id_value)
            next_id = int(self._next_of[bead_id])
            prev_id = int(self._prev_of[bead_id])
            slice_id = int(self._slice_of[bead_id])
            if slice_id < 0 or slice_id >= self.n_slices:
                raise InvariantViolation("slice index lies outside [0, M)")

            if next_id == NONE:
                if self.sector is not Sector.G or bead_id != self.worm_head:
                    raise InvariantViolation("only the worm head may lack next")
                if np.any(self._image_to_next[bead_id] != 0):
                    raise InvariantViolation("a missing link must have zero image")
            else:
                if (
                    next_id < 0
                    or next_id >= capacity
                    or not self._active[next_id]
                ):
                    raise InvariantViolation("next must reference an active bead")
                if int(self._prev_of[next_id]) != bead_id:
                    raise InvariantViolation("next/prev reciprocity is broken")
                expected_next = (slice_id + 1) % self.n_slices
                if int(self._slice_of[next_id]) != expected_next:
                    raise InvariantViolation("a link does not advance one time slice")
                occupied[slice_id] += 1

            if prev_id == NONE:
                if self.sector is not Sector.G or bead_id != self.worm_tail:
                    raise InvariantViolation("only the worm tail may lack prev")
            else:
                if (
                    prev_id < 0
                    or prev_id >= capacity
                    or not self._active[prev_id]
                ):
                    raise InvariantViolation("prev must reference an active bead")
                if int(self._next_of[prev_id]) != bead_id:
                    raise InvariantViolation("prev/next reciprocity is broken")

        if self.sector is Sector.Z:
            if not np.all(occupied == occupied[0]):
                raise InvariantViolation("Z-sector link occupation is not constant")
        else:
            for slice_id in range(self.n_slices):
                previous = (slice_id - 1) % self.n_slices
                expected_jump = 0
                if slice_id == int(self._slice_of[self.worm_tail]):
                    expected_jump += 1
                if slice_id == int(self._slice_of[self.worm_head]):
                    expected_jump -= 1
                if int(occupied[slice_id] - occupied[previous]) != expected_jump:
                    raise InvariantViolation(
                        "G-sector link occupation has the wrong endpoint jump"
                    )

        unvisited = set(int(bead_id) for bead_id in active_ids)
        if self.sector is Sector.G:
            open_ids = self.open_chain_ids()
            if len(open_ids) < 2:
                raise InvariantViolation("the open component needs at least one link")
            if len(set(open_ids)) != len(open_ids):
                raise InvariantViolation("the open component visits a bead twice")
            if open_ids[0] != self.worm_tail or open_ids[-1] != self.worm_head:
                raise InvariantViolation("open component endpoints are inconsistent")
            for bead_id in open_ids:
                if bead_id not in unvisited:
                    raise InvariantViolation("open component overlaps itself")
                unvisited.remove(bead_id)
        while unvisited:
            start = min(unvisited)
            ids = self.cycle_ids(start)
            if len(ids) % self.n_slices != 0:
                raise InvariantViolation(
                    "closed-cycle length is not a multiple of n_slices"
                )
            if len(set(ids)) != len(ids):
                raise InvariantViolation("a cycle visits a bead more than once")
            for bead_id in ids:
                if bead_id not in unvisited:
                    raise InvariantViolation("closed cycles overlap")
                unvisited.remove(bead_id)
            winding = winding_from_images(self._image_to_next[list(ids)])
            if not np.issubdtype(np.asarray(winding).dtype, np.integer):
                raise InvariantViolation("cycle winding must be integral")


def initialize_configuration(config: SimulationConfig) -> Configuration:
    """Construct the deterministic closed-sector initial state."""

    if config.initialization.strategy != "straight_worldlines":
        raise ValueError("unsupported initialization strategy")
    return Configuration.from_straight_worldlines(
        particle_count=config.initialization.particle_count,
        n_slices=config.discretization.n_slices,
        ndim=config.system.ndim,
        box_length=config.system.box_length,
    )


def sample_free_closed_worldline(
    config: SimulationConfig,
    rng: np.random.Generator,
    *,
    winding: int | None = None,
    tail_tolerance: float = 1.0e-14,
) -> Configuration:
    """Draw an independent one-particle path from the exact free measure."""

    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if config.system.ndim != 1:
        raise ValueError("free closed-path sampling supports ndim = 1")
    if config.potential.pair != "none" or config.potential.external != "none":
        raise ValueError("free closed-path sampling requires zero potential")
    if config.initialization.particle_count != 1:
        raise ValueError("free closed-path reference requires particle_count = 1")

    if winding is None:
        distribution = winding_distribution(
            config.system.box_length,
            config.system.lambda_kin,
            config.system.beta,
            tail_tolerance=tail_tolerance,
        )
        selected_winding = distribution.sample(rng)
    elif type(winding) is int:
        selected_winding = winding
    else:
        raise TypeError("winding must be an integer or None")

    start = rng.uniform(0.0, config.system.box_length, size=1)
    end = start + config.system.box_length * selected_winding
    unwrapped_path = sample_brownian_bridge(
        start,
        end,
        config.discretization.n_slices,
        config.system.lambda_kin,
        config.tau,
        rng,
    )
    return Configuration.from_closed_worldline(
        unwrapped_path,
        config.system.box_length,
    )
