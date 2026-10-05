"""Auditable log-space Metropolis sampler for the extended Z/G ensemble."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from .config import MOVE_NAMES, SimulationConfig
from .configuration import Configuration
from .measure import PrimitiveTargetMeasure, assert_local_matches_full
from .moves import (
    AdvanceMove,
    CloseMove,
    DisplaceMove,
    InsertMove,
    Move,
    OpenMove,
    RecedeMove,
    RemoveMove,
    SwapMove,
    WiggleMove,
)
from .types import AcceptanceBreakdown, MoveStatus, ProposalPatch, ProposalRatio


def _trace_snapshot(state: Configuration) -> Mapping[str, Any]:
    """Detached, read-only data: neither the mapping nor arrays can be edited."""

    snapshot = state.snapshot()
    for name, value in snapshot.items():
        if isinstance(value, np.ndarray):
            snapshot[name] = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    return MappingProxyType(snapshot)


@dataclass(slots=True)
class MoveCounter:
    """Mutable summary for one update class."""

    selected: int = 0
    not_applicable: int = 0
    structurally_invalid: int = 0
    action_evaluated: int = 0
    accepted: int = 0
    sum_kinetic: float = 0.0
    sum_potential: float = 0.0
    sum_chemical: float = 0.0
    sum_sector_measure: float = 0.0
    sum_proposal_selection: float = 0.0

    @property
    def acceptance_given_evaluated(self) -> float:
        if self.action_evaluated == 0:
            return float("nan")
        return self.accepted / self.action_evaluated

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoveCounter":
        allowed = set(cls.__dataclass_fields__)
        if set(data) != allowed:
            raise ValueError("move counter fields do not match checkpoint schema")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result of one atomic update-class selection and proposal attempt."""

    move_name: str
    status: MoveStatus
    patch: ProposalPatch | None
    breakdown: AcceptanceBreakdown | None
    proposal_ratio: ProposalRatio | None = None
    before: Mapping[str, Any] | None = field(default=None, compare=False, repr=False)
    after: Mapping[str, Any] | None = field(default=None, compare=False, repr=False)
    selection_mode: str = "random"

    @property
    def accepted(self) -> bool:
        return bool(self.breakdown and self.breakdown.accepted)

    def candidate(self) -> Configuration | None:
        """Rebuild this proposal from its trace, without touching live state.

        Rejected proposals also have candidates. Inapplicable/invalid moves
        return None. Call step(trace=True) to retain the required snapshot.
        """

        if self.before is None:
            raise ValueError("candidate() requires step(trace=True)")
        if self.patch is None or self.status is not MoveStatus.PROPOSED:
            return None
        return Configuration.from_snapshot(self.before).preview(self.patch)


class Sampler:
    """Compose moves and target measure without embedding model formulas."""

    def __init__(
        self,
        config: SimulationConfig,
        state: Configuration,
        rng: np.random.Generator,
        *,
        measure: PrimitiveTargetMeasure | None = None,
        verify_local_delta: bool = False,
    ) -> None:
        if not isinstance(state, Configuration):
            raise TypeError("state must be a Configuration")
        state.validate_against(config)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        self.config = config
        self.state = state
        self.rng = rng
        self.measure = measure or PrimitiveTargetMeasure(config)
        self.verify_local_delta = bool(verify_local_delta)
        probabilities = config.moves.weights.normalized()
        self._move_names = tuple(MOVE_NAMES)
        self._probabilities = np.array(
            [probabilities[name] for name in self._move_names],
            dtype=np.float64,
        )
        self._moves: dict[str, Move] = {}
        if probabilities["wiggle"] > 0.0:
            self._moves["wiggle"] = WiggleMove(
                config, probabilities["wiggle"]
            )
        if probabilities["displace"] > 0.0:
            self._moves["displace"] = DisplaceMove(
                config, probabilities["displace"]
            )
        if probabilities["open"] > 0.0:
            self._moves["open"] = OpenMove(
                config,
                probabilities["open"],
                probabilities["close"],
            )
            self._moves["close"] = CloseMove(
                config,
                probabilities["close"],
                probabilities["open"],
            )
        if probabilities["insert"] > 0.0:
            self._moves["insert"] = InsertMove(
                config,
                probabilities["insert"],
                probabilities["remove"],
            )
            self._moves["remove"] = RemoveMove(
                config,
                probabilities["remove"],
                probabilities["insert"],
            )
        if probabilities["advance"] > 0.0:
            self._moves["advance"] = AdvanceMove(
                config,
                probabilities["advance"],
                probabilities["recede"],
            )
            self._moves["recede"] = RecedeMove(
                config,
                probabilities["recede"],
                probabilities["advance"],
            )
        if probabilities["swap"] > 0.0:
            self._moves["swap"] = SwapMove(
                config, probabilities["swap"]
            )
        self.counters = {name: MoveCounter() for name in self._move_names}
        self.sector_visits = {"Z": 0, "G": 0}
        self.atomic_steps = 0

    def step(self, *, move: str | None = None, trace: bool = False) -> StepResult:
        """Attempt one update, optionally specifying its family for teaching.

        Specifying move skips only the selector draw. Acceptance still uses
        the configured forward/reverse selection weights; a manually arranged
        sequence is not the configured random ensemble sampler.
        """

        if move is not None:
            if not isinstance(move, str) or move not in self._move_names:
                raise ValueError(f"unknown move {move!r}; choose from {', '.join(self._move_names)}")
            if move not in self._moves:
                raise ValueError(f"move {move!r} is disabled by configuration (zero weight)")
        before = _trace_snapshot(self.state) if trace else None
        result = self._step(move)
        if move is not None:
            result = replace(result, selection_mode="specified")
        if trace:
            result = replace(result, before=before, after=_trace_snapshot(self.state))
        return result

    def _step(self, move_name: str | None = None) -> StepResult:
        """Execute one update selection and at most one Metropolis decision."""

        self.sector_visits[self.state.sector.value] += 1
        if move_name is None:
            move_name = str(self.rng.choice(self._move_names, p=self._probabilities))
        counter = self.counters[move_name]
        counter.selected += 1
        self.atomic_steps += 1
        move = self._moves.get(move_name)
        if move is None:
            counter.not_applicable += 1
            return StepResult(
                move_name=move_name,
                status=MoveStatus.NOT_APPLICABLE,
                patch=None,
                breakdown=None,
            )

        patch = move.propose(self.state, self.rng)
        if patch.status is MoveStatus.NOT_APPLICABLE:
            counter.not_applicable += 1
            return StepResult(move_name, patch.status, patch, None)
        if patch.status is MoveStatus.STRUCTURALLY_INVALID:
            counter.structurally_invalid += 1
            return StepResult(move_name, patch.status, patch, None)

        target = self.measure.delta_local(self.state, patch)
        if self.verify_local_delta:
            assert_local_matches_full(
                target,
                self.measure.delta_full(self.state, patch),
            )
        proposal = move.proposal_ratio(patch)
        log_ratio = target.total + proposal.total
        if not math.isfinite(log_ratio):
            raise FloatingPointError(
                f"nonfinite log ratio: move={move_name}, revision={self.state.revision}, "
                f"target={target}, proposal={proposal}"
            )
        uniform = max(float(self.rng.random()), np.finfo(np.float64).tiny)
        log_uniform = math.log(uniform)
        accepted = log_uniform < min(0.0, log_ratio)
        breakdown = AcceptanceBreakdown(
            kinetic=target.kinetic,
            potential=target.potential,
            chemical=target.chemical,
            sector_measure=target.sector_measure,
            proposal_selection=proposal.total,
            log_uniform=log_uniform,
            accepted=accepted,
        )
        counter.action_evaluated += 1
        counter.sum_kinetic += target.kinetic
        counter.sum_potential += target.potential
        counter.sum_chemical += target.chemical
        counter.sum_sector_measure += target.sector_measure
        counter.sum_proposal_selection += proposal.total
        if accepted:
            self.state.apply(patch)
            counter.accepted += 1
        if (
            accepted
            and self.config.run.invariant_check_interval_steps > 0
            and self.atomic_steps
            % self.config.run.invariant_check_interval_steps
            == 0
        ):
            self.state.validate()
        return StepResult(move_name, patch.status, patch, breakdown, proposal)

    def sweep(
        self,
        *,
        trace: bool = False,
        after_step: Callable[[StepResult], None] | None = None,
    ) -> tuple[StepResult, ...]:
        """Execute the configured fixed number of atomic steps."""

        results = []
        for _ in range(self.config.moves.steps_per_sweep):
            result = self.step(trace=trace)
            if after_step is not None:
                after_step(result)
            results.append(result)
        return tuple(results)

    def counters_snapshot(self) -> dict[str, dict[str, int | float]]:
        return {name: counter.as_dict() for name, counter in self.counters.items()}

    def restore_counters(
        self,
        snapshot: dict[str, dict[str, Any]],
        *,
        atomic_steps: int,
        sector_visits: dict[str, int],
    ) -> None:
        if set(snapshot) != set(self._move_names):
            raise ValueError("checkpoint move names do not match configuration")
        self.counters = {
            name: MoveCounter.from_dict(dict(snapshot[name]))
            for name in self._move_names
        }
        if type(atomic_steps) is not int or atomic_steps < 0:
            raise ValueError("atomic_steps must be a nonnegative integer")
        if set(sector_visits) != {"Z", "G"} or any(
            type(value) is not int or value < 0
            for value in sector_visits.values()
        ):
            raise ValueError("sector_visits must contain nonnegative Z/G counts")
        if sum(sector_visits.values()) != atomic_steps:
            raise ValueError("sector visit counts must sum to atomic_steps")
        self.sector_visits = dict(sector_visits)
        self.atomic_steps = atomic_steps
