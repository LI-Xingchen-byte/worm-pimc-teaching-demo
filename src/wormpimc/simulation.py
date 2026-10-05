"""Public lifecycle facade for reproducible extended-ensemble simulations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .checkpoint import (
    CheckpointPayload,
    read_checkpoint,
    write_checkpoint,
)
from .config import SimulationConfig
from .configuration import Configuration, initialize_configuration
from .estimators import EstimatorManager
from .measure import PrimitiveTargetMeasure
from .output import (
    create_run_directory,
    initial_metadata,
    utc_now,
    write_blocking_diagnostics,
    write_json_atomic,
    write_green_histogram,
    write_move_statistics,
    write_scalar_observables,
    write_sector_residence,
)
from .sampler import Sampler, StepResult


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Compact completion record returned by the public facade."""

    status: str
    run_directory: Path
    warmup_completed: int
    measurement_completed: int
    measurement_count: int
    latest_checkpoint: Path | None


class Simulation:
    """Own initialization, sampling phases, checkpointing, and finalization."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        state: Configuration | None = None,
        rng: np.random.Generator | None = None,
        verify_local_delta: bool = False,
    ) -> None:
        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig")
        self.config = config
        self.state = state if state is not None else initialize_configuration(config)
        self.rng = rng if rng is not None else np.random.default_rng(config.run.seed)
        self.target = PrimitiveTargetMeasure(config)
        self.sampler = Sampler(
            config,
            self.state,
            self.rng,
            measure=self.target,
            verify_local_delta=verify_local_delta,
        )
        self.estimators = EstimatorManager(config)
        self.warmup_completed = 0
        self.measurement_completed = 0
        self.phase = (
            "warmup" if config.run.warmup_sweeps > 0 else "measurement"
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        verify_local_delta: bool = False,
    ) -> "Simulation":
        """Restore every continuation field from a validated checkpoint."""

        payload = read_checkpoint(path)
        bit_generator_name = payload.rng_state.get("bit_generator")
        if bit_generator_name != "PCG64":
            raise ValueError(
                f"unsupported checkpoint RNG algorithm: {bit_generator_name}"
            )
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = payload.rng_state
        simulation = cls(
            payload.config,
            state=payload.state,
            rng=rng,
            verify_local_delta=verify_local_delta,
        )
        simulation.phase = payload.phase
        simulation.warmup_completed = payload.warmup_completed
        simulation.measurement_completed = payload.measurement_completed
        simulation.sampler.restore_counters(
            payload.move_counters,
            atomic_steps=payload.atomic_steps,
            sector_visits=payload.sector_visits,
        )
        simulation.estimators.restore(payload.estimator_state)
        return simulation

    @property
    def completed_sweeps(self) -> int:
        return self.warmup_completed + self.measurement_completed

    @property
    def target_sweeps(self) -> int:
        return self.config.run.warmup_sweeps + self.config.run.measurement_sweeps

    def step(self, *, move: str | None = None, trace: bool = False) -> StepResult:
        """Attempt one update without advancing phases, sweeps or measurements.

        With trace=True the result includes read-only before/after snapshots
        and can reconstruct the candidate, including rejected proposals.
        Set move='open', etc. to demonstrate one configured update family;
        this skips move selection, not the Metropolis acceptance decision.
        Use the default move=None for normal random sampling.
        """

        return self.sampler.step(move=move, trace=trace)

    def advance(
        self,
        *,
        max_sweeps: int | None = None,
        after_sweep: Callable[["Simulation"], None] | None = None,
        trace: bool = False,
        after_step: Callable[[StepResult], None] | None = None,
    ) -> int:
        """Advance phases and measurements in memory; return sweeps executed.

        after_step receives every attempt, including warmup and rejection.
        Use trace=True for snapshots. Callbacks are observers: do not mutate
        the simulation or consume its RNG. No trace history is kept here.
        """

        if max_sweeps is not None and (
            type(max_sweeps) is not int or max_sweeps < 0
        ):
            raise ValueError("max_sweeps must be a nonnegative integer or None")
        budget = max_sweeps
        executed = 0

        while self.warmup_completed < self.config.run.warmup_sweeps:
            if budget is not None and executed >= budget:
                self.phase = "warmup"
                return executed
            self.phase = "warmup"
            self.sampler.sweep(trace=trace, after_step=after_step)
            self.warmup_completed += 1
            executed += 1
            if after_sweep is not None:
                after_sweep(self)

        while self.measurement_completed < self.config.run.measurement_sweeps:
            if budget is not None and executed >= budget:
                self.phase = "measurement"
                return executed
            self.phase = "measurement"
            self.sampler.sweep(trace=trace, after_step=after_step)
            self.measurement_completed += 1
            if (
                self.measurement_completed
                % self.config.run.measurement_stride
                == 0
            ):
                self.estimators.measure(self.state, self.target)
            executed += 1
            if after_sweep is not None:
                after_sweep(self)

        self.phase = "completed"
        return executed

    def results(self) -> dict[str, Any]:
        """Return detached, JSON-ready physical results without file I/O.

        scalars maps observable names to estimates and their metadata;
        green_function contains finite-bin G and beta-minus g1 rows.
        summary reports scheduled opportunities as well as Z measurements.
        Missing/inapplicable values are None; normalization and uncertainty
        status strings explain why. The existing CSV schemas are unchanged.
        """

        def clean(row: dict[str, Any]) -> dict[str, Any]:
            return {
                key: None if value == "" or (
                    isinstance(value, float) and not math.isfinite(value)
                ) else value
                for key, value in row.items()
            }

        return {
            "summary": self.summary(),
            "scalars": {row["observable"]: clean(row) for row in self.estimators.rows()},
            "green_function": [clean(row) for row in self.estimators.green_rows()],
        }

    def checkpoint(self, path: str | Path) -> Path:
        """Persist a complete continuation state."""

        payload = CheckpointPayload(
            config=self.config,
            state=self.state,
            rng_state=self.rng.bit_generator.state,
            phase=self.phase,
            warmup_completed=self.warmup_completed,
            measurement_completed=self.measurement_completed,
            atomic_steps=self.sampler.atomic_steps,
            sector_visits=dict(self.sampler.sector_visits),
            move_counters=self.sampler.counters_snapshot(),
            estimator_state=self.estimators.snapshot(),
        )
        return write_checkpoint(path, payload)

    def summary(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "warmup_completed": self.warmup_completed,
            "warmup_target": self.config.run.warmup_sweeps,
            "measurement_completed": self.measurement_completed,
            "measurement_target": self.config.run.measurement_sweeps,
            "measurement_count": self.estimators.measurement_count,
            "measurement_opportunities": (
                self.estimators.total_measurement_count
            ),
            "atomic_steps": self.sampler.atomic_steps,
            "sector_visits": dict(self.sampler.sector_visits),
            "sector": self.state.sector.value,
            "revision": self.state.revision,
            "config_hash": self.config.config_hash,
        }

    def run(
        self,
        *,
        output_root: str | Path | None = None,
        run_directory: str | Path | None = None,
        max_sweeps: int | None = None,
        checkpoint_source: str | Path | None = None,
    ) -> SimulationResult:
        """Run or continue the configured lifecycle in one run directory."""

        if run_directory is None:
            directory = create_run_directory(
                self.config, output_root=output_root
            )
            is_new = True
        else:
            directory = Path(run_directory)
            if not directory.is_dir():
                raise ValueError(f"run directory does not exist: {directory}")
            (directory / "checkpoints").mkdir(exist_ok=True)
            is_new = False

        metadata_path = directory / "metadata.json"
        status_path = directory / "status.json"
        log_path = directory / "run.log"
        if is_new:
            (directory / "input.toml").write_text(
                self.config.to_toml(), encoding="utf-8"
            )
            write_json_atomic(
                directory / "resolved_config.json",
                self.config.resolved_dict(),
            )
            metadata = initial_metadata(self.config)
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            resolved = json.loads(
                (directory / "resolved_config.json").read_text(encoding="utf-8")
            )
            if metadata.get("config_hash") != self.config.config_hash:
                raise ValueError("run directory config hash does not match checkpoint")
            if resolved != self.config.resolved_dict():
                raise ValueError(
                    "run directory resolved configuration does not match checkpoint"
                )
        metadata["run_status"] = self.phase
        metadata["checkpoint_source"] = (
            str(Path(checkpoint_source).resolve())
            if checkpoint_source is not None
            else metadata.get("checkpoint_source")
        )
        write_json_atomic(metadata_path, metadata)
        latest_checkpoint: Path | None = None

        def current_status(failure_category: str | None = None) -> dict[str, object]:
            return {
                "phase": self.phase,
                "current_sweep": self.completed_sweeps,
                "target_sweep": self.target_sweeps,
                "warmup_completed": self.warmup_completed,
                "measurement_completed": self.measurement_completed,
                "current_sector": self.state.sector.value,
                "latest_checkpoint": (
                    str(latest_checkpoint.relative_to(directory))
                    if latest_checkpoint is not None
                    else None
                ),
                "last_update_time": utc_now(),
                "failure_category": failure_category,
            }

        def save_checkpoint() -> Path:
            nonlocal latest_checkpoint
            checkpoint_path = (
                directory
                / "checkpoints"
                / f"checkpoint_{self.completed_sweeps:08d}.npz"
            )
            latest_checkpoint = self.checkpoint(checkpoint_path)
            write_json_atomic(
                directory / "checkpoints" / "latest.json",
                {
                    "checkpoint": latest_checkpoint.name,
                    "completed_sweeps": self.completed_sweeps,
                    "phase": self.phase,
                },
            )
            return latest_checkpoint

        def after_sweep(_: "Simulation") -> None:
            if (
                self.completed_sweeps
                % self.config.run.checkpoint_interval_sweeps
                == 0
            ):
                save_checkpoint()
                write_json_atomic(status_path, current_status())

        write_json_atomic(status_path, current_status())
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"{utc_now()} start phase={self.phase} "
                f"completed_sweeps={self.completed_sweeps}\n"
            )
        try:
            self.advance(max_sweeps=max_sweeps, after_sweep=after_sweep)
            save_checkpoint()
            if self.phase == "completed":
                write_scalar_observables(
                    directory / "scalar_observables.csv", self.estimators
                )
                write_green_histogram(
                    directory / "green_function.csv", self.estimators
                )
                write_blocking_diagnostics(
                    directory / "blocking_diagnostics.csv", self.estimators
                )
                write_move_statistics(
                    directory / "move_statistics.csv", self.sampler.counters
                )
                write_sector_residence(
                    directory / "sector_residence.csv",
                    self.sampler.sector_visits,
                )
                metadata["run_status"] = "completed"
                metadata["end_time"] = utc_now()
            else:
                metadata["run_status"] = self.phase
            write_json_atomic(metadata_path, metadata)
            write_json_atomic(status_path, current_status())
        except KeyboardInterrupt:
            self.phase = "interrupted"
            save_checkpoint()
            metadata["run_status"] = "interrupted"
            metadata["end_time"] = utc_now()
            write_json_atomic(metadata_path, metadata)
            write_json_atomic(status_path, current_status("KeyboardInterrupt"))
        except Exception as exc:
            metadata["run_status"] = "failed"
            metadata["end_time"] = utc_now()
            write_json_atomic(metadata_path, metadata)
            write_json_atomic(status_path, current_status(type(exc).__name__))
            raise
        finally:
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"{utc_now()} stop phase={self.phase} "
                    f"completed_sweeps={self.completed_sweeps}\n"
                )

        return SimulationResult(
            status=self.phase,
            run_directory=directory.resolve(),
            warmup_completed=self.warmup_completed,
            measurement_completed=self.measurement_completed,
            measurement_count=self.estimators.measurement_count,
            latest_checkpoint=(
                latest_checkpoint.resolve()
                if latest_checkpoint is not None
                else None
            ),
        )
