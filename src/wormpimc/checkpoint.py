"""Versioned, checksummed continuation checkpoints for Z/G simulations."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .configuration import Configuration
from .version import __version__


CHECKPOINT_SCHEMA_VERSION = "1.3"
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = {"1.2", CHECKPOINT_SCHEMA_VERSION}
ARRAY_NAMES = (
    "positions",
    "slice_of",
    "next_of",
    "prev_of",
    "image_to_next",
    "active",
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete, corrupt, or incompatible."""


@dataclass(frozen=True, slots=True)
class CheckpointPayload:
    """Complete continuation state independent of run-directory presentation."""

    config: SimulationConfig
    state: Configuration
    rng_state: dict[str, Any]
    phase: str
    warmup_completed: int
    measurement_completed: int
    atomic_steps: int
    sector_visits: dict[str, int]
    move_counters: dict[str, dict[str, Any]]
    estimator_state: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"checkpoint value is not JSON serializable: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _checksum(metadata: dict[str, Any], arrays: dict[str, NDArray]) -> str:
    digest = hashlib.sha256(_canonical_json(metadata).encode("utf-8"))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_checkpoint(path: str | Path, payload: CheckpointPayload) -> Path:
    """Write, re-read, validate, and atomically replace one checkpoint."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = payload.state.snapshot()
    arrays = {
        name: np.asarray(snapshot[name])
        for name in ARRAY_NAMES
    }
    metadata: dict[str, Any] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "package_version": __version__,
        "config_hash": payload.config.config_hash,
        "config_input": payload.config.input_dict(),
        "resolved_config": payload.config.resolved_dict(),
        "phase": payload.phase,
        "warmup_completed": payload.warmup_completed,
        "measurement_completed": payload.measurement_completed,
        "atomic_steps": payload.atomic_steps,
        "sector_visits": payload.sector_visits,
        "configuration": {
            "box_length": snapshot["box_length"],
            "n_slices": snapshot["n_slices"],
            "sector": snapshot["sector"],
            "worm_head": snapshot["worm_head"],
            "worm_tail": snapshot["worm_tail"],
            "revision": snapshot["revision"],
        },
        "rng_state": payload.rng_state,
        "move_counters": payload.move_counters,
        "estimator_state": payload.estimator_state,
    }
    metadata["checksum"] = _checksum(metadata, arrays)
    temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata_json=np.array(_canonical_json(metadata)),
                **arrays,
            )
        read_checkpoint(temporary)
        os.replace(temporary, checkpoint_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return checkpoint_path


def read_checkpoint(path: str | Path) -> CheckpointPayload:
    """Load a checkpoint and reject schema, hash, or invariant failures."""

    checkpoint_path = Path(path)
    try:
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            files = set(archive.files)
            expected = {"metadata_json", *ARRAY_NAMES}
            if files != expected:
                raise CheckpointError("checkpoint array fields do not match schema")
            metadata = json.loads(str(archive["metadata_json"].item()))
            arrays = {
                name: np.array(archive[name], copy=True) for name in ARRAY_NAMES
            }
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"could not read checkpoint: {exc}") from exc

    checksum = metadata.pop("checksum", None)
    if not isinstance(checksum, str) or checksum != _checksum(metadata, arrays):
        raise CheckpointError("checkpoint checksum mismatch")
    if metadata.get("checkpoint_schema_version") not in (
        SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
    ):
        raise CheckpointError("unsupported checkpoint schema version")
    try:
        config = SimulationConfig.from_mapping(metadata["config_input"])
    except Exception as exc:
        raise CheckpointError("checkpoint contains invalid configuration") from exc
    if config.config_hash != metadata.get("config_hash"):
        raise CheckpointError("checkpoint config hash mismatch")
    if config.resolved_dict() != metadata.get("resolved_config"):
        raise CheckpointError("checkpoint resolved configuration mismatch")
    configuration_metadata = metadata.get("configuration")
    if not isinstance(configuration_metadata, dict):
        raise CheckpointError("checkpoint configuration metadata is invalid")
    try:
        state = Configuration.from_snapshot(
            {**arrays, **configuration_metadata}
        )
    except Exception as exc:
        raise CheckpointError("checkpoint worldline invariants failed") from exc

    for counter_name in (
        "warmup_completed",
        "measurement_completed",
        "atomic_steps",
    ):
        value = metadata.get(counter_name)
        if type(value) is not int or value < 0:
            raise CheckpointError(f"invalid checkpoint counter: {counter_name}")
    phase = metadata.get("phase")
    if phase not in {"warmup", "measurement", "completed", "interrupted"}:
        raise CheckpointError("invalid checkpoint phase")
    rng_state = metadata.get("rng_state")
    move_counters = metadata.get("move_counters")
    sector_visits = metadata.get("sector_visits")
    estimator_state = metadata.get("estimator_state")
    if not isinstance(rng_state, dict):
        raise CheckpointError("invalid RNG state")
    if not isinstance(sector_visits, dict):
        raise CheckpointError("invalid sector residence state")
    if not isinstance(move_counters, dict) or not isinstance(
        estimator_state, dict
    ):
        raise CheckpointError("invalid accumulator state")
    return CheckpointPayload(
        config=config,
        state=state,
        rng_state=rng_state,
        phase=phase,
        warmup_completed=metadata["warmup_completed"],
        measurement_completed=metadata["measurement_completed"],
        atomic_steps=metadata["atomic_steps"],
        sector_visits=sector_visits,
        move_counters=move_counters,
        estimator_state=estimator_state,
    )
