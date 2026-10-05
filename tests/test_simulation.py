from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from wormpimc import Simulation, SimulationConfig, __version__
from wormpimc.checkpoint import CheckpointError, read_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "closed_sampler.toml"


def short_config() -> SimulationConfig:
    base = SimulationConfig.from_toml(CONFIG_PATH).input_dict()
    base["moves"]["steps_per_sweep"] = 8
    base["run"]["warmup_sweeps"] = 4
    base["run"]["measurement_sweeps"] = 6
    base["run"]["measurement_stride"] = 2
    base["run"]["checkpoint_interval_sweeps"] = 3
    return SimulationConfig.from_mapping(base)


def short_worm_config() -> SimulationConfig:
    base = SimulationConfig.from_toml(
        PROJECT_ROOT / "configs" / "worm_sampler.toml"
    ).input_dict()
    base["moves"]["steps_per_sweep"] = 12
    base["run"]["warmup_sweeps"] = 4
    base["run"]["measurement_sweeps"] = 6
    base["run"]["measurement_stride"] = 1
    base["run"]["checkpoint_interval_sweeps"] = 3
    return SimulationConfig.from_mapping(base)


def test_same_seed_runs_are_stepwise_reproducible() -> None:
    config = short_config()
    first = Simulation(config, verify_local_delta=True)
    second = Simulation(config, verify_local_delta=True)

    first.advance()
    second.advance()

    assert first.state.state_digest(include_revision=True) == (
        second.state.state_digest(include_revision=True)
    )
    assert first.rng.bit_generator.state == second.rng.bit_generator.state
    assert first.sampler.counters_snapshot() == second.sampler.counters_snapshot()
    assert first.estimators.snapshot() == second.estimators.snapshot()
    assert first.summary() == second.summary()


def test_checkpoint_resume_matches_continuous_run(tmp_path: Path) -> None:
    config = short_config()
    continuous = Simulation(config)
    continuous.advance()

    split = Simulation(config)
    split.advance(max_sweeps=5)
    checkpoint = split.checkpoint(tmp_path / "split.npz")
    resumed = Simulation.from_checkpoint(checkpoint)
    resumed.advance()

    assert resumed.state.state_digest(include_revision=True) == (
        continuous.state.state_digest(include_revision=True)
    )
    assert resumed.rng.bit_generator.state == continuous.rng.bit_generator.state
    assert resumed.sampler.counters_snapshot() == (
        continuous.sampler.counters_snapshot()
    )
    assert resumed.estimators.snapshot() == continuous.estimators.snapshot()
    assert resumed.summary() == continuous.summary()


def test_run_directory_contains_continuation_and_result_artifacts(
    tmp_path: Path,
) -> None:
    config = short_config()
    result = Simulation(config).run(output_root=tmp_path)
    run_directory = result.run_directory

    assert result.status == "completed"
    for relative in (
        "input.toml",
        "resolved_config.json",
        "metadata.json",
        "status.json",
        "run.log",
        "move_statistics.csv",
        "sector_residence.csv",
        "scalar_observables.csv",
        "green_function.csv",
        "blocking_diagnostics.csv",
        "checkpoints/latest.json",
    ):
        assert (run_directory / relative).is_file(), relative
    status = json.loads(
        (run_directory / "status.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (run_directory / "metadata.json").read_text(encoding="utf-8")
    )
    assert status["phase"] == "completed"
    assert metadata["run_status"] == "completed"
    assert metadata["output_schema_version"] == "3.0"
    assert metadata["package_version"] == __version__
    assert metadata["uncertainty_method"] == "dyadic_blocking_ratio_delta"
    assert metadata["config_hash"] == config.config_hash
    assert SimulationConfig.from_toml(
        run_directory / "input.toml"
    ).config_hash == config.config_hash
    with (run_directory / "green_function.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        green_rows = list(csv.DictReader(handle))
    assert len(green_rows) == (
        config.discretization.n_slices
        * config.estimators.green_spatial_bins
    )
    assert {row["n_measurement_opportunities"] for row in green_rows} == {"3"}
    assert {row["n_z_measurements"] for row in green_rows} == {"3"}
    assert {row["normalization_status"] for row in green_rows} == {
        "not_sampled_topology_disabled"
    }


def test_partial_run_directory_can_be_resumed(tmp_path: Path) -> None:
    config = short_config()
    partial_result = Simulation(config).run(
        output_root=tmp_path,
        max_sweeps=5,
    )
    assert partial_result.status == "measurement"
    assert partial_result.latest_checkpoint is not None

    resumed = Simulation.from_checkpoint(partial_result.latest_checkpoint)
    completed = resumed.run(
        run_directory=partial_result.run_directory,
        checkpoint_source=partial_result.latest_checkpoint,
    )

    assert completed.status == "completed"
    assert completed.measurement_count == 3
    status = json.loads(
        (completed.run_directory / "status.json").read_text(encoding="utf-8")
    )
    assert status["current_sweep"] == 10


def test_checkpoint_checksum_detects_array_corruption(tmp_path: Path) -> None:
    simulation = Simulation(short_config())
    simulation.advance(max_sweeps=1)
    path = simulation.checkpoint(tmp_path / "valid.npz")
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["positions"][0, 0] += 0.125
    corrupted = tmp_path / "corrupted.npz"
    with corrupted.open("wb") as handle:
        np.savez_compressed(handle, **payload)

    with pytest.raises(CheckpointError, match="checksum mismatch"):
        read_checkpoint(corrupted)


def test_checkpoint_round_trip_preserves_live_g_sector(tmp_path: Path) -> None:
    data = SimulationConfig.from_toml(
        PROJECT_ROOT / "configs" / "ideal_bose_gas.toml"
    ).input_dict()
    data["discretization"]["n_slices"] = 16
    data["moves"]["max_segment_links"] = 4
    config = SimulationConfig.from_mapping(data)
    simulation = Simulation(config)
    for _ in range(500):
        simulation.step()
        if simulation.state.sector.value == "G":
            break
    assert simulation.state.sector.value == "G"

    path = simulation.checkpoint(tmp_path / "g-sector.npz")
    restored = Simulation.from_checkpoint(path)

    assert restored.state.state_digest(include_revision=True) == (
        simulation.state.state_digest(include_revision=True)
    )
    assert restored.state.open_chain_ids() == simulation.state.open_chain_ids()
    assert restored.rng.bit_generator.state == simulation.rng.bit_generator.state


def test_worm_checkpoint_resume_matches_uninterrupted_chain(
    tmp_path: Path,
) -> None:
    config = short_worm_config()
    continuous = Simulation(config, verify_local_delta=True)
    continuous.advance()

    split = Simulation(config, verify_local_delta=True)
    split.advance(max_sweeps=5)
    path = split.checkpoint(tmp_path / "worm-split.npz")
    resumed = Simulation.from_checkpoint(path, verify_local_delta=True)
    resumed.advance()

    assert resumed.state.state_digest(include_revision=True) == (
        continuous.state.state_digest(include_revision=True)
    )
    assert resumed.rng.bit_generator.state == continuous.rng.bit_generator.state
    assert resumed.sampler.counters_snapshot() == (
        continuous.sampler.counters_snapshot()
    )
    assert resumed.sampler.sector_visits == continuous.sampler.sector_visits
    assert resumed.estimators.snapshot() == continuous.estimators.snapshot()
