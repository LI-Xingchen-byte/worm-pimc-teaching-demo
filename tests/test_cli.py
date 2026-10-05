from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wormpimc.cli import main
from wormpimc.config import SimulationConfig
from wormpimc.simulation import Simulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_check_config_human_output(capsys: Any) -> None:
    exit_code = main(
        ["check-config", str(CONFIG_DIR / "free_particle.toml")]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Configuration valid" in captured.out
    assert "tau:" in captured.out
    assert "config_hash:" in captured.out
    assert captured.err == ""


def test_check_config_json_output(capsys: Any) -> None:
    exit_code = main(
        [
            "check-config",
            str(CONFIG_DIR / "gaussian_repulsion.toml"),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "valid"
    assert payload["resolved_config"]["potential"]["pair"] == (
        "periodized_gaussian"
    )
    assert len(payload["config_hash"]) == 64
    assert captured.err == ""


def test_invalid_config_returns_exit_code_two(
    tmp_path: Path,
    capsys: Any,
) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text("[system]\nndim = 1\n", encoding="utf-8")

    exit_code = main(["check-config", str(path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "configuration error:" in captured.err


def test_check_config_has_no_run_directory_side_effect(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        ["check-config", str(CONFIG_DIR / "free_particle.toml")]
    )

    assert exit_code == 0
    assert not (tmp_path / "runs").exists()


def _write_short_run_config(tmp_path: Path) -> Path:
    data = SimulationConfig.from_toml(
        CONFIG_DIR / "closed_sampler.toml"
    ).input_dict()
    data["moves"]["steps_per_sweep"] = 6
    data["run"]["warmup_sweeps"] = 2
    data["run"]["measurement_sweeps"] = 4
    data["run"]["measurement_stride"] = 2
    data["run"]["checkpoint_interval_sweeps"] = 2
    config = SimulationConfig.from_mapping(data)
    path = tmp_path / "short.toml"
    path.write_text(config.to_toml(), encoding="utf-8")
    return path


def test_run_and_summarize_commands(tmp_path: Path, capsys: Any) -> None:
    config_path = _write_short_run_config(tmp_path)
    output_root = tmp_path / "output"

    exit_code = main(
        ["run", str(config_path), "--output-root", str(output_root)]
    )
    run_output = capsys.readouterr()

    assert exit_code == 0
    assert "status: completed" in run_output.out
    run_directory = next(output_root.iterdir())

    summary_exit = main(["summarize", str(run_directory), "--json"])
    summary_output = capsys.readouterr()
    payload = json.loads(summary_output.out)

    assert summary_exit == 0
    assert payload["status"]["phase"] == "completed"
    assert payload["metadata"]["run_status"] == "completed"


def test_resume_command_continues_owning_run_directory(
    tmp_path: Path,
    capsys: Any,
) -> None:
    config_path = _write_short_run_config(tmp_path)
    config = SimulationConfig.from_toml(config_path)
    partial = Simulation(config).run(output_root=tmp_path / "runs", max_sweeps=3)
    assert partial.latest_checkpoint is not None

    exit_code = main(["resume", str(partial.latest_checkpoint)])
    output = capsys.readouterr()

    assert exit_code == 0
    assert "status: completed" in output.out
    assert (partial.run_directory / "scalar_observables.csv").is_file()
