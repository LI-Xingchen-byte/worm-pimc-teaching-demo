from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = PROJECT_ROOT / "examples"
REFERENCES = PROJECT_ROOT / "tests" / "reference_examples"


@pytest.mark.parametrize(
    ("script", "arguments", "heading"),
    [
        (
            "free_brownian_bridge.py",
            ["--samples", "1000", "--links", "8"],
            "Free Brownian-bridge marginal check",
        ),
        (
            "closed_worldline.py",
            ["--samples", "1000"],
            "Exact free closed-worldline experiment",
        ),
    ],
)
def test_teaching_example_runs_as_plain_script(
    script: str,
    arguments: list[str],
    heading: str,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(REFERENCES / script), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert heading in result.stdout
    assert "verification: PASS" in result.stdout


def test_closed_sampler_example_runs_as_plain_script(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(REFERENCES / "closed_sampler.py"),
            "--output-root",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Closed-sector interacting PIMC reference run" in result.stdout
    assert "verification: PASS" in result.stdout
    assert len(list(tmp_path.iterdir())) == 1


def test_worm_sampler_example_runs_as_plain_script(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(REFERENCES / "worm_sampler.py"),
            "--output-root",
            str(tmp_path),
            "--verify-local-delta",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Interacting full Worm PIMC reference run" in result.stdout
    assert "verification: PASS" in result.stdout
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("script,heading", [
    ("simulate.py", "In-memory Worm PIMC demonstration"),
    ("observe_step.py", "Measurement opportunities: 0"),
])
def test_python_entry_examples_need_no_config_or_output_files(script, heading, tmp_path):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script)], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert heading in result.stdout
    assert not list(tmp_path.iterdir())


def test_external_density_example_runs_without_plotting_or_output(tmp_path):
    environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"))
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "periodic_external.py"),
         "--sweeps", "32", "--no-plot"], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Periodic Fourier external field demonstration" in result.stdout
    assert "Integral rho dx = <N>:" in result.stdout
    assert not list(tmp_path.iterdir())
