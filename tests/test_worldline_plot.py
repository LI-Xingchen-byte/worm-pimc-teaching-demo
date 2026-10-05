"""Geometry correctness and passive rendering, including seam and Swap cases."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from wormpimc import Configuration, Simulation, SimulationConfig
from wormpimc.sampler import StepResult, _trace_snapshot
from wormpimc.types import MoveStatus


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "worldline_plot", ROOT / "examples/worldline_plot.py"
)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


@pytest.fixture
def pyplot(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    yield plt
    plt.close("all")


@pytest.mark.parametrize(
    "x0,x1,image,expected",
    [
        (3.5, 0.5, 1, [((3.5, 0), (4, 0.5)), ((0, 0.5), (0.5, 1))]),
        (0.5, 3.5, -1, [((0.5, 0), (0, 0.5)), ((4, 0.5), (3.5, 1))]),
        (0, 3, -1, [((4, 0), (3, 1))]),
        (3, 0, 1, [((3, 0), (4, 1))]),
        (1, 3, 0, [((1, 0), (3, 1))]),
    ],
)
def test_spatial_seam_segments_have_exact_interpolated_times(x0, x1, image, expected):
    np.testing.assert_allclose(
        plot.split_periodic_link(x0, x1, image, 4, 0, 1), expected
    )


@pytest.mark.parametrize("image", [-3, -2, -1, 0, 1, 2, 3])
def test_multimage_links_preserve_signed_displacement(image):
    pieces = plot.split_periodic_link(1, 3, image, 4, 0.875, 1.0)
    assert sum(end[0] - start[0] for start, end in pieces) == pytest.approx(
        2 + image * 4
    )
    assert sum(end[1] - start[1] for start, end in pieces) == pytest.approx(0.125)
    for start, end in pieces:
        assert 0 <= start[0] <= 4 and 0 <= end[0] <= 4
        assert 0.875 <= start[1] < end[1] <= 1


def test_rendering_and_candidate_reconstruction_do_not_change_trajectory(pyplot):
    config = SimulationConfig.small_system(n_slices=8, seed=1)
    rendered, plain = Simulation(config), Simulation(config)
    for name in (
        "open",
        "swap",
        "close",
        "insert",
        "advance",
        "recede",
        "remove",
        "wiggle",
        "displace",
    ):
        event = rendered.step(move=name, trace=True)
        expected = plain.step(move=name, trace=True)
        figure = plot.plot_step(event, tau=config.tau)
        figure.canvas.draw()
        assert len(figure.axes) == 3
        assert event == expected
        assert rendered.rng.bit_generator.state == plain.rng.bit_generator.state
        assert rendered.state.state_digest(
            include_revision=True
        ) == plain.state.state_digest(include_revision=True)
        assert rendered.sampler.counters_snapshot() == plain.sampler.counters_snapshot()
        pyplot.close(figure)


def test_rejected_candidate_and_inapplicable_panel_are_distinct(pyplot):
    sim = Simulation(SimulationConfig.small_system(n_slices=8))
    event = sim.step(move="close", trace=True)
    figure = plot.plot_step(event, tau=sim.config.tau)
    assert not figure.axes[1].lines
    assert "No valid proposal" in [t.get_text() for t in figure.axes[1].texts]
    sim.step(move="open")
    rejected = sim.step(move="close", trace=True)
    assert not rejected.accepted and rejected.candidate() is not None
    figure = plot.plot_step(rejected, tau=sim.config.tau)
    assert "[Z]" in figure.axes[1].get_title()
    assert "[G]" in figure.axes[2].get_title()
    for before, after in zip(figure.axes[0].lines, figure.axes[2].lines):
        np.testing.assert_array_equal(before.get_xydata(), after.get_xydata())


def test_swap_highlights_actual_connectivity_changes(pyplot):
    sim = Simulation(SimulationConfig.small_system(n_slices=8, seed=1))
    sim.step(move="open")
    event = sim.step(move="swap", trace=True)
    assert event.accepted
    candidate = event.candidate().snapshot()
    connectivity_changes = {
        i
        for i, (a, b) in enumerate(zip(event.before["next_of"], candidate["next_of"]))
        if a != b
    }
    assert connectivity_changes
    assert connectivity_changes <= plot.changed_links(event.before, candidate)
    figure = plot.plot_step(event, tau=sim.config.tau, show_ids=True)
    assert any(line.get_color() == "#287bb5" for line in figure.axes[1].lines)


def test_last_time_link_is_lifted_to_beta_and_images_are_not_minimum_image(pyplot):
    state = Configuration.from_closed_worldline(
        np.array([[1.0], [1.0], [1.0], [1.0], [9.0]]), 4.0
    )
    snapshot = _trace_snapshot(state)
    event = StepResult(
        "geometry",
        MoveStatus.NOT_APPLICABLE,
        None,
        None,
        before=snapshot,
        after=snapshot,
    )
    figure = plot.plot_step(event, tau=0.25)
    pieces = [line.get_xydata() for line in figure.axes[0].lines[2:]]
    assert all(
        0 <= point[0] <= 4 and 0 <= point[1] <= 1 for piece in pieces for point in piece
    )
    assert all(piece[1, 1] - piece[0, 1] <= 0.25 for piece in pieces)
    assert sum(piece[1, 0] - piece[0, 0] for piece in pieces) == pytest.approx(8)


def test_plot_requires_trace(pyplot):
    sim = Simulation(SimulationConfig.small_system())
    with pytest.raises(ValueError, match="trace=True"):
        plot.plot_step(sim.step(), tau=sim.config.tau)


def test_external_field_strips_and_density_plot_are_passive(pyplot):
    from wormpimc.estimators import DensityProfile
    from wormpimc.potentials import external_from_config

    config = SimulationConfig.small_system(n_slices=8, external="fourier",
                                          external_offset=0.5, external_cosine=[-0.5])
    rendered, plain = Simulation(config), Simulation(config)
    density = DensityProfile(config)
    rendered.advance(max_sweeps=25, after_sweep=density.after_sweep)
    plain.advance(max_sweeps=25)
    event = rendered.step(trace=True)
    expected = plain.step(trace=True)
    field = external_from_config(config)
    figure = plot.plot_step(event, tau=config.tau, external=field)
    figure.canvas.draw()
    assert len(figure.axes) == 6
    for ax in figure.axes[3:]:
        line = ax.lines[0]
        assert line.get_ydata() == pytest.approx(field.energy(line.get_xdata()))
        assert ax.get_xlim() == (0, config.system.box_length)
    second = plot.plot_density(density, field)
    second.canvas.draw()
    assert len(second.axes) == 2
    assert event == expected
    assert rendered.state.state_digest(include_revision=True) == plain.state.state_digest(include_revision=True)
    assert rendered.rng.bit_generator.state == plain.rng.bit_generator.state


def test_visualization_cli_headless_from_another_directory(pyplot, tmp_path):
    environment = dict(
        os.environ, PYTHONPATH=str(ROOT / "src"), MPLCONFIGDIR=str(tmp_path / "mpl")
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/visualize_updates.py"),
            "--moves",
            "open",
            "close",
            "--output",
            str(tmp_path / "figures"),
            "--no-show",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "open proposed accepted: True specified" in result.stdout
    assert "close proposed accepted: False specified" in result.stdout
    assert {path.name for path in (tmp_path / "figures").glob("*.png")} == {
        "001_open.png",
        "002_close.png",
    }
