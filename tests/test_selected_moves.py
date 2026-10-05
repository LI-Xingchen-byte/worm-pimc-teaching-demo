"""Specified updates condition the existing move kernel, not a new sampler."""

from dataclasses import replace
import math

import numpy as np
import pytest

from wormpimc import MoveStatus, Simulation, SimulationConfig
from wormpimc.config import MOVE_NAMES


@pytest.mark.parametrize("name", MOVE_NAMES)
def test_requested_move_is_the_only_family_attempted(name):
    sim = Simulation(SimulationConfig.small_system(n_slices=8))
    event = sim.step(move=name, trace=True)
    assert event.move_name == name
    assert event.selection_mode == "specified"
    assert sim.sampler.counters[name].selected == 1
    assert sum(c.selected for c in sim.sampler.counters.values()) == 1
    assert sim.summary()["measurement_opportunities"] == sim.completed_sweeps == 0


def test_conditioned_step_matches_random_step_after_the_selector_draw():
    config = SimulationConfig.small_system(n_slices=8)
    config = replace(
        config,
        moves=replace(config.moves, weights=replace(config.moves.weights, open=3.0)),
    )
    random, specified = Simulation(config), Simulation(config)
    for _ in range(200):
        probabilities = config.moves.weights.normalized()
        selected = str(
            specified.rng.choice(MOVE_NAMES, p=[probabilities[n] for n in MOVE_NAMES])
        )
        expected = random.step(trace=True)
        event = specified.step(move=selected, trace=True)
        assert replace(event, selection_mode="random") == expected
        assert specified.rng.bit_generator.state == random.rng.bit_generator.state
        assert specified.state.state_digest(
            include_revision=True
        ) == random.state.state_digest(include_revision=True)
    assert specified.sampler.counters_snapshot() == random.sampler.counters_snapshot()
    assert specified.sampler.sector_visits == random.sampler.sector_visits


def test_selected_pair_retains_configured_selection_correction():
    config = SimulationConfig.small_system(n_slices=8)
    config = replace(
        config,
        moves=replace(config.moves, weights=replace(config.moves.weights, open=3.0)),
    )
    event = Simulation(config).step(move="open")
    ratio = event.proposal_ratio
    assert (
        ratio.log_move_selection_reverse - ratio.log_move_selection_forward
        == pytest.approx(-math.log(3))
    )


@pytest.mark.parametrize("name", ["unknown", "Open", "random", "", 3, ["open"]])
def test_bad_move_does_not_consume_rng_or_change_counters(name):
    sim = Simulation(SimulationConfig.small_system())
    initial = (
        sim.summary(),
        sim.rng.bit_generator.state,
        sim.sampler.counters_snapshot(),
    )
    with pytest.raises(ValueError, match="unknown move"):
        sim.step(move=name, trace=True)
    assert (
        sim.summary(),
        sim.rng.bit_generator.state,
        sim.sampler.counters_snapshot(),
    ) == initial


def test_disabled_move_is_rejected_before_mutation():
    config = SimulationConfig.small_system()
    config = replace(
        config,
        moves=replace(
            config.moves, weights=replace(config.moves.weights, open=0.0, close=0.0)
        ),
    )
    sim = Simulation(config)
    rng = sim.rng.bit_generator.state
    with pytest.raises(ValueError, match="disabled"):
        sim.step(move="open")
    assert sim.sampler.atomic_steps == 0
    assert sim.rng.bit_generator.state == rng


def test_inapplicable_selected_move_neither_prepares_state_nor_draws_acceptance():
    sim = Simulation(SimulationConfig.small_system())
    rng = sim.rng.bit_generator.state
    before = sim.state.state_digest(include_revision=True)
    event = sim.step(move="close", trace=True)
    assert event.status is MoveStatus.NOT_APPLICABLE
    assert event.breakdown is None
    assert event.candidate() is None
    assert "sector G" in event.patch.invalid_reason
    assert sim.rng.bit_generator.state == rng
    assert sim.state.state_digest(include_revision=True) == before


def test_specified_rejection_keeps_live_state_and_trace_on_off_match():
    config = SimulationConfig.small_system(n_slices=8)
    plain, traced = Simulation(config), Simulation(config)
    outcomes = []
    for name in ("open", "close"):
        expected = plain.step(move=name)
        event = traced.step(move=name, trace=True)
        assert event == expected
        outcomes.append(event.accepted)
        assert traced.rng.bit_generator.state == plain.rng.bit_generator.state
        assert traced.state.state_digest() == plain.state.state_digest()
    assert outcomes == [True, False]
    np.testing.assert_array_equal(event.before["positions"], event.after["positions"])
    assert event.candidate().sector != traced.state.sector
