"""Public Python workflows, validation boundaries, and passive step tracing."""

from dataclasses import fields, replace
import json

import numpy as np
import pytest

from wormpimc import (
    ConfigError,
    Configuration,
    MoveStatus,
    Simulation,
    SimulationConfig,
)
from wormpimc.sampler import Sampler
from wormpimc.types import ProposalPatch, TargetDelta


def small_config(**kwargs):
    return SimulationConfig.small_system(
        n_slices=8,
        warmup_sweeps=2,
        measurement_sweeps=10,
        steps_per_sweep=12,
        green_spatial_bins=4,
        **kwargs,
    )


@pytest.mark.parametrize(
    "section,changes,match",
    [
        ("system", {"beta": -1.0}, "system.beta"),
        ("system", {"beta": float("nan")}, "system.beta"),
        ("system", {"box_length": float("inf")}, "system.box_length"),
        ("system", {"lambda_kin": 0.0}, "system.lambda_kin"),
        ("system", {"ndim": 2}, "system.ndim"),
        ("system", {"chemical_potential": 0.0}, "chemical_potential"),
        ("discretization", {"n_slices": 0}, "n_slices"),
        ("discretization", {"n_slices": True}, "n_slices"),
        ("moves", {"max_segment_links": 8}, "max_segment_links"),
        ("moves", {"worm_sector_weight": 0.0}, "worm_sector_weight"),
        ("run", {"measurement_stride": 0}, "measurement_stride"),
        ("estimators", {"green_spatial_bins": 1}, "green_spatial_bins"),
        ("potential", {"epsilon": 1.0}, "not allowed"),
    ],
)
def test_python_construction_and_replacement_cannot_bypass_validation(
    section, changes, match
):
    config = small_config()
    changed_section = replace(getattr(config, section), **changes)
    with pytest.raises(ConfigError, match=match):
        replace(config, **{section: changed_section})
    arguments = {field.name: getattr(config, field.name) for field in fields(config)}
    arguments[section] = changed_section
    with pytest.raises(ConfigError, match=match):
        SimulationConfig(**arguments)


def test_python_move_weights_and_section_types_are_validated():
    config = small_config()
    for changes, message in [
        ({"close": 0.0}, "open.*close"),
        ({"wiggle": -1.0}, "nonnegative"),
    ]:
        moves = replace(config.moves, weights=replace(config.moves.weights, **changes))
        with pytest.raises(ConfigError, match=message):
            replace(config, moves=moves)
    with pytest.raises(ConfigError, match="expected SystemConfig"):
        replace(config, system={"beta": 1.0})


@pytest.mark.parametrize(
    "pair,parameters",
    [
        ("none", {}),
        ("periodized_gaussian", {"epsilon": 1.0, "sigma": 0.5}),
    ],
)
def test_small_system_is_self_contained_and_round_trips(
    pair, parameters, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config = small_config(pair=pair, **parameters)
    assert SimulationConfig.from_mapping(config.input_dict()) == config
    assert config.resolved_dict()["discretization"]["tau"] == config.tau
    assert set(config.moves.weights.normalized()) == set(
        Sampler(config, Simulation(config).state, np.random.default_rng(1)).counters
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"pair": "periodized_gaussian"}, "requires"),
        ({"epsilon": 1.0}, "not allowed"),
        ({"n_slices": 2}, ">= 3"),
        ({"n_slices": 4.5}, "integer"),
        ({"seed": -1}, "run.seed"),
    ],
)
def test_small_system_reports_invalid_arguments(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        SimulationConfig.small_system(**kwargs)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"box_length": 9.0}, "state.box_length"),
        ({"n_slices": 16}, "state.n_slices"),
        ({"ndim": 2}, "state.ndim"),
    ],
)
def test_injected_state_must_match_config_in_both_entry_points(changes, match):
    config = small_config()
    arguments = dict(particle_count=2, n_slices=8, ndim=1, box_length=4.0)
    arguments.update(changes)
    state = Configuration.from_straight_worldlines(**arguments)
    with pytest.raises(ConfigError, match=match):
        Simulation(config, state=state)
    with pytest.raises(ConfigError, match=match):
        Sampler(config, state, np.random.default_rng(1))


def test_injected_live_particle_number_need_not_match_initialization():
    config = small_config()
    state = Configuration.from_straight_worldlines(
        particle_count=1,
        n_slices=8,
        ndim=1,
        box_length=4.0,
    )
    assert Simulation(config, state=state).state.n_particles == 1


def test_trace_is_passive_and_reconstructs_accepted_and_rejected_candidates():
    config = small_config(pair="periodized_gaussian", epsilon=1.0, sigma=0.5)
    plain, traced = Simulation(config), Simulation(config)
    outcomes = set()
    first_event = None
    for _ in range(300):
        expected = plain.step()
        event = traced.step(trace=True)
        assert event == expected
        assert expected.before is expected.after is None
        assert traced.state.state_digest(
            include_revision=True
        ) == plain.state.state_digest(include_revision=True)
        before = Configuration.from_snapshot(event.before)
        after = Configuration.from_snapshot(event.after)
        assert after.state_digest(include_revision=True) == traced.state.state_digest(
            include_revision=True
        )
        candidate = event.candidate()
        if event.status is MoveStatus.PROPOSED:
            assert candidate is not None
            assert event.breakdown.accepted == (
                event.breakdown.log_uniform < min(0.0, event.breakdown.log_ratio)
            )
            assert event.breakdown.proposal_selection == event.proposal_ratio.total
            if event.accepted:
                assert candidate.state_digest() == after.state_digest()
                outcomes.add("accepted")
            else:
                outcomes.add("rejected")
        else:
            assert candidate is None
            assert event.breakdown is event.proposal_ratio is None
            outcomes.add("inapplicable")
        if not event.accepted:
            assert before.state_digest(include_revision=True) == after.state_digest(
                include_revision=True
            )
        if first_event is None:
            first_event = event
            saved_digest = before.state_digest(include_revision=True)
    assert outcomes == {"accepted", "rejected", "inapplicable"}
    assert (
        Configuration.from_snapshot(first_event.before).state_digest(
            include_revision=True
        )
        == saved_digest
    )
    assert traced.rng.bit_generator.state == plain.rng.bit_generator.state
    assert traced.sampler.counters_snapshot() == plain.sampler.counters_snapshot()
    assert traced.summary() == plain.summary()
    assert traced.completed_sweeps == traced.estimators.total_measurement_count == 0
    with pytest.raises(TypeError):
        first_event.before["revision"] = 999
    with pytest.raises(ValueError):
        first_event.before["positions"][0, 0] = 99
    with pytest.raises(ValueError):
        first_event.before["positions"].setflags(write=True)
    with pytest.raises(ValueError, match="trace=True"):
        expected.candidate()


def test_trace_preserves_structurally_invalid_event(monkeypatch):
    simulation = Simulation(small_config())

    def invalid(state, rng):
        return ProposalPatch(
            "invalid-demo",
            state.revision,
            MoveStatus.STRUCTURALLY_INVALID,
            invalid_reason="deliberately invalid proposal",
        )

    for move in simulation.sampler._moves.values():
        monkeypatch.setattr(move, "propose", invalid)
    event = simulation.step(trace=True)
    assert event.status is MoveStatus.STRUCTURALLY_INVALID
    assert event.patch.invalid_reason == "deliberately invalid proposal"
    assert event.candidate() is None
    assert (
        Configuration.from_snapshot(event.before).state_digest()
        == Configuration.from_snapshot(event.after).state_digest()
    )


def test_nonfinite_acceptance_ratio_cannot_silently_accept(monkeypatch):
    simulation = Simulation(small_config())
    monkeypatch.setattr(
        simulation.target, "delta_local", lambda *_: TargetDelta(float("nan"), 0, 0, 0)
    )
    before = simulation.state.state_digest(include_revision=True)
    with pytest.raises(
        FloatingPointError, match="nonfinite log ratio.*move=.*revision="
    ):
        for _ in range(100):
            simulation.step()
    assert simulation.state.state_digest(include_revision=True) == before


def test_in_memory_results_and_traced_advance_preserve_measurement_lifecycle(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config = small_config()
    plain, traced = Simulation(config), Simulation(config)
    initial = traced.results()
    assert initial["scalars"]["particle_number"]["mean"] is None
    assert initial["scalars"]["particle_number"]["standard_error"] is None
    json.dumps(initial, allow_nan=False)
    events = []
    traced.advance(max_sweeps=2, trace=True, after_step=events.append)
    assert traced.estimators.total_measurement_count == 0
    traced.advance(trace=True, after_step=events.append)
    plain.advance()
    assert len(events) == config.moves.steps_per_sweep * traced.target_sweeps
    assert traced.summary() == plain.summary()
    assert traced.rng.bit_generator.state == plain.rng.bit_generator.state
    assert traced.state.state_digest(include_revision=True) == plain.state.state_digest(
        include_revision=True
    )
    assert traced.sampler.counters_snapshot() == plain.sampler.counters_snapshot()
    assert traced.estimators.snapshot() == plain.estimators.snapshot()
    result = traced.results()
    assert result == plain.results()
    assert result["summary"]["measurement_opportunities"] == 10
    assert initial["summary"]["measurement_opportunities"] == 0
    number = result["scalars"]["particle_number"]
    assert number["mean"] == traced.estimators.accumulators["particle_number"].mean
    assert number["standard_error"] is None
    assert number["uncertainty_status"] == "insufficient_blocking_levels"
    assert number["normalization_status"] == "physical_z_conditional"
    assert number["estimator_variant"] == "link_occupation"
    assert len(result["green_function"]) == 32
    json.dumps(result, allow_nan=False)
    result["scalars"]["particle_number"]["mean"] = -99
    result["green_function"][0]["count"] = -99
    assert traced.results() == plain.results()
    assert not list(tmp_path.iterdir())


def test_checkpoint_after_traced_advance_resumes_exactly(tmp_path):
    config = small_config()
    continuous = Simulation(config)
    continuous.advance()
    split = Simulation(config)
    split.advance(max_sweeps=5, trace=True, after_step=lambda event: event.candidate())
    path = split.checkpoint(tmp_path / "traced.npz")
    resumed = Simulation.from_checkpoint(path)
    resumed.advance()
    assert resumed.results() == continuous.results()
    assert resumed.rng.bit_generator.state == continuous.rng.bit_generator.state
    assert resumed.state.state_digest(
        include_revision=True
    ) == continuous.state.state_digest(include_revision=True)
    assert resumed.estimators.snapshot() == continuous.estimators.snapshot()
