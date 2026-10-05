from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from wormpimc.config import SimulationConfig
from wormpimc.configuration import StalePatchError, initialize_configuration
from wormpimc.measure import PrimitiveTargetMeasure, assert_local_matches_full
from wormpimc.moves import DisplaceMove, WiggleMove


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "closed_sampler.toml"


def components(target: object) -> tuple[float, float, float, float]:
    return (
        float(getattr(target, "kinetic")),
        float(getattr(target, "potential")),
        float(getattr(target, "chemical")),
        float(getattr(target, "sector_measure")),
    )


def test_wiggle_local_delta_and_bridge_cancellation() -> None:
    config = SimulationConfig.from_toml(CONFIG_PATH)
    state = initialize_configuration(config)
    measure = PrimitiveTargetMeasure(config)
    move = WiggleMove(config, config.moves.weights.normalized()["wiggle"])
    digest_before = state.state_digest(include_revision=True)
    patch = move.propose(state, np.random.default_rng(1122))

    local = measure.delta_local(state, patch)
    full = measure.delta_full(state, patch)
    proposal = move.proposal_ratio(patch)

    assert_local_matches_full(local, full)
    assert local.kinetic + proposal.total == pytest.approx(0.0, abs=1.0e-12)
    assert state.state_digest(include_revision=True) == digest_before


def test_displace_is_symmetric_and_preserves_kinetic_weight() -> None:
    config = SimulationConfig.from_toml(CONFIG_PATH)
    state = initialize_configuration(config)
    measure = PrimitiveTargetMeasure(config)
    move = DisplaceMove(
        config, config.moves.weights.normalized()["displace"]
    )
    patch = move.propose(state, np.random.default_rng(3344))
    local = measure.delta_local(state, patch)

    assert_local_matches_full(local, measure.delta_full(state, patch))
    assert local.kinetic == pytest.approx(0.0, abs=1.0e-12)
    assert move.proposal_ratio(patch).total == 0.0


@pytest.mark.parametrize("move_name", ["wiggle", "displace"])
def test_forward_reverse_round_trip_restores_state_and_log_ratio(
    move_name: str,
) -> None:
    config = SimulationConfig.from_toml(CONFIG_PATH)
    state = initialize_configuration(config)
    measure = PrimitiveTargetMeasure(config)
    probability = config.moves.weights.normalized()[move_name]
    move = (
        WiggleMove(config, probability)
        if move_name == "wiggle"
        else DisplaceMove(config, probability)
    )
    original_digest = state.state_digest()
    patch = move.propose(state, np.random.default_rng(5566))
    forward_target = measure.delta_local(state, patch)
    forward_proposal = move.proposal_ratio(patch)
    forward_log_ratio = forward_target.total + forward_proposal.total
    state.apply(patch)

    reverse = patch.reversed(base_revision=state.revision)
    reverse_target = measure.delta_local(state, reverse)
    reverse_proposal = move.proposal_ratio(reverse)
    reverse_log_ratio = reverse_target.total + reverse_proposal.total
    state.apply(reverse)

    assert forward_log_ratio + reverse_log_ratio == pytest.approx(
        0.0, abs=2.0e-11
    )
    assert tuple(-x for x in components(forward_target)) == pytest.approx(
        components(reverse_target), abs=2.0e-11
    )
    assert state.state_digest() == original_digest
    assert state.revision == 2


def test_stale_patch_is_rejected() -> None:
    config = SimulationConfig.from_toml(CONFIG_PATH)
    state = initialize_configuration(config)
    move = WiggleMove(config, config.moves.weights.normalized()["wiggle"])
    patch = move.propose(state, np.random.default_rng(7788))
    state.apply(patch)

    with pytest.raises(StalePatchError, match="patch revision"):
        state.apply(patch)
