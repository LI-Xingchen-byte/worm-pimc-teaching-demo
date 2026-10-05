from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wormpimc.config import SimulationConfig
from wormpimc.configuration import Configuration, initialize_configuration
from wormpimc.measure import PrimitiveTargetMeasure, assert_local_matches_full
from wormpimc.moves import (
    AdvanceMove,
    CloseMove,
    InsertMove,
    OpenMove,
    RecedeMove,
    RemoveMove,
    SwapMove,
)
from wormpimc.types import MoveStatus, Sector
from wormpimc.simulation import Simulation
from wormpimc.worm_measure import (
    advance_ratio,
    close_ratio,
    insert_ratio,
    open_ratio,
    recede_ratio,
    remove_ratio,
    swap_ratio,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "ideal_bose_gas.toml"
INTERACTING_CONFIG_PATH = PROJECT_ROOT / "configs" / "worm_sampler.toml"


def _moves(config: SimulationConfig) -> dict[str, object]:
    probabilities = config.moves.weights.normalized()
    return {
        "open": OpenMove(config, probabilities["open"], probabilities["close"]),
        "close": CloseMove(config, probabilities["close"], probabilities["open"]),
        "insert": InsertMove(config, probabilities["insert"], probabilities["remove"]),
        "remove": RemoveMove(config, probabilities["remove"], probabilities["insert"]),
        "advance": AdvanceMove(config, probabilities["advance"], probabilities["recede"]),
        "recede": RecedeMove(config, probabilities["recede"], probabilities["advance"]),
        "swap": SwapMove(config, probabilities["swap"]),
    }


def _assert_round_trip(
    state: Configuration,
    forward_move: object,
    reverse_move: object,
    seed: int,
) -> None:
    measure = PrimitiveTargetMeasure(SimulationConfig.from_toml(CONFIG_PATH))
    original = state.state_digest()
    patch = forward_move.propose(state, np.random.default_rng(seed))
    assert patch.status is MoveStatus.PROPOSED
    forward_target = measure.delta_local(state, patch)
    assert_local_matches_full(forward_target, measure.delta_full(state, patch))
    forward_proposal = forward_move.proposal_ratio(patch)
    state.apply(patch)

    reverse = patch.reversed(base_revision=state.revision)
    reverse_target = measure.delta_local(state, reverse)
    assert_local_matches_full(reverse_target, measure.delta_full(state, reverse))
    reverse_proposal = reverse_move.proposal_ratio(reverse)
    state.apply(reverse)

    assert forward_target.total + reverse_target.total == pytest.approx(
        0.0, abs=5.0e-11
    )
    assert forward_proposal.total + reverse_proposal.total == pytest.approx(
        0.0, abs=5.0e-11
    )
    assert state.state_digest() == original
    state.validate()


def _assert_matches_oracle(
    actual_target: object,
    actual_proposal: object,
    oracle: object,
) -> None:
    assert (
        actual_target.kinetic,
        actual_target.potential,
        actual_target.chemical,
        actual_target.sector_measure,
    ) == pytest.approx(
        (
            oracle.target.kinetic,
            oracle.target.potential,
            oracle.target.chemical,
            oracle.target.sector_measure,
        ),
        abs=5.0e-11,
    )
    assert (
        actual_proposal.log_q_forward,
        actual_proposal.log_q_reverse,
        actual_proposal.log_move_selection_forward,
        actual_proposal.log_move_selection_reverse,
    ) == pytest.approx(
        (
            oracle.proposal.log_q_forward,
            oracle.proposal.log_q_reverse,
            oracle.proposal.log_move_selection_forward,
            oracle.proposal.log_move_selection_reverse,
        ),
        abs=5.0e-11,
    )


def test_open_close_exact_patch_round_trip() -> None:
    config = SimulationConfig.from_toml(CONFIG_PATH)
    moves = _moves(config)
    _assert_round_trip(
        initialize_configuration(config), moves["open"], moves["close"], 101
    )


def test_insert_remove_exact_patch_round_trip_including_empty_z() -> None:
    config = SimulationConfig.from_toml(CONFIG_PATH)
    moves = _moves(config)
    empty = Configuration(
        positions=np.zeros((1, 1)),
        slice_of=np.zeros(1, dtype=np.int64),
        next_of=np.full(1, -1, dtype=np.int64),
        prev_of=np.full(1, -1, dtype=np.int64),
        image_to_next=np.zeros((1, 1), dtype=np.int64),
        active=np.zeros(1, dtype=np.bool_),
        box_length=config.system.box_length,
        n_slices=config.discretization.n_slices,
    )
    assert empty.n_particles == 0
    _assert_round_trip(empty, moves["insert"], moves["remove"], 202)


def test_advance_recede_exact_patch_round_trip() -> None:
    config = SimulationConfig.from_toml(CONFIG_PATH)
    moves = _moves(config)
    state = initialize_configuration(config)
    inserted = moves["insert"].propose(state, np.random.default_rng(303))
    state.apply(inserted)

    _assert_round_trip(state, moves["advance"], moves["recede"], 404)


def test_recede_advance_exact_patch_round_trip() -> None:
    config = SimulationConfig.from_toml(CONFIG_PATH)
    moves = _moves(config)
    state = initialize_configuration(config)
    state.apply(moves["insert"].propose(state, np.random.default_rng(505)))
    while state.open_chain_links <= config.moves.max_segment_links:
        state.apply(
            moves["advance"].propose(state, np.random.default_rng(state.revision + 600))
        )

    _assert_round_trip(state, moves["recede"], moves["advance"], 707)


def test_swap_exact_patch_round_trip() -> None:
    config = SimulationConfig.from_toml(CONFIG_PATH)
    moves = _moves(config)
    state = initialize_configuration(config)
    state.apply(moves["open"].propose(state, np.random.default_rng(808)))

    _assert_round_trip(state, moves["swap"], moves["swap"], 909)


def test_swap_can_create_a_two_particle_permutation_cycle() -> None:
    config = SimulationConfig.from_toml(CONFIG_PATH)
    moves = _moves(config)
    point = np.array([[1.0]] * (config.discretization.n_slices + 1))
    state = Configuration.from_closed_worldlines(
        [point, point], config.system.box_length
    )
    state.apply(moves["open"].propose(state, np.random.default_rng(111)))
    closed_ids = set(state.cycle_ids(state.cycle_starts()[0]))

    swap_patch = None
    for seed in range(200):
        candidate = moves["swap"].propose(state, np.random.default_rng(seed))
        if (
            candidate.status is MoveStatus.PROPOSED
            and int(candidate.metadata_dict()["candidate_id"]) in closed_ids
        ):
            swap_patch = candidate
            break
    assert swap_patch is not None
    state.apply(swap_patch)
    close_patch = moves["close"].propose(state, np.random.default_rng(222))
    assert close_patch.status is MoveStatus.PROPOSED
    state.apply(close_patch)

    assert state.sector is Sector.Z
    assert len(state.cycle_starts()) == 1
    assert len(state.cycle_ids(state.cycle_starts()[0])) == (
        2 * config.discretization.n_slices
    )


def test_short_full_worm_chain_visits_both_sectors_and_permutations() -> None:
    data = SimulationConfig.from_toml(CONFIG_PATH).input_dict()
    data["system"]["box_length"] = 2.0
    data["discretization"]["n_slices"] = 16
    data["moves"]["max_segment_links"] = 4
    config = SimulationConfig.from_mapping(data)
    simulation = Simulation(config)
    residence = {Sector.Z: 0, Sector.G: 0}
    longest_cycle = 0
    for _ in range(500):
        simulation.step()
        residence[simulation.state.sector] += 1
        if simulation.state.sector is Sector.Z:
            longest_cycle = max(
                longest_cycle,
                max(
                    (
                        len(simulation.state.cycle_ids(start))
                        for start in simulation.state.cycle_starts()
                    ),
                    default=0,
                ),
            )

    assert residence[Sector.Z] > 0
    assert residence[Sector.G] > 0
    assert simulation.sampler.counters["swap"].accepted > 0
    assert longest_cycle > config.discretization.n_slices


def test_live_topology_components_match_gate_e_oracles() -> None:
    config = SimulationConfig.from_toml(INTERACTING_CONFIG_PATH)
    moves = _moves(config)
    probabilities = config.moves.weights.normalized()
    measure = PrimitiveTargetMeasure(config)
    maximum = config.moves.max_segment_links
    coefficient = config.worm_measure_coefficient
    volume = config.system.box_length**config.system.ndim

    z_state = initialize_configuration(config)
    open_patch = moves["open"].propose(z_state, np.random.default_rng(1201))
    metadata = open_patch.metadata_dict()
    target = measure.delta_full(z_state, open_patch)
    _assert_matches_oracle(
        target,
        moves["open"].proposal_ratio(open_patch),
        open_ratio(
            segment_log_kinetic=float(metadata["segment_log_kinetic"]),
            endpoint_log_free_density=float(
                metadata["endpoint_log_free_density"]
            ),
            delta_potential_action=-target.potential,
            chemical_potential=config.system.chemical_potential,
            tau=config.tau,
            segment_links=int(metadata["segment_links"]),
            diagonal_bead_count=z_state.number_of_beads,
            max_segment_links=maximum,
            worm_measure_coefficient=coefficient,
            move_probability=probabilities["open"],
            reverse_move_probability=probabilities["close"],
        ),
    )
    z_state.apply(open_patch)
    close_patch = moves["close"].propose(
        z_state, np.random.default_rng(1202)
    )
    metadata = close_patch.metadata_dict()
    target = measure.delta_full(z_state, close_patch)
    _assert_matches_oracle(
        target,
        moves["close"].proposal_ratio(close_patch),
        close_ratio(
            segment_log_kinetic=float(metadata["segment_log_kinetic"]),
            endpoint_log_free_density=float(
                metadata["endpoint_log_free_density"]
            ),
            delta_potential_action=-target.potential,
            chemical_potential=config.system.chemical_potential,
            tau=config.tau,
            segment_links=int(metadata["segment_links"]),
            final_diagonal_bead_count=(
                z_state.number_of_beads + len(close_patch.added_beads)
            ),
            max_segment_links=maximum,
            worm_measure_coefficient=coefficient,
            move_probability=probabilities["close"],
            reverse_move_probability=probabilities["open"],
        ),
    )

    insert_state = initialize_configuration(config)
    insert_patch = moves["insert"].propose(
        insert_state, np.random.default_rng(1301)
    )
    metadata = insert_patch.metadata_dict()
    target = measure.delta_full(insert_state, insert_patch)
    _assert_matches_oracle(
        target,
        moves["insert"].proposal_ratio(insert_patch),
        insert_ratio(
            segment_log_kinetic=float(metadata["segment_log_kinetic"]),
            delta_potential_action=-target.potential,
            chemical_potential=config.system.chemical_potential,
            tau=config.tau,
            segment_links=int(metadata["segment_links"]),
            volume=volume,
            n_slices=config.discretization.n_slices,
            max_segment_links=maximum,
            worm_measure_coefficient=coefficient,
            move_probability=probabilities["insert"],
            reverse_move_probability=probabilities["remove"],
        ),
    )
    insert_state.apply(insert_patch)
    remove_patch = moves["remove"].propose(
        insert_state, np.random.default_rng(1302)
    )
    metadata = remove_patch.metadata_dict()
    target = measure.delta_full(insert_state, remove_patch)
    _assert_matches_oracle(
        target,
        moves["remove"].proposal_ratio(remove_patch),
        remove_ratio(
            segment_log_kinetic=float(metadata["segment_log_kinetic"]),
            delta_potential_action=-target.potential,
            chemical_potential=config.system.chemical_potential,
            tau=config.tau,
            segment_links=int(metadata["segment_links"]),
            volume=volume,
            n_slices=config.discretization.n_slices,
            max_segment_links=maximum,
            worm_measure_coefficient=coefficient,
            move_probability=probabilities["remove"],
            reverse_move_probability=probabilities["insert"],
        ),
    )

    advance_state = initialize_configuration(config)
    advance_state.apply(
        moves["insert"].propose(advance_state, np.random.default_rng(1401))
    )
    advance_patch = moves["advance"].propose(
        advance_state, np.random.default_rng(1402)
    )
    metadata = advance_patch.metadata_dict()
    target = measure.delta_full(advance_state, advance_patch)
    _assert_matches_oracle(
        target,
        moves["advance"].proposal_ratio(advance_patch),
        advance_ratio(
            segment_log_kinetic=float(metadata["segment_log_kinetic"]),
            delta_potential_action=-target.potential,
            chemical_potential=config.system.chemical_potential,
            tau=config.tau,
            segment_links=int(metadata["segment_links"]),
            max_segment_links=maximum,
            move_probability=probabilities["advance"],
            reverse_move_probability=probabilities["recede"],
        ),
    )
    advance_state.apply(advance_patch)
    while advance_state.open_chain_links <= maximum:
        advance_state.apply(
            moves["advance"].propose(
                advance_state,
                np.random.default_rng(1500 + advance_state.revision),
            )
        )
    recede_patch = moves["recede"].propose(
        advance_state, np.random.default_rng(1501)
    )
    metadata = recede_patch.metadata_dict()
    target = measure.delta_full(advance_state, recede_patch)
    _assert_matches_oracle(
        target,
        moves["recede"].proposal_ratio(recede_patch),
        recede_ratio(
            segment_log_kinetic=float(metadata["segment_log_kinetic"]),
            delta_potential_action=-target.potential,
            chemical_potential=config.system.chemical_potential,
            tau=config.tau,
            segment_links=int(metadata["segment_links"]),
            max_segment_links=maximum,
            move_probability=probabilities["recede"],
            reverse_move_probability=probabilities["advance"],
        ),
    )

    swap_state = initialize_configuration(config)
    swap_state.apply(
        moves["open"].propose(swap_state, np.random.default_rng(1601))
    )
    swap_patch = moves["swap"].propose(
        swap_state, np.random.default_rng(1602)
    )
    assert swap_patch.status is MoveStatus.PROPOSED
    metadata = swap_patch.metadata_dict()
    target = measure.delta_full(swap_state, swap_patch)
    _assert_matches_oracle(
        target,
        moves["swap"].proposal_ratio(swap_patch),
        swap_ratio(
            old_segment_log_kinetic=float(
                metadata["old_segment_log_kinetic"]
            ),
            new_segment_log_kinetic=float(
                metadata["new_segment_log_kinetic"]
            ),
            log_candidate_normalizer_before=float(
                metadata["log_candidate_normalizer_before"]
            ),
            log_candidate_normalizer_after=float(
                metadata["log_candidate_normalizer_after"]
            ),
            delta_potential_action=-target.potential,
            move_probability=probabilities["swap"],
            reverse_move_probability=probabilities["swap"],
        ),
    )
