from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from wormpimc.config import ConfigError, SimulationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


@pytest.mark.parametrize(
    "name",
    [
        "free_particle.toml",
        "closed_worldline.toml",
        "closed_sampler.toml",
        "ideal_bose_gas.toml",
        "gaussian_repulsion.toml",
        "worm_sampler.toml",
        "worm_statistics.toml",
    ],
)
def test_example_configs_are_valid(name: str) -> None:
    config = SimulationConfig.from_toml(CONFIG_DIR / name)

    assert config.tau == pytest.approx(
        config.system.beta / config.discretization.n_slices
    )
    assert sum(config.moves.weights.normalized().values()) == pytest.approx(1.0)
    assert len(config.config_hash) == 64


def test_configuration_is_immutable() -> None:
    config = SimulationConfig.from_toml(CONFIG_DIR / "free_particle.toml")

    with pytest.raises(FrozenInstanceError):
        config.system.beta = 3.0  # type: ignore[misc]


def test_hash_is_stable_for_the_same_file() -> None:
    first = SimulationConfig.from_toml(CONFIG_DIR / "gaussian_repulsion.toml")
    second = SimulationConfig.from_toml(CONFIG_DIR / "gaussian_repulsion.toml")

    assert first.config_hash == second.config_hash
    assert first.canonical_json() == second.canonical_json()


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    text = (CONFIG_DIR / "free_particle.toml").read_text(encoding="utf-8")
    text = text.replace("ndim = 1", "ndim = 1\nmystery = 7", 1)
    path = tmp_path / "unknown.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="system: unknown field"):
        SimulationConfig.from_toml(path)


def test_segment_length_must_be_smaller_than_slice_count(
    tmp_path: Path,
) -> None:
    text = (CONFIG_DIR / "free_particle.toml").read_text(encoding="utf-8")
    text = text.replace("max_segment_links = 8", "max_segment_links = 32", 1)
    path = tmp_path / "long-segment.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="must be smaller"):
        SimulationConfig.from_toml(path)


def test_noninteracting_chemical_potential_must_be_negative(
    tmp_path: Path,
) -> None:
    text = (CONFIG_DIR / "free_particle.toml").read_text(encoding="utf-8")
    text = text.replace(
        "chemical_potential = -4.0",
        "chemical_potential = 0.0",
        1,
    )
    path = tmp_path / "bad-mu.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="must be negative"):
        SimulationConfig.from_toml(path)


def test_resolved_config_contains_derived_values() -> None:
    config = SimulationConfig.from_toml(CONFIG_DIR / "ideal_bose_gas.toml")
    resolved = config.resolved_dict()

    assert resolved["schema_version"] == 4
    assert resolved["discretization"]["tau"] == config.tau
    assert resolved["initialization"]["particle_count"] == 2
    assert resolved["moves"]["displacement_scale"] == 0.5
    assert resolved["moves"]["worm_sector_weight"] == 1.0
    assert resolved["moves"]["worm_measure_coefficient"] == pytest.approx(
        1.0 / (8.0 * 64 * 8)
    )
    assert resolved["moves"]["selection_strategy"] == "all_moves"
    assert sum(resolved["moves"]["probabilities"].values()) == pytest.approx(1.0)
    assert resolved["estimators"]["green_spatial_bins"] == 64
    assert resolved["estimators"]["green_bin_width"] == pytest.approx(8.0 / 64)
    assert resolved["estimators"]["green_equal_time_side"] == "beta_minus"


def test_normalized_toml_round_trip_preserves_hash(tmp_path: Path) -> None:
    original = SimulationConfig.from_toml(CONFIG_DIR / "closed_sampler.toml")
    path = tmp_path / "normalized.toml"
    path.write_text(original.to_toml(), encoding="utf-8")

    restored = SimulationConfig.from_toml(path)

    assert restored.config_hash == original.config_hash
    assert restored.input_dict() == original.input_dict()


def test_inverse_move_weights_must_be_enabled_in_pairs(tmp_path: Path) -> None:
    text = (CONFIG_DIR / "ideal_bose_gas.toml").read_text(encoding="utf-8")
    text = text.replace("close = 1.0", "close = 0.0", 1)
    path = tmp_path / "unpaired.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="open.*close"):
        SimulationConfig.from_toml(path)


def test_green_histogram_requires_at_least_two_bins(tmp_path: Path) -> None:
    text = (CONFIG_DIR / "free_particle.toml").read_text(encoding="utf-8")
    text = text.replace("green_spatial_bins = 64", "green_spatial_bins = 1")
    path = tmp_path / "one-bin.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="green_spatial_bins.*>= 2"):
        SimulationConfig.from_toml(path)
