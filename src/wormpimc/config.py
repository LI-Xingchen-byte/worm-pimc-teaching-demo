"""Strict, immutable configuration model for the Worm PIMC project.

This module constructs and validates input. Worldline construction and
Monte Carlo transitions belong to the configuration and sampler modules.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar


ROOT_TABLES = frozenset(
    {
        "system",
        "discretization",
        "potential",
        "initialization",
        "moves",
        "estimators",
        "run",
        "output",
    }
)
MOVE_NAMES = (
    "wiggle",
    "displace",
    "open",
    "close",
    "insert",
    "remove",
    "advance",
    "recede",
    "swap",
)
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_SEED = 2**64 - 1


class ConfigError(ValueError):
    """Raised when an input file violates the versioned configuration schema."""


@dataclass(frozen=True, slots=True)
class SystemConfig:
    """Physical system parameters in the code-unit convention."""

    ndim: int
    box_length: float
    lambda_kin: float
    beta: float
    chemical_potential: float


@dataclass(frozen=True, slots=True)
class DiscretizationConfig:
    """Imaginary-time discretization parameters."""

    n_slices: int
    action: str


@dataclass(frozen=True, slots=True)
class PotentialConfig:
    """External and pair-potential selection."""

    pair: str
    external: str
    epsilon: float | None
    sigma: float | None
    image_tolerance: float | None
    external_offset: float | None = None
    external_cosine: tuple[float, ...] | None = None
    external_sine: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class InitializationConfig:
    """Deterministic starting-state controls for the closed-sector sampler."""

    particle_count: int
    strategy: str


@dataclass(frozen=True, slots=True)
class MoveWeights:
    """Immutable unnormalized weights for the nine reserved move classes."""

    wiggle: float
    displace: float
    open: float
    close: float
    insert: float
    remove: float
    advance: float
    recede: float
    swap: float

    def as_dict(self) -> dict[str, float]:
        """Return a new mapping in the canonical move order."""

        return {name: getattr(self, name) for name in MOVE_NAMES}

    def normalized(self) -> dict[str, float]:
        """Return normalized move-selection probabilities."""

        values = self.as_dict()
        total = math.fsum(values.values())
        if not math.isfinite(total) or total <= 0.0:
            raise ConfigError("moves.weights: at least one weight must be positive")
        return {name: value / total for name, value in values.items()}


@dataclass(frozen=True, slots=True)
class MovesConfig:
    """Move-selection and local-segment controls."""

    steps_per_sweep: int
    max_segment_links: int
    displacement_scale: float
    worm_sector_weight: float
    selection_strategy: str
    weights: MoveWeights


@dataclass(frozen=True, slots=True)
class EstimatorsConfig:
    """Histogram controls that affect estimator state and output schemas."""

    green_spatial_bins: int


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Finite run-length and reproducibility settings."""

    seed: int
    warmup_sweeps: int
    measurement_sweeps: int
    measurement_stride: int
    checkpoint_interval_sweeps: int
    invariant_check_interval_steps: int


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output location and user-facing run label."""

    root: str
    label: str


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Fully validated, immutable schema-4 simulation configuration."""

    SCHEMA_VERSION: ClassVar[int] = 4

    system: SystemConfig
    discretization: DiscretizationConfig
    potential: PotentialConfig
    initialization: InitializationConfig
    moves: MovesConfig
    estimators: EstimatorsConfig
    run: RunConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        """Validate Python construction and dataclass replacement like TOML."""

        sections = (
            ("system", SystemConfig, _parse_system),
            ("discretization", DiscretizationConfig, _parse_discretization),
            ("potential", PotentialConfig, _parse_potential),
            ("initialization", InitializationConfig, _parse_initialization),
            ("moves", MovesConfig, _parse_moves),
            ("estimators", EstimatorsConfig, _parse_estimators),
            ("run", RunConfig, _parse_run),
            ("output", OutputConfig, _parse_output),
        )
        for name, expected_type, parse in sections:
            section = getattr(self, name)
            if not isinstance(section, expected_type):
                raise ConfigError(f"{name}: expected {expected_type.__name__}")
            values = asdict(section)
            if name == "potential":
                values = {key: value for key, value in values.items() if value is not None}
            # Reuse the same validators and normalization as file-based input.
            object.__setattr__(self, name, parse(values))
        self._validate_cross_constraints()

    @classmethod
    def small_system(
        cls,
        *,
        beta: float = 1.0,
        box_length: float = 4.0,
        lambda_kin: float = 0.5,
        chemical_potential: float = -2.0,
        n_slices: int = 16,
        particle_count: int = 2,
        pair: str = "none",
        epsilon: float | None = None,
        sigma: float | None = None,
        external: str = "none",
        external_offset: float | None = None,
        external_cosine: list[float] | tuple[float, ...] | None = None,
        external_sine: list[float] | tuple[float, ...] | None = None,
        seed: int = 97531,
        warmup_sweeps: int = 20,
        measurement_sweeps: int = 100,
        steps_per_sweep: int = 20,
        green_spatial_bins: int = 8,
    ) -> "SimulationConfig":
        """Configure a small 1D periodic full-Worm demonstration.

        Defaults describe an ideal gas, not a convergence prescription.
        For Gaussian repulsion set pair='periodized_gaussian', epsilon and
        sigma explicitly. particle_count specifies initialization, not fixed N.
        For a static periodic field set external='fourier', external_offset,
        external_cosine and/or external_sine (harmonics start at n=1).
        Inspect input_dict()/resolved_dict(); use dataclasses.replace for
        advanced settings. No files are read or created here.
        """

        slices = _integer(n_slices, "discretization.n_slices", minimum=3)
        potential: dict[str, Any] = {"pair": pair, "external": external}
        for key, value in (
            ("external_offset", external_offset),
            ("external_cosine", external_cosine),
            ("external_sine", external_sine),
        ):
            if value is not None:
                potential[key] = value
        if epsilon is not None:
            potential["epsilon"] = epsilon
        if sigma is not None:
            potential["sigma"] = sigma
        if pair == "periodized_gaussian":
            potential["image_tolerance"] = 1.0e-12
        return cls.from_mapping(
            {
                "system": {
                    "ndim": 1,
                    "beta": beta,
                    "box_length": box_length,
                    "lambda_kin": lambda_kin,
                    "chemical_potential": chemical_potential,
                },
                "discretization": {"n_slices": slices, "action": "primitive"},
                "potential": potential,
                "initialization": {
                    "particle_count": particle_count,
                    "strategy": "straight_worldlines",
                },
                "moves": {
                    "steps_per_sweep": steps_per_sweep,
                    "max_segment_links": min(6, slices - 1),
                    "displacement_scale": 0.5,
                    "worm_sector_weight": 1.0,
                    "selection_strategy": "all_moves",
                    "weights": {
                        name: 4.0 if name == "wiggle" else 1.0
                        for name in MOVE_NAMES
                    },
                },
                "estimators": {"green_spatial_bins": green_spatial_bins},
                "run": {
                    "seed": seed,
                    "warmup_sweeps": warmup_sweeps,
                    "measurement_sweeps": measurement_sweeps,
                    "measurement_stride": 1,
                    "checkpoint_interval_sweeps": 25,
                    "invariant_check_interval_steps": 1,
                },
                "output": {"root": "runs", "label": "small-system"},
            }
        )

    @property
    def tau(self) -> float:
        """Imaginary-time step derived from beta and the number of slices."""

        return self.system.beta / self.discretization.n_slices

    @property
    def worm_measure_coefficient(self) -> float:
        """Return ``C_G = C_0 / (V M s_max)`` in code units."""

        volume = self.system.box_length**self.system.ndim
        return self.moves.worm_sector_weight / (
            volume
            * self.discretization.n_slices
            * self.moves.max_segment_links
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "SimulationConfig":
        """Load and validate a TOML configuration file."""

        config_path = Path(path)
        try:
            with config_path.open("rb") as handle:
                data = tomllib.load(handle)
        except FileNotFoundError as exc:
            raise ConfigError(f"configuration file not found: {config_path}") from exc
        except OSError as exc:
            raise ConfigError(
                f"could not read configuration file {config_path}: {exc}"
            ) from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc

        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SimulationConfig":
        """Validate an already parsed mapping against schema version 4."""

        root = _mapping(data, "root")
        _check_keys(root, ROOT_TABLES, ROOT_TABLES, "root")

        system = _parse_system(_table(root, "system", "root"))
        discretization = _parse_discretization(
            _table(root, "discretization", "root")
        )
        potential = _parse_potential(_table(root, "potential", "root"))
        initialization = _parse_initialization(
            _table(root, "initialization", "root")
        )
        moves = _parse_moves(_table(root, "moves", "root"))
        estimators = _parse_estimators(_table(root, "estimators", "root"))
        run = _parse_run(_table(root, "run", "root"))
        output = _parse_output(_table(root, "output", "root"))

        config = cls(
            system=system,
            discretization=discretization,
            potential=potential,
            initialization=initialization,
            moves=moves,
            estimators=estimators,
            run=run,
            output=output,
        )
        return config

    def _validate_cross_constraints(self) -> None:
        if self.moves.max_segment_links >= self.discretization.n_slices:
            raise ConfigError(
                "moves.max_segment_links must be smaller than "
                "discretization.n_slices"
            )

        for forward, reverse in (
            ("open", "close"),
            ("insert", "remove"),
            ("advance", "recede"),
        ):
            forward_weight = getattr(self.moves.weights, forward)
            reverse_weight = getattr(self.moves.weights, reverse)
            if (forward_weight > 0.0) != (reverse_weight > 0.0):
                raise ConfigError(
                    f"moves.weights.{forward} and moves.weights.{reverse} "
                    "must either both be positive or both be zero"
                )

        if (
            self.potential.pair == "none"
            and self.potential.external == "none"
            and self.system.chemical_potential >= 0.0
        ):
            raise ConfigError(
                "system.chemical_potential must be negative for the "
                "noninteracting, untrapped grand-canonical Bose gas"
            )

        if self.potential.pair == "none" and self.potential.external == "fourier":
            lower_bound = fourier_lower_bound(self.potential)
            if self.system.chemical_potential >= lower_bound:
                raise ConfigError(
                    "system.chemical_potential must be below the conservative "
                    f"Fourier lower bound ({lower_bound:.17g}) for the "
                    "noninteracting gas; this sufficient condition may exclude "
                    "otherwise stable models"
                )

    def resolved_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-serializable resolved configuration."""

        potential = {
            "pair": self.potential.pair,
            "external": self.potential.external,
            "epsilon": self.potential.epsilon,
            "sigma": self.potential.sigma,
            "image_tolerance": self.potential.image_tolerance,
        }
        if self.potential.external == "fourier":
            potential.update(
                external_offset=self.potential.external_offset,
                external_cosine=list(self.potential.external_cosine),
                external_sine=list(self.potential.external_sine),
            )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "system": {
                "ndim": self.system.ndim,
                "box_length": self.system.box_length,
                "lambda_kin": self.system.lambda_kin,
                "beta": self.system.beta,
                "chemical_potential": self.system.chemical_potential,
            },
            "discretization": {
                "n_slices": self.discretization.n_slices,
                "action": self.discretization.action,
                "tau": self.tau,
            },
            "potential": potential,
            "initialization": {
                "particle_count": self.initialization.particle_count,
                "strategy": self.initialization.strategy,
            },
            "moves": {
                "steps_per_sweep": self.moves.steps_per_sweep,
                "max_segment_links": self.moves.max_segment_links,
                "displacement_scale": self.moves.displacement_scale,
                "worm_sector_weight": self.moves.worm_sector_weight,
                "worm_measure_coefficient": self.worm_measure_coefficient,
                "selection_strategy": self.moves.selection_strategy,
                "weights": self.moves.weights.as_dict(),
                "probabilities": self.moves.weights.normalized(),
            },
            "estimators": {
                "green_spatial_bins": self.estimators.green_spatial_bins,
                "green_bin_width": (
                    self.system.box_length
                    / self.estimators.green_spatial_bins
                ),
                "green_equal_time_side": "beta_minus",
            },
            "run": {
                "seed": self.run.seed,
                "warmup_sweeps": self.run.warmup_sweeps,
                "measurement_sweeps": self.run.measurement_sweeps,
                "measurement_stride": self.run.measurement_stride,
                "checkpoint_interval_sweeps": (
                    self.run.checkpoint_interval_sweeps
                ),
                "invariant_check_interval_steps": (
                    self.run.invariant_check_interval_steps
                ),
            },
            "output": {
                "root": self.output.root,
                "label": self.output.label,
            },
        }

    def input_dict(self) -> dict[str, Any]:
        """Return strict schema-4 input fields without derived values."""

        resolved = self.resolved_dict()
        resolved.pop("schema_version")
        discretization = dict(resolved["discretization"])
        discretization.pop("tau")
        resolved["discretization"] = discretization
        moves = dict(resolved["moves"])
        moves.pop("probabilities")
        moves.pop("worm_measure_coefficient")
        resolved["moves"] = moves
        estimators = dict(resolved["estimators"])
        estimators.pop("green_bin_width")
        estimators.pop("green_equal_time_side")
        resolved["estimators"] = estimators
        potential = {
            key: value
            for key, value in dict(resolved["potential"]).items()
            if value is not None
        }
        resolved["potential"] = potential
        return resolved

    def to_toml(self) -> str:
        """Serialize a normalized schema-4 input for run provenance."""

        def number(value: float) -> str:
            text = format(float(value), ".17g")
            if "." not in text and "e" not in text.lower():
                text += ".0"
            return text

        lines = [
            "[system]",
            f"ndim = {self.system.ndim}",
            f"box_length = {number(self.system.box_length)}",
            f"lambda_kin = {number(self.system.lambda_kin)}",
            f"beta = {number(self.system.beta)}",
            f"chemical_potential = {number(self.system.chemical_potential)}",
            "",
            "[discretization]",
            f"n_slices = {self.discretization.n_slices}",
            f"action = {json.dumps(self.discretization.action)}",
            "",
            "[potential]",
            f"pair = {json.dumps(self.potential.pair)}",
            f"external = {json.dumps(self.potential.external)}",
        ]
        if self.potential.pair == "periodized_gaussian":
            assert self.potential.epsilon is not None
            assert self.potential.sigma is not None
            assert self.potential.image_tolerance is not None
            lines.extend(
                [
                    f"epsilon = {number(self.potential.epsilon)}",
                    f"sigma = {number(self.potential.sigma)}",
                    f"image_tolerance = {number(self.potential.image_tolerance)}",
                ]
            )
        if self.potential.external == "fourier":
            lines.extend(
                [
                    f"external_offset = {number(self.potential.external_offset)}",
                    "external_cosine = ["
                    + ", ".join(number(x) for x in self.potential.external_cosine)
                    + "]",
                    "external_sine = ["
                    + ", ".join(number(x) for x in self.potential.external_sine)
                    + "]",
                ]
            )
        lines.extend(
            [
                "",
                "[initialization]",
                f"particle_count = {self.initialization.particle_count}",
                f"strategy = {json.dumps(self.initialization.strategy)}",
                "",
                "[moves]",
                f"steps_per_sweep = {self.moves.steps_per_sweep}",
                f"max_segment_links = {self.moves.max_segment_links}",
                f"displacement_scale = {number(self.moves.displacement_scale)}",
                "worm_sector_weight = "
                f"{number(self.moves.worm_sector_weight)}",
                f"selection_strategy = {json.dumps(self.moves.selection_strategy)}",
                "",
                "[moves.weights]",
            ]
        )
        for name, value in self.moves.weights.as_dict().items():
            lines.append(f"{name} = {number(value)}")
        lines.extend(
            [
                "",
                "[estimators]",
                "green_spatial_bins = "
                f"{self.estimators.green_spatial_bins}",
                "",
                "[run]",
                f"seed = {self.run.seed}",
                f"warmup_sweeps = {self.run.warmup_sweeps}",
                f"measurement_sweeps = {self.run.measurement_sweeps}",
                f"measurement_stride = {self.run.measurement_stride}",
                "checkpoint_interval_sweeps = "
                f"{self.run.checkpoint_interval_sweeps}",
                "invariant_check_interval_steps = "
                f"{self.run.invariant_check_interval_steps}",
                "",
                "[output]",
                f"root = {json.dumps(self.output.root)}",
                f"label = {json.dumps(self.output.label)}",
                "",
            ]
        )
        return "\n".join(lines)

    def canonical_json(self) -> str:
        """Return the canonical representation used for hashing."""

        return json.dumps(
            self.resolved_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def config_hash(self) -> str:
        """SHA-256 digest of the canonical resolved configuration."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected a table")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{path}: all keys must be strings")
    return value


def _table(parent: Mapping[str, Any], key: str, parent_path: str) -> Mapping[str, Any]:
    if key not in parent:
        raise ConfigError(f"{parent_path}: missing required table {key!r}")
    return _mapping(parent[key], f"{parent_path}.{key}")


def _check_keys(
    table: Mapping[str, Any],
    allowed: frozenset[str] | set[str],
    required: frozenset[str] | set[str],
    path: str,
) -> None:
    unknown = sorted(set(table) - set(allowed))
    missing = sorted(set(required) - set(table))
    if unknown:
        raise ConfigError(f"{path}: unknown field(s): {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"{path}: missing required field(s): {', '.join(missing)}")


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ConfigError(f"{path}: expected an integer")
    if value < minimum:
        raise ConfigError(f"{path}: must be >= {minimum}")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path}: expected a finite number")
    return result


def _positive_number(value: Any, path: str) -> float:
    result = _finite_number(value, path)
    if result <= 0.0:
        raise ConfigError(f"{path}: must be > 0")
    return result


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: expected a non-empty string")
    return value


def _parse_system(table: Mapping[str, Any]) -> SystemConfig:
    keys = {
        "ndim",
        "box_length",
        "lambda_kin",
        "beta",
        "chemical_potential",
    }
    _check_keys(table, keys, keys, "system")

    ndim = _integer(table["ndim"], "system.ndim", minimum=1)
    if ndim != 1:
        raise ConfigError("system.ndim: Gate B currently supports only ndim = 1")

    return SystemConfig(
        ndim=ndim,
        box_length=_positive_number(table["box_length"], "system.box_length"),
        lambda_kin=_positive_number(table["lambda_kin"], "system.lambda_kin"),
        beta=_positive_number(table["beta"], "system.beta"),
        chemical_potential=_finite_number(
            table["chemical_potential"], "system.chemical_potential"
        ),
    )


def _parse_discretization(table: Mapping[str, Any]) -> DiscretizationConfig:
    keys = {"n_slices", "action"}
    _check_keys(table, keys, keys, "discretization")

    action = _text(table["action"], "discretization.action")
    if action != "primitive":
        raise ConfigError(
            "discretization.action: Gate B currently supports only 'primitive'"
        )

    return DiscretizationConfig(
        n_slices=_integer(
            table["n_slices"], "discretization.n_slices", minimum=1
        ),
        action=action,
    )


def _parse_potential(table: Mapping[str, Any]) -> PotentialConfig:
    field_keys = {"external_offset", "external_cosine", "external_sine"}
    allowed = {"pair", "external", "epsilon", "sigma", "image_tolerance"} | field_keys
    required = {"pair"}
    _check_keys(table, allowed, required, "potential")

    pair = _text(table["pair"], "potential.pair")
    external = _text(table.get("external", "none"), "potential.external")
    if external not in {"none", "fourier"}:
        raise ConfigError(
            "potential.external: supported values are 'none' and 'fourier'"
        )
    field = {}
    if external == "none":
        if field_keys & set(table):
            raise ConfigError("potential: external fields are not allowed when external = 'none'")
    else:
        field = {
            "external_offset": _finite_number(
                table.get("external_offset", 0.0), "potential.external_offset"
            ),
            "external_cosine": _coefficient_sequence(
                table.get("external_cosine", ()), "potential.external_cosine"
            ),
            "external_sine": _coefficient_sequence(
                table.get("external_sine", ()), "potential.external_sine"
            ),
        }
        # Reject overflow in the bound even for interacting configurations.
        fourier_lower_bound(PotentialConfig(pair, external, None, None, None, **field))

    if pair == "none":
        model_fields = {"epsilon", "sigma", "image_tolerance"} & set(table)
        if model_fields:
            names = ", ".join(sorted(model_fields))
            raise ConfigError(
                f"potential: field(s) {names} are not allowed when pair = 'none'"
            )
        return PotentialConfig(
            pair=pair,
            external=external,
            epsilon=None,
            sigma=None,
            image_tolerance=None,
            **field,
        )

    if pair != "periodized_gaussian":
        raise ConfigError(
            "potential.pair: supported values are 'none' and "
            "'periodized_gaussian'"
        )

    required_model = {"epsilon", "sigma", "image_tolerance"}
    missing = sorted(required_model - set(table))
    if missing:
        raise ConfigError(
            "potential: periodized_gaussian requires field(s): "
            + ", ".join(missing)
        )

    tolerance = _positive_number(
        table["image_tolerance"], "potential.image_tolerance"
    )
    if tolerance >= 1.0:
        raise ConfigError("potential.image_tolerance: must be < 1")

    return PotentialConfig(
        pair=pair,
        external=external,
        epsilon=_positive_number(table["epsilon"], "potential.epsilon"),
        sigma=_positive_number(table["sigma"], "potential.sigma"),
        image_tolerance=tolerance,
        **field,
    )


def _coefficient_sequence(value: Any, path: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{path}: expected an array of finite numbers")
    return tuple(_finite_number(x, f"{path}[{i}]") for i, x in enumerate(value))


def fourier_lower_bound(potential: PotentialConfig) -> float:
    """Sufficient lower bound c0 - sum(hypot(a_n, b_n)), not the ground energy."""
    from itertools import zip_longest

    try:
        radius = math.fsum(
            math.hypot(a, b)
            for a, b in zip_longest(
                potential.external_cosine, potential.external_sine, fillvalue=0.0
            )
        )
        bound = potential.external_offset - radius
        maximum_magnitude = abs(potential.external_offset) + radius
    except OverflowError as exc:
        raise ConfigError("potential: Fourier amplitude bound must be finite") from exc
    if not math.isfinite(bound) or not math.isfinite(maximum_magnitude):
        raise ConfigError("potential: Fourier amplitude bound must be finite")
    return bound


def _parse_initialization(
    table: Mapping[str, Any],
) -> InitializationConfig:
    keys = {"particle_count", "strategy"}
    _check_keys(table, keys, keys, "initialization")
    strategy = _text(table["strategy"], "initialization.strategy")
    if strategy != "straight_worldlines":
        raise ConfigError(
            "initialization.strategy: schema 4 supports only "
            "'straight_worldlines'"
        )
    return InitializationConfig(
        particle_count=_integer(
            table["particle_count"],
            "initialization.particle_count",
            minimum=1,
        ),
        strategy=strategy,
    )


def _parse_moves(table: Mapping[str, Any]) -> MovesConfig:
    allowed = {
        "steps_per_sweep",
        "max_segment_links",
        "displacement_scale",
        "worm_sector_weight",
        "selection_strategy",
        "weights",
    }
    _check_keys(table, allowed, allowed, "moves")

    strategy = _text(table["selection_strategy"], "moves.selection_strategy")
    if strategy != "all_moves":
        raise ConfigError(
            "moves.selection_strategy: Gate B currently supports only 'all_moves'"
        )

    weight_table = _table(table, "weights", "moves")
    move_keys = set(MOVE_NAMES)
    _check_keys(weight_table, move_keys, move_keys, "moves.weights")
    weight_values = {
        name: _finite_number(weight_table[name], f"moves.weights.{name}")
        for name in MOVE_NAMES
    }
    negative = [name for name, value in weight_values.items() if value < 0.0]
    if negative:
        raise ConfigError(
            "moves.weights: weights must be nonnegative; invalid: "
            + ", ".join(negative)
        )
    weights = MoveWeights(**weight_values)
    weights.normalized()

    return MovesConfig(
        steps_per_sweep=_integer(
            table["steps_per_sweep"], "moves.steps_per_sweep", minimum=1
        ),
        max_segment_links=_integer(
            table["max_segment_links"], "moves.max_segment_links", minimum=2
        ),
        displacement_scale=_positive_number(
            table["displacement_scale"], "moves.displacement_scale"
        ),
        worm_sector_weight=_positive_number(
            table["worm_sector_weight"],
            "moves.worm_sector_weight",
        ),
        selection_strategy=strategy,
        weights=weights,
    )


def _parse_estimators(table: Mapping[str, Any]) -> EstimatorsConfig:
    keys = {"green_spatial_bins"}
    _check_keys(table, keys, keys, "estimators")
    return EstimatorsConfig(
        green_spatial_bins=_integer(
            table["green_spatial_bins"],
            "estimators.green_spatial_bins",
            minimum=2,
        )
    )


def _parse_run(table: Mapping[str, Any]) -> RunConfig:
    keys = {
        "seed",
        "warmup_sweeps",
        "measurement_sweeps",
        "measurement_stride",
        "checkpoint_interval_sweeps",
        "invariant_check_interval_steps",
    }
    _check_keys(table, keys, keys, "run")

    seed = _integer(table["seed"], "run.seed")
    if seed > MAX_SEED:
        raise ConfigError(f"run.seed: must be <= {MAX_SEED}")

    return RunConfig(
        seed=seed,
        warmup_sweeps=_integer(
            table["warmup_sweeps"], "run.warmup_sweeps"
        ),
        measurement_sweeps=_integer(
            table["measurement_sweeps"], "run.measurement_sweeps", minimum=1
        ),
        measurement_stride=_integer(
            table["measurement_stride"], "run.measurement_stride", minimum=1
        ),
        checkpoint_interval_sweeps=_integer(
            table["checkpoint_interval_sweeps"],
            "run.checkpoint_interval_sweeps",
            minimum=1,
        ),
        invariant_check_interval_steps=_integer(
            table["invariant_check_interval_steps"],
            "run.invariant_check_interval_steps",
            minimum=1,
        ),
    )


def _parse_output(table: Mapping[str, Any]) -> OutputConfig:
    keys = {"root", "label"}
    _check_keys(table, keys, keys, "output")

    root = _text(table["root"], "output.root")
    label = _text(table["label"], "output.label")
    if not LABEL_PATTERN.fullmatch(label):
        raise ConfigError(
            "output.label: use ASCII letters, digits, dot, underscore, or hyphen; "
            "the first character must be alphanumeric"
        )
    return OutputConfig(root=root, label=label)
