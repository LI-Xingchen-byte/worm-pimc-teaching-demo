"""Public package surface for the Worm PIMC reference implementation."""

from .config import ConfigError, SimulationConfig
from .configuration import (
    Configuration,
    InvariantViolation,
    StalePatchError,
    sample_free_closed_worldline,
)
from .geometry import (
    centered_displacement,
    image_resolved_displacement,
    wrap,
)
from .propagator import (
    FreeWalkSample,
    PeriodicBridgeSample,
    brownian_bridge_log_density,
    brownian_bridge_moments,
    free_log_density,
    image_resolved_log_density,
    periodic_log_density,
    sample_brownian_bridge,
    sample_free_walk,
    sample_periodic_bridge,
    winding_distribution,
)
from .simulation import Simulation, SimulationResult
from .sampler import StepResult
from .types import MoveStatus, Sector
from .version import __version__

__all__ = [
    "ConfigError",
    "Configuration",
    "FreeWalkSample",
    "InvariantViolation",
    "PeriodicBridgeSample",
    "MoveStatus",
    "Sector",
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "StalePatchError",
    "StepResult",
    "brownian_bridge_log_density",
    "brownian_bridge_moments",
    "centered_displacement",
    "free_log_density",
    "image_resolved_displacement",
    "image_resolved_log_density",
    "periodic_log_density",
    "sample_brownian_bridge",
    "sample_free_closed_worldline",
    "sample_free_walk",
    "sample_periodic_bridge",
    "winding_distribution",
    "wrap",
    "__version__",
]
