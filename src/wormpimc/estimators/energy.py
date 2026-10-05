"""Primitive thermodynamic energy estimators for diagonal configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..configuration import Configuration
from ..geometry import image_resolved_displacement
from ..measure import PrimitiveTargetMeasure
from ..types import Sector


@dataclass(frozen=True, slots=True)
class ThermodynamicEnergy:
    """Per-configuration kinetic, potential, and total energy estimates."""

    kinetic: float
    potential: float
    total: float


def thermodynamic_energy(
    state: Configuration,
    target: PrimitiveTargetMeasure,
) -> ThermodynamicEnergy:
    """Evaluate ``eq-thermodynamic-energy-estimator`` in sector Z."""

    if state.sector is not Sector.Z:
        raise ValueError("thermodynamic energy requires sector Z")
    config = target.config
    if (
        state.n_slices != config.discretization.n_slices
        or state.ndim != config.system.ndim
        or state.box_length != config.system.box_length
    ):
        raise ValueError("state and target use different geometry")
    squared_displacements: list[float] = []
    for bead_id in state.active_ids:
        next_id = int(state.next_of[bead_id])
        displacement = np.asarray(
            image_resolved_displacement(
                state.positions[bead_id],
                state.positions[next_id],
                state.image_to_next[bead_id],
                state.box_length,
            ),
            dtype=np.float64,
        )
        squared_displacements.append(float(np.dot(displacement, displacement)))
    squared_links = math.fsum(squared_displacements)
    tau = config.tau
    kinetic = (
        state.ndim * state.n_particles / (2.0 * tau)
        - squared_links
        / (
            4.0
            * config.system.lambda_kin
            * state.n_slices
            * tau
            * tau
        )
    )
    potential = math.fsum(
        target.slice_potential_energy(state, slice_id)
        for slice_id in range(state.n_slices)
    ) / state.n_slices
    return ThermodynamicEnergy(
        kinetic=kinetic,
        potential=potential,
        total=kinetic + potential,
    )
