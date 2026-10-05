"""Independent finite-box grand-canonical ideal gas correlation reference."""

import numpy as np

from .ideal_reference import ideal_bose_reference


def ideal_correlations(
    *,
    spatial_bins=8,
    modes=(1, 2, 3),
    beta=1.0,
    box_length=4.0,
    lambda_kin=0.5,
    chemical_potential=-0.7,
    mode_cutoff=16
):
    reference = ideal_bose_reference(
        beta=beta,
        box_length=box_length,
        lambda_kin=lambda_kin,
        chemical_potential=chemical_potential,
        mode_cutoff=mode_cutoff,
    )
    if type(spatial_bins) is not int or spatial_bins < 1:
        raise ValueError("spatial_bins must be a positive integer")
    modes = tuple(modes)
    if not modes or any(type(mode) is not int or mode <= 0 for mode in modes):
        raise ValueError("nonzero positive integer modes required")
    momentum = np.arange(-mode_cutoff, mode_cutoff + 1)
    energy = lambda_kin * (2 * np.pi * momentum / box_length) ** 2
    q = np.exp(beta * (chemical_potential - energy))
    occupancy = q / (-np.expm1(beta * (chemical_potential - energy)))
    n = occupancy.sum()
    if n <= 0:
        raise ValueError("density underflow; reference cannot be normalized")
    # Wick: g2(r)=1+|sum_j n_j exp(i*k_j*r)|²/<N>².
    # Integrate each Fourier term over the bin exactly, not at its midpoint.
    differences = momentum[:, None] - momentum[None, :]
    coefficients = occupancy[:, None] * occupancy[None, :]
    bins = []
    for index in range(spatial_bins):
        center = (index + 0.5) / spatial_bins
        average_cosine = np.cos(2 * np.pi * differences * center) * np.sinc(
            differences / spatial_bins
        )
        bins.append(float(1 + np.sum(coefficients * average_cosine) / n**2))
    structure = [
        (
            float(1 + np.dot(occupancy[:-mode], occupancy[mode:]) / n)
            if mode < len(occupancy)
            else 1.0
        )
        for mode in modes
    ]
    return {
        "mean_N": float(n),
        "g2": bins,
        "modes": list(modes),
        "structure_factor": structure,
        "g2_at_zero": 2.0,
        "g2_spatial_average": float(1 + np.dot(occupancy, occupancy) / n**2),
        "reference_truncation": reference,
    }
