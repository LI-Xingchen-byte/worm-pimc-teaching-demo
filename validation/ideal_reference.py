"""Finite-box grand-canonical ideal Bose reference from momentum occupations.

Deliberately independent of all wormpimc modules. Energies use hbar²/(2m)=lambda.
"""

from __future__ import annotations

import math

import numpy as np


def ideal_bose_reference(
    *,
    beta: float = 1.0,
    box_length: float = 4.0,
    lambda_kin: float = 0.5,
    chemical_potential: float = -0.7,
    mode_cutoff: int = 16,
    number_cutoff: int = 64,
) -> dict:
    """Return moments and P(N) with explicit mode and number-tail diagnostics.

    mu must be strictly below the zero-energy ground state. The momentum-tail
    bounds use (K+1+j)^2 >= (K+1)^2 + j*(2K+3). P(N) is NOT renormalized after
    truncation; the omitted number probability is reported separately.
    """
    for name, value in (
        ("beta", beta),
        ("box_length", box_length),
        ("lambda_kin", lambda_kin),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(chemical_potential) or chemical_potential >= 0:
        raise ValueError("chemical_potential must be finite and negative")
    for value in (mode_cutoff, number_cutoff):
        if type(value) is not int or value < 1:
            raise ValueError("cutoffs must be positive integers")
    if number_cutoff < 4:
        raise ValueError("number_cutoff must be at least 4 to report P0 through P4")
    a = lambda_kin * (2 * math.pi / box_length) ** 2
    modes = np.arange(-mode_cutoff, mode_cutoff + 1)
    energies = a * modes**2
    q = np.exp(beta * (chemical_potential - energies))
    occupations = q / (-np.expm1(beta * (chemical_potential - energies)))
    mean = float(occupations.sum())
    variance = float(np.sum(occupations * (1 + occupations)))
    probabilities = np.zeros(number_cutoff + 1)
    probabilities[0] = 1.0
    for probability in q:
        geometric = (1 - probability) * probability ** np.arange(number_cutoff + 1)
        probabilities = np.convolve(probabilities, geometric)[: number_cutoff + 1]
    m = mode_cutoff + 1
    q_next = math.exp(beta * (chemical_potential - a * m * m))
    r = math.exp(-beta * a * (2 * m + 1))
    tail_n = 2 * q_next / ((1 - q_next) * (1 - r))
    tail_e = (
        2
        * a
        * q_next
        / (1 - q_next)
        * (m * m / (1 - r) + 2 * m * r / (1 - r) ** 2 + r * (1 + r) / (1 - r) ** 3)
    )
    values = {
        "N": mean,
        "N2": mean**2 + variance,
        "E": float(np.dot(energies, occupations)),
        "exchange_particles": float(np.sum(occupations - q)),
        **{f"P{i}": float(probabilities[i]) for i in range(min(5, len(probabilities)))},
        "Pge5": max(0.0, 1 - float(probabilities[:5].sum())),
    }
    return {
        "values": values,
        "number_probabilities": probabilities.tolist(),
        "number_tail_probability": max(0.0, 1 - float(probabilities.sum())),
        "mode_cutoff": mode_cutoff,
        "number_cutoff": number_cutoff,
        "omitted_mean_N_upper_bound": tail_n,
        "omitted_energy_upper_bound": tail_e,
        "variance_N": variance,
    }
