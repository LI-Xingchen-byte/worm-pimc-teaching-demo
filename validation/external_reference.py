"""Independent finite-M ideal Bose reference in a static Fourier field.

Use a uniform spatial quadrature and a directly summed periodic heat kernel.
This file deliberately imports no wormpimc modules. Grid refinement is needed
separately from imaginary-time refinement; this is not a continuum reference.
"""

from __future__ import annotations

import math

import numpy as np


def primitive_external_reference(
    *,
    beta=1.0,
    box_length=4.0,
    lambda_kin=0.5,
    chemical_potential=-0.7,
    n_slices=8,
    grid_points=64,
    offset=0.5,
    cosine=(-0.5,),
    sine=(),
) -> dict:
    """Return canonical N=1 and grand-canonical ideal density/energy.

    The symmetric one-link transfer matrix is dx*exp(-tau*V/2)*K*exp(-tau*V/2).
    Its eigenvalues t give cycle weights t**M and ideal Bose occupations
    q/(1-q), q=exp(beta*mu)*t**M. Energy differentiates the complete transfer
    matrix at fixed M, rather than substituting continuum eigenenergies.
    """
    for name, value in (
        ("beta", beta),
        ("box_length", box_length),
        ("lambda_kin", lambda_kin),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        type(n_slices) is not int
        or n_slices < 1
        or type(grid_points) is not int
        or grid_points < 8
    ):
        raise ValueError("invalid time/spatial grid")
    if not math.isfinite(chemical_potential) or not math.isfinite(offset):
        raise ValueError("mu and offset must be finite")
    if any(not math.isfinite(x) for x in (*cosine, *sine)):
        raise ValueError("coefficients must be finite")
    if grid_points <= 2 * max(len(cosine), len(sine)):
        raise ValueError("spatial grid cannot resolve the supplied Fourier modes")
    dx = box_length / grid_points
    x = (np.arange(grid_points) + 0.5) * dx
    v = np.full(grid_points, float(offset))
    for n, a in enumerate(cosine, 1):
        v += a * np.cos(2 * np.pi * n * x / box_length)
    for n, b in enumerate(sine, 1):
        v += b * np.sin(2 * np.pi * n * x / box_length)
    tau = beta / n_slices
    displacement = x[:, None] - x[None, :]
    images = (
        math.ceil(math.sqrt(-4 * lambda_kin * tau * math.log(1e-16)) / box_length) + 1
    )
    kernel = np.zeros_like(displacement)
    derivative = np.zeros_like(displacement)
    for image in range(-images, images + 1):
        distance = displacement + image * box_length
        term = np.exp(-(distance**2) / (4 * lambda_kin * tau)) / math.sqrt(
            4 * math.pi * lambda_kin * tau
        )
        kernel += term
        derivative += term * (
            -0.5 / beta + distance**2 * n_slices / (4 * lambda_kin * beta**2)
        )
    endpoint_sum = v[:, None] + v[None, :]
    factor = dx * np.exp(-tau * endpoint_sum / 2)
    transfer = factor * kernel
    d_transfer = factor * (derivative - kernel * endpoint_sum / (2 * n_slices))
    eigenvalues, vectors = np.linalg.eigh(transfer)
    if eigenvalues[0] < -1e-12:
        raise ValueError("quadrature transfer matrix is not positive")
    keep = eigenvalues > 1e-14 * eigenvalues[-1]
    eigenvalues, vectors = eigenvalues[keep], vectors[:, keep]
    log_weights = n_slices * np.log(eigenvalues)
    log_q = beta * chemical_potential + log_weights
    if np.max(log_q) >= 0:
        raise ValueError("grand-canonical reference is not stable")
    occupations = np.exp(log_q) / (-np.expm1(log_q))
    log_derivative = np.einsum("ij,ij->j", vectors, d_transfer @ vectors) / eigenvalues
    density = (vectors**2 @ occupations) / dx
    canonical = np.exp(log_weights - np.max(log_weights))
    canonical /= canonical.sum()
    return {
        "variant": "independent_periodic_quadrature_symmetric_primitive",
        "grid_points": grid_points,
        "n_slices": n_slices,
        "x": x.tolist(),
        "density": density.tolist(),
        "particle_number": float(occupations.sum()),
        "total_energy": float(-n_slices * np.dot(occupations, log_derivative)),
        "potential_energy": float(dx * np.dot(density, v)),
        "canonical_density": ((vectors**2 @ canonical) / dx).tolist(),
        "canonical_energy": float(-n_slices * np.dot(canonical, log_derivative)),
    }
