"""Potential-energy models independent of worldline topology and moves."""

from __future__ import annotations

import math
from itertools import zip_longest
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from .config import PotentialConfig, SimulationConfig
from .geometry import centered_displacement


class PotentialModel(Protocol):
    """Interface consumed by the primitive target measure."""

    name: str

    def pair_energy(self, displacement: ArrayLike) -> float:
        """Return the periodic energy of one unordered pair."""

    def total_energy(self, positions: ArrayLike) -> float:
        """Return the potential energy of one time slice."""


class ZeroPotential:
    """No external or pair potential."""

    name = "none"

    def pair_energy(self, displacement: ArrayLike) -> float:
        return 0.0

    def total_energy(self, positions: ArrayLike) -> float:
        points = np.asarray(positions, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError("positions must have shape (n_particles, ndim)")
        if not np.all(np.isfinite(points)):
            raise ValueError("positions must contain only finite values")
        return 0.0


class PeriodizedGaussianPotential:
    """One-dimensional smooth repulsion from derivations.md eq-periodized-gaussian."""

    name = "periodized_gaussian"

    def __init__(
        self,
        *,
        epsilon: float,
        sigma: float,
        box_length: float,
        image_tolerance: float,
    ) -> None:
        self.epsilon = _positive_finite(epsilon, "epsilon")
        self.sigma = _positive_finite(sigma, "sigma")
        self.box_length = _positive_finite(box_length, "box_length")
        self.image_tolerance = _positive_finite(
            image_tolerance, "image_tolerance"
        )
        if self.image_tolerance >= 1.0:
            raise ValueError("image_tolerance must be smaller than one")

    def _dimensionless_sum(self, displacement: float) -> tuple[float, int, float]:
        centered = float(
            centered_displacement(displacement, 0.0, self.box_length)
        )
        denominator = 2.0 * self.sigma * self.sigma
        total = math.exp(-(centered * centered) / denominator)
        half_width = 0
        omitted_bound = math.inf
        while half_width < 1_000_000:
            first_index = half_width + 1
            positive_distance = centered + first_index * self.box_length
            negative_distance = centered - first_index * self.box_length
            positive = math.exp(
                -(positive_distance * positive_distance) / denominator
            )
            negative = math.exp(
                -(negative_distance * negative_distance) / denominator
            )
            next_positive = positive_distance + self.box_length
            next_negative = negative_distance - self.box_length
            positive_ratio = math.exp(
                -(
                    next_positive * next_positive
                    - positive_distance * positive_distance
                )
                / denominator
            )
            negative_ratio = math.exp(
                -(
                    next_negative * next_negative
                    - negative_distance * negative_distance
                )
                / denominator
            )
            positive_tail = (
                positive / (1.0 - positive_ratio)
                if positive_ratio < 1.0
                else math.inf
            )
            negative_tail = (
                negative / (1.0 - negative_ratio)
                if negative_ratio < 1.0
                else math.inf
            )
            omitted_bound = positive_tail + negative_tail
            if omitted_bound <= self.image_tolerance * total:
                return total, half_width, omitted_bound
            total += positive + negative
            half_width += 1
        raise RuntimeError("periodized Gaussian exceeded one million images")

    def pair_energy(self, displacement: ArrayLike) -> float:
        delta = np.asarray(displacement, dtype=np.float64)
        if delta.ndim == 0:
            scalar = float(delta)
        elif delta.shape == (1,):
            scalar = float(delta[0])
        else:
            raise ValueError(
                "periodized_gaussian currently supports one-dimensional points"
            )
        if not math.isfinite(scalar):
            raise ValueError("displacement must be finite")
        image_sum, _, _ = self._dimensionless_sum(scalar)
        return self.epsilon * image_sum

    def image_diagnostics(self, displacement: float) -> tuple[int, float]:
        """Return included half-width and relative omitted-tail bound."""

        total, half_width, omitted = self._dimensionless_sum(displacement)
        return half_width, omitted / total

    def total_energy(self, positions: ArrayLike) -> float:
        points = np.asarray(positions, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 1:
            raise ValueError(
                "periodized_gaussian expects shape (n_particles, 1)"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("positions must contain only finite values")
        total = 0.0
        for first in range(points.shape[0]):
            for second in range(first + 1, points.shape[0]):
                total += self.pair_energy(points[first] - points[second])
        return total


class FourierExternalPotential:
    """Static periodic one-body field; coefficient index zero means harmonic 1."""

    name = "fourier"

    def __init__(
        self,
        *,
        box_length: float,
        offset: float = 0.0,
        cosine: ArrayLike = (),
        sine: ArrayLike = (),
    ) -> None:
        self.box_length = _positive_finite(box_length, "box_length")
        self.offset = float(offset)
        if not math.isfinite(self.offset):
            raise ValueError("offset must be finite")
        arrays = []
        for values in (cosine, sine):
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise ValueError("Fourier coefficients must be finite one-dimensional arrays")
            arrays.append(array.copy())
        self.cosine, self.sine = arrays
        self.cosine.setflags(write=False)
        self.sine.setflags(write=False)
        try:
            bound = abs(self.offset) + math.fsum(
                math.hypot(a, b) for a, b in zip_longest(self.cosine, self.sine, fillvalue=0.0)
            )
        except OverflowError as exc:
            raise ValueError("Fourier energy bound must be finite") from exc
        if not math.isfinite(bound):
            raise ValueError("Fourier energy bound must be finite")

    def energy(self, x: ArrayLike) -> float | np.ndarray:
        """Evaluate at scalar/array x, with coordinates wrapped onto [0,L)."""
        points = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(points)):
            raise ValueError("positions must contain only finite values")
        phase = ((points % self.box_length) / self.box_length) * (2.0 * np.pi)
        value = np.full(points.shape, self.offset, dtype=np.float64)
        for n, coefficient in enumerate(self.cosine, start=1):
            value += coefficient * np.cos(n * phase)
        for n, coefficient in enumerate(self.sine, start=1):
            value += coefficient * np.sin(n * phase)
        return float(value) if value.ndim == 0 else value

    def total_energy(self, positions: ArrayLike) -> float:
        points = np.asarray(positions, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 1:
            raise ValueError("external field expects shape (n_particles, 1)")
        return math.fsum(self.energy(points[:, 0]))


class ZeroExternalPotential(FourierExternalPotential):
    """The zero one-body field, with the same plotting/evaluation interface."""

    name = "none"

    def __init__(self, *, box_length: float) -> None:
        super().__init__(box_length=box_length)


class CompositePotential:
    """Add a one-body field to a pair model without exposing topology."""

    def __init__(self, pair: PotentialModel, external: FourierExternalPotential) -> None:
        self.pair = pair
        self.external = external
        self.name = f"{pair.name}+{external.name}"

    def pair_energy(self, displacement: ArrayLike) -> float:
        return self.pair.pair_energy(displacement)

    def total_energy(self, positions: ArrayLike) -> float:
        return math.fsum((self.pair.total_energy(positions), self.external.total_energy(positions)))


def external_from_config(config: SimulationConfig) -> FourierExternalPotential:
    """Build the configured one-body field, independently of the pair model."""
    potential = config.potential
    if potential.external == "none":
        return ZeroExternalPotential(box_length=config.system.box_length)
    if potential.external == "fourier":
        return FourierExternalPotential(
            box_length=config.system.box_length,
            offset=potential.external_offset,
            cosine=potential.external_cosine,
            sine=potential.external_sine,
        )
    raise ValueError(f"unsupported external potential: {potential.external}")


def potential_from_config(config: SimulationConfig) -> PotentialModel:
    """Construct the configured potential through an explicit factory."""

    potential: PotentialConfig = config.potential
    if potential.pair == "none":
        pair: PotentialModel = ZeroPotential()
    elif potential.pair == "periodized_gaussian":
        assert potential.epsilon is not None
        assert potential.sigma is not None
        assert potential.image_tolerance is not None
        pair = PeriodizedGaussianPotential(
            epsilon=potential.epsilon,
            sigma=potential.sigma,
            box_length=config.system.box_length,
            image_tolerance=potential.image_tolerance,
        )
    else:
        raise ValueError(f"unsupported pair potential: {potential.pair}")
    if potential.external == "none":
        return pair
    return CompositePotential(pair, external_from_config(config))


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result
