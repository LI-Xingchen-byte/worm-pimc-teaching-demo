"""Free-particle density matrices and Brownian-bridge proposals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import (
    image_resolved_displacement,
    link_image_from_unwrapped,
    wrap,
)


FloatArray: TypeAlias = NDArray[np.float64]


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _point(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a scalar or one-dimensional point")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def free_log_density(
    displacement: ArrayLike,
    lambda_kin: float,
    imaginary_time: float,
) -> float:
    """Log of the free density matrix for one image-resolved displacement."""

    lam = _positive_finite(lambda_kin, "lambda_kin")
    time = _positive_finite(imaginary_time, "imaginary_time")
    delta = _point(displacement, "displacement")
    ndim = delta.size
    return float(
        -0.5 * ndim * math.log(4.0 * math.pi * lam * time)
        - np.dot(delta, delta) / (4.0 * lam * time)
    )


def image_resolved_log_density(
    start_wrapped: ArrayLike,
    end_wrapped: ArrayLike,
    image: ArrayLike,
    box_length: float,
    lambda_kin: float,
    imaginary_time: float,
) -> float:
    """Log density for a periodic link carrying an explicit image integer."""

    delta = image_resolved_displacement(
        start_wrapped,
        end_wrapped,
        image,
        box_length,
    )
    return free_log_density(delta, lambda_kin, imaginary_time)


@dataclass(frozen=True, slots=True)
class WindingDistribution:
    """Truncated, normalized 1D free-particle winding distribution."""

    values: NDArray[np.int64]
    probabilities: FloatArray
    omitted_probability_bound: float

    @property
    def mean_square(self) -> float:
        return float(np.dot(self.probabilities, self.values.astype(float) ** 2))

    @property
    def fourth_moment(self) -> float:
        return float(np.dot(self.probabilities, self.values.astype(float) ** 4))

    def sample(self, rng: np.random.Generator) -> int:
        """Draw one winding number from the represented distribution."""

        return int(rng.choice(self.values, p=self.probabilities))


@dataclass(frozen=True, slots=True)
class PeriodicImageDistribution:
    """Truncated normalized image distribution for one spatial coordinate."""

    values: NDArray[np.int64]
    probabilities: FloatArray
    log_density: float
    omitted_probability_bound: float

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.values, p=self.probabilities))


@dataclass(frozen=True, slots=True)
class PeriodicBridgeSample:
    """Periodic bridge path and the explicit image on every link."""

    wrapped_path: FloatArray
    link_images: NDArray[np.int64]
    total_image: NDArray[np.int64]
    log_proposal_density: float


@dataclass(frozen=True, slots=True)
class FreeWalkSample:
    """Unconditioned image-resolved free walk including its start point."""

    wrapped_path: FloatArray
    link_images: NDArray[np.int64]
    log_proposal_density: float


def _periodic_image_distribution_1d(
    start: float,
    end: float,
    box_length: float,
    lambda_kin: float,
    imaginary_time: float,
    tail_tolerance: float,
) -> PeriodicImageDistribution:
    length = _positive_finite(box_length, "box_length")
    lam = _positive_finite(lambda_kin, "lambda_kin")
    time = _positive_finite(imaginary_time, "imaginary_time")
    tolerance = _positive_finite(tail_tolerance, "tail_tolerance")
    if tolerance >= 1.0:
        raise ValueError("tail_tolerance must be smaller than one")
    delta_scaled = (float(end) - float(start)) / length
    coefficient = length * length / (4.0 * lam * time)
    half_width = 0
    omitted_probability_bound = math.inf
    included_sum = 0.0
    weights = np.empty(0, dtype=np.float64)
    while half_width < 1_000_000:
        values = np.arange(-half_width, half_width + 1, dtype=np.int64)
        shifted = values.astype(np.float64) + delta_scaled
        weights = np.exp(-coefficient * shifted * shifted)
        included_sum = math.fsum(float(value) for value in weights)
        first = half_width + 1
        right_first = math.exp(-coefficient * (first + delta_scaled) ** 2)
        left_first = math.exp(-coefficient * (-first + delta_scaled) ** 2)
        right_ratio = math.exp(
            -coefficient * (2.0 * (first + delta_scaled) + 1.0)
        )
        left_ratio = math.exp(
            -coefficient * (2.0 * (first - delta_scaled) + 1.0)
        )
        omitted_weight_bound = (
            right_first / (1.0 - right_ratio)
            + left_first / (1.0 - left_ratio)
        )
        omitted_probability_bound = omitted_weight_bound / included_sum
        if omitted_probability_bound <= tolerance:
            break
        half_width += 1
    else:
        raise RuntimeError("periodic image support exceeded one million images")

    probabilities = weights / included_sum
    prefactor_log = -0.5 * math.log(4.0 * math.pi * lam * time)
    values.setflags(write=False)
    probabilities.setflags(write=False)
    return PeriodicImageDistribution(
        values=values,
        probabilities=probabilities,
        log_density=prefactor_log + math.log(included_sum),
        omitted_probability_bound=omitted_probability_bound,
    )


def periodic_log_density(
    start_wrapped: ArrayLike,
    end_wrapped: ArrayLike,
    box_length: float,
    lambda_kin: float,
    imaginary_time: float,
    *,
    tail_tolerance: float = 1.0e-14,
) -> float:
    """Log of the periodic free density summed over all endpoint images."""

    start = _point(start_wrapped, "start_wrapped")
    end = _point(end_wrapped, "end_wrapped")
    if start.shape != end.shape:
        raise ValueError("start and end must have the same dimension")
    per_axis_tolerance = tail_tolerance / start.size
    return math.fsum(
        _periodic_image_distribution_1d(
            float(start[axis]),
            float(end[axis]),
            box_length,
            lambda_kin,
            imaginary_time,
            per_axis_tolerance,
        ).log_density
        for axis in range(start.size)
    )


def sample_periodic_bridge(
    start_wrapped: ArrayLike,
    end_wrapped: ArrayLike,
    n_links: int,
    box_length: float,
    lambda_kin: float,
    tau: float,
    rng: np.random.Generator,
    *,
    tail_tolerance: float = 1.0e-14,
) -> PeriodicBridgeSample:
    """Sample the image mixture and conditional bridge from Section 8.1."""

    if type(n_links) is not int or n_links < 1:
        raise ValueError("n_links must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    start = _point(start_wrapped, "start_wrapped")
    end = _point(end_wrapped, "end_wrapped")
    if start.shape != end.shape:
        raise ValueError("start and end must have the same dimension")
    per_axis_tolerance = tail_tolerance / start.size
    distributions = tuple(
        _periodic_image_distribution_1d(
            float(start[axis]),
            float(end[axis]),
            box_length,
            lambda_kin,
            n_links * tau,
            per_axis_tolerance,
        )
        for axis in range(start.size)
    )
    total_image = np.array(
        [distribution.sample(rng) for distribution in distributions],
        dtype=np.int64,
    )
    unwrapped = sample_brownian_bridge(
        start,
        end + float(box_length) * total_image,
        n_links,
        lambda_kin,
        tau,
        rng,
    )
    wrapped = np.asarray(wrap(unwrapped, box_length), dtype=np.float64)
    link_images = np.empty((n_links, start.size), dtype=np.int64)
    log_links = 0.0
    for index in range(n_links):
        link_images[index] = link_image_from_unwrapped(
            unwrapped[index],
            unwrapped[index + 1],
            box_length,
        )
        log_links += image_resolved_log_density(
            wrapped[index],
            wrapped[index + 1],
            link_images[index],
            box_length,
            lambda_kin,
            tau,
        )
    wrapped.setflags(write=False)
    link_images.setflags(write=False)
    total_image.setflags(write=False)
    log_endpoint = math.fsum(
        distribution.log_density for distribution in distributions
    )
    return PeriodicBridgeSample(
        wrapped_path=wrapped,
        link_images=link_images,
        total_image=total_image,
        log_proposal_density=log_links - log_endpoint,
    )


def sample_free_walk(
    start_wrapped: ArrayLike,
    n_links: int,
    box_length: float,
    lambda_kin: float,
    tau: float,
    rng: np.random.Generator,
) -> FreeWalkSample:
    """Sample an unconditioned free walk with explicit periodic link images."""

    if type(n_links) is not int or n_links < 1:
        raise ValueError("n_links must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    start = _point(start_wrapped, "start_wrapped")
    length = _positive_finite(box_length, "box_length")
    lam = _positive_finite(lambda_kin, "lambda_kin")
    time_step = _positive_finite(tau, "tau")
    unwrapped = np.empty((n_links + 1, start.size), dtype=np.float64)
    unwrapped[0] = start
    standard_deviation = math.sqrt(2.0 * lam * time_step)
    for index in range(1, n_links + 1):
        unwrapped[index] = unwrapped[index - 1] + rng.normal(
            0.0,
            standard_deviation,
            size=start.size,
        )
    wrapped = np.asarray(wrap(unwrapped, length), dtype=np.float64)
    images = np.empty((n_links, start.size), dtype=np.int64)
    log_density = 0.0
    for index in range(n_links):
        images[index] = link_image_from_unwrapped(
            unwrapped[index],
            unwrapped[index + 1],
            length,
        )
        log_density += image_resolved_log_density(
            wrapped[index],
            wrapped[index + 1],
            images[index],
            length,
            lam,
            time_step,
        )
    wrapped.setflags(write=False)
    images.setflags(write=False)
    return FreeWalkSample(wrapped, images, log_density)


def winding_distribution(
    box_length: float,
    lambda_kin: float,
    beta: float,
    *,
    tail_tolerance: float = 1.0e-14,
) -> WindingDistribution:
    """Return ``P(W) proportional to exp[-(LW)^2/(4 lambda beta)]``.

    The returned support is enlarged until a geometric upper bound on the
    omitted two-sided probability is no larger than ``tail_tolerance``.
    """

    length = _positive_finite(box_length, "box_length")
    lam = _positive_finite(lambda_kin, "lambda_kin")
    inverse_temperature = _positive_finite(beta, "beta")
    tolerance = _positive_finite(tail_tolerance, "tail_tolerance")
    if tolerance >= 1.0:
        raise ValueError("tail_tolerance must be smaller than one")

    coefficient = length * length / (4.0 * lam * inverse_temperature)
    half_width = 0
    omitted_bound = math.inf
    while half_width < 1_000_000:
        first_omitted = math.exp(-coefficient * (half_width + 1) ** 2)
        ratio_bound = math.exp(-coefficient * (2 * half_width + 3))
        omitted_bound = 2.0 * first_omitted / (1.0 - ratio_bound)
        if omitted_bound <= tolerance:
            break
        half_width += 1
    else:
        raise RuntimeError("winding support exceeded one million images")

    values = np.arange(-half_width, half_width + 1, dtype=np.int64)
    weights = np.exp(-coefficient * values.astype(np.float64) ** 2)
    probabilities = weights / math.fsum(float(value) for value in weights)
    values.setflags(write=False)
    probabilities.setflags(write=False)
    return WindingDistribution(values, probabilities, omitted_bound)


def sample_brownian_bridge(
    start: ArrayLike,
    end: ArrayLike,
    n_links: int,
    lambda_kin: float,
    tau: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Sample an unwrapped free Brownian bridge, including both endpoints."""

    if type(n_links) is not int or n_links < 1:
        raise ValueError("n_links must be a positive integer")
    lam = _positive_finite(lambda_kin, "lambda_kin")
    time_step = _positive_finite(tau, "tau")
    start_point = _point(start, "start")
    end_point = _point(end, "end")
    if start_point.shape != end_point.shape:
        raise ValueError("start and end must have the same dimension")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    path = np.empty((n_links + 1, start_point.size), dtype=np.float64)
    path[0] = start_point
    path[-1] = end_point
    for index in range(1, n_links):
        remaining_links = n_links - index + 1
        mean = path[index - 1] + (
            end_point - path[index - 1]
        ) / remaining_links
        variance = (
            2.0 * lam * time_step * (remaining_links - 1) / remaining_links
        )
        path[index] = rng.normal(mean, math.sqrt(variance))
    return path


def brownian_bridge_log_density(
    path: ArrayLike,
    lambda_kin: float,
    tau: float,
) -> float:
    """Log proposal density of the intermediate points given both endpoints."""

    lam = _positive_finite(lambda_kin, "lambda_kin")
    time_step = _positive_finite(tau, "tau")
    points = np.asarray(path, dtype=np.float64)
    if points.ndim == 1:
        points = points[:, np.newaxis]
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 1:
        raise ValueError("path must contain at least two finite points")
    if not np.all(np.isfinite(points)):
        raise ValueError("path must contain only finite values")

    log_density = 0.0
    n_links = points.shape[0] - 1
    ndim = points.shape[1]
    endpoint = points[-1]
    for index in range(1, n_links):
        remaining_links = n_links - index + 1
        mean = points[index - 1] + (
            endpoint - points[index - 1]
        ) / remaining_links
        variance = (
            2.0 * lam * time_step * (remaining_links - 1) / remaining_links
        )
        residual = points[index] - mean
        log_density += -0.5 * (
            ndim * math.log(2.0 * math.pi * variance)
            + float(np.dot(residual, residual)) / variance
        )
    return log_density


def brownian_bridge_moments(
    start: ArrayLike,
    end: ArrayLike,
    n_links: int,
    lambda_kin: float,
    tau: float,
) -> tuple[FloatArray, FloatArray]:
    """Return analytic marginal means and per-coordinate variances."""

    if type(n_links) is not int or n_links < 1:
        raise ValueError("n_links must be a positive integer")
    lam = _positive_finite(lambda_kin, "lambda_kin")
    time_step = _positive_finite(tau, "tau")
    start_point = _point(start, "start")
    end_point = _point(end, "end")
    if start_point.shape != end_point.shape:
        raise ValueError("start and end must have the same dimension")

    fractions = np.arange(n_links + 1, dtype=np.float64) / n_links
    means = start_point + fractions[:, np.newaxis] * (end_point - start_point)
    indices = np.arange(n_links + 1, dtype=np.float64)
    variances = 2.0 * lam * time_step * indices * (n_links - indices) / n_links
    return means, variances
