"""Periodic geometry primitives fixed by derivations.md Section 4."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _finite_array(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _scalar_or_array(value: ArrayLike, result: FloatArray) -> float | FloatArray:
    if np.ndim(value) == 0:
        return float(result)
    return result


def wrap(value: ArrayLike, box_length: float) -> float | FloatArray:
    """Wrap coordinates into ``[0, L)`` using the documented floor rule."""

    length = _positive_finite(box_length, "box_length")
    coordinate = _finite_array(value, "value")
    result = coordinate - length * np.floor(coordinate / length)
    return _scalar_or_array(value, result)


def centered_displacement(
    end: ArrayLike,
    start: ArrayLike,
    box_length: float,
) -> float | FloatArray:
    """Return ``end - start`` in ``[-L/2, L/2)``.

    Both half-box ties map to ``-L/2``, matching derivations.md Section 4.1.
    """

    length = _positive_finite(box_length, "box_length")
    end_array = _finite_array(end, "end")
    start_array = _finite_array(start, "start")
    delta = end_array - start_array
    result = delta - length * np.floor(delta / length + 0.5)
    return _scalar_or_array(delta, result)


def image_resolved_displacement(
    start_wrapped: ArrayLike,
    end_wrapped: ArrayLike,
    image: ArrayLike,
    box_length: float,
) -> float | FloatArray:
    """Return the unwrapped link displacement ``r' - r + L n``."""

    length = _positive_finite(box_length, "box_length")
    start = _finite_array(start_wrapped, "start_wrapped")
    end = _finite_array(end_wrapped, "end_wrapped")
    image_array = np.asarray(image)
    if not np.issubdtype(image_array.dtype, np.integer):
        rounded = np.rint(image_array)
        if not np.allclose(image_array, rounded, rtol=0.0, atol=1.0e-12):
            raise ValueError("image must contain integers")
        image_array = rounded
    result = end - start + length * image_array.astype(np.float64)
    return _scalar_or_array(result, result)


def link_image_from_unwrapped(
    start_unwrapped: ArrayLike,
    end_unwrapped: ArrayLike,
    box_length: float,
) -> int | IntArray:
    """Recover the integer link image from two unwrapped endpoints."""

    length = _positive_finite(box_length, "box_length")
    start = _finite_array(start_unwrapped, "start_unwrapped")
    end = _finite_array(end_unwrapped, "end_unwrapped")
    start_wrapped = np.asarray(wrap(start, length))
    end_wrapped = np.asarray(wrap(end, length))
    raw = ((end - start) - (end_wrapped - start_wrapped)) / length
    rounded = np.rint(raw)
    if not np.allclose(raw, rounded, rtol=0.0, atol=1.0e-10):
        raise ValueError("unwrapped endpoints do not define an integer image")
    result = rounded.astype(np.int64)
    if np.ndim(start_unwrapped) == 0 and np.ndim(end_unwrapped) == 0:
        return int(result)
    return result


def winding_from_images(images: ArrayLike) -> int | IntArray:
    """Sum directed link images to obtain a cycle winding vector."""

    image_array = np.asarray(images)
    if not np.issubdtype(image_array.dtype, np.integer):
        rounded = np.rint(image_array)
        if not np.allclose(image_array, rounded, rtol=0.0, atol=1.0e-12):
            raise ValueError("images must contain integers")
        image_array = rounded
    result = np.sum(image_array, axis=0, dtype=np.int64)
    if result.ndim == 0:
        return int(result)
    return result
