from __future__ import annotations

import numpy as np
import pytest

from wormpimc.geometry import (
    centered_displacement,
    image_resolved_displacement,
    link_image_from_unwrapped,
    winding_from_images,
    wrap,
)


def test_wrap_uses_half_open_box() -> None:
    values = np.array([-0.1, 0.0, 10.0, 20.1])

    assert np.asarray(wrap(values, 10.0)) == pytest.approx(
        [9.9, 0.0, 0.0, 0.1]
    )


def test_centered_displacement_assigns_both_half_box_ties_to_negative() -> None:
    assert centered_displacement(5.0, 0.0, 10.0) == -5.0
    assert centered_displacement(-5.0, 0.0, 10.0) == -5.0


def test_explicit_image_is_not_replaced_by_minimum_image() -> None:
    assert image_resolved_displacement(9.0, 1.0, 1, 10.0) == 2.0
    assert image_resolved_displacement(9.0, 1.0, 0, 10.0) == -8.0


def test_link_image_round_trip_and_winding() -> None:
    assert link_image_from_unwrapped(9.5, 10.5, 10.0) == 1
    images = np.array([[1], [0], [-1], [2]], dtype=np.int64)

    assert np.asarray(winding_from_images(images)).tolist() == [2]


@pytest.mark.parametrize("box_length", [0.0, -1.0, float("inf")])
def test_invalid_box_length_is_rejected(box_length: float) -> None:
    with pytest.raises(ValueError, match="box_length"):
        wrap(0.0, box_length)
