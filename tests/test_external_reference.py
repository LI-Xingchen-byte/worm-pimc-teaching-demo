"""Independent transfer reference: exact limits and spatial refinement."""

import numpy as np
import pytest

from validation.external_reference import primitive_external_reference
from validation.ideal_reference import ideal_bose_reference


def test_single_particle_target_integrates_to_independent_primitive_density():
    from itertools import product
    import math

    from wormpimc import Configuration, SimulationConfig
    from wormpimc.measure import PrimitiveTargetMeasure
    from wormpimc.propagator import image_resolved_log_density

    # Exhaustively integrate a tiny three-slice grid, including all kinetic
    # image mixtures. Expected density comes from the independent transfer
    # matrix, rather than from the production action or potential helpers.
    config = SimulationConfig.small_system(
        n_slices=3,
        external="fourier",
        external_offset=0.5,
        external_cosine=[-0.5],
        external_sine=[0.1],
    )
    target = PrimitiveTargetMeasure(config)
    size = 8
    reference = primitive_external_reference(
        n_slices=3,
        grid_points=size,
        chemical_potential=-2,
        offset=0.5,
        cosine=(-0.5,),
        sine=(0.1,),
    )
    x = np.array(reference["x"])
    log_kernel = np.zeros((size, size))
    for i, j in product(range(size), repeat=2):
        log_kernel[i, j] = math.log(
            math.fsum(
                math.exp(
                    image_resolved_log_density(
                        [x[i]],
                        [x[j]],
                        [image],
                        4.0,
                        0.5,
                        config.tau,
                    )
                )
                for image in range(-2, 3)
            )
        )
    counts = np.zeros(size)
    partition = 0.0
    for indices in product(range(size), repeat=3):
        closed_indices = (*indices, indices[0])
        state = Configuration.from_closed_worldline(x[list(closed_indices), None], 4.0)
        components = target.full_components(state)
        # Sum over link images without changing the target's potential terms.
        log_weight = components.total - components.chemical - components.kinetic
        log_weight += math.fsum(
            log_kernel[i, j] for i, j in zip(closed_indices[:-1], closed_indices[1:])
        )
        weight = math.exp(log_weight)
        partition += weight
        counts += weight * np.bincount(indices, minlength=size) / 3
    density = counts / (partition * (4 / size))
    assert density == pytest.approx(reference["canonical_density"], abs=1e-11)


def test_constant_field_reference_matches_exact_momentum_sum():
    offset = 0.4
    result = primitive_external_reference(
        offset=offset, cosine=(), chemical_potential=-0.7
    )
    exact = ideal_bose_reference(chemical_potential=-1.1)
    assert result["particle_number"] == pytest.approx(exact["values"]["N"], abs=1e-11)
    assert result["total_energy"] == pytest.approx(
        exact["values"]["E"] + offset * exact["values"]["N"], abs=1e-10
    )
    assert result["potential_energy"] == pytest.approx(
        offset * result["particle_number"]
    )
    assert result["density"] == pytest.approx(
        np.full(64, result["particle_number"] / 4)
    )


def test_nonconstant_field_reference_grid_refinement_normalization_and_phase_translation():
    coarse = primitive_external_reference(
        grid_points=64, cosine=(-0.5, 0.1), sine=(0.2,)
    )
    fine = primitive_external_reference(
        grid_points=128, cosine=(-0.5, 0.1), sine=(0.2,)
    )
    assert coarse["particle_number"] == pytest.approx(
        fine["particle_number"], abs=1e-10
    )
    assert coarse["total_energy"] == pytest.approx(fine["total_energy"], abs=1e-10)
    assert sum(fine["density"]) * 4 / 128 == pytest.approx(fine["particle_number"])
    assert sum(fine["canonical_density"]) * 4 / 128 == pytest.approx(1)
    assert max(fine["density"]) > 1.5 * min(fine["density"])
    first = primitive_external_reference(cosine=(-0.5,))
    translated = primitive_external_reference(cosine=(), sine=(-0.5,))
    assert first["particle_number"] == pytest.approx(translated["particle_number"])
    assert np.roll(first["density"], 16) == pytest.approx(
        translated["density"], abs=1e-10
    )
