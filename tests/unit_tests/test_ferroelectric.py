import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openferro.engine.ferroelectric import (
    get_inhomo_strain_dipole_interaction,
    get_short_range_3rdnn_isotropic,
    homo_strain_dipole_interaction,
    self_energy_onsite_isotropic,
    short_range_1stnn_isotropic,
    short_range_2ednn_isotropic,
)
from scientific_helpers import (
    assert_eager_jit_parity,
    assert_float32_float64_parity,
    assert_force_matches_finite_difference,
)


pytestmark = pytest.mark.scientific

FIRST_SHELL = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
SECOND_SHELL = (
    (1, 1, 0),
    (-1, 1, 0),
    (1, 0, 1),
    (-1, 0, 1),
    (0, 1, 1),
    (0, -1, 1),
)
THIRD_SHELL = (
    (1, 1, 1),
    (1, -1, 1),
    (-1, 1, 1),
    (-1, -1, 1),
)


def _nonsymmetric_field(shape=(3, 3, 3, 3)):
    index = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    return np.sin(index * 0.37) + 0.13 * np.cos(index * 0.19)


def _reference_short_range(field, shifts, parameters):
    values = np.asarray(field)
    parameters = np.asarray(parameters)
    total = 0.0

    for site in np.ndindex(values.shape[:3]):
        for shift in shifts:
            neighbor = tuple(
                (site[axis] - shift[axis]) % values.shape[axis]
                for axis in range(3)
            )
            direction = -np.asarray(shift, dtype=np.float64)
            direction /= np.linalg.norm(direction)

            if len(parameters) == 2 and len(shifts) == 3:
                j1, j2 = parameters
                matrix = np.diag(j1 + (j2 - j1) * np.abs(direction))
            elif len(parameters) == 3:
                j3, j4, j5 = parameters
                diagonal = j4 + np.sqrt(2.0) * (j3 - j4) * np.abs(direction)
                matrix = 2.0 * j5 * np.outer(direction, direction)
                np.fill_diagonal(matrix, diagonal)
            else:
                j6, j7 = parameters
                matrix = 3.0 * j7 * np.outer(direction, direction)
                np.fill_diagonal(matrix, j6)

            total += values[site] @ matrix @ values[neighbor]
    return total


def test_onsite_energy_matches_sitewise_polynomial_and_force():
    field = jnp.asarray(_nonsymmetric_field())
    parameters = jnp.array([5.502, 110.4, -163.1])
    squared = np.asarray(field) ** 2
    reference = parameters[0] * squared.sum()
    reference += parameters[1] * np.square(squared.sum(axis=-1)).sum()
    reference += parameters[2] * (
        squared[..., 0] * squared[..., 1]
        + squared[..., 1] * squared[..., 2]
        + squared[..., 2] * squared[..., 0]
    ).sum()

    np.testing.assert_allclose(
        self_energy_onsite_isotropic(field, parameters), reference
    )
    assert_force_matches_finite_difference(
        lambda value: self_energy_onsite_isotropic(value, parameters), field
    )
    assert_eager_jit_parity(self_energy_onsite_isotropic, field, parameters)
    assert_float32_float64_parity(
        self_energy_onsite_isotropic, field * 0.01, parameters
    )


@pytest.mark.parametrize(
    ("engine", "shifts", "parameters"),
    [
        (short_range_1stnn_isotropic, FIRST_SHELL, (-2.648, 3.894)),
        (short_range_2ednn_isotropic, SECOND_SHELL, (0.898, -0.789, 0.562)),
        (get_short_range_3rdnn_isotropic(), THIRD_SHELL, (0.358, 0.179)),
    ],
)
def test_short_range_shells_match_source_matrices_and_forces(
    engine, shifts, parameters
):
    field = jnp.asarray(_nonsymmetric_field())
    parameters = jnp.asarray(parameters)
    reference = _reference_short_range(field, shifts, parameters)

    np.testing.assert_allclose(engine(field, parameters), reference)
    assert_force_matches_finite_difference(
        lambda value: engine(value, parameters), field
    )
    assert_eager_jit_parity(engine, field, parameters)
    assert_float32_float64_parity(engine, field, parameters)


def test_combined_short_range_has_inversion_and_cubic_permutation_symmetry():
    field = jnp.asarray(_nonsymmetric_field())
    p1 = jnp.array([-2.648, 3.894])
    p2 = jnp.array([0.898, -0.789, 0.562])
    p3 = jnp.array([0.358, 0.179])
    engine3 = get_short_range_3rdnn_isotropic()

    def energy(value):
        return (
            short_range_1stnn_isotropic(value, p1)
            + short_range_2ednn_isotropic(value, p2)
            + engine3(value, p3)
        )

    permuted = jnp.transpose(field, (1, 0, 2, 3))[..., jnp.array([1, 0, 2])]
    np.testing.assert_allclose(energy(-field), energy(field))
    np.testing.assert_allclose(energy(permuted), energy(field))


def test_homogeneous_strain_dipole_matches_explicit_matrix_and_forces():
    strain = jnp.array([0.03, -0.02, 0.01, 0.04, -0.05, 0.06])
    field = jnp.asarray(_nonsymmetric_field()) * 0.05
    parameters = jnp.array([-211.0, -19.3, -7.75])
    b1xx, b1yy, b4yz = np.asarray(parameters)
    eta = np.asarray(strain)
    matrix = np.array(
        [
            [b1xx * eta[0] + b1yy * (eta[1] + eta[2]), b4yz * eta[5], b4yz * eta[4]],
            [b4yz * eta[5], b1xx * eta[1] + b1yy * (eta[0] + eta[2]), b4yz * eta[3]],
            [b4yz * eta[4], b4yz * eta[3], b1xx * eta[2] + b1yy * (eta[0] + eta[1])],
        ]
    )
    field_np = np.asarray(field)
    reference = 0.5 * np.sum(field_np * (field_np @ matrix.T))

    np.testing.assert_allclose(
        homo_strain_dipole_interaction(strain, field, parameters), reference
    )
    assert_force_matches_finite_difference(
        lambda value: homo_strain_dipole_interaction(value, field, parameters),
        strain,
    )
    assert_force_matches_finite_difference(
        lambda value: homo_strain_dipole_interaction(strain, value, parameters),
        field,
    )
    assert_eager_jit_parity(
        homo_strain_dipole_interaction, strain, field, parameters
    )


def test_inhomogeneous_strain_dipole_translation_and_mutual_forces():
    local_displacement = jnp.asarray(_nonsymmetric_field()) * 0.01
    dipole = jnp.asarray(_nonsymmetric_field()) * 0.03 + 0.02
    parameters = jnp.array([-211.0, -19.3, -7.75])
    engine = get_inhomo_strain_dipole_interaction(enable_jit=False)

    np.testing.assert_allclose(
        engine(jnp.ones_like(local_displacement), dipole, parameters), 0.0
    )
    assert_force_matches_finite_difference(
        lambda value: engine(value, dipole, parameters), local_displacement
    )
    assert_force_matches_finite_difference(
        lambda value: engine(local_displacement, value, parameters), dipole
    )
    assert_eager_jit_parity(engine, local_displacement, dipole, parameters)
