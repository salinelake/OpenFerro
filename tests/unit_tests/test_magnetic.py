import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openferro.engine import magnetic
from openferro.engine.magnetic import (
    Dzyaloshinskii_Moriya_energy,
    external_field_energy,
    get_isotropic_exchange_energy_engine,
)
from openferro.lattice import BodyCenteredCubic3D, SimpleCubic3D
from openferro.units import Constants
from scientific_helpers import (
    assert_eager_jit_parity,
    assert_float32_float64_parity,
    assert_force_matches_finite_difference,
    unique_periodic_bond_sum,
)


SC_FIRST_SHELL_SHIFTS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
BCC_SHELL_SHIFTS = (
    ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (1, 0, 1), (1, 1, 0)),
    (
        (1, 2, 1),
        (2, 1, 1),
        (1, 1, 2),
        (-1, 0, 1),
        (0, 1, -1),
        (1, -1, 0),
    ),
    (
        (1, 2, 2),
        (0, 2, 1),
        (0, 1, 2),
        (-1, 1, 1),
        (2, 1, 2),
        (1, 0, 2),
        (2, 0, 1),
        (1, -1, 1),
        (2, 2, 1),
        (2, 1, 0),
        (1, 2, 0),
        (1, 1, -1),
    ),
)


def test_external_magnetic_field_uses_all_three_components():
    field = jnp.ones((1, 1, 1, 3))
    external_field = jnp.array([1.0, 2.0, 3.0])

    energy = external_field_energy(field, external_field)
    force = -jax.grad(external_field_energy, argnums=0)(field, external_field)

    np.testing.assert_allclose(energy, -6.0)
    np.testing.assert_allclose(
        force, jnp.broadcast_to(external_field, force.shape)
    )


def test_external_magnetic_field_rejects_ambiguous_parameter_shape():
    field = jnp.ones((1, 1, 1, 3))

    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        external_field_energy(field, jnp.array([[1.0, 2.0, 3.0]]))


def test_dzyaloshinskii_moriya_fails_explicitly_and_is_not_exported():
    with pytest.raises(NotImplementedError, match="bond and orientation convention"):
        Dzyaloshinskii_Moriya_energy(
            jnp.ones((1, 1, 1, 3)), jnp.array([1.0])
        )

    assert "Dzyaloshinskii_Moriya_energy" not in magnetic.__all__


@pytest.mark.scientific
def test_simple_cubic_exchange_matches_independent_unique_bond_sum():
    lattice = SimpleCubic3D(5, 4, 3)
    field = jnp.arange(5 * 4 * 3 * 3, dtype=jnp.float64).reshape((5, 4, 3, 3))
    field = (field - 37.0) / 19.0
    coupling = jnp.array([0.73])
    engine = get_isotropic_exchange_energy_engine(lattice.first_shell_roller)

    reference = -float(coupling[0]) * unique_periodic_bond_sum(
        field, SC_FIRST_SHELL_SHIFTS
    )
    np.testing.assert_allclose(engine(field, coupling), reference)
    assert_force_matches_finite_difference(
        lambda value: engine(value, coupling), field
    )
    assert_eager_jit_parity(engine, field, coupling)
    assert_float32_float64_parity(engine, field, coupling)


@pytest.mark.scientific
@pytest.mark.parametrize("shell_index", range(4))
def test_bcc_exchange_shells_match_independent_bond_sums(shell_index):
    lattice = BodyCenteredCubic3D(7, 7, 7)
    rollers = (
        lattice.first_shell_roller,
        lattice.second_shell_roller,
        lattice.third_shell_roller,
        lattice.fourth_shell_roller,
    )[shell_index]
    field = np.zeros((7, 7, 7, 3))
    field[1, 2, 3] = [0.2, -0.7, 1.1]
    field[4, 1, 5] = [-0.3, 0.8, 0.6]
    field += [0.11, -0.05, 0.17]
    coupling = jnp.array([1.25])
    engine = get_isotropic_exchange_energy_engine(rollers)

    reference = -float(coupling[0]) * unique_periodic_bond_sum(
        field, BCC_SHELL_SHIFTS[shell_index]
    )
    np.testing.assert_allclose(engine(jnp.asarray(field), coupling), reference)


@pytest.mark.scientific
def test_exchange_source_conversion_and_legacy_compatibility():
    lattice = BodyCenteredCubic3D(4, 3, 5)
    moment = 2.23
    source_j = 1.33767484769984 * Constants.mRy
    field = jnp.full((4, 3, 5, 3), jnp.array([0.0, 0.0, moment]))
    unique_engine = get_isotropic_exchange_energy_engine(lattice.first_shell_roller)
    legacy_engine = get_isotropic_exchange_energy_engine(
        lattice.first_shell_roller, bond_counting="ordered"
    )

    unique_coupling = jnp.array([2.0 * source_j / moment**2])
    legacy_coupling = jnp.array([source_j / moment**2])
    expected = -lattice.nsites * 8 * source_j
    np.testing.assert_allclose(unique_engine(field, unique_coupling), expected)
    np.testing.assert_allclose(
        legacy_engine(field, legacy_coupling),
        unique_engine(field, unique_coupling),
    )


@pytest.mark.scientific
def test_small_periodic_cells_keep_displacement_multiplicity():
    lattice = SimpleCubic3D(2, 2, 2)
    field = jnp.ones((2, 2, 2, 3))
    engine = get_isotropic_exchange_energy_engine(lattice.second_shell_roller)

    # Opposite physical displacements can alias the same periodic site, but
    # remain distinct bonds in the periodic multigraph.
    expected_bonds = 8 * 6
    np.testing.assert_allclose(engine(field, jnp.array([1.0])), -3 * expected_bonds)


def test_exchange_rejects_unknown_bond_counting():
    lattice = SimpleCubic3D(3, 3, 3)
    with pytest.raises(ValueError, match="unique.*ordered"):
        get_isotropic_exchange_energy_engine(
            lattice.first_shell_roller, bond_counting="ambiguous"
        )
