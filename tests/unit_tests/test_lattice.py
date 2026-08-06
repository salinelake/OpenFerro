import json

import jax.numpy as jnp
import numpy as np
import pytest

from openferro.engine.ewald import (
    estimate_dipole_dipole_ewald_memory,
    get_dipole_dipole_ewald,
)
from openferro.lattice import (
    BodyCenteredCubic3D,
    BravaisLattice3D,
    Hexagonal3D,
)


@pytest.mark.parametrize(
    "lattice",
    [
        BravaisLattice3D(2, 3, 4),
        BravaisLattice3D(
            2,
            2,
            2,
            a1=jnp.array([0.0, 1.0, 0.0]),
            a2=jnp.array([-1.0, 0.0, 0.0]),
            a3=jnp.array([0.0, 0.0, 1.0]),
        ),
        BravaisLattice3D(
            2,
            2,
            2,
            a1=jnp.array([1.0, 0.0, 0.0]),
            a2=jnp.array([0.0, 0.0, 1.0]),
            a3=jnp.array([0.0, 1.0, 0.0]),
        ),
        BravaisLattice3D(
            3,
            2,
            5,
            a1=jnp.array([1.2, 0.0, 0.0]),
            a2=jnp.array([0.2, 1.1, 0.0]),
            a3=jnp.array([0.1, 0.3, 0.9]),
        ),
        BodyCenteredCubic3D(2, 2, 2),
        Hexagonal3D(2, 3, 1),
    ],
)
def test_reciprocal_vectors_are_dual_to_primitive_vectors(lattice):
    duality = lattice.latt_vec @ lattice.reciprocal_latt_vec.T

    np.testing.assert_allclose(duality, 2 * np.pi * np.eye(3), rtol=1e-6, atol=1e-6)


def test_reciprocal_vectors_do_not_depend_on_supercell_size():
    small = BravaisLattice3D(1, 1, 1)
    large = BravaisLattice3D(7, 5, 3)

    np.testing.assert_allclose(small.reciprocal_latt_vec, large.reciprocal_latt_vec)


@pytest.mark.parametrize(
    "vectors",
    [
        (
            jnp.array([0.0, 1.0, 0.0]),
            jnp.array([-1.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
        ),
        (
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.2, 1.0, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
        ),
    ],
)
def test_ewald_rejects_rotated_and_skew_cells(vectors):
    lattice = BravaisLattice3D(1, 1, 1, *vectors)

    with pytest.raises(NotImplementedError, match="rotated and skew"):
        get_dipole_dipole_ewald(lattice)


def test_ewald_memory_estimate_is_json_serializable():
    estimate = estimate_dipole_dipole_ewald_memory(
        BravaisLattice3D(3, 2, 2), dtype=jnp.float32
    )

    assert json.loads(json.dumps(estimate))["shape"] == [3, 2, 2]
    assert type(estimate["nsites"]) is int
    assert all(type(value) is int for value in estimate["arrays"].values())
