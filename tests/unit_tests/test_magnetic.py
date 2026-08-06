import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openferro.engine import magnetic
from openferro.engine.magnetic import (
    Dzyaloshinskii_Moriya_energy,
    external_field_energy,
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
