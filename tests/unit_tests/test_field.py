import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openferro.field import (
    FieldR3,
    FieldRn,
    FieldSO3,
    FieldScalar,
    GlobalStrain,
    LocalStrain3D,
)
from openferro.lattice import SimpleCubic3D
from openferro.parallelism import DeviceMesh
from openferro.system import System


def test_system_add_field_aliases_and_defaults():
    system = System(SimpleCubic3D(2, 1, 1))

    scalar = system.add_field("scalar")
    vector = system.add_field("vector", ftype="R3", value=[1.0, 2.0, 3.0])
    general = system.add_field("general", ftype="Rn", dim=2, value=4.0)
    local_strain = system.add_field("lstrain", ftype="LocalStrain3D")
    scalar_alias = system.add_field("scalar_alias", ftype="FieldScalar")
    vector_alias = system.add_field("vector_alias", ftype="FieldR3")

    assert isinstance(scalar, FieldScalar)
    assert isinstance(vector, FieldR3)
    assert isinstance(general, FieldRn)
    assert isinstance(local_strain, LocalStrain3D)
    assert isinstance(scalar_alias, FieldScalar)
    assert isinstance(vector_alias, FieldR3)
    assert scalar.get_values().shape == (2, 1, 1, 1)
    assert vector.get_values().shape == (2, 1, 1, 3)
    assert general.get_values().shape == (2, 1, 1, 2)
    np.testing.assert_allclose(
        vector.get_values(),
        jnp.broadcast_to(jnp.array([1.0, 2.0, 3.0]), vector.get_values().shape),
    )
    np.testing.assert_allclose(general.get_values(), 4.0)
    np.testing.assert_allclose(scalar.get_mass(), 1.0)


def test_system_add_field_failure_does_not_register_partial_field():
    system = System(SimpleCubic3D(1, 1, 1))

    with pytest.raises(ValueError, match="positive integer"):
        system.add_field("invalid_dim", ftype="Rn")
    with pytest.raises(ValueError, match="cannot broadcast"):
        system.add_field("invalid_value", ftype="Rn", dim=3, value=[1.0, 2.0])

    with pytest.raises(ValueError, match="does not exist"):
        system.get_field_by_ID("invalid_dim")
    with pytest.raises(ValueError, match="does not exist"):
        system.get_field_by_ID("invalid_value")


def test_so3_constructor_has_a_valid_default_orientation():
    spin = FieldSO3(SimpleCubic3D(2, 1, 1), "spin")

    np.testing.assert_allclose(jnp.linalg.norm(spin.get_values(), axis=-1), 1.0)
    np.testing.assert_allclose(spin.get_values()[..., 2], 1.0)


def test_so3_values_preserve_configured_magnitude_and_default_to_no_mass():
    system = System(SimpleCubic3D(2, 1, 1))
    spin = system.add_field("spin", ftype="SO3", value=[0.0, 0.0, 2.0])

    assert isinstance(spin, FieldSO3)
    np.testing.assert_allclose(jnp.linalg.norm(spin.get_values(), axis=-1), 1.0)
    with pytest.raises(ValueError, match="Mass is not set"):
        spin.get_mass()

    mesh = DeviceMesh(devices=jax.devices()[:1], num_rows=1, num_cols=1)
    spin.to_multi_devs(mesh)
    sharding = spin.get_values().sharding
    magnitudes = jnp.array([[[2.0]], [[3.0]]])
    spin.set_magnitude(magnitudes)
    new_values = jnp.broadcast_to(jnp.array([3.0, 0.0, 0.0]), spin.get_values().shape)
    spin.set_values(new_values)
    spin.set_local_value((0, 0, 0), [0.0, 4.0, 0.0])

    np.testing.assert_allclose(jnp.linalg.norm(spin.get_values(), axis=-1), magnitudes)
    np.testing.assert_allclose(spin.get_values()[0, 0, 0], [0.0, 2.0, 0.0])
    assert spin.get_values().sharding == sharding
    assert spin.get_magnitude().sharding == sharding
    spin.perturb(sigma=0.1, seed=11)
    np.testing.assert_allclose(jnp.linalg.norm(spin.get_values(), axis=-1), magnitudes)
    assert spin.get_values().sharding == sharding


def test_so3_rejects_zero_values_without_mutating_state():
    system = System(SimpleCubic3D(1, 1, 1))
    spin = system.add_field("spin", ftype="SO3")
    original = spin.get_values()

    with pytest.raises(ValueError, match="finite and nonzero"):
        spin.set_values(jnp.zeros_like(original))
    with pytest.raises(ValueError, match="finite and nonzero"):
        spin.set_local_value((0, 0, 0), [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="strictly positive"):
        spin.set_magnitude(0.0)

    np.testing.assert_allclose(spin.get_values(), original)
    np.testing.assert_allclose(spin.get_magnitude(), 1.0)


def test_local_value_update_validates_input_and_preserves_sharding():
    system = System(SimpleCubic3D(2, 1, 1))
    field = system.add_field("u", ftype="Rn", dim=2)
    mesh = DeviceMesh(devices=jax.devices()[:1], num_rows=1, num_cols=1)
    field.to_multi_devs(mesh)
    sharding = field.get_values().sharding

    field.set_local_value((1, 0, 0), [2.0, 3.0])

    np.testing.assert_allclose(field.get_values()[1, 0, 0], [2.0, 3.0])
    assert field.get_values().sharding == sharding
    with pytest.raises(ValueError, match="shape"):
        field.set_local_value((1, 0, 0), [1.0])
    with pytest.raises(IndexError, match="outside shape"):
        field.set_local_value((2, 0, 0), [1.0, 2.0])


def test_real_field_value_setter_rejects_invalid_state_without_mutation():
    field = FieldRn(SimpleCubic3D(2, 1, 1), "u", dim=2)
    original = field.get_values().copy()

    with pytest.raises(ValueError, match="shape"):
        field.set_values(jnp.zeros((2, 1, 1, 1)))
    with pytest.raises(ValueError, match="finite"):
        field.set_values(original.at[0, 0, 0, 0].set(jnp.nan))
    with pytest.raises(ValueError, match="real-valued"):
        field.set_values(original.astype(jnp.complex64) + 1.0j)

    np.testing.assert_array_equal(field.get_values(), original)


def test_global_strain_value_setter_rejects_invalid_state_without_mutation():
    strain = GlobalStrain(SimpleCubic3D(1, 1, 1), "gstrain")
    original = strain.get_values().copy()

    with pytest.raises(ValueError, match="shape"):
        strain.set_values(jnp.zeros(5))
    with pytest.raises(ValueError, match="finite"):
        strain.set_values(original.at[0].set(jnp.inf))
    with pytest.raises(ValueError, match="real-valued"):
        strain.set_values(original.astype(jnp.complex64) + 1.0j)

    np.testing.assert_array_equal(strain.get_values(), original)


def test_real_field_value_setters_promote_integers_and_preserve_sharding():
    system = System(SimpleCubic3D(2, 1, 1))
    field = system.add_field("u", ftype="Rn", dim=2)
    strain = system.add_global_strain()
    mesh = DeviceMesh(devices=jax.devices()[:1], num_rows=1, num_cols=1)
    system.move_fields_to_multi_devs(mesh)
    field_sharding = field.get_values().sharding
    strain_sharding = strain.get_values().sharding

    field.set_values(jnp.ones(field.get_values().shape, dtype=jnp.float32))
    field.set_values(np.full(field.get_values().shape, 2, dtype=np.int32))
    strain.set_values(np.arange(6, dtype=np.int32))

    assert field.get_values().dtype == jnp.float32
    assert jnp.issubdtype(strain.get_values().dtype, jnp.floating)
    assert field.get_values().sharding == field_sharding
    assert strain.get_values().sharding == strain_sharding
    np.testing.assert_allclose(field.get_values(), 2.0)
    np.testing.assert_allclose(strain.get_values(), np.arange(6))
