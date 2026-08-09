import jax.numpy as jnp
import numpy as np
import pytest

from openferro.field import FieldSO3, GlobalStrain, LocalStrain3D
from openferro.lattice import SimpleCubic3D
from openferro.system import System


def _quadratic_energy(field, parameters):
    return parameters[0] * jnp.sum(field**2)


def _bilinear_energy(field1, field2, parameters):
    return parameters[0] * jnp.sum(field1 * field2)


def test_add_global_strain_is_transactional_for_duplicates():
    system = System(SimpleCubic3D(1, 1, 1))
    field = system.add_global_strain(value=jnp.arange(6.0), mass=2.0)
    pressure = system.get_interaction_by_ID("pV")

    with pytest.raises(ValueError, match="already exists"):
        system.add_global_strain(value=jnp.zeros(6), mass=3.0)

    assert system.get_field_by_ID("gstrain") is field
    assert system.get_interaction_by_ID("pV") is pressure
    np.testing.assert_allclose(field.get_values(), jnp.arange(6.0))
    np.testing.assert_allclose(field.get_mass(), 2.0)


def test_add_global_strain_validates_before_mutating_system():
    system = System(SimpleCubic3D(1, 1, 1))

    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        system.add_global_strain(value=jnp.zeros(5))

    with pytest.raises(ValueError, match="finite"):
        system.add_global_strain(value=[0.0, 0.0, 0.0, 0.0, 0.0, jnp.nan])

    with pytest.raises(ValueError, match="does not exist"):
        system.get_field_by_ID("gstrain")
    with pytest.raises(ValueError, match="does not exist"):
        system.get_interaction_by_ID("pV")

    with pytest.raises(ValueError, match="pressure_volume"):
        system.add_global_strain(pressure_volume="unsupported")
    assert "gstrain" not in {field.ID for field in system.get_all_fields()}


def test_preexisting_pressure_term_blocks_global_strain_without_partial_field():
    system = System(SimpleCubic3D(1, 1, 1))
    pressure_sentinel = object()
    system._self_interaction_dict["pV"] = pressure_sentinel

    with pytest.raises(ValueError, match="Pressure interaction"):
        system.add_global_strain()

    assert "gstrain" not in {field.ID for field in system.get_all_fields()}
    assert system.get_interaction_by_ID("pV") is pressure_sentinel

def test_pressure_registration_failure_rolls_back_global_strain(monkeypatch):
    system = System(SimpleCubic3D(1, 1, 1))

    def fail_after_partial_registration(pressure, volume_mode="determinant"):
        del pressure
        del volume_mode
        system._self_interaction_dict["pV"] = object()
        raise RuntimeError("pressure setup failed")

    monkeypatch.setattr(system, "add_pressure", fail_after_partial_registration)
    with pytest.raises(RuntimeError, match="pressure setup failed"):
        system.add_global_strain()

    assert "gstrain" not in {field.ID for field in system.get_all_fields()}
    assert "pV" not in system.interaction_dict


def test_reference_example_field_calls_remain_compatible():
    lattice = SimpleCubic3D(2, 2, 2)
    ferroelectric = System(lattice)

    dipole = ferroelectric.add_field(
        ID="dipole", ftype="Rn", dim=3, value=0.0, mass=200.0
    )
    local_strain = ferroelectric.add_field(
        ID="lstrain", ftype="LocalStrain3D", value=0.0, mass=200.0
    )
    global_strain = ferroelectric.add_global_strain(
        value=jnp.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0]), mass=1600.0
    )

    magnetic = System(lattice)
    spin = magnetic.add_field(ID="spin", ftype="SO3", value=jnp.array([0, 0, 1]))
    spin.set_magnitude(2.23)

    assert dipole.get_values().shape == (2, 2, 2, 3)
    assert isinstance(local_strain, LocalStrain3D)
    assert isinstance(global_strain, GlobalStrain)
    assert isinstance(spin, FieldSO3)
    np.testing.assert_allclose(jnp.linalg.norm(spin.get_values(), axis=-1), 2.23)


def test_integer_and_boolean_initializers_create_differentiable_state():
    system = System(SimpleCubic3D(1, 1, 1))
    scalar = system.add_field("scalar", value=1)
    vector = system.add_field("vector", ftype="R3", value=[1, 2, 3])
    boolean = system.add_field("boolean", value=True)
    strain = system.add_global_strain(value=[0, 1, 2, 0, 0, 0])

    for interaction_id, field_id in (
        ("scalar_energy", "scalar"),
        ("vector_energy", "vector"),
        ("boolean_energy", "boolean"),
        ("strain_energy", "gstrain"),
    ):
        system.add_self_interaction(
            interaction_id,
            field_id,
            _quadratic_energy,
            [1.0],
            enable_jit=False,
        )

    system.update_force()

    for field in (scalar, vector, boolean, strain):
        assert jnp.issubdtype(field.get_values().dtype, jnp.floating)
        assert bool(jnp.all(jnp.isfinite(field.get_force())))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_update_force_accumulates_terms_and_uses_updated_parameters(dtype):
    system = System(SimpleCubic3D(2, 1, 1))
    field1 = system.add_field(
        "a", value=jnp.asarray(2.0, dtype=dtype), mass=None
    )
    field2 = system.add_field(
        "b", value=jnp.asarray(3.0, dtype=dtype), mass=None
    )
    first = system.add_self_interaction(
        "a_first", "a", _quadratic_energy, [0.5]
    )
    system.add_self_interaction("a_second", "a", _quadratic_energy, [1.5])
    system.add_mutual_interaction("ab", "a", "b", _bilinear_energy, [2.0])

    def expected_forces():
        self_first = system.calc_force_by_ID("a_first")
        self_second = system.calc_force_by_ID("a_second")
        mutual1, mutual2 = system.calc_force_by_ID("ab")
        return self_first + self_second + mutual1, mutual2

    expected1, expected2 = expected_forces()
    system.update_force()
    np.testing.assert_allclose(field1.get_force(), expected1)
    np.testing.assert_allclose(field2.get_force(), expected2)

    first.set_parameters([2.5])
    expected1, expected2 = expected_forces()
    system.update_force()
    np.testing.assert_allclose(field1.get_force(), expected1)
    np.testing.assert_allclose(field2.get_force(), expected2)
