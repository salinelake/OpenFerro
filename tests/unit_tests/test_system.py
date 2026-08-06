import jax.numpy as jnp
import numpy as np
import pytest

from openferro.field import FieldSO3, GlobalStrain, LocalStrain3D
from openferro.lattice import SimpleCubic3D
from openferro.system import System


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

    with pytest.raises(ValueError, match="does not exist"):
        system.get_field_by_ID("gstrain")
    with pytest.raises(ValueError, match="does not exist"):
        system.get_interaction_by_ID("pV")


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

    def fail_after_partial_registration(pressure):
        del pressure
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
