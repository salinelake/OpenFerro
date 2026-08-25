import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openferro.field import FieldRn, MaskedFieldRn
from openferro.integrator.md import (
    GradientDescentIntegrator,
    LangevinIntegrator,
    LeapFrogIntegrator,
)
from openferro.lattice import SimpleCubic3D
from openferro.parallelism import DeviceMesh
from openferro.simulation import MDMinimize
from openferro.system import System
from openferro.units import Constants


def _mask():
    mask = np.zeros((3, 2, 1), dtype=bool)
    mask[1:, :, 0] = True
    return mask


@pytest.mark.parametrize("dim", [None, 0, -1, 1.5, True])
def test_masked_field_requires_positive_integer_dimension(dim):
    with pytest.raises(ValueError, match="positive integer"):
        MaskedFieldRn(SimpleCubic3D(3, 2, 1), "u", dim, _mask())


def test_masked_field_validates_immutable_boolean_mask():
    lattice = SimpleCubic3D(3, 2, 1)

    with pytest.raises(TypeError, match="Boolean dtype"):
        MaskedFieldRn(lattice, "u", 2, _mask().astype(np.int8))
    with pytest.raises(ValueError, match="shape"):
        MaskedFieldRn(lattice, "u", 2, np.ones((3, 2), dtype=bool))
    with pytest.raises(ValueError, match="at least one"):
        MaskedFieldRn(lattice, "u", 2, np.zeros((3, 2, 1), dtype=bool))

    field = MaskedFieldRn(lattice, "u", 2, jnp.asarray(_mask()))
    assert field.n_active_sites == 4
    assert field.active_dof == 8
    with pytest.raises(AttributeError):
        field.active_mask = jnp.ones_like(field.active_mask)


def test_system_masked_aliases_and_nonmasked_mask_rejection():
    lattice = SimpleCubic3D(3, 2, 1)
    system = System(lattice)
    first = system.add_field(
        "first", "MaskedRn", dim=2, value=3.0, active_mask=_mask()
    )
    second = system.add_field(
        "second", "MaskedFieldRn", dim=1, active_mask=_mask()
    )

    assert isinstance(first, MaskedFieldRn)
    assert isinstance(second, MaskedFieldRn)
    np.testing.assert_array_equal(first.get_values()[~_mask()], 0.0)
    with pytest.raises(ValueError, match="require active_mask"):
        system.add_field("missing", "MaskedRn", dim=1)
    with pytest.raises(ValueError, match="only valid"):
        system.add_field("ordinary", "Rn", dim=1, active_mask=_mask())
    with pytest.raises(ValueError, match="positive integer"):
        system.add_field("bad_dim", "MaskedRn", dim=True, active_mask=_mask())


def test_all_public_mutations_preserve_exact_zero_exterior_and_mass():
    mask = _mask()
    field = MaskedFieldRn(SimpleCubic3D(3, 2, 1), "u", 2, mask)
    masses = np.arange(1, 7, dtype=float).reshape(3, 2, 1)
    field.set_mass(masses)

    values = jnp.arange(12.0).reshape(3, 2, 1, 2) + 1.0
    field.set_values(values)
    np.testing.assert_array_equal(field.get_values()[~mask], 0.0)
    field.set_local_value((0, 0, 0), [99.0, 98.0])
    np.testing.assert_array_equal(field.get_values()[0, 0, 0], 0.0)
    field.set_local_value((2, 1, 0), [7.0, 8.0])
    np.testing.assert_array_equal(field.get_values()[2, 1, 0], [7.0, 8.0])

    field.set_velocity(jnp.ones_like(values) * 4.0)
    np.testing.assert_array_equal(field.get_velocity()[~mask], 0.0)
    field.init_velocity("gaussian", temperature=250.0, seed=7)
    np.testing.assert_array_equal(field.get_velocity()[~mask], 0.0)

    field.set_force(jnp.ones_like(values) * 2.0)
    field.accumulate_force(jnp.ones_like(values) * 3.0)
    np.testing.assert_array_equal(field.get_force()[~mask], 0.0)
    np.testing.assert_allclose(field.get_force()[mask], 5.0)
    np.testing.assert_array_equal(field.get_mass()[..., 0], masses)
    assert bool(jnp.all(field.get_mass() > 0.0))


def test_masked_statistics_remain_padded_but_temperature_uses_active_dof():
    mask = _mask()
    field = MaskedFieldRn(SimpleCubic3D(3, 2, 1), "u", 2, mask)
    values = jnp.arange(12.0).reshape(3, 2, 1, 2)
    masses = jnp.arange(1.0, 7.0).reshape(3, 2, 1)
    velocities = (values + 1.0) / 10.0
    field.set_values(values)
    field.set_mass(masses)
    field.set_velocity(velocities)

    stored = np.asarray(field.get_values())
    np.testing.assert_allclose(field.mean, stored.mean(axis=(0, 1, 2)))
    np.testing.assert_allclose(field.var, stored.var(axis=(0, 1, 2)))
    expected_ke = 0.5 * np.sum(
        np.asarray(field.get_mass()) * np.asarray(field.get_velocity()) ** 2
    )
    np.testing.assert_allclose(field.get_kinetic_energy(), expected_ke)
    np.testing.assert_allclose(
        field.get_temperature(),
        2.0 * expected_ke / (Constants.kb * field.active_dof),
    )


def test_all_true_mask_matches_fieldrn_state_and_langevin_step():
    lattice = SimpleCubic3D(2, 2, 1)
    mask = np.ones((2, 2, 1), dtype=bool)
    ordinary = FieldRn(lattice, "ordinary", 2)
    masked = MaskedFieldRn(lattice, "masked", 2, mask)
    values = jnp.arange(8.0).reshape(2, 2, 1, 2) / 10.0
    velocity = values - 0.2
    force = 0.3 - values
    mass = np.asarray([[[1.0], [2.0]], [[3.0], [4.0]]])
    for field in (ordinary, masked):
        field.set_values(values)
        field.set_mass(mass)
        field.set_velocity(velocity)
        field.set_force(force)

    key = jax.random.PRNGKey(11)
    integrator = LangevinIntegrator(0.01, 275.0, 0.5)
    integrator.step(key, ordinary)
    integrator.step(key, masked)

    np.testing.assert_array_equal(masked.get_values(), ordinary.get_values())
    np.testing.assert_array_equal(masked.get_velocity(), ordinary.get_velocity())
    np.testing.assert_array_equal(masked.get_force(), ordinary.get_force())
    np.testing.assert_allclose(masked.mean, ordinary.mean)
    np.testing.assert_allclose(masked.var, ordinary.var)
    np.testing.assert_allclose(masked.get_kinetic_energy(), ordinary.get_kinetic_energy())
    np.testing.assert_allclose(masked.get_temperature(), ordinary.get_temperature())


@pytest.mark.parametrize("kind", ["gradient", "leapfrog", "langevin"])
def test_builtin_integrators_hold_constraint_for_one_hundred_steps(kind):
    mask = _mask()
    field = MaskedFieldRn(SimpleCubic3D(3, 2, 1), "u", 2, mask)
    field.set_mass(2.0)
    field.set_values(jnp.ones(field.shape))
    field.set_force(jnp.ones(field.shape) * 0.2)

    if kind == "gradient":
        integrator = GradientDescentIntegrator(0.001)
    else:
        field.set_velocity(jnp.ones(field.shape) * 0.1)
        integrator = (
            LeapFrogIntegrator(0.001)
            if kind == "leapfrog"
            else LangevinIntegrator(0.001, 300.0, 0.1)
        )

    for step in range(100):
        if kind == "langevin":
            integrator.step(jax.random.PRNGKey(step), field)
        else:
            integrator.step(field)
        np.testing.assert_array_equal(field.get_values()[~mask], 0.0)
        if kind != "gradient":
            np.testing.assert_array_equal(field.get_velocity()[~mask], 0.0)


def test_fast_integrator_commits_project_without_public_setters(monkeypatch):
    mask = _mask()
    field = MaskedFieldRn(SimpleCubic3D(3, 2, 1), "u", 2, mask)
    field.set_velocity(jnp.zeros(field.shape))

    def unexpected(*args, **kwargs):
        del args, kwargs
        raise AssertionError("public setter was called")

    monkeypatch.setattr(field, "set_values", unexpected)
    monkeypatch.setattr(field, "set_velocity", unexpected)
    field._set_values_from_integrator(jnp.ones(field.shape))
    field._set_velocity_from_integrator(jnp.ones(field.shape))
    np.testing.assert_array_equal(field.get_values()[~mask], 0.0)
    np.testing.assert_array_equal(field.get_velocity()[~mask], 0.0)


def test_raw_interaction_force_can_be_nonzero_outside_but_assembled_force_is_zero():
    mask = _mask()
    system = System(SimpleCubic3D(3, 2, 1))
    field = system.add_field("u", "MaskedRn", dim=1, active_mask=mask)

    def linear_energy(values, parameters):
        return parameters[0] * jnp.sum(values)

    system.add_self_interaction("linear", "u", linear_energy, [2.0])
    raw = system.calc_force_by_ID("linear")
    system.update_force()

    np.testing.assert_allclose(raw, -2.0)
    np.testing.assert_allclose(field.get_force()[mask], -2.0)
    np.testing.assert_array_equal(field.get_force()[~mask], 0.0)


def test_minimizer_convergence_uses_projected_force():
    mask = _mask()
    inactive = jnp.asarray(~mask)[..., None]
    system = System(SimpleCubic3D(3, 2, 1))
    field = system.add_field("u", "MaskedRn", dim=1, mass=1.0, active_mask=mask)

    def exterior_linear_energy(values, parameters):
        del parameters
        return jnp.sum(values * inactive)

    system.add_self_interaction(
        "outside", "u", exterior_linear_energy, enable_jit=False
    )
    field.set_integrator("optimization", dt=0.01)
    minimizer = MDMinimize(system, max_iter=2, tol=1.0e-12)
    minimizer.run(variable_cell=False)

    assert minimizer.converged
    assert minimizer.iterations == 0
    assert minimizer.max_force_by_field["u"] == 0.0


def test_projected_commits_preserve_partition_sharding():
    mask = _mask()
    field = MaskedFieldRn(SimpleCubic3D(3, 2, 1), "u", 2, mask)
    field.set_mass(1.0)
    field.set_velocity(jnp.zeros(field.shape))
    field.set_force(jnp.zeros(field.shape))
    mesh = DeviceMesh(devices=jax.devices()[:1], num_rows=1, num_cols=1)
    field.to_multi_devs(mesh)
    sharding = mesh.partition_sharding()

    field.set_values(jnp.ones(field.shape))
    field.set_velocity(jnp.ones(field.shape))
    field.set_force(jnp.ones(field.shape))
    field._set_values_from_integrator(jnp.ones(field.shape))
    field._set_velocity_from_integrator(jnp.ones(field.shape))
    GradientDescentIntegrator(0.001).step(field)

    for array in (
        field.active_mask,
        field.get_values(),
        field.get_mass(),
        field.get_velocity(),
        field.get_force(),
    ):
        assert array.sharding == sharding
    np.testing.assert_array_equal(field.get_values()[~mask], 0.0)
    np.testing.assert_array_equal(field.get_velocity()[~mask], 0.0)
    np.testing.assert_array_equal(field.get_force()[~mask], 0.0)
