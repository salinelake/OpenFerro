import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openferro.field import FieldRn, GlobalStrain
from openferro.integrator.md import (
    GradientDescentIntegrator,
    LangevinIntegrator,
    LangevinIntegrator_Strain,
    LeapFrogIntegrator,
    LeapFrogIntegrator_Strain,
)
from openferro.lattice import BravaisLattice3D
from openferro.simulation import SimulationNPTLangevin, SimulationNVTLangevin
from openferro.system import System
from openferro.units import Constants


pytestmark = pytest.mark.scientific


def _field(value, velocity=0.0, force=0.0, mass=1.0, shape=(1, 1, 1)):
    field = FieldRn(BravaisLattice3D(*shape), "x", 1)
    field.set_values(jnp.full(shape + (1,), value, dtype=jnp.float64))
    field.set_mass(mass)
    field.set_velocity(jnp.full(shape + (1,), velocity, dtype=jnp.float64))
    field.set_force(jnp.full(shape + (1,), force, dtype=jnp.float64))
    return field


def test_gradient_descent_and_leapfrog_exact_one_step():
    gradient_field = _field(1.0, force=0.8, mass=2.0)
    GradientDescentIntegrator(0.1).step(gradient_field)
    np.testing.assert_allclose(gradient_field.get_values(), 1.04)

    leapfrog_field = _field(1.0, velocity=-0.3, force=0.8, mass=2.0)
    integrator = LeapFrogIntegrator(0.1)
    integrator.step(leapfrog_field)
    np.testing.assert_allclose(leapfrog_field.get_velocity(), -0.26)
    np.testing.assert_allclose(leapfrog_field.get_values(), 0.974)
    assert integrator.velocity_time_offset == -0.5


def test_lfmiddle_exact_one_step_matches_independent_operator_sequence():
    field = _field(0.7, velocity=-0.4, force=0.9, mass=2.5)
    key = jax.random.PRNGKey(19)
    integrator = LangevinIntegrator(dt=0.08, temp=325.0, tau=0.6)

    x0 = np.asarray(field.get_values())
    v0 = np.asarray(field.get_velocity())
    force = np.asarray(field.get_force())
    mass = np.asarray(field.get_mass())
    v_kick = v0 + force / mass * integrator.dt
    x_half = x0 + 0.5 * v_kick * integrator.dt
    gaussian = np.asarray(
        jax.random.normal(key, field.get_values().shape, dtype=jnp.float64)
    )
    thermal = gaussian * np.sqrt(Constants.kb * integrator.temp / mass)
    v_expected = float(integrator.z1) * v_kick + float(integrator.z2) * thermal
    x_expected = x_half + 0.5 * v_expected * integrator.dt

    integrator.step(key, field)
    np.testing.assert_allclose(field.get_values(), x_expected)
    np.testing.assert_allclose(field.get_velocity(), v_expected)
    assert integrator.velocity_time_offset == -0.5


def _harmonic_leapfrog_error(dt, final_time=4.0):
    field = _field(1.0, velocity=0.5 * dt, mass=1.0)
    integrator = LeapFrogIntegrator(dt)
    energies = []
    nsteps = round(final_time / dt)
    for _ in range(nsteps):
        field.set_force(-field.get_values())
        integrator.step(field)
        acceleration = -np.asarray(field.get_values())
        on_step_velocity = np.asarray(field.get_velocity()) + 0.5 * acceleration * dt
        energies.append(
            0.5 * float(np.sum(np.asarray(field.get_values()) ** 2))
            + 0.5 * float(np.sum(on_step_velocity**2))
        )
    exact = np.cos(final_time)
    return abs(float(field.get_values().item()) - exact), np.asarray(energies)


def test_leapfrog_harmonic_convergence_and_bounded_energy_error():
    coarse_error, coarse_energy = _harmonic_leapfrog_error(0.04)
    fine_error, fine_energy = _harmonic_leapfrog_error(0.02)

    assert coarse_error / fine_error > 3.5
    assert np.max(np.abs(fine_energy - 0.5)) < 2.0e-4


def test_lfmiddle_preserves_canonical_kinetic_and_configurational_scales():
    shape = (32, 32, 2)
    temperature = 1.0 / Constants.kb
    field = _field(0.0, mass=1.0, shape=shape)
    key_x, key_v, key_step = jax.random.split(jax.random.PRNGKey(73), 3)
    field.set_values(jax.random.normal(key_x, shape + (1,), dtype=jnp.float64))
    field.set_velocity(jax.random.normal(key_v, shape + (1,), dtype=jnp.float64))
    field.set_force(-field.get_values())

    LangevinIntegrator(0.02, temperature, 0.5).step(key_step, field)
    configurational = float(jnp.mean(field.get_values() ** 2))
    kinetic = float(jnp.mean(field.get_velocity() ** 2))

    assert configurational == pytest.approx(1.0, rel=0.07)
    assert kinetic == pytest.approx(1.0, rel=0.07)


def test_strain_integrators_apply_normal_and_shear_masks():
    lattice = BravaisLattice3D(1, 1, 1)
    field = GlobalStrain(lattice, "gstrain")
    field.set_mass(2.0)
    field.set_velocity(jnp.ones(6))
    field.set_force(jnp.ones(6))
    field.set_values(jnp.zeros(6))

    LeapFrogIntegrator_Strain(0.1, freeze_x=True).step(field)
    np.testing.assert_allclose(field.get_velocity(), [0.0, 1.05, 1.05, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(field.get_values(), [0.0, 0.105, 0.105, 0.0, 0.0, 0.0])

    field.set_velocity(jnp.ones(6))
    field.set_force(jnp.zeros(6))
    LangevinIntegrator_Strain(
        0.1, temp=0.0, tau=1.0, freeze_z=True
    ).step(jax.random.PRNGKey(0), field)
    assert field.get_velocity()[2] == 0.0
    np.testing.assert_allclose(field.get_velocity()[3:], 0.0)


def _harmonic_system(seed=42):
    lattice = BravaisLattice3D(2, 1, 1)
    system = System(lattice)
    field = system.add_field("x", ftype="Rn", dim=1, value=0.0, mass=1.7)
    field.set_values(jnp.array([[[[0.4]]], [[[-0.2]]]], dtype=jnp.float64))
    field.set_velocity(jnp.array([[[[0.1]]], [[[0.3]]]], dtype=jnp.float64))

    def harmonic_energy(values, parameters):
        return 0.5 * parameters[0] * jnp.sum(values**2)

    system.add_self_interaction("harmonic", "x", harmonic_energy, [2.3])
    field.set_integrator("isothermal", dt=0.015, temp=250.0, tau=0.4)
    return system, field, SimulationNVTLangevin(system, seed=seed)


@pytest.mark.stochastic
def test_langevin_manual_state_and_key_restore_matches_uninterrupted_run():
    _, full_field, full = _harmonic_system(seed=91)
    full.run(12)

    _, first_field, first = _harmonic_system(seed=91)
    first.run(5)
    saved_values = first_field.get_values().copy()
    saved_velocity = first_field.get_velocity().copy()
    saved_key = first.get_random_key().copy()

    _, resumed_field, resumed = _harmonic_system(seed=0)
    resumed_field.set_values(saved_values)
    resumed_field.set_velocity(saved_velocity)
    resumed.reset_random_key(key=saved_key)
    resumed.run(7)

    np.testing.assert_array_equal(resumed_field.get_values(), full_field.get_values())
    np.testing.assert_array_equal(resumed_field.get_velocity(), full_field.get_velocity())
    np.testing.assert_array_equal(resumed.get_random_key(), full.get_random_key())


def test_zero_temperature_npt_relaxes_to_determinant_pressure_solution():
    lattice = BravaisLattice3D(1, 1, 1)
    system = System(lattice)
    strain = system.add_global_strain(value=jnp.zeros(6), mass=1.0)
    system.add_homo_elastic_interaction(
        "elastic", "gstrain", B11=10.0, B12=0.0, B44=4.0
    )
    simulation = SimulationNPTLangevin(system, pressure=0.0, seed=5)
    system.get_interaction_by_ID("pV").set_parameters(jnp.array([0.1, 1.0]))
    strain.set_integrator("isothermal", dt=0.005, temp=0.0, tau=0.1)
    simulation.init_velocity("zero")

    simulation.run(1400)

    # For isotropic e, each stationarity equation is
    # B11 * e + pressure * (1 + e)^2 = 0.
    roots = np.roots([0.1, 10.2, 0.1])
    expected = roots[np.argmin(np.abs(roots))]
    np.testing.assert_allclose(strain.get_values()[:3], expected, atol=3.0e-5)
    np.testing.assert_allclose(strain.get_values()[3:], 0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: GradientDescentIntegrator(0.0),
        lambda: LeapFrogIntegrator(-0.1),
        lambda: LangevinIntegrator(0.1, temp=-1.0, tau=1.0),
        lambda: LangevinIntegrator(0.1, temp=1.0, tau=0.0),
    ],
)
def test_md_integrators_reject_invalid_configuration(constructor):
    with pytest.raises(ValueError):
        constructor()


@pytest.mark.parametrize("mass", [0.0, -1.0, np.inf, np.nan])
def test_field_rejects_invalid_inertial_mass(mass):
    field = FieldRn(BravaisLattice3D(1, 1, 1), "x", 1)
    with pytest.raises(ValueError, match="strictly positive"):
        field.set_mass(mass)
