import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openferro.field import FieldSO3
from openferro.integrator.llg import (
    ConservativeLLSIBIntegrator,
    LLSIBIntegrator,
    LLSIBLangevinIntegrator,
)
from openferro.lattice import SimpleCubic3D
from openferro.simulation import SimulationNVE
from openferro.system import System
from openferro.units import Constants


pytestmark = pytest.mark.scientific


def _spin_field(initial, shape=(1, 1, 1)):
    field = FieldSO3(SimpleCubic3D(*shape), "spin")
    values = np.broadcast_to(np.asarray(initial, dtype=np.float64), shape + (3,)).copy()
    field.set_values(jnp.asarray(values))
    field.set_force(jnp.zeros_like(field.get_values()))
    return field


def _constant_force_updater(field, vector, seen=None):
    vector = jnp.asarray(vector, dtype=field.get_values().dtype)

    def update():
        if seen is not None:
            seen.append(np.asarray(field.get_values()).copy())
        field.set_force(jnp.broadcast_to(vector, field.get_values().shape))

    return update


def test_conservative_sib_constant_field_precession_and_raw_midpoint():
    field = _spin_field([1.0, 0.0, 0.0])
    seen = []
    update = _constant_force_updater(field, [0.0, 0.0, 1.0], seen)
    integrator = ConservativeLLSIBIntegrator(
        dt=0.1, gamma=1.0, max_iter=100, tol=1.0e-13
    )

    integrator.step(field, update)

    angle = 2.0 * np.arctan(0.05)
    np.testing.assert_allclose(
        field.get_values()[0, 0, 0],
        [np.cos(angle), np.sin(angle), 0.0],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert len(seen) == 2
    np.testing.assert_allclose(np.linalg.norm(seen[0], axis=-1), 1.0)
    assert np.linalg.norm(seen[1], axis=-1).item() < 1.0
    assert integrator.last_converged == {"predictor": True, "corrector": True}


def test_conservative_sib_preserves_two_spin_invariants():
    field = _spin_field(
        np.array([[[[1.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0]]]]),
        shape=(2, 1, 1),
    )

    def update():
        values = field.get_values()
        force = jnp.zeros_like(values)
        force = force.at[0, 0, 0].set(values[1, 0, 0])
        force = force.at[1, 0, 0].set(values[0, 0, 0])
        field.set_force(force)

    initial = np.asarray(field.get_values())[:, 0, 0]
    initial_total = initial.sum(axis=0)
    initial_energy = -np.dot(initial[0], initial[1])
    integrator = ConservativeLLSIBIntegrator(
        dt=0.04, gamma=1.0, max_iter=100, tol=1.0e-13
    )
    for _ in range(100):
        integrator.step(field, update)

    final = np.asarray(field.get_values())[:, 0, 0]
    np.testing.assert_allclose(np.linalg.norm(final, axis=-1), 1.0, atol=2.0e-12)
    np.testing.assert_allclose(final.sum(axis=0), initial_total, atol=2.0e-10)
    np.testing.assert_allclose(-np.dot(final[0], final[1]), initial_energy, atol=2.0e-10)


def _state_dependent_field(spin):
    x, y, z = spin
    return np.array([0.2 + 0.4 * y, -0.1 + 0.3 * z, 1.0 + 0.2 * x])


def _reference_llsib_step(initial, dt, alpha, gamma):
    M = np.asarray(initial, dtype=np.float64)
    renormalized_gamma = gamma / (1.0 + alpha**2)

    def effective_field(spin):
        field = _state_dependent_field(spin)
        return field + alpha * np.cross(spin, field)

    predictor = M.copy()
    field0 = effective_field(M)
    for _ in range(1000):
        updated = M - dt * renormalized_gamma * np.cross(
            (M + predictor) / 2.0, field0
        )
        if np.linalg.norm(updated - predictor) < 1.0e-14:
            predictor = updated
            break
        predictor = updated

    midpoint = (M + predictor) / 2.0
    field_midpoint = effective_field(midpoint)
    result = predictor.copy()
    for _ in range(1000):
        updated = M - dt * renormalized_gamma * np.cross(
            (M + result) / 2.0, field_midpoint
        )
        if np.linalg.norm(updated - result) < 1.0e-14:
            result = updated
            break
        result = updated
    return result / np.linalg.norm(result)


def test_damped_sib_uses_midpoint_spin_and_aligns_with_field():
    initial = np.array([1.0, 0.2, -0.1])
    initial /= np.linalg.norm(initial)
    field = _spin_field(initial)
    seen = []

    def update():
        spin = np.asarray(field.get_values()[0, 0, 0])
        seen.append(spin.copy())
        field.set_force(
            jnp.asarray(_state_dependent_field(spin)).reshape((1, 1, 1, 3))
        )

    integrator = LLSIBIntegrator(
        dt=0.07, alpha=0.6, gamma=1.0, max_iter=100, tol=1.0e-13
    )
    expected = _reference_llsib_step(initial, 0.07, 0.6, 1.0)
    integrator.step(field, update)

    np.testing.assert_allclose(
        field.get_values()[0, 0, 0], expected, rtol=3.0e-12, atol=3.0e-12
    )
    assert np.linalg.norm(seen[1]) < 1.0

    constant_field = _spin_field([1.0, 0.0, 0.0])
    constant_update = _constant_force_updater(constant_field, [0.0, 0.0, 1.0])
    before = -float(constant_field.get_values()[0, 0, 0, 2])
    integrator.step(constant_field, constant_update)
    after = -float(constant_field.get_values()[0, 0, 0, 2])
    assert after < before


def test_sib_reports_bounded_nonconvergence(caplog):
    field = _spin_field([1.0, 0.0, 0.0])
    update = _constant_force_updater(field, [0.0, 0.0, 2.0])
    integrator = ConservativeLLSIBIntegrator(
        dt=1.0, gamma=1.0, max_iter=1, tol=1.0e-15
    )

    with caplog.at_level(logging.WARNING):
        integrator.step(field, update)

    assert integrator.last_converged == {"predictor": False, "corrector": False}
    assert integrator.last_iterations == {"predictor": 1, "corrector": 1}
    assert "did not converge" in caplog.text


@pytest.mark.stochastic
def test_stochastic_sib_is_reproducible_and_preserves_norm():
    first = _spin_field([0.3, -0.4, 0.5], shape=(4, 3, 2))
    second = _spin_field([0.3, -0.4, 0.5], shape=(4, 3, 2))
    key = jax.random.PRNGKey(37)
    first_integrator = LLSIBLangevinIntegrator(
        0.01, temp=300.0, alpha=0.5, gamma=1.0, max_iter=50, tol=1.0e-10
    )
    second_integrator = LLSIBLangevinIntegrator(
        0.01, temp=300.0, alpha=0.5, gamma=1.0, max_iter=50, tol=1.0e-10
    )
    first_update = _constant_force_updater(first, [0.1, -0.2, 0.7])
    second_update = _constant_force_updater(second, [0.1, -0.2, 0.7])

    first_integrator.step(key, first, first_update)
    second_integrator.step(key, second, second_update)

    np.testing.assert_array_equal(first.get_values(), second.get_values())
    np.testing.assert_allclose(
        np.linalg.norm(first.get_values(), axis=-1), 1.0, atol=2.0e-12
    )


@pytest.mark.stochastic
def test_stochastic_sib_preserves_single_spin_boltzmann_observable():
    shape = (32, 32, 1)
    beta_field = 1.0
    rng = np.random.default_rng(12)
    uniform = rng.random(np.prod(shape))
    z = np.log(
        np.exp(-beta_field)
        + uniform * (np.exp(beta_field) - np.exp(-beta_field))
    ) / beta_field
    phi = rng.uniform(0.0, 2.0 * np.pi, np.prod(shape))
    radial = np.sqrt(1.0 - z**2)
    values = np.stack([radial * np.cos(phi), radial * np.sin(phi), z], axis=-1)
    field = _spin_field(values.reshape(shape + (3,)), shape=shape)
    update = _constant_force_updater(field, [0.0, 0.0, 1.0])
    integrator = LLSIBLangevinIntegrator(
        dt=0.01,
        temp=1.0 / Constants.kb,
        alpha=1.0,
        gamma=1.0,
        max_iter=40,
        tol=1.0e-9,
    )
    key = jax.random.PRNGKey(88)
    for _ in range(40):
        key, subkey = jax.random.split(key)
        integrator.step(subkey, field, update)

    expected = 1.0 / np.tanh(beta_field) - 1.0 / beta_field
    observed = float(jnp.mean(field.get_values()[..., 2]))
    assert observed == pytest.approx(expected, abs=0.06)
    np.testing.assert_allclose(
        np.linalg.norm(field.get_values(), axis=-1), 1.0, atol=2.0e-12
    )


def test_multiple_so3_fields_fail_before_order_dependent_dynamics():
    system = System(SimpleCubic3D(1, 1, 1))
    first = system.add_field("first", ftype="SO3")
    second = system.add_field("second", ftype="SO3")
    first.set_integrator("adiabatic", dt=0.01)
    second.set_integrator("adiabatic", dt=0.01)

    with pytest.raises(NotImplementedError, match="simultaneous SIB"):
        SimulationNVE(system).run(1)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: ConservativeLLSIBIntegrator(0.0),
        lambda: ConservativeLLSIBIntegrator(0.1, gamma=0.0),
        lambda: ConservativeLLSIBIntegrator(0.1, max_iter=0),
        lambda: ConservativeLLSIBIntegrator(0.1, tol=0.0),
        lambda: LLSIBIntegrator(0.1, alpha=-0.1),
        lambda: LLSIBLangevinIntegrator(0.1, temp=-1.0, alpha=0.1),
    ],
)
def test_llg_integrators_reject_invalid_configuration(constructor):
    with pytest.raises(ValueError):
        constructor()
