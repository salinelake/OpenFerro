import logging

import jax.numpy as jnp
import numpy as np

from openferro.lattice import SimpleCubic3D
from openferro.simulation import MDMinimize
from openferro.system import System


def _linear_energy(field, parameters):
    del parameters
    return jnp.sum(field)


def _quartic_energy(field, parameters):
    del parameters
    return jnp.sum(field**4)


def test_fixed_cell_minimization_ignores_frozen_global_strain_force():
    system = System(SimpleCubic3D(1, 1, 1))
    active = system.add_field("active")
    system.add_global_strain()
    system.add_self_interaction(
        "frozen_strain_force",
        "gstrain",
        _linear_energy,
        enable_jit=False,
    )
    active.set_integrator("optimization", dt=0.01)
    minimizer = MDMinimize(system, max_iter=3, tol=1e-6)

    minimizer.run(variable_cell=False)

    assert minimizer.converged
    assert minimizer.iterations == 0
    assert minimizer.max_force_by_field == {"active": 0.0}
    np.testing.assert_allclose(system.get_field_by_ID("gstrain").get_force(), -1.0)


def test_minimization_does_not_move_an_initially_converged_state():
    system = System(SimpleCubic3D(1, 1, 1))
    field = system.add_field("active", value=0.1)
    system.add_self_interaction(
        "quartic", "active", _quartic_energy, enable_jit=False
    )
    field.set_integrator("optimization", dt=1000.0)
    minimizer = MDMinimize(system, max_iter=1, tol=0.005)

    minimizer.run(variable_cell=False)

    assert minimizer.converged
    assert minimizer.iterations == 0
    np.testing.assert_allclose(field.get_values(), 0.1)
    np.testing.assert_allclose(field.get_force(), -0.004)
    np.testing.assert_allclose(minimizer.max_force_by_field["active"], 0.004)


def test_minimization_checks_force_at_the_updated_state():
    system = System(SimpleCubic3D(1, 1, 1))
    field = system.add_field("active", value=0.1)
    system.add_self_interaction(
        "quartic", "active", _quartic_energy, enable_jit=False
    )
    field.set_integrator("optimization", dt=1000.0)
    minimizer = MDMinimize(system, max_iter=1, tol=0.003)

    minimizer.run(variable_cell=False)

    final_value = float(field.get_values().item())
    expected_force = -4.0 * final_value**3
    assert not minimizer.converged
    assert minimizer.iterations == 1
    np.testing.assert_allclose(final_value, -3.9)
    np.testing.assert_allclose(field.get_force(), expected_force)
    np.testing.assert_allclose(
        minimizer.max_force_by_field["active"], abs(expected_force)
    )


def test_minimization_reports_nonconvergence(caplog):
    system = System(SimpleCubic3D(1, 1, 1))
    active = system.add_field("active")
    system.add_self_interaction(
        "constant_force",
        "active",
        _linear_energy,
        enable_jit=False,
    )
    active.set_integrator("optimization", dt=0.01)
    minimizer = MDMinimize(system, max_iter=2, tol=1e-6)

    with caplog.at_level(logging.WARNING):
        minimizer.run(variable_cell=False)

    assert not minimizer.converged
    assert minimizer.iterations == 2
    assert minimizer.max_force_by_field == {"active": 1.0}
    assert "did not converge after 2 iterations" in caplog.text
