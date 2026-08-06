import logging

import jax.numpy as jnp
import numpy as np

from openferro.lattice import SimpleCubic3D
from openferro.simulation import MDMinimize
from openferro.system import System


def _linear_energy(field, parameters):
    del parameters
    return jnp.sum(field)


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
    assert minimizer.iterations == 1
    assert minimizer.max_force_by_field == {"active": 0.0}
    np.testing.assert_allclose(system.get_field_by_ID("gstrain").get_force(), -1.0)


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
