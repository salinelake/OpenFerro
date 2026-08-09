import jax.numpy as jnp
import numpy as np

from openferro.lattice import SimpleCubic3D
from openferro.simulation import SimulationNVE, SimulationNVTLangevin
from openferro.system import System


class _RecordingIntegrator:
    def __init__(self):
        self.keys = []

    def step(self, key, field):
        del field
        self.keys.append(np.asarray(key).copy())


def _sample_velocities(seed):
    system = System(SimpleCubic3D(16, 1, 1))
    first = system.add_field("first")
    second = system.add_field("second")
    simulation = SimulationNVE(system, seed=seed)

    simulation.init_velocity(mode="gaussian", temp=300.0)

    return np.asarray(first.get_velocity()), np.asarray(second.get_velocity())


def _recording_simulation(seed=42, key=None):
    system = System(SimpleCubic3D(1, 1, 1))
    field = system.add_field("field")
    integrator = _RecordingIntegrator()
    field.integrator = integrator
    return SimulationNVTLangevin(system, seed=seed, key=key), integrator


def test_velocity_streams_are_reproducible_and_independent_by_field():
    first_a, second_a = _sample_velocities(seed=17)
    first_b, second_b = _sample_velocities(seed=17)
    first_c, second_c = _sample_velocities(seed=23)

    np.testing.assert_array_equal(first_a, first_b)
    np.testing.assert_array_equal(second_a, second_b)
    assert not np.array_equal(first_a, second_a)
    assert not np.array_equal(first_a, first_c)
    assert not np.array_equal(second_a, second_c)


def test_repeated_runs_continue_the_random_stream():
    simulation, integrator = _recording_simulation(seed=17)
    simulation.run(nsteps=1)
    simulation.run(nsteps=1)

    reference, reference_integrator = _recording_simulation(seed=17)
    reference.run(nsteps=2)

    assert not np.array_equal(integrator.keys[0], integrator.keys[1])
    np.testing.assert_array_equal(integrator.keys, reference_integrator.keys)


def test_saved_random_key_restores_the_next_step_and_seed_can_reset():
    simulation, integrator = _recording_simulation(seed=17)
    simulation.run(nsteps=1)
    saved_key = jnp.array(simulation.get_random_key())
    simulation.run(nsteps=1)

    restarted, restarted_integrator = _recording_simulation(key=saved_key)
    restarted.run(nsteps=1)
    np.testing.assert_array_equal(restarted_integrator.keys[0], integrator.keys[1])

    simulation.run(nsteps=1, seed=17)
    np.testing.assert_array_equal(integrator.keys[2], integrator.keys[0])
