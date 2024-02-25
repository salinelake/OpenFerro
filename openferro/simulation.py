import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from openferro.units import Constants

class MDMinimize:
    def __init__(self, system, max_iter=100, tol=1e-5, dt=0.01):
        self.system = system
        self.max_iter = max_iter
        self.tol = tol
        self.dt = dt
    def _step(self):
        self.system.update_force()
        for field in self.system.get_all_fields():
            x0 = field.get_values()
            f0 = field.get_force()
            x0 += self.dt * f0 / field.get_mass()
            field.set_values(x0)
    def minimize(self):
        for i in range(self.max_iter):
            self._step()
            converged = []
            for field in self.system.get_all_fields():
                if jnp.max(jnp.abs(field.get_force())) < self.tol:
                    converged.append(True)
                else:
                    converged.append(False)
            if all(converged):
                break


class SimulationNVE:
    """
    The base class to define a simulation. A simulation describes the time evolution of a system.
    """
    def __init__(self, system, dt=0.01, temperature=0.0):
        self.system = system
        self.dt = dt
        self.temperature = temperature
        self.kbT = Constants.kb * temperature
    
    def init_velocity(self, mode='zero'):
        for field in self.system.get_all_fields():
            field.init_velocity(mode=mode, temperature=self.temperature)
    
    def _step(self):
        dt = self.dt
        for field in self.system.get_all_fields():
            x0 = field.get_values()
            v0 = field.get_velocity()
            a0 = field.get_force() / field.get_mass()
            x0 += v0 * dt + 0.5 * a0 * dt**2
            v0 += 0.5 * a0 * dt
            field.set_values(x0)
            field.set_velocity(v0)
        self.system.update_force()
        for field in self.system.get_all_fields():
            v1 = field.get_velocity()
            a1 = field.get_force() / field.get_mass()
            v1 += 0.5 * a1 * dt
            field.set_velocity(v1)

    def step(self, nsteps=1):
        self.system.update_force()
        for i in range(nsteps):
            self._step()

class SimulationNVTLangevin(SimulationNVE):
    """
    A class to define a simulation using the Langevin equation. A Langevin simulation evolves the system in time using the Langevin equation.
    """
    def __init__(self, system, dt=0.01, temperature=0.0, gamma=1.0):
        super().__init__(system, dt, temperature)
        self.gamma = gamma
        self.z1 = jnp.exp( -dt * gamma )
        self.z2 = ( 1 - jnp.exp( -2 * dt * gamma ))**0.5

    def _step(self, key):
        dt = self.dt
        self.system.update_force()
        for field in self.system.get_all_fields():
            mass = field.get_mass()
            key, subkey = jax.random.split(key)
            ##
            a0 = field.get_force() / mass
            v0 = field.get_velocity()
            x0 = field.get_values()
            v0 += dt * a0
            x0 += 0.5 * dt * v0
            gaussian = jax.random.normal(subkey, v0.shape)
            v0 = self.z1 * v0 + self.z2 * gaussian * (self.kbT/ mass)**0.5
            x0 += 0.5 * dt * v0
            field.set_values(x0)
            field.set_velocity(v0)
    
    def step(self, nsteps=1):
        key = jax.random.PRNGKey(np.random.randint(0, 1000000))
        for i in range(nsteps):
            key, subkey = jax.random.split(key)
            self._step(key)


