"""
Classes which define the integrator of molecular dynamics simulation. 
"""
# This file is part of OpenFerro.


import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap


class Integrator:
    """
    The base class to define an integrator. An integrator evolves the system in time.
    """
    def __init__(self, dt):
        self.dt = dt
        self.force_engine = None
        self.buffer = None
    def set_force_engine(self, force_engine):
        self.force_engine = force_engine
    def step(self, x, v, force_engine):
        pass

class VerletIntegrator(Integrator):
    """
    A class to define a Verlet integrator. A Verlet integrator evolves the system in time using the Verlet algorithm.
    """
    def __init__(self, dt):
        super().__init__(dt)
    def step(self, x, v, force_engine):
        force = force_engine(x)
        x = x + v * self.dt + 0.5 * force * self.dt**2
        force_new = force_engine(x)
        v = v + 0.5 * (force + force_new) * self.dt
        return x, v

class LangevinIntegrator_Rn(Integrator):
    """
    A class to define a Langevin integrator. A Langevin integrator evolves the system in time using the Langevin equation.
    """
    def __init__(self, dt, temperature, gamma):
        super().__init__(dt)
        self.temperature = temperature
        self.gamma = gamma
    def step(self, x, v, force_engine):
        noise = jnp.sqrt(2 * self.temperature * self.gamma / self.dt) * jnp.random.normal(x.shape)
        force = force_engine(x)
        v = v - force * self.dt - self.gamma * v * self.dt + noise
        x = x + v * self.dt
        return x, v