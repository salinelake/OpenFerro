import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap

def LangevinIntegrator(x, v, )



class SimulationNVE:
    """
    The class to define a simulation. A simulation describes the time evolution of a system.
    """
    def __init__(self, system, dt=0.01, temperature=0.0):
        self.system = system
        self.dt = dt
        self.temperature = temperature
    def step()

class