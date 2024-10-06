"""
Classes which define the time evolution of physical systems. 
"""
# This file is part of OpenFerro.
from time import time as timer

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from openferro.units import Constants
from openferro.field import GlobalStrain
class MDMinimize:
    def __init__(self, system, max_iter=100, tol=1e-5, dt=0.01):
        self.system = system
        self.max_iter = max_iter
        self.tol = tol
        self.dt = dt
    def _step(self, variable_cell):
        self.system.update_force()
        for field in self.system.get_all_fields():
            if (variable_cell is False) and isinstance(field, GlobalStrain):
                    continue
            x0 = field.get_values()
            f0 = field.get_force()
            x0 += self.dt * f0 / field.get_mass()
            field.set_values(x0)
    def minimize(self, variable_cell=True, pressure=None):
        ## sanity check
        if variable_cell:
            if pressure is None:
                raise ValueError('Please specify pressure for variable-cell structural minimization')
            else:
                # self.system.get_interaction_by_name('pV').set_parameter_by_name(
                #     'p', pressure * Constants.bar)  # bar -> eV/Angstrom^3
                pV_param = self.system.get_interaction_by_name('pV').get_parameters()
                pV_param_new = [pressure * Constants.bar, pV_param[1]]
                self.system.get_interaction_by_name('pV').set_parameters(pV_param_new)
        else:
            if pressure is not None:
                raise ValueError('Specifying pressure is not allowed for fixed-cell structural minimization')
        ## structural relaxation
        for i in range(self.max_iter):
            self._step(variable_cell)
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
            if isinstance(field, GlobalStrain):
                continue
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
    def __init__(self, system, dt=0.01, temperature=0.0, tau=0.1):
        super().__init__(system, dt, temperature)
        self.gamma = 1.0 / tau
        self.z1 = jnp.exp(-dt * self.gamma)
        self.z2 = jnp.sqrt(1 - jnp.exp(-2 * dt * self.gamma))
 
    def _step(self, key, profile=False):
        dt = self.dt
        self.system.update_force(profile=profile)
        all_fields = [ field for field in self.system.get_all_fields() if not isinstance(field, GlobalStrain) ]
        keys = jax.random.split(key, len(all_fields))
        for field, subkey in zip(all_fields, keys):
            if profile:
                t2 = timer()
            mass = field.get_mass()
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
            if profile:
                t3 = timer()
                jax.block_until_ready(field.get_values())
                print('time for updating field %s:' % type(field), t3-t2)
        return

    def step(self, nsteps=1, profile=False):
        key = jax.random.PRNGKey(np.random.randint(0, 1000000))
        keys = jax.random.split(key, nsteps)
        for subkey in keys:
            if profile:
                t0 = timer()
            self._step(subkey, profile)
            if profile:
                t1 = timer()
                print('time for one step:', t1-t0)
        return

class SimulationNPTLangevin(SimulationNVTLangevin):
    """
    A class to define a simulation using the Langevin equation. A Langevin simulation evolves the system in time using the Langevin equation.
    """
    def __init__(self, system, dt=0.01, temperature=0.0, pressure=0.0, tau=0.1, tauP=1.0):
        super().__init__(system, dt, temperature)
        self.pressure = pressure
        # self.system.get_interaction_by_name('pV').set_parameter_by_name(
        #     'p', pressure * Constants.bar)  # bar -> eV/Angstrom^3  
        pV_param = self.system.get_interaction_by_name('pV').get_parameters()
        pV_param_new = [pressure * Constants.bar, pV_param[1]]
        self.system.get_interaction_by_name('pV').set_parameters(pV_param_new)
        self.gamma = 1.0 / tau
        self.gammaP = 1.0 / tauP
        self.z1 = jnp.exp( -dt * self.gamma )
        self.z2 = ( 1 - jnp.exp( -2 * dt * self.gamma ))**0.5
        self.z1P = jnp.exp( -dt * self.gammaP )
        self.z2P = ( 1 - jnp.exp( -2 * dt * self.gammaP ))**0.5
         
    def _step(self, key, profile=False):
        dt = self.dt
        self.system.update_force(profile=profile)
        all_fields = [ field for field in self.system.get_all_fields() ]
        keys = jax.random.split(key, len(all_fields))
        for field, subkey in zip(all_fields, keys):
            if profile:
                t2 = timer()
            mass = field.get_mass()
            a0 = field.get_force() / mass
            v0 = field.get_velocity()
            x0 = field.get_values()
            v0 += dt * a0
            x0 += 0.5 * dt * v0
            gaussian = jax.random.normal(subkey, v0.shape)  # slow
            if isinstance(field, GlobalStrain):
                v0 = self.z1P * v0 + self.z2P * gaussian * (self.kbT/ mass)**0.5
            else:
                v0 = self.z1 * v0 + self.z2 * gaussian * (self.kbT/ mass)**0.5
            x0 += 0.5 * dt * v0
            field.set_values(x0)
            field.set_velocity(v0)
            if profile:
                t3 = timer()
                jax.block_until_ready(field.get_values())
                print('time for updating field %s:' % type(field), t3-t2)
        return