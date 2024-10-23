"""
Classes which define the time evolution of physical systems. 
"""
# This file is part of OpenFerro.
from time import time as timer
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from openferro.units import Constants
from openferro.field import GlobalStrain
class MDMinimize:
    def __init__(self, system, max_iter=100, tol=1e-5, dt=0.01):
        self.system = system
        self.max_iter = max_iter
        self.tol = tol
        self.dt = dt
        self.all_fields = self.system.get_all_fields()
        self.all_integrators = [ field.integrator_class['optimization'](dt) for field in self.all_fields ]
        
    def _step(self, variable_cell):
        self.system.update_force()
        for field, integrator in zip(self.all_fields, self.all_integrators):
            if (variable_cell is False) and isinstance(field, GlobalStrain):
                    continue
            integrator.step(field)
            
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
            for field in self.all_fields:
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
        self.all_fields = [ field for field in self.system.get_all_fields() if not isinstance(field, GlobalStrain) ]
        self.all_integrators = [ field.integrator_class['adiabatic'](dt) for field in self.all_fields ]
        
    def init_velocity(self, mode='zero'):
        for field in self.all_fields:
            field.init_velocity(mode=mode, temperature=self.temperature)
    
    def step(self, nsteps=1):
        for i in range(nsteps):
            self.system.update_force()
            for field, integrator in zip(self.all_fields, self.all_integrators):
                integrator.step(field)


class SimulationNVTLangevin(SimulationNVE):
    """
    A class to define a simulation using the Langevin equation. A Langevin simulation evolves the system in time using the Langevin equation.
    """
    def __init__(self, system, dt=0.01, temperature=0.0, tau=0.1):
        super().__init__(system, dt, temperature)
        self.gamma = 1.0 / tau
        self.z1 = jnp.exp(-dt * self.gamma)
        self.z2 = jnp.sqrt(1 - jnp.exp(-2 * dt * self.gamma))
        self.all_fields = [ field for field in self.system.get_all_fields() if not isinstance(field, GlobalStrain) ]
        self.all_integrators = [ field.integrator_class['isothermal'](dt, temperature, tau) for field in self.all_fields ]

    def _step(self, key, profile=False):
        self.system.update_force(profile=profile)
        keys = jax.random.split(key, len(self.all_fields))
        for field, integrator, subkey in zip(self.all_fields, self.all_integrators, keys):
            if profile:
                t0 = timer()
            integrator.step(subkey, field)
            if profile:
                jax.block_until_ready(field.get_values())
                print('time for updating field %s:' % type(field), timer()-t0)
        return

    def step(self, nsteps=1, profile=False):
        key = jax.random.PRNGKey(np.random.randint(0, 1000000))
        keys = jax.random.split(key, nsteps)
        for subkey in keys:
            if profile:
                t0 = timer()
            self._step(subkey, profile)
            if profile:
                print('Total time for one step:', timer()-t0)
        return

class SimulationNPTLangevin(SimulationNVTLangevin):
    """
    A class to define a simulation using the Langevin equation. A Langevin simulation evolves the system in time using the Langevin equation.
    """
    def __init__(self, system, dt=0.01, temperature=0.0, pressure=0.0, tau=0.1, tauP=1.0):
        super().__init__(system, dt, temperature)
        ## set pressure
        self.pressure = pressure
        pV_param = self.system.get_interaction_by_name('pV').get_parameters()
        pV_param_new = [pressure * Constants.bar, pV_param[1]]
        self.system.get_interaction_by_name('pV').set_parameters(pV_param_new)
        ## set integrators
        self.all_fields = self.system.get_all_fields()
        self.all_integrators = []
        for field in self.all_fields:
            if isinstance(field, GlobalStrain):
                self.all_integrators.append(field.integrator_class['isothermal'](dt, temperature, tauP))
            else:
                self.all_integrators.append(field.integrator_class['isothermal'](dt, temperature, tau))