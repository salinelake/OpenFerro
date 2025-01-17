"""
Classes which define the time evolution of physical systems.

Notes
-----
This file is part of OpenFerro.
"""

from time import time as timer
import logging
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from openferro.units import Constants
from openferro.field import GlobalStrain
from openferro.reporter import Thermo_Reporter, Field_Reporter

class Simulation:
    """
    The base class to define a simulation.
    
    A simulation controls the time evolution of a system. This class does not implement any time 
    integration algorithm. Each field object has its own integrator. The class only calls the step 
    method of each integrator and controls output.

    Parameters
    ----------
    system : System
        The physical system to simulate
    """
    def __init__(self, system):
        self.system = system
        self.reporters = []
    
    def clear_reporters(self):
        self.reporters = []

    def add_thermo_reporter(self, file='thermo.log', log_interval=100, global_strain=False, excess_stress=False, volume=False, potential_energy=False, kinetic_energy=False, temperature=False):
        """
        Add a reporter to output global thermodynamic information.

        Parameters
        ----------
        file : str, optional
            Output file name, by default 'thermo.log'
        log_interval : int, optional
            Number of steps between outputs, by default 100
        global_strain : bool, optional
            Whether to output global strain, by default False
        excess_stress : bool, optional
            Whether to output excess stress, by default False
        volume : bool, optional
            Whether to output volume, by default False
        potential_energy : bool, optional
            Whether to output potential energy, by default False
        kinetic_energy : bool, optional
            Whether to output kinetic energy, by default False
        temperature : bool, optional
            Whether to output temperature, by default False
        """
        self.reporters.append(Thermo_Reporter(file, log_interval, global_strain, excess_stress, volume, potential_energy, kinetic_energy, temperature))
    
    def add_field_reporter(self, file_prefix, field_ID, log_interval=100, field_average=True, dump_field=False):
        """
        Add a reporter to dump the values of a given field.

        Parameters
        ----------
        file_prefix : str
            Prefix for output files
        field_ID : str
            ID of field to report
        log_interval : int, optional
            Number of steps between outputs, by default 100
        field_average : bool, optional
            Whether to output field averages, by default True
        dump_field : bool, optional
            Whether to dump full field values, by default False
        """
        self.reporters.append(Field_Reporter(file_prefix, field_ID, log_interval, field_average, dump_field))

    def initialize_reporters(self):
        """Initialize all reporters."""
        for reporter in self.reporters:
            reporter.initialize(self.system)

    def remove_all_reporters(self):
        self.reporters = []

    def reset_reporters(self):
        """Reset the counters of all reporters."""
        for reporter in self.reporters:
            reporter.counter = -1
    
    def step_reporters(self):
        """Step all reporters."""
        for reporter in self.reporters:
            reporter.step(self.system)
    

    def init_velocity(self, mode='zero', temp=None):
        for field in self.all_fields:
            field.init_velocity(mode=mode, temperature=temp)

    def _step(self):
        """
        Update the system by one time step. To be implemented by subclasses.
        """
        pass
    
    def run(self):
        """
        Run the simulation for a given number of steps or until convergence. To be implemented by subclasses.
        """
        pass
        
class MDMinimize(Simulation):
    """
    Class for energy minimization using molecular dynamics.

    Parameters
    ----------
    system : System
        The physical system to minimize
    max_iter : int, optional
        Maximum number of iterations, by default 100
    tol : float, optional
        Force tolerance for convergence, by default 1e-5
    """
    def __init__(self, system, max_iter=100, tol=1e-5 ):
        super().__init__(system)
        self.max_iter = max_iter
        self.tol = tol
        self.all_fields = self.system.get_all_fields()
            
    def _step(self, variable_cell):
        """
        Update the field by one time step.

        Parameters
        ----------
        variable_cell : bool
            Whether to allow cell parameters to vary
        """
        SO3_fields = self.system.get_all_SO3_fields()
        non_SO3_fields = self.system.get_all_non_SO3_fields()
        if len(non_SO3_fields) > 0:
            ## update the force for all fields. 
            ## Force will not be updated again while integrating each non-SO3 field with simple explicit integrator. 
            self.system.update_force()
            for field in non_SO3_fields:
                if (variable_cell is False) and isinstance(field, GlobalStrain):
                    continue
                field.integrator.step(field)
        if len(SO3_fields) > 0:
            ## Force updater will be passed to the integrator of each SO3 fields because implicit methods are used.
            ## So the force will not be updated here. 
            for field in SO3_fields:
                field.integrator.step(field, force_updater=self.system.update_force)
            
    def run(self, variable_cell=True, pressure=None):
        """
        Run the minimization.

        Parameters
        ----------
        variable_cell : bool, optional
            Whether to allow cell parameters to vary, by default True
        pressure : float, optional
            External pressure in bar, required if variable_cell=True

        Raises
        ------
        ValueError
            If pressure not specified for variable cell minimization
            If pressure specified for fixed cell minimization
            If integrator not set for any field
        """
        ## sanity check
        if variable_cell:
            if pressure is None:
                raise ValueError('Please specify pressure for variable-cell structural minimization')
            else:
                # self.system.get_interaction_by_ID('pV').set_parameter_by_ID(
                #     'p', pressure * Constants.bar)  # bar -> eV/Angstrom^3
                pV_param = self.system.get_interaction_by_ID('pV').get_parameters()
                pV_param_new = [pressure * Constants.bar, pV_param[1]]
                self.system.get_interaction_by_ID('pV').set_parameters(pV_param_new)
            for field in self.all_fields:
                if field.integrator is None:
                    raise ValueError('Please set the integrator for the field %s for variable-cell structural minimization' % type(field))
        else:
            if pressure is not None:
                raise ValueError('Specifying pressure is not allowed for fixed-cell structural minimization')
            for field in [field for field in self.all_fields if not isinstance(field, GlobalStrain)]:
                if field.integrator is None:
                    raise ValueError('Please set the integrator for the field %s for fixed-cell structural minimization' % type(field))
        ## structural relaxation
        self.initialize_reporters()
        for i in range(self.max_iter):
            self._step(variable_cell)
            self.step_reporters()
            converged = []
            for field in self.all_fields:
                if jnp.max(jnp.abs(field.get_force())) < self.tol:
                    converged.append(True)
                else:
                    converged.append(False)
            if all(converged):
                break

class SimulationNVE(Simulation):
    """
    Class for NVE (microcanonical) molecular dynamics simulation.

    Parameters
    ----------
    system : System
        The physical system to simulate
    """
    def __init__(self, system):
        super().__init__(system)
        ## get all fields, excluding the global strain field
        self.SO3_fields = self.system.get_all_SO3_fields()
        self.non_SO3_fields = [field for field in self.system.get_all_non_SO3_fields() if not isinstance(field, GlobalStrain)]
        self.all_fields = self.SO3_fields + self.non_SO3_fields
        self.nfields = len(self.all_fields)

    def _step(self, profile=False):
        """
        Update the field by one step.

        Parameters
        ----------
        profile : bool, optional
            Whether to profile timing, by default False
        """
        if len(self.non_SO3_fields) > 0:
            ## update the force for all fields. 
            ## Force will not be updated again while integrating each non-SO3 field with simple explicit integrator. 
            self.system.update_force(profile=profile)
            for field in self.non_SO3_fields:
                if profile:
                    t0 = timer()
                field.integrator.step(field)
                if profile:
                    jax.block_until_ready(field.get_values())
                    logging.info('Time for updating field {}: {:.8f}s'.format(type(field), timer()-t0))
        if len(self.SO3_fields) > 0:
            ## Force updater will be passed to the integrator of each SO3 fields because implicit methods are used.
            ## So the force will not be updated here. 
            for field in self.SO3_fields:
                if profile:
                    t0 = timer()
                field.integrator.step(field, force_updater=self.system.update_force)
                if profile:
                    jax.block_until_ready(field.get_values())
                    logging.info('Time for updating field {}: {:.8f}s'.format(type(field), timer()-t0))
    def run(self, nsteps=1, profile=False):
        """
        Run the simulation.

        Parameters
        ----------
        nsteps : int, optional
            Number of steps to run, by default 1
        profile : bool, optional
            Whether to profile timing, by default False

        Raises
        ------
        ValueError
            If integrator not set for any field
        """
        ## sanity check
        for field in self.all_fields:
            if field.integrator is None:
                raise ValueError('Please set the integrator for the field %s before running the simulation' % type(field))
        ## run the simulation
        self.initialize_reporters()
        for i in range(nsteps):
            self._step(profile=profile)
            self.step_reporters()

class SimulationNVTLangevin(SimulationNVE):
    """
    Class for NVT molecular dynamics simulation using Langevin dynamics.

    Parameters
    ----------
    system : System
        The physical system to simulate
    """
    def __init__(self, system):
        super().__init__(system)

    def _step(self, keys, profile=False):
        """
        Update the field by one step.

        Parameters
        ----------
        keys : array_like
            Random keys for Langevin dynamics
        profile : bool, optional
            Whether to profile timing, by default False
        """
        keys_SO3 = keys[:len(self.SO3_fields)]
        keys_non_SO3 = keys[len(self.SO3_fields):]
        if len(self.non_SO3_fields) > 0:
            self.system.update_force(profile=profile)
            for field, subkey in zip(self.non_SO3_fields, keys_non_SO3):
                if profile:
                    t0 = timer()
                field.integrator.step(subkey, field)
                if profile:
                    jax.block_until_ready(field.get_values())
                    logging.info('Time for updating field {}: {:.8f}s'.format(type(field), timer()-t0))
        if len(self.SO3_fields) > 0:
            for field, subkey in zip(self.SO3_fields, keys_SO3):
                if profile:
                    t0 = timer()
                field.integrator.step(subkey, field, force_updater=self.system.update_force)
                if profile:
                    jax.block_until_ready(field.get_values())
                    logging.info('Time for updating field {}: {:.8f}s'.format(type(field), timer()-t0))
        return

    def run(self, nsteps=1, profile=False):
        """
        Run the simulation.

        Parameters
        ----------
        nsteps : int, optional
            Number of steps to run, by default 1
        profile : bool, optional
            Whether to profile timing, by default False

        Raises
        ------
        ValueError
            If integrator not set for any field
        """
        ## sanity check
        for field in self.all_fields:
            if field.integrator is None:
                raise ValueError('Please set the integrator for the field %s before running the simulation' % type(field))
        ## generate all the needed random keys in advance
        key = jax.random.PRNGKey(np.random.randint(0, 1000000))
        keys = jax.random.split(key, nsteps * self.nfields)
        ## run the simulation
        self.initialize_reporters()
        for id_step in range(nsteps):
            if profile:
                t0 = timer()
            subkeys = keys[id_step * self.nfields:(id_step+1) * self.nfields]
            self._step(subkeys, profile)
            self.step_reporters()
            if profile:
                logging.info('Total time for one step: {:.8f}s'.format(timer()-t0))
        return

class SimulationNPTLangevin(SimulationNVTLangevin):
    """
    Class for NPT molecular dynamics simulation using Langevin dynamics.

    Parameters
    ----------
    system : System
        The physical system to simulate
    pressure : float, optional
        External pressure in bar, by default 0.0
    """
    def __init__(self, system, pressure=0.0):
        super().__init__(system)
        ## set pressure
        self.pressure = pressure
        pV_param = self.system.get_interaction_by_ID('pV').get_parameters()
        pV_param_new = [pressure * Constants.bar, pV_param[1]]
        self.system.get_interaction_by_ID('pV').set_parameters(pV_param_new)
        ## get all fields, including the global strain field
        self.SO3_fields = self.system.get_all_SO3_fields()
        self.non_SO3_fields = self.system.get_all_non_SO3_fields()
        self.all_fields = self.SO3_fields + self.non_SO3_fields
        self.nfields = len(self.all_fields)
