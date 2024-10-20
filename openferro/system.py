"""
Classes which define the physical system. 
"""
# This file is part of OpenFerro.
from time import time as timer

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from openferro.field import *
from openferro.interaction import *
from openferro.units import Constants
from openferro.engine import pV_energy
from openferro.parallelism import DeviceMesh
import warnings

class System:
    """
    A class to define a physical system. A system is a lattice with fields and a Hamiltonian.
    """
    def __init__(self, lattice, pbc=True ):
        self.lattice = lattice
        self.pbc = pbc
        self._fields_dict = {}
        self._self_interaction_dict = {}
        self._mutual_interaction_dict = {}
        self._triple_interaction_dict = {}
    def __repr__(self):
        return f"System with lattice {self.lattice} and fields {self._fields_dict.keys()}"
    
    """
    Methods for fields
    """

    def get_field_by_name(self, name):
        if name in self._fields_dict:
            return self._fields_dict[name]
        else:
            raise ValueError('Field with the name {} does not exist.'.format(name))

    def get_all_fields(self):
        return [self._fields_dict[name] for name in self._fields_dict.keys()]

    def move_fields_to_multi_devs(self, mesh: DeviceMesh):
        """
        Move all fields to given devices.
        """
        for name in self._fields_dict.keys():
            self._fields_dict[name].to_multi_devs(mesh)
        
    def add_field(self, name, ftype='scalar', dim=None, unit=None, value=None, mass=1.0):
        """
        Add a field to the system.
        TODO: clean up
        """
        if name in self._fields_dict:
            raise ValueError("Field with this name already exists. Pick another name")
        if name == 'gstrain':
            assert ftype == 'global_strain', "The name 'gstrain' is only compatible with global strain field"
        if ftype == 'global_strain':
            if name != 'gstrain':
                warnings.warn("The name of global strain has to be 'gstrain'. The name you entered is discarded. ")
                name = 'gstrain'
            if value is not None:
                assert len(value) == 6, "Global strain must be a 6D vector"
                init = jnp.array(value, dtype=jnp.float32)
            else:
                init = jnp.zeros(6, dtype=jnp.float32)
            self._fields_dict[name] = GlobalStrain(self.lattice, name)
            self._fields_dict[name].set_values(jnp.zeros((1,1,1, 6)) + init)
            self.add_pressure(0)
        elif ftype == 'scalar':
            init = value if value is not None else 0.0
            self._fields_dict[name] = FieldScalar(self.lattice, name, unit)
            self._fields_dict[name].set_values(jnp.zeros(self.lattice.size) + init)
        elif ftype == 'Rn':
            init = jnp.array(value, dtype=jnp.float32) if value is not None else jnp.zeros(dim, dtype=jnp.float32)
            self._fields_dict[name] = FieldRn(self.lattice, name, dim, unit)
            self._fields_dict[name].set_values(jnp.zeros((self.lattice.size[0], self.lattice.size[1], self.lattice.size[2], dim)) + init)
        elif ftype == 'SO3':
            init = jnp.array(value, dtype=jnp.float32) if value is not None else jnp.zeros(2, dtype=jnp.float32)
            self._fields_dict[name] = FieldSO3(self.lattice, name)
            self._fields_dict[name].set_values(jnp.zeros((self.lattice.size[0], self.lattice.size[1], self.lattice.size[2], 2)) + init)
        elif ftype == 'local_strain':
            init = jnp.array(value, dtype=jnp.float32) if value is not None else jnp.zeros(3, dtype=jnp.float32)
            self._fields_dict[name] = LocalStrain(self.lattice, name)
            self._fields_dict[name].set_values(jnp.zeros((self.lattice.size[0], self.lattice.size[1], self.lattice.size[2], 3)) + init)
        else:
            raise ValueError("Unknown field type. ")
        if mass is not None:
            self._fields_dict[name].set_mass(mass)
        return self._fields_dict[name]


    """
    Methods for interactions
    """
    
    @property
    def interaction_dict(self):
        return {**self._self_interaction_dict, **self._mutual_interaction_dict, **self._triple_interaction_dict}

    def get_interaction_by_name(self, interaction_name):
        """
        Get an interaction by name.
        Args:
            interaction_name (string): name of the interaction
        """
        if interaction_name in self._self_interaction_dict:
            return self._self_interaction_dict[interaction_name]
        elif interaction_name in self._mutual_interaction_dict:
            return self._mutual_interaction_dict[interaction_name]
        else:
            raise ValueError("Interaction with this name does not exist. Existing interactions: ", 
                self._self_interaction_dict.keys(), self._mutual_interaction_dict.keys())
 
    def _add_interaction_sanity_check(self, name):
        if name == 'pV':
            raise ValueError("The interaction name 'pV' is internal. The term pV in the Hamiltonian will be added automatically when you add a global strain.")
        if name in self.interaction_dict:
            raise ValueError("Interaction with this name already exists. Pick another name.")
        return

    def add_self_interaction(self, name, field_name, energy_engine, parameters=None, enable_jit=True):
        '''
        Add a self-interaction term to the Hamiltonian.
        Args:
            name (string): name of the interaction
            field_name (string): name of the field
            energy_engine (function): a function that takes the field as input and returns the interaction energy
            parameters (list): parameters for the interaction
            enable_jit (bool): whether to use JIT compilation
        '''
        self._add_interaction_sanity_check(name)
        interaction = self_interaction( field_name)
        interaction.set_energy_engine(energy_engine, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        if parameters is not None:
            interaction.set_parameters(parameters)
        self._self_interaction_dict[name] = interaction
        return interaction

    def add_mutual_interaction(self, name, field_name1, field_name2, energy_engine,  parameters=None, enable_jit=True):
        '''
        Add a mutual interaction term to the Hamiltonian.
        Args:
            name (string): name of the interaction
            field_name1 (string): name of the first field
            field_name2 (string): name of the second field
            energy_engine (function): a function that takes the fields as input and returns the interaction energy
            parameters (list): parameters for the interaction
            enable_jit (bool): whether to use JIT compilation
        '''
        self._add_interaction_sanity_check(name)
        interaction = mutual_interaction( field_name1, field_name2)
        interaction.set_energy_engine(energy_engine, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        if parameters is not None:
            interaction.set_parameters(parameters)
        self._mutual_interaction_dict[name] = interaction
        return interaction
    
    def add_triple_interaction(self, name, field_name1, field_name2, field_name3, energy_engine, parameters=None, enable_jit=True):
        '''
        Add a triple interaction term to the Hamiltonian.
        '''
        self._add_interaction_sanity_check(name)
        interaction = triple_interaction(field_name1, field_name2, field_name3)
        interaction.set_energy_engine(energy_engine, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        if parameters is not None:
            interaction.set_parameters(parameters)
        self._triple_interaction_dict[name] = interaction
        return interaction

    def add_pressure(self, pressure):
        '''
        Add a pressure term to the Hamiltonian.
        Args:
            name (string): name of the interaction
            pressure (float): pressure in bars
            field_name (string): name of the field
            parameters (list): parameters for the interaction
            enable_jit (bool): whether to use JIT compilation
        '''
        _pres = pressure * Constants.bar  # bar -> eV/Angstrom^3
        ## interaction name sanity check
        name = 'pV'
        if name in self.interaction_dict:
            raise ValueError("pV term already exists in the Hamiltonian.")
        ## field name sanity check
        field_name = 'gstrain'
        field = self.get_field_by_name(field_name)
        if not isinstance(field, GlobalStrain):
            raise ValueError("I find a field named gstrain, but it is not a global strain field. Please rename the field or remove it. Then add the global strain field to the system.")
        ## add the interaction
        interaction = self_interaction(field_name)
        interaction.set_energy_engine(energy_engine=pV_energy, enable_jit=True)
        interaction.create_force_engine(enable_jit=True)
        parameters = [_pres, self.lattice.ref_volume] 
        interaction.set_parameters(parameters)
        self._self_interaction_dict[name] = interaction
        return interaction

    """
    Methods for energy and force calculation
    """

    def calc_energy_by_name(self, interaction_name):
        if interaction_name in self._self_interaction_dict:
            interaction = self.get_interaction_by_name(interaction_name)
            field = self.get_field_by_name(interaction.field_name)
            energy = interaction.calc_energy(field)
        elif interaction_name in self._mutual_interaction_dict:
            interaction = self.get_interaction_by_name(interaction_name)
            field1 = self.get_field_by_name(interaction.field_name1)
            field2 = self.get_field_by_name(interaction.field_name2)
            energy = interaction.calc_energy(field1, field2)
        elif interaction_name in self._triple_interaction_dict:
            interaction = self.get_interaction_by_name(interaction_name)
            field1 = self.get_field_by_name(interaction.field_name1)
            field2 = self.get_field_by_name(interaction.field_name2)
            field3 = self.get_field_by_name(interaction.field_name3)
            energy = interaction.calc_energy(field1, field2, field3)
        else:
            raise ValueError("Interaction with this name does not exist. Existing interactions: ", 
                self._self_interaction_dict.keys(), self._mutual_interaction_dict.keys())
        return energy            

    def calc_force_by_name(self, interaction_name):
        if interaction_name in self._self_interaction_dict:
            interaction = self.get_interaction_by_name(interaction_name)
            field = self.get_field_by_name(interaction.field_name)
            force = interaction.calc_force(field)
        elif interaction_name in self._mutual_interaction_dict:
            interaction = self.get_interaction_by_name(interaction_name)
            field1 = self.get_field_by_name(interaction.field_name1)
            field2 = self.get_field_by_name(interaction.field_name2)
            force = interaction.calc_force(field1, field2)
        elif interaction_name in self._triple_interaction_dict:
            interaction = self.get_interaction_by_name(interaction_name)
            field1 = self.get_field_by_name(interaction.field_name1)
            field2 = self.get_field_by_name(interaction.field_name2)
            field3 = self.get_field_by_name(interaction.field_name3)
            force = interaction.calc_force(field1, field2, field3)
        else:
            raise ValueError("Interaction with this name does not exist. Existing interactions: ", 
                self._self_interaction_dict.keys(), self._mutual_interaction_dict.keys())
        return force

    def calc_total_self_energy(self):
        return sum([self.calc_energy_by_name(interaction_name) for interaction_name in self._self_interaction_dict])

    def calc_total_mutual_interaction(self):
        return sum([self.calc_energy_by_name(interaction_name) for interaction_name in self._mutual_interaction_dict])

    def calc_total_triple_interaction(self):
        return sum([self.calc_energy_by_name(interaction_name) for interaction_name in self._triple_interaction_dict])

    def calc_total_potential_energy(self):
        return self.calc_total_self_energy() + self.calc_total_mutual_interaction() + self.calc_total_triple_interaction()

    def calc_kinetic_energy(self):
        """
        Calculate the kinetic energy of the system. 
        TODO: move energy calculation to field class
        """
        kinetic_energy = 0.0
        for field in self.get_all_fields():
            velocity = field.get_velocity()
            mass = field.get_mass()
            kinetic_energy += 0.5 * jnp.sum(mass * velocity**2)
        return kinetic_energy

    def calc_temp_by_name(self, name):
        """
        Calculate the temperature of a field.
        TODO: move temperature calculation to field class
        """
        field = self.get_field_by_name(name)
        velocity = field.get_velocity()
        mass = field.get_mass()
        return jnp.mean(mass * velocity**2) / ( Constants.kb)

    def calc_excess_stress(self):
        '''
        Get instantaneous stress - applied stress (e.g. from hydrostatic pressure)
        TODO: move stress calculation to gstrain class
        '''
        field = self.get_field_by_name('gstrain')
        field.zero_force()
        for interaction_name in self._self_interaction_dict:
            interaction = self._self_interaction_dict[interaction_name]
            if interaction.field_name == 'gstrain':
                force = interaction.calc_force(field)
                field.accumulate_force(force)
        for interaction_name in self._mutual_interaction_dict:
            interaction = self.get_interaction_by_name(interaction_name)
            if interaction.field_name1 == 'gstrain':
                assert (interaction.field_name2 != 'gstrain')
                field2 = self.get_field_by_name(interaction.field_name2)
                force1, force2 = interaction.calc_force(field, field2)
                field.accumulate_force(force1)
            elif interaction.field_name2 == 'gstrain':
                field1 = self.get_field_by_name(interaction.field_name1)
                force1, force2 = interaction.calc_force(field1, field)
                field.accumulate_force(force2)
        return field.get_force() / self.lattice.ref_volume / Constants.bar 
    
    """
    Methods for updating the gradient force 
    """

    def update_force_from_self_interaction(self, profile=False):
        """
        update the gradient force felt by each field from self interactions.
        """
        for interaction_name in self._self_interaction_dict:
            if profile:
                t0 = timer()
            interaction = self._self_interaction_dict[interaction_name]
            field = self.get_field_by_name(interaction.field_name)
            force = interaction.calc_force(field)
            field.accumulate_force(force)
            if profile:
                jax.block_until_ready(field.get_force())
                print('time for updating force from %s:' % interaction_name, timer()-t0)
        return
    
    def update_force_from_mutual_interaction(self, profile=False):
        """
        update the gradient force felt by each field from mutual interactions.
        """
        for interaction_name in self._mutual_interaction_dict:
            if profile:
                t0 = timer()
            interaction = self.get_interaction_by_name(interaction_name)
            field1 = self.get_field_by_name(interaction.field_name1)
            field2 = self.get_field_by_name(interaction.field_name2)
            force1, force2 = interaction.calc_force(field1, field2)
            field1.accumulate_force(force1)
            field2.accumulate_force(force2)
            if profile:
                jax.block_until_ready(field2.get_force())
                print('time for updating force from %s:' % interaction_name, timer()-t0)
        return
    
    def update_force_from_triple_interaction(self, profile=False):
        """
        update the gradient force felt by each field from triple interactions.
        """
        for interaction_name in self._triple_interaction_dict:
            if profile:
                t0 = timer()
            interaction = self.get_interaction_by_name(interaction_name)
            field1 = self.get_field_by_name(interaction.field_name1)
            field2 = self.get_field_by_name(interaction.field_name2)
            field3 = self.get_field_by_name(interaction.field_name3)
            force1, force2, force3 = interaction.calc_force(field1, field2, field3)
            field1.accumulate_force(force1)
            field2.accumulate_force(force2)
            field3.accumulate_force(force3)
            if profile:
                jax.block_until_ready(field3.get_force())
                print('time for updating force from %s:' % interaction_name, timer()-t0)


    def update_force(self, profile=False):
        """
        update the gradient force felt by each field from all interactions.
        """
        ## zero force
        for field in self.get_all_fields():
            if profile:
                t0 = timer()
            field.zero_force()
            if profile:
                jax.block_until_ready(field.get_force())
                print('time for zeroing force of %s:' % field.name, timer()-t0)

        ## update force from all interactions
        self.update_force_from_self_interaction(profile=profile)
        self.update_force_from_mutual_interaction(profile=profile)
        self.update_force_from_triple_interaction(profile=profile)
        return

class RingPolymerSystem(System):
    """
    A class to define a ring polymer system for path-integral molecular dynamics simulations.
    """
    def __init__(self, lattice, nbeads=1, pbc=True):
        super().__init__(lattice, name)
        self.nbeads = nbeads
        raise NotImplementedError("Ring polymer system is not implemented yet.")