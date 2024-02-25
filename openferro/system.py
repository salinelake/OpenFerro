"""
Classes which define the physical system. 
"""
# This file is part of OpenFerro.


import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from openferro.field import FieldSO3, FieldRn, FieldScalar
from openferro.interaction import self_interaction, mutual_interaction
from openferro.units import Constants
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
    def __repr__(self):
        return f"System with lattice {self.lattice} and fields {self._fields_dict.keys()}"
     
    ## setter and getter methods for fields
    def add_field(self, name, ftype='scalar', dim=None, unit=None, value=None, mass=1.0):
        if name in self._fields_dict:
            raise ValueError("Field with this name already exists. Pick another name")
        if ftype == 'scalar':
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
        elif ftype == 'global_strain':
            if value is not None:
                assert len(value) == 6, "Global strain must be a 6D vector"
                init = jnp.array(value, dtype=jnp.float32)
            else:
                init = jnp.zeros(6, dtype=jnp.float32)
            self._fields_dict[name] = GlobalStrain(self.lattice, name)
            self._fields_dict[name].set_values(jnp.zeros((1,1,1, 6)) + init)
        else:
            raise ValueError("Unknown field type. Choose from 'scalar', 'Rn', 'SO3'")
        if mass is not None:
            self._fields_dict[name].set_mass(mass)
        return self._fields_dict[name]

    def get_field_by_name(self, name):
        return self._fields_dict[name]

    def get_all_fields(self):
        return [self._fields_dict[name] for name in self._fields_dict.keys()]
    
    ## setter and getter methods for interactions
    def add_self_interaction(self, name, field_name, energy_engine, parameters=None, enable_jit=True):
        '''
        Add a self-interaction term to the Hamiltonian.
        Args:
            name (string): name of the interaction
            field_name (string): name of the field
            energy_engine (function): a function that takes the field as input and returns the interaction energy
            parameters (dict): parameters for the interaction
            enable_jit (bool): whether to use JIT compilation
        '''
        if name in self._self_interaction_dict or name in self._mutual_interaction_dict:
            raise ValueError("Interaction with this name already exists. Pick another name.")
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
            parameters (dict): parameters for the interaction
            enable_jit (bool): whether to use JIT compilation
        '''
        if name in self._self_interaction_dict or name in self._mutual_interaction_dict:
            raise ValueError("Interaction with this name already exists. Pick another name.")
        interaction = mutual_interaction( field_name1, field_name2)
        interaction.set_energy_engine(energy_engine, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        if parameters is not None:
            interaction.set_parameters(parameters)
        self._mutual_interaction_dict[name] = interaction
        return interaction

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
 
    ## calculators for energy and forces
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
        else:
            raise ValueError("Interaction with this name does not exist. Existing interactions: ", 
                self._self_interaction_dict.keys(), self._mutual_interaction_dict.keys())
        return force

    def calc_total_self_energy(self):
        return sum([self.calc_energy_by_name(interaction_name) for interaction_name in self._self_interaction_dict])

    def calc_total_mutual_interaction(self):
        return sum([self.calc_interaction_by_name(interaction_name) for interaction_name in self._mutual_interaction_dict])

    def calc_potential_energy(self):
        return self.calc_total_self_energy() + self.calc_total_mutual_interaction()

    def calc_kinetic_energy(self):
        kinetic_energy = 0.0
        for field in self.get_all_fields():
            velocity = field.get_velocity()
            mass = field.get_mass()
            kinetic_energy += 0.5 * jnp.sum(mass * velocity**2)
        return kinetic_energy

    def calc_temp_by_name(self, name):
        field = self.get_field_by_name(name)
        velocity = field.get_velocity()
        mass = field.get_mass()
        return jnp.mean(mass * velocity**2) / ( Constants.kb)

    def update_force(self):
        for field in self.get_all_fields():
            field.zero_force()
        for interaction_name in self._self_interaction_dict:
            interaction = self._self_interaction_dict[interaction_name]
            field = self.get_field_by_name(interaction.field_name)
            force = interaction.calc_force(field)
            field.accumulate_force(force)
        for interaction_name in self._mutual_interaction_dict:
            interaction = self.get_interaction_by_name(interaction_name)
            field1 = self.get_field_by_name(interaction.field_name1)
            field2 = self.get_field_by_name(interaction.field_name2)
            force1, force2 = interaction.calc_force(field1, field2)
            field1.accumulate_force(force1)
            field2.accumulate_force(force2)


class RingPolymerSystem(System):
    """
    A class to define a ring polymer system for path-integral molecular dynamics simulations.
    """
    def __init__(self, lattice, nbeads=1, pbc=True):
        super().__init__(lattice, name)
        self.nbeads = nbeads