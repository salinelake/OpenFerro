"""
Classes which define the physical system. 
"""
# This file is part of OpenFerro.


import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from openferro.field import FieldSO3, FieldRn, FieldScalar
from openferro.interaction import self_interaction, mutual_interaction

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
    def add_field(self, name, ftype='scalar', dim=None, unit=None, value=None):
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
        else:
            raise ValueError("Unknown field type. Choose from 'scalar', 'Rn', 'SO3'")
        return self._fields_dict[name]

    def get_field_by_name(self, name):
        return self._fields_dict[name]

    def get_all_fields(self):
        return [self._fields_dict[name] for name in self._fields_dict.keys()]
    
    ## setter and getter methods for interactions
    def add_self_interaction(self, name, field_name, energy_engine, parameters=None):
        '''
        Add a self-interaction term to the Hamiltonian.
        Args:
            field_name (string): name of the field
            interaction (function): a function that takes the field as input and returns the interaction energy
        '''
        if name in self._self_interaction_dict or name in self._mutual_interaction_dict:
            raise ValueError("Interaction with this name already exists. Pick another name.")
        interaction = self_interaction( field_name)
        interaction.set_energy_engine(energy_engine)
        if parameters is not None:
            interaction.set_parameters(parameters)
        self._self_interaction_dict[name] = interaction
        return interaction

    def add_mutual_interaction(self, name, field_name1, field_name2, energy_engine,  parameters=None):
        '''
        Add a mutual interaction term to the Hamiltonian.
        Args:
            field_name1 (string): name of the first field
            field_name2 (string): name of the second field
            interaction (function): a function that takes the fields as input and returns the interaction energy
        '''
        if name in self._self_interaction_dict or name in self._mutual_interaction_dict:
            raise ValueError("Interaction with this name already exists. Pick another name.")
        interaction = mutual_interaction( field_name1, field_name2)
        interaction.set_energy_engine(energy_engine)
        if parameters is not None:
            interaction.set_parameters(parameters)
        self._mutual_interaction_dict[name] = interaction
        return interaction

    def get_interaction_by_name(self, interaction_name):
        if interaction_name in self._self_interaction_dict:
            return self._self_interaction_dict[interaction_name]
        elif interaction_name in self._mutual_interaction_dict:
            return self._mutual_interaction_dict[interaction_name]
        else:
            raise ValueError("Interaction with this name does not exist. Existing interactions: ", 
                self._self_interaction_dict.keys(), self._mutual_interaction_dict.keys())
 
    ## calculators for energy and forces
    def calculate_energy_by_name(self, interaction_name):
        if interaction_name in self._self_interaction_dict:
            interaction = self.get_interaction_by_name(interaction_name)
            field = self.get_field_by_name(interaction.field_name).get_field()
            energy = interaction.calculate_energy(field)
        elif interaction_name in self._mutual_interaction_dict:
            interaction = self.get_interaction_by_name(interaction_name)
            field1 = self.get_field_by_name(interaction.field_name1).get_field()
            field2 = self.get_field_by_name(interaction.field_name2).get_field()
            energy = interaction.calculate_energy(field1, field2)
        else:
            raise ValueError("Interaction with this name does not exist. Existing interactions: ", 
                self._self_interaction_dict.keys(), self._mutual_interaction_dict.keys())
        return energy            

    def calculate_force_by_name(self, interaction_name):
        if interaction_name in self._self_interaction_dict:
            interaction = self.get_interaction_by_name(interaction_name)
            field = self.get_field_by_name(interaction.field_name).get_field()
            force = interaction.calculate_force(field)
        elif interaction_name in self._mutual_interaction_dict:
            interaction = self.get_interaction_by_name(interaction_name)
            field1 = self.get_field_by_name(interaction.field_name1).get_field()
            field2 = self.get_field_by_name(interaction.field_name2).get_field()
            force = interaction.calculate_force(field1, field2)
        else:
            raise ValueError("Interaction with this name does not exist. Existing interactions: ", 
                self._self_interaction_dict.keys(), self._mutual_interaction_dict.keys())
        return force

    def calculate_total_self_energy(self):
        return sum([self.calculate_energy_by_name(interaction_name) for interaction_name in self._self_interaction_dict])

    def calculate_total_mutual_interaction(self):
        return sum([self.calculate_interaction_by_name(interaction_name) for interaction_name in self._mutual_interaction_dict])

    def calculate_total_energy(self):
        return calculate_total_self_energy() + calculate_total_interaction()


class RingPolymerSystem(System):
    """
    A class to define a ring polymer system for path-integral molecular dynamics simulations.
    """
    def __init__(self, lattice, nbeads=1, pbc=True):
        super().__init__(lattice, name)
        self.nbeads = nbeads