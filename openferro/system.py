"""
Classes which define the physical system. 
"""
# This file is part of OpenFerro.


import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from field import FieldSO3, FieldRn, FieldScalar


class System:
    """
    A class to define a physical system. A system is a lattice with fields and a Hamiltonian.
    """
    def __init__(self, lattice, pbc=True):
        self.lattice = lattice
        self.pbc = pbc
        self._fields_dict = {}
        self._self_interaction_list = []
        self._mutual_interaction_list = []

    def __repr__(self):
        return f"System with lattice {self.lattice} and fields {self._fields_dict.keys()}"
    
    ## setter and getter methods for fields
    def add_field(name, type='scalar', dim=None, unit=None):
        if name in self._fields_dict:
            raise ValueError("Field with this name already exists. Pick another name")
        if type == 'scalar':
            self._fields_dict[name] = FieldScalar(self.lattice, name, unit)
        elif type == 'Rn':
            self._fields_dict[name] = FieldRn(self.lattice, name, dim, unit)
        elif type == 'SO3':
            self._fields_dict[name] = FieldSO3(self.lattice, name)
        else:
            raise ValueError("Unknown field type. Choose from 'scalar', 'Rn', 'SO3'")
        return self._fields_dict[name]

    def get_field_by_name(self, name):
        return self._fields_dict[name]

    def get_all_fields(self):
        return [self._fields_dict[name] for name in self._fields_dict.keys()]
    
    ## setter and getter methods for interactions
    def add_self_interaction(self, field_name, interaction, interaction_name=None):
        '''
        Add a self-interaction term to the Hamiltonian.
        Args:
            field_name (string): name of the field
            interaction (function): a function that takes the field as input and returns the interaction energy
        '''
        current_interaction_names = [_dict['interaction_name'] for _dict in self._self_interaction_list]
        if interaction_name is None:
            interaction_name = 'self-interaction-{:d}'.format(len(current_interaction_names))
        if interaction_name in current_interaction_names:
            raise ValueError("Interaction with this name already exists. Pick another name. Existing self-interactions: ", current_interaction_names)
        interaction_dict = {
            'interaction_name':interaction_name, 
            'field_name': field_name, 
            'energy': interaction, 
            'energy_gradient': grad(interaction, argnums=0)
            }
        self._self_interaction_list.append(interaction_dict)
        return interaction_dict

    def add_mutual_interaction(self, field_name1, field_name2, interaction, interaction_name=None):
        '''
        Add a mutual interaction term to the Hamiltonian.
        Args:
            field_name1 (string): name of the first field
            field_name2 (string): name of the second field
            interaction (function): a function that takes the fields as input and returns the interaction energy
        '''
        current_interaction_names = [_dict['interaction_name'] for _dict in self._mutual_interaction_list]
        if interaction_name is None:
            interaction_name = 'mutual-interaction-{:d}'.format(len(current_interaction_names))
        if interaction_name in current_interaction_names:
            raise ValueError("Interaction with this name already exists. Pick another name. Existing mutual-interactions: ", current_interaction_names)
        interaction_dict = {
            'interaction_name':interaction_name,
            'field_name1': field_name1,
            'field_name2': field_name2,
            'energy': interaction,
            'energy_gradient': grad(interaction, argnums=(0, 1))
            }
        self._mutual_interaction_list.append(interaction_dict)
        return interaction_dict

    def get_self_interaction_by_name(self, interaction_name):
        self_interaction_names = [_dict['interaction_name'] for _dict in self._self_interaction_list]
        if interaction_name not in self_interaction_names:
            raise ValueError("Interaction with this name does not exist. Existing self-interactions: ", self_interaction_names)
        return self._self_interaction_list[self_interaction_names.index(interaction_name)]
    
    def get_mutual_interaction_by_name(self, interaction_name):
        mutual_interaction_names = [_dict['interaction_name'] for _dict in self._mutual_interaction_list]
        if interaction_name not in mutual_interaction_names:
            raise ValueError("Interaction with this name does not exist. Existing mutual-interactions: ", mutual_interaction_names)
        return self._mutual_interaction_list[mutual_interaction_names.index(interaction_name)]
    
    ## calculators for energy and forces
    def calculate_self_energy_by_name(self, interaction_name):
        interaction_dict = self.get_self_interaction_by_name(interaction_name)
        field = self.get_field_by_name(interaction_dict['field_name'])
        energy = interaction_dict['energy'](field)
        return energy            

    def calculate_self_force_by_name(self, interaction_name):
        interaction_dict = self.get_self_interaction_by_name(interaction_name)
        field = self.get_field_by_name(interaction_dict['field_name'])
        energy_gradient = interaction_dict['energy_gradient'](field)
        return - energy_gradient

    def calculate_mutual_interaction_by_name(self, name1, name2):
        interaction_dict =  self.get_mutual_interaction_by_name(interaction_name)
        field1 = self.get_field_by_name(interaction_dict['field_name1'])
        field2 = self.get_field_by_name(interaction_dict['field_name2'])
        energy = interaction_dict['energy'](field1, field2)
        return energy

    def calculate_mutual_force_by_name(self, interaction_name):
        interaction_dict = self.get_mutual_interaction_by_name(interaction_name)
        field1 = self.get_field_by_name(interaction_dict['field_name1'])
        field2 = self.get_field_by_name(interaction_dict['field_name2'])
        energy_gradient = interaction_dict['energy_gradient'](field1, field2)
        return (- energy_gradient[0], - energy_gradient[1])

    def calculate_total_self_energy(self):
        return sum([self.calculate_self_energy_by_name(interaction_name) for interaction_name in self._self_interaction_list])

    def calculate_total_mutual_interaction(self):
        return sum([self.calculate_mutual_interaction_by_name(interaction_name) for interaction_name in self._mutual_interaction_list])

    def calculate_total_energy(self):
        return calculate_total_self_energy() + calculate_total_interaction()


class RingPolymerSystem(System):
    """
    A class to define a ring polymer system for path-integral molecular dynamics simulations.
    """
    def __init__(self, lattice, nbeads=1, pbc=True):
        super().__init__(lattice, name)
        self.nbeads = nbeads