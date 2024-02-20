"""
Classes which define the physical system. 
"""
# This file is part of OpenFerro.


import numpy as np
import jax.numpy as jnp
from field import FieldSO3, FieldRn, FieldScalar


class System:
    """
    A class to define a physical system. A system is a lattice with fields and a Hamiltonian.
    """
    def __init__(self, lattice, pbc=True):
        self.lattice = lattice
        self.pbc = pbc
        self._fields_dict = {}
        self._self_energy_dict = {}
        self._interaction_dict = {}

    def __repr__(self):
        return f"System with lattice {self.lattice} and fields {self._fields_dict.keys()}"
    
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

    def get_field_by_name(self, name):
        return self._fields_dict[name]
    
    def add_self_energy(self, name, energy):
        self._self_energy_dict[name] = energy
 
    def add_interaction(self, name, interaction):
        self._interaction_dict[name] = interaction

    def calculate_self_energy_by_name(self, name):
        pass

    def calculate_interaction_by_name(self, name1, name2):
        pass

    def calculate_total_self_energy(self):
        pass

    def calculate_total_interaction(self):
        pass

    def calculate_total_energy(self):
        return calculate_total_self_energy() + calculate_total_interaction()


class RingPolymerSystem(System):
    """
    A class to define a ring polymer system for path-integral molecular dynamics simulations.
    """
    def __init__(self, lattice, nbeads=1, pbc=True):
        super().__init__(lattice, name)
        self.nbeads = nbeads