"""
Classes which define the "interaction" between fields. Each interaction is associated with a term in the Hamiltonian. 
Each interaction stores a function "self.energy_engine" that calculates the energy of the interaction and a function "force engine" that calculates the force of the interaction.
Only the energy engine is required. The force engine is optional. If the force engine is not set, the force will be calculated by automatic differentiation of the energy engine.
"""
# This file is part of OpenFerro.

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap

class interaction_base:
    """
    The base class to specify the interaction between fields.
    """
    def __init__(self, parameters=None):
        self.parameters = {} if parameters is None else parameters
        self.energy_engine = None
        self.force_engine = None
    def set_parameters(self, parameters):
        self.parameters = parameters
    def get_parameters(self):
        return self.parameters  ## TODO
    def set_parameter_by_name(self, name, value):
        self.parameters[name] = value
    def get_parameter_by_name(self, name):
        if name not in self.parameters:
            raise ValueError("Parameter with this name does not exist. Existing parameters: ", self.parameters.keys())
        return self.parameters[name]
    def set_energy_engine(self, energy_engine, enable_jit=True):
        if enable_jit:
            self.energy_engine = jit(energy_engine)
        else:
            self.energy_engine = energy_engine
    def calc_energy(self):
        pass
    def calc_force(self):
        pass


class self_interaction(interaction_base):
    """
    A class to specify the self-interaction of a field.
    """
    def __init__(self, field_name, parameters=None):
        super().__init__( parameters)
        self.field_name = field_name
    def create_force_engine(self, enable_jit=True):
        if self.energy_engine is None:
            raise ValueError("Energy engine is not set. Set energy engine before creating force engine.")
        if enable_jit:
            self.force_engine =  jit(grad(self.energy_engine, argnums=0 )) 
        else:
            self.force_engine =  grad(self.energy_engine, argnums=0 )
    def calc_energy(self, field):
        field_values = field.get_values()
        return self.energy_engine(field_values, self.parameters)
    def calc_force(self, field):
        field_values = field.get_values()
        gradient = self.force_engine(field_values, self.parameters)
        return -gradient

class mutual_interaction:
    """
    A class to specify the  mutual interaction between two fields.
    """
    def __init__(self, field_name1, field_name2, parameters=None):
        super().__init__( parameters)
        self.field_name1 = field_name1
        self.field_name2 = field_name2
    def create_force_engine(self, enable_jit=True):
        if self.energy_engine is None:
            raise ValueError("Energy engine is not set. Set energy engine before creating force engine.")
        if enable_jit:
            self.force_engine =  jit(grad(self.energy_engine, argnums=(0, 1) )) 
        else:
            self.force_engine =  grad(self.energy_engine, argnums=(0, 1) )
    def calc_energy(self, field1, field2):
        f1 = field1.get_values()
        f2 = field2.get_values()
        return self.energy_engine(f1, f2, self.parameters)
    def calc_force(self, field1, field2):
        f1 = field1.get_values()
        f2 = field2.get_values()
        gradient = self.force_engine(f1, f2, self.parameters)
        return (- gradient[0], - gradient[1])

class triple_interaction:
    """
    A class to specify the  mutual interaction between three fields.
    """
    def __init__(self, field_name1, field_name2, field_name3, parameters=None):
        super().__init__( parameters)
        self.field_name1 = field_name1
        self.field_name2 = field_name2
        self.field_name3 = field_name3
    def create_force_engine(self, enable_jit=True):
        if self.energy_engine is None:
            raise ValueError("Energy engine is not set. Set energy engine before creating force engine.")
        if enable_jit:
            self.force_engine =  jit(grad(self.energy_engine, argnums=(0, 1, 2) )) 
        else:
            self.force_engine =  grad(self.energy_engine, argnums=(0, 1, 2) )
    def calc_energy(self, field1, field2, field3):
        f1 = field1.get_values()
        f2 = field2.get_values()
        f3 = field3.get_values()
        return self.energy_engine(f1, f2, f3, self.parameters)
    def calc_force(self, field1, field2, field3):
        f1 = field1.get_values()
        f2 = field2.get_values()
        f3 = field3.get_values()
        gradient = self.force_engine(f1, f2, f3, self.parameters)
        return (- gradient[0], - gradient[1], - gradient[2])


