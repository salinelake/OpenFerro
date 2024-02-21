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
        return self.parameters  ## TODO: to list
    def set_parameter_by_name(self, name, value):
        self.parameters[name] = value
    def get_parameter_by_name(self, name):
        if name not in self.parameters:
            raise ValueError("Parameter with this name does not exist. Existing parameters: ", self.parameters.keys())
        return self.parameters[name]
    def set_energy_engine(self, energy_engine):
        pass
    def calculate_energy(self):
        pass
    def calculate_force(self):
        pass


class self_interaction(interaction_base):
    """
    A class to specify the self-interaction of a field.
    """
    def __init__(self, field_name, parameters=None):
        super().__init__( parameters)
        self.field_name = field_name
    def set_energy_engine(self, energy_engine):
        self.energy_engine = energy_engine
        self.force_engine =  grad(energy_engine, argnums=0 ) 
    def calculate_energy(self, field):
        return self.energy_engine(field, self.parameters)
    def calculate_force(self, field):
        gradient = self.force_engine(field, self.parameters)
        return -gradient

class mutual_interaction:
    """
    A class to specify the  mutual interaction between two fields.
    """
    def __init__(self, field_name1, field_name2, parameters=None):
        super().__init__( parameters)
        self.field_name1 = field_name1
        self.field_name2 = field_name2
    def set_energy_engine(self, energy_engine):
        self.energy_engine = energy_engine
        self.force_engine =  grad(energy_engine, argnums=(0, 1)) 
    def calculate_energy(self, field1, field2):
        return self.energy_engine(field1, field2, self.parameters)
    def calculate_force(self, field1, field2):
        gradient = self.force_engine(field1, field2, self.parameters)
        return (- gradient[0], - gradient[1])


def self_energy_R3_onsite_isotropic(field, parameters):
    """
    Returns the isotropic self-energy of a 3D field.
    See [Zhong, W., David Vanderbilt, and K. M. Rabe. Physical Review B 52.9 (1995): 6301.] for the meaning of the parameters.
    """
    k2 = parameters['k2']
    alpha = parameters['alpha']
    gamma = parameters['gamma']
    offset = parameters['offset']

    field2 = (field-offset) ** 2
    energy = k2 * jnp.sum(field2)
    energy += alpha * jnp.sum( (field2.sum(axis=-1))**2 )
    energy += gamma * jnp.sum(
        field2[...,0]*field2[...,1] + field2[...,1]*field2[...,2] + field2[...,2]*field2[...,0]
        )
    return energy




def self_energy_R1_onsite(field, parameters):
    """
    Returns the isotropic self-energy of a field.
    See [Zhong, W., David Vanderbilt, and K. M. Rabe. Physical Review B 52.9 (1995): 6301.] for the meaning of the parameters.
    """
    k2 = parameters['k2']
    alpha = parameters['alpha']
    offset = parameters['offset']

    energy = k2 * jnp.sum((field-offset)**2)
    energy += alpha * jnp.sum((field-offset)**4 )
    return energy

def self_energy_R3_neighbor_pbc(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a R^3 field defined on a lattice with periodic boundary conditions.
    """
    J_1 = parameters['J_1']
    J_2 = parameters['J_2']
    J_3 = parameters['J_3']
    offset = parameters['offset']

    f = field - offset
    f_1p = jnp.roll( f, 1, axis=0) 
    energy = jnp.sum( jnp.dot(f_1p, J1) * f )  
    f_2p = jnp.roll( f, 1, axis=1)
    energy += jnp.sum( jnp.dot(f_2p, J2) * f )
    f_3p = jnp.roll( f, 1, axis=2)
    energy += jnp.sum( jnp.dot(f_3p, J3) * f )
    return energy

def self_energy_R1_neighbor_pbc(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a R^3 field defined on a lattice with periodic boundary conditions.
    """
    J_1 = parameters['J1']
    J_2 = parameters['J2']
    J_3 = parameters['J3']
    offset = parameters['offset']

    f = field - offset
    f_1p = jnp.roll( f, 1, axis=0) 
    energy = jnp.sum( f_1p * f * J_1 )  
    f_2p = jnp.roll( f, 1, axis=1)
    energy += jnp.sum( f_1p * f * J_2 )  
    f_3p = jnp.roll( f, 1, axis=2)
    energy += jnp.sum( f_1p * f * J_3 )  
    return energy

def self_energy_dipole_dipole(field, parameters):
    """
    Returns the dipole-dipole interaction energy of a field with Ewald summation.
    """
    pass

def mutual_energy_R3(field1, field2, parameters):
    """
    Returns the mutual interaction energy of two R^3 fields.
    """
    pass