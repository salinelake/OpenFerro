import numpy as np
import jax.numpy as jnp

@jit
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

@jit
def self_energy_R1_onsite(field, parameters):
    """
    Returns the isotropic self-energy of a field.
    See [Zhong, W., David Vanderbilt, and K. M. Rabe. Physical Review B 52.9 (1995): 6301.] for the meaning of the parameters.
    """
    k2 = parameters['k2']
    alpha = parameters['alpha']
    gamma = parameters['gamma']
    offset = parameters['offset']

    energy = k2 * jnp.sum((field-offset)**2)
    energy += alpha * jnp.sum((field-offset)**4 )
    return energy

@jit
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

@jit
def self_energy_dipole_dipole(field, parameters):
    """
    Returns the dipole-dipole interaction energy of a field with Ewald summation.
    """
    pass

@jit
def mutual_energy_R3(field1, field2, parameters):
    """
    Returns the mutual interaction energy of two R^3 fields.
    """
    pass