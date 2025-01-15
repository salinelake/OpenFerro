"""
Functions that define a term in the magnetic Hamiltonian. They will be added into <class interaction> for automatic differentiation.
"""
# This file is part of OpenFerro.


import numpy as np
import jax.numpy as jnp

def get_isotropic_exchange_energy_engine(rollers):
    """Returns the exchange energy engine for a R^3 field defined on a lattice with periodic boundary conditions.

    Parameters
    ----------
    rollers : list
        List of jnp.roll functions specifying the neighbors

    Returns
    -------
    callable
        Energy engine function
    """
    def energy_engine(field, parameters):
        coupling = - parameters[0] * 2  #  the double counting should be made under adopted convention
        energy = 0
        for roller in rollers:
            field_rolled = roller(field)
            energy += jnp.sum(field * field_rolled)
        return coupling * energy
    return energy_engine
 
def cubic_anisotropy_energy(field, parameters):
    """Returns the anisotropy energy of the field.

    Parameters
    ----------
    field : ndarray
        The magnetic field
    parameters : ndarray
        Array containing K1 and K2 anisotropy constants

    Returns
    -------
    float
        The anisotropy energy: -K1*(mx^2*my^2 + my^2*mz^2 + mx^2*mz^2) - K2*mx^2*my^2*mz^2
    """
    K1 = parameters[0]
    K2 = parameters[1]
    energy = -K1 * (field[:,:,:,0]**2 * field[:,:,:,1]**2 + field[:,:,:,1]**2 * field[:,:,:,2]**2 + field[:,:,:,0]**2 * field[:,:,:,2]**2).sum()
    energy += -K2 * (field[:,:,:,0]**2 * field[:,:,:,1]**2 * field[:,:,:,2]**2).sum()
    return energy

def Dzyaloshinskii_Moriya_energy(field, parameters):
    """Returns the Dzyaloshinskii-Moriya energy of the field.

    Parameters
    ----------
    field : ndarray
        The magnetic field
    parameters : ndarray
        Array of parameters

    Returns
    -------
    float
        The Dzyaloshinskii-Moriya energy
    """
    pass

def external_field_energy(field, parameters):
    """Returns the external field energy of the field.

    Parameters
    ----------
    field : ndarray
        The magnetic field
    parameters : ndarray
        Array containing the external field B_ext

    Returns
    -------
    float
        The external field energy: -field·B_ext
    """
    B_ext = parameters[0]
    energy = - jnp.sum(field * B_ext)
    return energy
