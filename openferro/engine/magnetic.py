"""
Functions that define a term in the magnetic Hamiltonian. They will be added into <class interaction> for automatic differentiation.
"""
# This file is part of OpenFerro.

import jax.numpy as jnp

__all__ = [
    "get_isotropic_exchange_energy_engine",
    "cubic_anisotropy_energy",
    "external_field_energy",
]


def get_isotropic_exchange_energy_engine(rollers, bond_counting="unique"):
    """Return an isotropic exchange engine for a periodic vector field.

    The default convention is
    ``E = -sum_(i,d in half-shell) J * m_i dot m_(i-d)``, so each physical
    undirected displacement bond occurs once. Rollers must therefore enumerate
    one half of a neighbor shell. ``bond_counting="ordered"`` retains the
    pre-Milestone-B convention and multiplies this sum by two.

    Parameters
    ----------
    rollers : list
        List of jnp.roll functions specifying the neighbors
    bond_counting : {"unique", "ordered"}, optional
        Pair-counting convention. ``"ordered"`` is provided only for explicit
        compatibility with earlier parameter sets.

    Returns
    -------
    callable
        Energy engine function
    """
    if bond_counting not in {"unique", "ordered"}:
        raise ValueError("bond_counting must be 'unique' or 'ordered'.")
    multiplicity = 1.0 if bond_counting == "unique" else 2.0

    def energy_engine(field, parameters):
        coupling = -parameters[0] * multiplicity
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
    raise NotImplementedError(
        "Dzyaloshinskii-Moriya energy requires a validated bond and orientation "
        "convention and is not implemented."
    )

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
    B_ext = jnp.asarray(parameters)
    if B_ext.shape != (3,):
        raise ValueError("External magnetic field parameters must have shape (3,).")
    energy = - jnp.sum(field * B_ext)
    return energy
