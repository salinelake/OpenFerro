"""
Functions that define a ferroelectric term in the Hamiltonian. They will be added into <class interaction> for automatic differentiation.
"""
# This file is part of OpenFerro.

import numpy as np
import jax.numpy as jnp
from jax import jit
from openferro.field import LocalStrain3D


def self_energy_onsite_isotropic(field, parameters):
    """
    Returns the isotropic self-energy of a 3D field.
    See Eq.(2-3) in [Zhong, W., David Vanderbilt, and K. M. Rabe. Physical Review B 52.9 (1995): 6301.] for meaning of the parameters.

    Parameters
    ----------
    field : jnp.array
        The field to calculate the energy
    parameters : jax.numpy array
        The parameters of the energy function

    Returns
    -------
    jnp.array
        The energy of the field
    """
    k2 = parameters[0]
    alpha = parameters[1]
    gamma = parameters[2]

    field2 = field ** 2
    energy = k2 * jnp.sum(field2)
    energy += alpha * jnp.sum( (field2.sum(axis=-1))**2 )
    energy += gamma * jnp.sum(
        field2[...,0]*field2[...,1] + field2[...,1]*field2[...,2] + field2[...,2]*field2[...,0]
        )
    return energy

def self_energy_onsite_scalar(field, parameters):
    """
    Returns the self-energy of a scalar field.
    E=  sum_i E_i.  (sum over the lattice sites i)
    E_i = k_2 * u_i^2 + alpha * u_i^4

    Parameters
    ----------
    field : jnp.array
        The field to calculate the energy
    parameters : jax.numpy array
        The parameters of the energy function

    Returns
    -------
    jnp.array
        The energy of the field
    """
    # k2 = parameters['k2']
    # alpha = parameters['alpha']
    # offset = parameters['offset']
    k2, alpha, offset = parameters

    energy = k2 * jnp.sum((field-offset)**2)
    energy += alpha * jnp.sum((field-offset)**4 )
    return energy

def short_range_1stnn_isotropic_scalar(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a R^3 field defined on a lattice with periodic boundary conditions.

    Parameters
    ----------
    field : jnp.array
        The field to calculate the energy
    parameters : jax.numpy array
        The parameters of the energy function

    Returns
    -------
    jnp.array
        The energy of the field
    """
    # j = parameters['j']
    # offset = parameters['offset']
    j, offset = parameters

    f = field - offset
    f_0 = jnp.roll( f, 1, axis=0) 
    f_1 = jnp.roll( f, 1, axis=1)
    f_2 = jnp.roll( f, 1, axis=2)
    energy = j * jnp.sum( f * (f_0+f_1+f_2))  
    return energy

def short_range_1stnn_isotropic(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a R^3 field defined on a isotropic lattice with periodic boundary conditions.

    Parameters
    ----------
    field : jnp.array
        The field to calculate the energy
    parameters : jax.numpy array
        The parameters of the energy function

    Returns
    -------
    jnp.array
        The energy of the field
    """
    j1 = parameters[0]  ## uni-axis interaction orthogonal to displacement direction
    j2 = parameters[1]  ## uni-axis interaction along displacement direction

    f = field
    energy = 0
    for axis in range(3):
        f_shifted = jnp.roll(f, 1, axis=axis)
        energy += j1 * jnp.sum(f * f_shifted)
        energy += (j2 - j1) * jnp.sum(f[..., axis] * f_shifted[..., axis])
    return energy

def short_range_1stnn_anisotropic(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a R^3 field defined on a anisotropic lattice with periodic boundary conditions.

    Parameters
    ----------
    field : jnp.array
        The field to calculate the energy
    parameters : jax.numpy array
        The parameters of the energy function

    Returns
    -------
    jnp.array
        The energy of the field
    """
    # J_1 = parameters['J_1']
    # J_2 = parameters['J_2']
    # J_3 = parameters['J_3']
    # offset = parameters['offset']
    J_1, J_2, J_3, offset = parameters

    f = field - offset
    f_1p = jnp.roll( f, 1, axis=0) 
    energy = jnp.sum( jnp.dot(f_1p, J_1) * f )  
    f_2p = jnp.roll( f, 1, axis=1)
    energy += jnp.sum( jnp.dot(f_2p, J_2) * f )
    f_3p = jnp.roll( f, 1, axis=2)
    energy += jnp.sum( jnp.dot(f_3p, J_3) * f )
    return energy
 
def short_range_2ednn_isotropic(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a R^3 field defined on a lattice with periodic boundary conditions.

    Parameters
    ----------
    field : jnp.array
        The field to calculate the energy
    parameters : jax.numpy array
        The parameters of the energy function

    Returns
    -------
    jnp.array
        The energy of the field
    """
    j3, j4, j5 = parameters
    
    f = field
    energy = 0.0

    for axis_pair in [(0,1), (0,2), (1,2)]:
        f1 = jnp.roll( f, (1, 1), axis=axis_pair) 
        f2 = jnp.roll( f, (1,-1), axis=axis_pair) 

        # Uni-axis interactions
        energy += j3 * jnp.sum(f * (f1 + f2))
        energy += (j4 - j3) * jnp.sum(f[..., 3 - axis_pair[0] - axis_pair[1]] * (f1 + f2)[..., 3 - axis_pair[0] - axis_pair[1]])

        # Orthogonal-axis interactions
        energy += j5 * jnp.sum(f[..., [axis_pair[0], axis_pair[1]]] * (f1 - f2)[..., [axis_pair[1], axis_pair[0]]])

    return energy

def get_short_range_3rdnn_isotropic():
    """
    Returns the engine of short-range interaction of third nearest neighbors for a R^3 field defined on a lattice with periodic boundary conditions.

    Returns
    -------
    function
        The interaction function
    """
    c_0 = jnp.eye(3)
    c_1 = jnp.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        ])
    c_2 = jnp.array([
        [ 0, -1,  1],
        [-1,  0, -1],
        [ 1, -1,  0],
        ])    
    c_3 = jnp.array([
        [ 0, -1, -1],
        [-1,  0,  1],
        [-1,  1,  0],
        ])
    c_4 = jnp.array([
        [ 0,  1, -1],
        [ 1,  0, -1],
        [-1, -1,  0],
        ])
    def short_range_3rdnn_isotropic(field, parameters):
        """
        Returns the short-range interaction of nearest neighbors for a R^3 field defined on a lattice with periodic boundary conditions.

        Parameters
        ----------
        field : jnp.array
            The field to calculate the energy
        parameters : jax.numpy array
            The parameters of the energy function

        Returns
        -------
        jnp.array
            The energy of the field
        """
        j6 = parameters[0]  ## uni-axis interaction 
        j7 = parameters[1]  ## orthogonal-axis interaction

        ## get R_ij_alpha * R_ij_beta for different i-j displacement
        r_1 = c_1 * j7 + c_0 * j6
        r_2 = c_2 * j7 + c_0 * j6
        r_3 = c_3 * j7 + c_0 * j6
        r_4 = c_4 * j7 + c_0 * j6

        ## sum up the interaction
        f = field
        fr_sum = jnp.dot( jnp.roll( f, ( 1, 1, 1), axis=(0,1,2)), r_1)
        fr_sum += jnp.dot( jnp.roll( f, ( 1,-1, 1), axis=(0,1,2)), r_2)
        fr_sum += jnp.dot( jnp.roll( f, (-1, 1, 1), axis=(0,1,2)), r_3)
        fr_sum += jnp.dot( jnp.roll( f, (-1,-1, 1), axis=(0,1,2)), r_4)
        energy = jnp.sum(f * fr_sum)
        return energy
    return short_range_3rdnn_isotropic

def homo_strain_dipole_interaction(global_strain, dipole_field, parameters):
    """
    Returns the homogeneous strain dipole interaction energy.

    Parameters
    ----------
    global_strain : jnp.array
        Shape=(6), the global strain of a supercell
    dipole_field : jnp.array
        Shape=(nx, ny, nz, 3), the dipole field
    parameters : jax.numpy array
        The parameters of the energy function containing:
        B1xx : float
            Elastic constant B1xx
        B1yy : float
            Elastic constant B1yy
        B4yz : float
            Elastic constant B4yz

    Returns
    -------
    jnp.array
        The homogeneous strain dipole interaction energy
    """
    B1xx, B1yy, B4yz = parameters
    
    gs = global_strain
    
    # Calculate coef_mat directly without creating B_tensor
    coef_mat = jnp.array([
        [B1xx*gs[0] + B1yy*(gs[1]+gs[2]), B4yz*gs[5], B4yz*gs[4]],
        [B4yz*gs[5], B1xx*gs[1] + B1yy*(gs[0]+gs[2]), B4yz*gs[3]],
        [B4yz*gs[4], B4yz*gs[3], B1xx*gs[2] + B1yy*(gs[0]+gs[1])]
    ])
    
    f = dipole_field 
    # energy = 0.5 * jnp.sum(jnp.dot(f, coef_mat) * f)
    energy =  coef_mat[0,0] * (f[...,0]**2).sum() * 0.5
    energy +=  coef_mat[1,1] * (f[...,1]**2).sum() * 0.5
    energy +=  coef_mat[2,2] * (f[...,2]**2).sum() * 0.5
    energy +=  coef_mat[0,1] * (f[...,0] * f[...,1]).sum()
    energy +=  coef_mat[0,2] * (f[...,0] * f[...,2]).sum()
    energy +=  coef_mat[1,2] * (f[...,1] * f[...,2]).sum()
    return energy

def get_inhomo_strain_dipole_interaction(enable_jit=True):
    """
    Returns the inhomogeneous strain dipole interaction function.

    Parameters
    ----------
    enable_jit : bool, optional
        Whether to enable JIT compilation, by default True

    Returns
    -------
    function
        The interaction function
    """
    get_local_strain = jit(LocalStrain3D.get_local_strain) if enable_jit else LocalStrain3D.get_local_strain
    def inhomo_strain_dipole_interaction(local_displacement, dipole_field, parameters):
        """
        Returns the inhomogeneous strain dipole interaction energy.

        Parameters
        ----------
        local_displacement : jnp.array
            Shape=(nx, ny, nz, 3), the local displacement field
        dipole_field : jnp.array
            Shape=(nx, ny, nz, 3), the dipole field
        parameters : jax.numpy array
            The parameters of the energy function containing:
            B1xx : float
                Elastic constant B1xx
            B1yy : float
                Elastic constant B1yy
            B4yz : float
                Elastic constant B4yz

        Returns
        -------
        jnp.array
            The inhomogeneous strain dipole interaction energy
        """
        B1xx, B1yy, B4yz = parameters
        ls = get_local_strain(local_displacement)  # (l1, l2, l3, 6)
        dp = dipole_field # (l1, l2, l3, 3)
        energy = ((B1xx * ls[...,0] + B1yy * (ls[...,1] + ls[...,2])) * dp[...,0]**2 ).sum()
        energy += ((B1xx * ls[...,1] + B1yy * (ls[...,0] + ls[...,2])) * dp[...,1]**2 ).sum()
        energy += ((B1xx * ls[...,2] + B1yy * (ls[...,0] + ls[...,1])) * dp[...,2]**2 ).sum()
        energy += 2 * (B4yz * ls[...,5] * dp[...,0] * dp[...,1]).sum()
        energy += 2 * (B4yz * ls[...,4] * dp[...,0] * dp[...,2]).sum()
        energy += 2 * (B4yz * ls[...,3] * dp[...,1] * dp[...,2]).sum()
        energy *= 0.5
        return energy
    return inhomo_strain_dipole_interaction

def dipole_efield_interaction(field, parameters):
    """
    Returns the dipole-electric field interaction energy.

    Parameters
    ----------
    field : jnp.array
        The field to calculate the energy
    parameters : jax.numpy array
        The parameters of the energy function

    Returns
    -------
    jnp.array
        The interaction energy
    """
    efield = parameters[:3]
    energy = - jnp.sum(efield * field)
    return energy

# def homo_strain_dipole_interaction(global_strain, dipole_field, parameters ):
#     """
#     Returns the homogeneous strain dipole interaction energy.
    
#     Args:
#         global_strain: jnp.array, shape=(6), the global strain of a supercell
#         dipole_field: jnp.array, shape=(nx, ny, nz, 3), the dipole field
#         parameters: jax.numpy array, the parameters of the energy function
#             'B1xx': float, elastic constant B1xx
#             'B1yy': float, elastic constant B1yy
#             'B4yz': float, elastic constant B4yz
#             'offset': float, offset of the dipole field

#     Returns:
#         jnp.array, the homogeneous strain dipole interaction energy
#     """
#     B1xx, B1yy, B4yz, offset = parameters
    
#     gs = global_strain
#     B1 = jnp.diag(jnp.array([B1xx, B1yy, B1yy]))
#     B2 = jnp.diag(jnp.array([B1yy, B1xx, B1yy]))
#     B3 = jnp.diag(jnp.array([B1yy, B1yy, B1xx]))
#     B4 = jnp.array([
#         [0,   0,  0],
#         [0,   0, B4yz],
#         [0,  B4yz,  0],
#     ])
#     B5 = jnp.array([
#         [0,   0, B4yz],
#         [0,   0,  0],
#         [B4yz,  0,  0],
#     ])
#     B6 = jnp.array([
#         [0,   B4yz, 0],
#         [B4yz,  0,  0],
#         [0,   0,  0],
#     ])
#     B_tensor = jnp.stack([B1, B2, B3, B4, B5, B6], axis=-1)  # (3,3,6)
    
#     ### get the homogeneous strain energy
#     coef_mat = (B_tensor * gs).sum(-1)
#     f = dipole_field - offset
#     energy = 0.5 * jnp.sum(jnp.dot(f, coef_mat) * f)   # There is an unexpected loss of precision when using float32... Why?  
#     return energy



# def inhomo_strain_dipole_interaction_old(local_displacement, dipole_field, parameters):
#     """
#     Returns the inhomogeneous strain dipole interaction energy.
    
#     Args:
#         local_displacement: jnp.array, shape=(nx, ny, nz, 3), the local displacement field
#         dipole_field: jnp.array, shape=(nx, ny, nz, 3), the dipole field
#         parameters: jax.numpy array, the parameters of the energy function
#             'B1xx': float, elastic constant B1xx
#             'B1yy': float, elastic constant B1yy
#             'B4yz': float, elastic constant B4yz

#     Returns:
#         jnp.array, the inhomogeneous strain dipole interaction energy
#     """
#     B1xx, B1yy, B4yz = parameters
    
#     B1 = jnp.diag(jnp.array([B1xx, B1yy, B1yy]))
#     B2 = jnp.diag(jnp.array([B1yy, B1xx, B1yy]))
#     B3 = jnp.diag(jnp.array([B1yy, B1yy, B1xx]))
#     B4 = jnp.array([
#         [0,   0,  0],
#         [0,   0, B4yz],
#         [0,  B4yz,  0],
#     ])
#     B5 = jnp.array([
#         [0,   0, B4yz],
#         [0,   0,  0],
#         [B4yz,  0,  0],
#     ])
#     B6 = jnp.array([
#         [0,   B4yz, 0],
#         [B4yz,  0,  0],
#         [0,   0,  0],
#     ])
#     B_tensor = jnp.stack([B1, B2, B3, B4, B5, B6], axis=1)  # (3,6,3)
    
#     ### get the inhomogeneous strain energy
#     local_strain = LocalStrain3D.get_local_strain(local_displacement)  # (l1, l2, l3, 6)
#     local_strain = jnp.dot(local_strain, B_tensor) # (l1, l2, l3, 3,3 )

#     f = dipole_field 
#     energy = 0.5 * jnp.sum(local_strain * f[:,:,:,None,:] * f[:,:,:,:,None])

#     return energy
