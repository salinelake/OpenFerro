"""
Functions that define a term in the Hamiltonian. They will be added into <class interaction> for automatic differentiation.
"""
# This file is part of OpenFerro.


import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from field import LocalStrain


def self_energy_onsite_isotropic(field, parameters):
    """
    Returns the isotropic self-energy of a 3D field.
    See Eq.(2-3) in [Zhong, W., David Vanderbilt, and K. M. Rabe. Physical Review B 52.9 (1995): 6301.] for meaning of the parameters.
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


def self_energy_onsite_scalar(field, parameters):
    """
    Returns the self-energy of a scalar field. 
    E= \sum_i E_i.  (sum over the lattice sites i)
    E_i = k_2 * u_i^2 + alpha * u_i^4 
    """
    k2 = parameters['k2']
    alpha = parameters['alpha']
    offset = parameters['offset']

    energy = k2 * jnp.sum((field-offset)**2)
    energy += alpha * jnp.sum((field-offset)**4 )
    return energy

def short_range_1stnn(field, parameters):
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

def short_range_1stnn_scalar(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a R^3 field defined on a lattice with periodic boundary conditions.
    """
    J_1 = parameters['J_1']
    J_2 = parameters['J_2']
    J_3 = parameters['J_3']
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

def homo_elastic_energy(global_strain, parameters):
    """
    Returns the homogeneous elastic energy of a strain field.
    """
    B11 = parameters['B11'] 
    B12 = parameters['B12']
    B44 = parameters['B44']
    N = parameters['N']

    ## get the homogeneous strain energy 
    gs = global_strain.flatten()
    homo_elastic_energy = 0.5 * B11 * jnp.sum(gs[:3]**2)
    homo_elastic_energy += B12 * (gs[0]*gs[1]+gs[1]*gs[2]+gs[2]*gs[0])
    homo_elastic_energy += 0.5 * B44 * jnp.sum(gs[3:]**2)
    homo_elastic_energy *= N
    return homo_elastic_energy

def elastic_energy(local_displacement, global_strain, parameters):
    """
    Returns the elastic energy of a strain field.
    """
    B11 = parameters['B11'] 
    B12 = parameters['B12']
    B44 = parameters['B44']
    g11 = B11 / 4
    g12 = B12 / 8
    g44 = B44 / 8

    ## get the homogeneous strain energy 
    gs = global_strain.flatten()
    N = local_strain.shape[0] * local_strain.shape[1] * local_strain.shape[2]
    homo_elastic_energy = 0.5 * B11 * jnp.sum(gs[:3]**2)
    homo_elastic_energy += B12 * (gs[0]*gs[1]+gs[1]*gs[2]+gs[2]*gs[0])
    homo_elastic_energy += 0.5 * B44 * jnp.sum(gs[3:]**2)
    homo_elastic_energy *= N
    
    ## get the inhomogeneous strain energy
    ls = local_displacement
    grad_0 = ls - jnp.roll( ls, 1, axis=0)     # v(R)-v(R-x)
    grad_1 = ls - jnp.roll( ls, 1, axis=1)     # v(R)-v(R-y)
    grad_2 = ls - jnp.roll( ls, 1, axis=2)     # v(R)-v(R-z)

    inhomo_elastic_energy = 2 * g11 * (jnp.sum(grad_0[...,0]**2) + jnp.sum(grad_1[...,1]**2) + jnp.sum(grad_2[...,2]**2))
    inhomo_elastic_energy +=  g12 * (grad_0[...,0] - jnp.roll(grad_0[...,0], -1, axis=0)) * (grad_1[...,1] - jnp.roll(grad_1[...,1], -1, axis=1))
    inhomo_elastic_energy +=  g12 * (grad_2[...,2] - jnp.roll(grad_2[...,2], -1, axis=2)) * (grad_1[...,1] - jnp.roll(grad_1[...,1], -1, axis=1))
    inhomo_elastic_energy +=  g12 * (grad_0[...,0] - jnp.roll(grad_0[...,0], -1, axis=0)) * (grad_2[...,2] - jnp.roll(grad_2[...,2], -1, axis=2))
    
    vyx_m = grad_1[...,0]
    vyx_p = - jnp.roll(vyx_m, -1, axis=1)
    vxy_m = grad_0[...,1]
    vxy_p = - jnp.roll(vxy_m, -1, axis=0)
    inhomo_elastic_energy +=  g44 * jnp.sum((vyx_m+vxy_m)**2)
    inhomo_elastic_energy +=  g44 * jnp.sum((vyx_p+vxy_m)**2)
    inhomo_elastic_energy +=  g44 * jnp.sum((vyx_m+vxy_p)**2)
    inhomo_elastic_energy +=  g44 * jnp.sum((vyx_p+vxy_p)**2)

    vzx_m = grad_2[...,0]
    vzx_p = - jnp.roll(vzx_m, -1, axis=2)
    vxz_m = grad_0[...,2]
    vxz_p = - jnp.roll(vxz_m, -1, axis=0)
    inhomo_elastic_energy +=  g44 * jnp.sum((vzx_m+vxz_m)**2)
    inhomo_elastic_energy +=  g44 * jnp.sum((vzx_p+vxz_m)**2)
    inhomo_elastic_energy +=  g44 * jnp.sum((vzx_m+vxz_p)**2)
    inhomo_elastic_energy +=  g44 * jnp.sum((vzx_p+vxz_p)**2)
    
    vyz_m = grad_1[...,2]
    vyz_p = - jnp.roll(vyz_m, -1, axis=1)
    vzy_m = grad_2[...,1]
    vzy_p = - jnp.roll(vzy_m, -1, axis=2)
    inhomo_elastic_energy +=  g44 * jnp.sum((vyz_m+vzy_m)**2)
    inhomo_elastic_energy +=  g44 * jnp.sum((vyz_p+vzy_m)**2)
    inhomo_elastic_energy +=  g44 * jnp.sum((vyz_m+vzy_p)**2)
    inhomo_elastic_energy +=  g44 * jnp.sum((vyz_p+vzy_p)**2)
    
    return homo_elastic_energy + inhomo_elastic_energy


def homo_strain_dipole_interaction(global_strain, dipole_field, parameters ):
    B1xx = parameters['B1xx']
    B1yy = parameters['B1yy']
    B4yz = parameters['B4yz']
    gs = global_strain.flatten()
    B1 = jnp.diag(jnp.array([B1xx, B1yy, B1yy]))
    B2 = jnp.diag(jnp.array([B1yy, B1xx, B1yy]))
    B3 = jnp.diag(jnp.array([B1yy, B1yy, B1xx]))
    B4 = jnp.array(
        [0,   0,   0],
        [0,   0,B4yz],
        [0,B4yz,   0],
    )
    B5 = jnp.array(
        [0,   0, B4yz],
        [0,   0,    0],
        [B4yz,0,    0],
    )
    B6 = jnp.array(
        [0,   B4yz, 0],
        [B4yz,   0, 0],
        [   0,   0, 0],
    )
    B_tensor = jnp.stack([B1, B2, B3, B4, B5, B6], axis=-1)
    
    ### get the homogeneous strain energy
    coef_mat = (B_tensor * gs).sum(-1)
    energy = 0.5 * jnp.sum(jnp.dot(dipole_field, coef_mat) * dipole_field)
    return energy

def inhomo_strain_dipole_interaction(local_displacement, dipole_field, parameters):
    B1xx = parameters['B1xx']
    B1yy = parameters['B1yy']
    B4yz = parameters['B4yz']
    gs = global_strain.flatten()
    B1 = jnp.diag(jnp.array([B1xx, B1yy, B1yy]))
    B2 = jnp.diag(jnp.array([B1yy, B1xx, B1yy]))
    B3 = jnp.diag(jnp.array([B1yy, B1yy, B1xx]))
    B4 = jnp.array(
        [0,   0,   0],
        [0,   0,B4yz],
        [0,B4yz,   0],
    )
    B5 = jnp.array(
        [0,   0, B4yz],
        [0,   0,    0],
        [B4yz,0,    0],
    )
    B6 = jnp.array(
        [0,   B4yz, 0],
        [B4yz,   0, 0],
        [   0,   0, 0],
    )
    B_tensor = jnp.stack([B1, B2, B3, B4, B5, B6], axis=1)  # (3,6,3)
    
    ### get the inhomogeneous strain energy
    local_strain = LocalStrain.get_local_strain(local_displacement)  # (l1, l2, l3, 6)
    local_strain = jnp.dot(local_strain, B_tensor) # (l1, l2, l3, 3,3 )

    energy = 0.5 * jnp.sum(local_strain * dipole_field[:,:,:,None,:] * dipole_field[:,:,:,:,None])

    return energy