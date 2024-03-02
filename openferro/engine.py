"""
Functions that define a term in the Hamiltonian. They will be added into <class interaction> for automatic differentiation.
"""
# This file is part of OpenFerro.


import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from openferro.field import LocalStrain


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

def short_range_1stnn_isotropic_scalar(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a R^3 field defined on a lattice with periodic boundary conditions.
    """
    j = parameters['j']
    offset = parameters['offset']

    f = field - offset
    f_0 = jnp.roll( f, 1, axis=0) 
    f_1 = jnp.roll( f, 1, axis=1)
    f_2 = jnp.roll( f, 1, axis=2)
    energy = jnp.sum( j * f * (f_0+f_1+f_2))  
    return energy

def short_range_1stnn_isotropic(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a R^3 field defined on a isotropic lattice with periodic boundary conditions.
    """
    j1 = parameters['j1']  ## uni-axis interaction orthogonal to displacement direction
    j2 = parameters['j2']  ## uni-axis interaction along displacement direction

    offset = parameters['offset']

    f = field - offset
    f_0 = jnp.roll( f, 1, axis=0) 
    f_1 = jnp.roll( f, 1, axis=1)
    f_2 = jnp.roll( f, 1, axis=2)

    energy  = jnp.sum( j1 * f * (f_0+f_1+f_2))
    energy += jnp.sum( (j2-j1) * f[...,0] * f_0[...,0])
    energy += jnp.sum( (j2-j1) * f[...,1] * f_1[...,1])
    energy += jnp.sum( (j2-j1) * f[...,2] * f_2[...,2])

    return energy

def short_range_1stnn_anisotropic(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a R^3 field defined on a anisotropic lattice with periodic boundary conditions.
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


def short_range_2ednn_isotropic(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a R^3 field defined on a lattice with periodic boundary conditions.
    """
    j3 = parameters['j3']  ## uni-axis interaction parallel to displacement plane
    j4 = parameters['j4']  ## uni-axis interaction orthogonal to displacement plane
    j5 = parameters['j5']  ## orthogonal-axis interaction on displacement plane
    offset = parameters['offset']

    f = field - offset
    fxy_1 = jnp.roll( f, (1, 1), axis=(0,1)) 
    fxy_2 = jnp.roll( f, (1,-1), axis=(0,1)) 
    fxz_1 = jnp.roll( f, (1, 1), axis=(0,2)) 
    fxz_2 = jnp.roll( f, (1,-1), axis=(0,2)) 
    fyz_1 = jnp.roll( f, (1, 1), axis=(1,2)) 
    fyz_2 = jnp.roll( f, (1,-1), axis=(1,2)) 
    ## uni-axis
    energy = jnp.sum( j3 * f * (fxy_1 + fxy_2 + fxz_1 + fxz_2 + fyz_1 + fyz_2 ) ) 
    energy += jnp.sum( (j4-j3) * f[..., 2] * (fxy_1 + fxy_2)[...,2] )
    energy += jnp.sum( (j4-j3) * f[..., 1] * (fxz_1 + fxz_2)[...,1] )
    energy += jnp.sum( (j4-j3) * f[..., 0] * (fyz_1 + fyz_2)[...,0] )
    ## orthogonal-axis
    energy += jnp.sum( j5 * f[..., [0,1]] * (fxy_1 - fxy_2)[...,[1,0]] )
    energy += jnp.sum( j5 * f[..., [0,2]] * (fxz_1 - fxz_2)[...,[2,0]] )
    energy += jnp.sum( j5 * f[..., [1,2]] * (fyz_1 - fyz_2)[...,[2,1]] )
    return energy

def short_range_3rdnn_isotropic(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a R^3 field defined on a lattice with periodic boundary conditions.
    """
    j6 = parameters['j6']  ## uni-axis interaction 
    j7 = parameters['j7']  ## orthogonal-axis interaction
    offset = parameters['offset']

    f = field - offset
    f_1 = jnp.roll( f, ( 1, 1, 1), axis=(0,1,2))
    f_2 = jnp.roll( f, ( 1,-1, 1), axis=(0,1,2))
    f_3 = jnp.roll( f, (-1, 1, 1), axis=(0,1,2))
    f_4 = jnp.roll( f, (-1,-1, 1), axis=(0,1,2))
    ## get R_ij_alpha * R_ij_beta for different i-j displacement
    r_1 = jnp.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        ])
    r_2 = jnp.array([
        [ 0, -1,  1],
        [-1,  0, -1],
        [ 1, -1,  0],
        ])    
    r_3 = jnp.array([
        [ 0, -1, -1],
        [-1,  0,  1],
        [-1,  1,  0],
        ])
    r_4 = jnp.array([
        [ 0,  1, -1],
        [ 1,  0, -1],
        [-1, -1,  0],
        ])
    r_1 = r_1 * j7 + jnp.eye(3) * j6
    r_2 = r_2 * j7 + jnp.eye(3) * j6
    r_3 = r_3 * j7 + jnp.eye(3) * j6
    r_4 = r_4 * j7 + jnp.eye(3) * j6
    fr_sum = jnp.dot(f_1, r_1) + jnp.dot(f_2, r_2) + jnp.dot(f_3, r_3) + jnp.dot(f_4, r_4)
    energy = jnp.sum(f * fr_sum)
    # energy = jnp.sum( j6 * f * (f_1 + f_2 + f_3 + f_4) )
    # fr_sum = jnp.dot(f_1, r_1) + jnp.dot(f_2, r_2) + jnp.dot(f_3, r_3) + jnp.dot(f_4, r_4)
    # energy += jnp.sum( j7 * f * fr_sum )
    return energy

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
    N = local_displacement.shape[0] * local_displacement.shape[1] * local_displacement.shape[2]
    homo_elastic_energy = 0.5 * B11 * jnp.sum(gs[:3]**2)
    homo_elastic_energy += B12 * (gs[0]*gs[1]+gs[1]*gs[2]+gs[2]*gs[0])
    homo_elastic_energy += 0.5 * B44 * jnp.sum(gs[3:]**2)
    homo_elastic_energy *= N
    
    ## get the inhomogeneous strain energy
    ls = local_displacement
    grad_0 = ls - jnp.roll( ls, 1, axis=0)     # v(R)-v(R-x)
    grad_1 = ls - jnp.roll( ls, 1, axis=1)     # v(R)-v(R-y)
    grad_2 = ls - jnp.roll( ls, 1, axis=2)     # v(R)-v(R-z)
    
    vxx_m = grad_0[...,0]
    vxx_p = - jnp.roll(vxx_m, -1, axis=0)
    vyy_m = grad_1[...,1]
    vyy_p = - jnp.roll(vyy_m, -1, axis=1)
    vzz_m = grad_2[...,2]
    vzz_p = - jnp.roll(vzz_m, -1, axis=2)
    inhomo_elastic_energy = 2 * g11 * (jnp.sum(vxx_m**2) + jnp.sum(vyy_m**2) + jnp.sum(vzz_m**2))
    inhomo_elastic_energy +=  g12 * jnp.sum((vxx_m + vxx_p) * (vyy_m + vyy_p))
    inhomo_elastic_energy +=  g12 * jnp.sum((vzz_m + vzz_p) * (vyy_m + vyy_p))
    inhomo_elastic_energy +=  g12 * jnp.sum((vxx_m + vxx_p) * (vzz_m + vzz_p))
    
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
    offset = parameters['offset']
    gs = global_strain.flatten()
    B1 = jnp.diag(jnp.array([B1xx, B1yy, B1yy]))
    B2 = jnp.diag(jnp.array([B1yy, B1xx, B1yy]))
    B3 = jnp.diag(jnp.array([B1yy, B1yy, B1xx]))
    B4 = jnp.array([
        [0,   0,  0],
        [0,   0, B4yz],
        [0,  B4yz,  0],
    ])
    B5 = jnp.array([
        [0,   0, B4yz],
        [0,   0,  0],
        [B4yz,  0,  0],
    ])
    B6 = jnp.array([
        [0,   B4yz, 0],
        [B4yz,  0,  0],
        [0,   0,  0],
    ])
    B_tensor = jnp.stack([B1, B2, B3, B4, B5, B6], axis=-1)
    
    ### get the homogeneous strain energy
    coef_mat = (B_tensor * gs).sum(-1)
    f = dipole_field - offset
    energy = 0.5 * jnp.sum(jnp.dot(f, coef_mat) * f)
    return energy

def inhomo_strain_dipole_interaction(local_displacement, dipole_field, parameters):
    B1xx = parameters['B1xx']
    B1yy = parameters['B1yy']
    B4yz = parameters['B4yz']
    offset = parameters['offset']

    B1 = jnp.diag(jnp.array([B1xx, B1yy, B1yy]))
    B2 = jnp.diag(jnp.array([B1yy, B1xx, B1yy]))
    B3 = jnp.diag(jnp.array([B1yy, B1yy, B1xx]))
    B4 = jnp.array([
        [0,   0,  0],
        [0,   0, B4yz],
        [0,  B4yz,  0],
    ])
    B5 = jnp.array([
        [0,   0, B4yz],
        [0,   0,  0],
        [B4yz,  0,  0],
    ])
    B6 = jnp.array([
        [0,   B4yz, 0],
        [B4yz,  0,  0],
        [0,   0,  0],
    ])
    B_tensor = jnp.stack([B1, B2, B3, B4, B5, B6], axis=1)  # (3,6,3)
    
    ### get the inhomogeneous strain energy
    local_strain = LocalStrain.get_local_strain(local_displacement)  # (l1, l2, l3, 6)
    local_strain = jnp.dot(local_strain, B_tensor) # (l1, l2, l3, 3,3 )

    f = dipole_field - offset
    energy = 0.5 * jnp.sum(local_strain * f[:,:,:,None,:] * f[:,:,:,:,None])

    return energy