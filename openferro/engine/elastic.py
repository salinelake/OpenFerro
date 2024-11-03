"""
Functions that define a term in the elastic energy. They will be added into <class interaction> for automatic differentiation.
"""
# This file is part of OpenFerro.

import numpy as np
import jax.numpy as jnp

def homo_elastic_energy(global_strain, parameters):
    """
    Returns the homogeneous elastic energy of a strain field.
    
    Args:
        global_strain: jnp.array, shape=(6), the global strain of a supercell
        parameters: jax.numpy array, the parameters of the energy function
    
    Returns:
        jnp.array, the homogeneous elastic energy
    """
    # B11 = parameters['B11'] 
    # B12 = parameters['B12']
    # B44 = parameters['B44']
    # N = parameters['N']
    B11, B12, B44, N = parameters

    ## get the homogeneous strain energy 
    gs = global_strain
    homo_elastic_energy = 0.5 * B11 * jnp.sum(gs[:3]**2)
    homo_elastic_energy += B12 * (gs[0]*gs[1]+gs[1]*gs[2]+gs[2]*gs[0])
    homo_elastic_energy += 0.5 * B44 * jnp.sum(gs[3:]**2)
    homo_elastic_energy *= N
    return homo_elastic_energy

def pV_energy(global_strain, parameters):
    """
    Returns pressure * (volume - reference volume)

    Args:
        global_strain: jnp.array, shape=(6), the global strain of a supercell
        parameters: jax.numpy array, the parameters of the energy function
    
    Returns:
        jnp.array, the pV energy
    """
    # pres = parameters['p']
    # vol_ref = parameters['V0']
    pres, vol_ref = parameters
    
    gs = global_strain
    pV = ( gs[:3].sum()) * pres * vol_ref
    return pV

def elastic_energy(local_displacement, global_strain, parameters):
    """
    Returns the elastic energy of a strain field.

    Args: 
        local_displacement: jnp.array, shape=(nx, ny, nz, 3), the local displacement field
        global_strain: jnp.array, shape=(6), the global strain of a supercell
        parameters: dict, the parameters of the energy function containing:
            'B11': float, elastic constant B11
            'B12': float, elastic constant B12
            'B44': float, elastic constant B44

    Returns:
        jnp.array, the total elastic energy (homogeneous + inhomogeneous)
    """
    # B11 = parameters['B11'] 
    # B12 = parameters['B12']
    # B44 = parameters['B44']
    B11, B12, B44 = parameters
    g11 = B11 / 4
    g12 = B12 / 8
    g44 = B44 / 8

    ## get the homogeneous strain energy 
    gs = global_strain
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
