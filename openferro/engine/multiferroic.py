"""derroic energy.

These functions will be added into <class interaction> for automatic differentiation.

Notes
-----
This file is part of OpenFerro.
"""
import jax.numpy as jnp
from jax import jit
from openferro.field import LocalStrain3D

##########################################################################################
## Utilities
##########################################################################################

def mean_field_on_interwaving_sublattice(field):
    """
    Returns the mean-field of a field on an interwaving sublattice. Say we have two cubic sublattices A and B. We have a field defined on sublattice A. 
    We want to calculate the mean-field (averaged over the 8 nearest neighbors of a B-site) of this field on sublattice B. 

    Let the lattice constant be unit length. We assume that the origin of sublattice B is at (0,0,0) and the origin of sublattice A is at (-0.5,-0.5,-0.5). 
    So (0,0,0) and (-0.5,-0.5,-0.5) will make a primitive cell. We will adopt this convention throughout this file. 
    Note such a convention is not unique. It is just for consistency among the energy engines defined for multiferroics, to avoid confusion over the annoying amounts of jnp.roll.
    
    Parameters
    ----------
    field : jnp.array
        The field to calculate the mean-field

    Returns
    -------
    jnp.array
        The mean-field of the field
    """
    mean_field = field + jnp.roll(field, [1,0,0], axis=[0,1,2])
    mean_field += jnp.roll(field, [0,-1,0], axis=[0,1,2])
    mean_field += jnp.roll(field, [0,0,-1], axis=[0,1,2])
    mean_field += jnp.roll(field, [1,-1,0], axis=[0,1,2])
    mean_field += jnp.roll(field, [1,0,-1], axis=[0,1,2])
    mean_field += jnp.roll(field, [0,-1,-1], axis=[0,1,2])
    mean_field += jnp.roll(field, [1,-1,-1], axis=[0,1,2])
    mean_field /= 8

    return mean_field

def mean_field_1stnn_on_single_lattice(field):
    """
    Returns the average of a field over the 6 nearest neighbors of a given site, on a cubic lattice.
    """
    mean_field = jnp.roll(field, [1,0,0], axis=[0,1,2])
    mean_field += jnp.roll(field, [-1,0,0], axis=[0,1,2])
    mean_field += jnp.roll(field, [0,1,0], axis=[0,1,2])
    mean_field += jnp.roll(field, [0,-1,0], axis=[0,1,2])
    mean_field += jnp.roll(field, [0,0,1], axis=[0,1,2])
    mean_field += jnp.roll(field, [0,0,-1], axis=[0,1,2])
    mean_field /= 6
    return mean_field

def mean_field_2ndnn_on_single_lattice(field):
    """
    Returns the average of a field over the 12 nearest neighbors of a given site, on a cubic lattice.
    """
    mean_field = jnp.roll(field, [1,1,0], axis=[0,1,2])
    mean_field += jnp.roll(field, [1,-1,0], axis=[0,1,2])
    mean_field += jnp.roll(field, [-1,1,0], axis=[0,1,2])
    mean_field += jnp.roll(field, [-1,-1,0], axis=[0,1,2])
    mean_field += jnp.roll(field, [1,0,1], axis=[0,1,2])
    mean_field += jnp.roll(field, [1,0,-1], axis=[0,1,2])
    mean_field += jnp.roll(field, [-1,0,1], axis=[0,1,2])
    mean_field += jnp.roll(field, [-1,0,-1], axis=[0,1,2])
    mean_field += jnp.roll(field, [0,1,1], axis=[0,1,2])
    mean_field += jnp.roll(field, [0,1,-1], axis=[0,1,2])
    mean_field += jnp.roll(field, [0,-1,1], axis=[0,1,2])
    mean_field += jnp.roll(field, [0,-1,-1], axis=[0,1,2])
    mean_field /= 12
    return mean_field

def mean_field_3rdnn_on_single_lattice(field):
    """
    Returns the average of a field over the 8 nearest neighbors of a given site, on a cubic lattice.
    """
    mean_field = jnp.roll(field, [1,1,1], axis=[0,1,2])
    mean_field += jnp.roll(field, [1,-1,1], axis=[0,1,2])
    mean_field += jnp.roll(field, [-1,1,1], axis=[0,1,2])
    mean_field += jnp.roll(field, [-1,-1,1], axis=[0,1,2])
    mean_field += jnp.roll(field, [1,1,-1], axis=[0,1,2])
    mean_field += jnp.roll(field, [1,-1,-1], axis=[0,1,2])
    mean_field += jnp.roll(field, [-1,1,-1], axis=[0,1,2])
    mean_field += jnp.roll(field, [-1,-1,-1], axis=[0,1,2])
    mean_field /= 8
    return mean_field

##########################################################################################
## Energies involving AFD field, strain and dipole field
##########################################################################################

def short_range_1stnn_uniaxial_quartic(field, parameters):
    """
    Returns the short-range interaction of nearest neighbors for a :math:`R^3` field defined on a isotropic lattice with periodic boundary conditions.
    The energy is given by:
    
    .. math::
        \sum_{i, \\alpha} K u^3_{i, \\alpha} (u_{i+\\alpha, \\alpha} + u_{i-\\alpha, \\alpha})


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
    K = parameters[0]  ## uni-axis interaction orthogonal to displacement direction

    f = field
    energy = 0
    for axis in range(3):
        f_shifted = jnp.roll(f[..., axis], 1, axis=axis) + jnp.roll(f[..., axis], -1, axis=axis)
        energy += jnp.sum(f[..., axis]**3 * f_shifted)
    return energy * K

def short_range_1stnn_trilinear_two_sublattices(field_A, field_B, parameters):
    """
    Returns the short-range interaction of nearest neighbors for two :math:`R^3` fields defined on interwaving sublattices (A and B sublattices) with periodic boundary conditions.
    The energy is given by:
    
    .. math::
        \sum_{i \\sim j, \\alpha \\neq \\beta} D_{ij,\\alpha\\beta} a_{j, \\alpha} b_{i, \\alpha} b_{i, \\beta}

    Here i is the index of sublattice A and j is the index of sublattice B. :math:`i\sim j` means that i and j are nearest neighbors. 
    :math:`D_{ij,\\alpha\\beta}` is the trilinear coupling constant.
    For interwaving cubic lattices, there are 8 such nearest neighbors for a given i. 

    Define :math:`a'_{i}` to be the mean-field (averaged over the 8 nearest neighbors of a B-site) of field_A on the B site-i. The energy is given by
    
    .. math::
        \sum_{i, \\alpha \\neq \\beta} 8 D^{\mathrm{nn}}_{\\alpha\\beta} a'_{i, \\alpha} b_{i, \\alpha} b_{i, \\beta}
    
    Parameters
    ----------
    field_A : jnp.array
        The field of sublattice A to calculate the energy
    field_B : jnp.array
        The field of sublattice B to calculate the energy
    parameters : jax.numpy array
        The parameters of the energy function

    Returns
    -------
    jnp.array
        The energy of the field
    """
    D = parameters[0]  ## trilinear interaction
    energy = 0
    ## the mean-field of field_A on a B-site. So we need to average the field_A over the 8 nearest neighbors of a B-site.
    mean_field_A = mean_field_on_interwaving_sublattice(field_A)

    mean_field_A *= field_B
    energy += jnp.sum(mean_field_A * jnp.roll(field_B, 1, axis=-1))
    energy += jnp.sum(mean_field_A * jnp.roll(field_B, -1, axis=-1))
    # energy += jnp.sum(mean_field_A[..., 0] * (field_B[..., 1] + field_B[..., 2]))
    # energy += jnp.sum(mean_field_A[..., 1] * (field_B[..., 2] + field_B[..., 0]))
    # energy += jnp.sum(mean_field_A[..., 2] * (field_B[..., 0] + field_B[..., 1]))
    return energy * D * 8  ## TODO: check if the factor 8 is correct

def short_range_1stnn_biquadratic_iiii_two_sublattices(field_A, field_B, parameters):
    """
    Returns the short-range interaction of nearest neighbors for two :math:`R^3` fields defined on interwaving sublattices (A and B sublattices) with periodic boundary conditions.

    Let the lattice constant be unit length. We assume that the origin of field_B is at (0,0,0) and the origin of field_A is at (-0.5,-0.5,-0.5). 
    So (0,0,0) and (-0.5,-0.5,-0.5) will make a primitive cell. We will adopt this convention implementing the sum over i and j. 
    Note such a convention is not unique. It is just for consistency among the energy engines defined for multiferroics, to avoid confusion over the annoying amounts of jnp.roll.

    Define :math:`a'_{i}` to be the mean-field (averaged over the 8 nearest neighbors of a B-site) of field_A on the B site-i.  
    
    The energy is given by:
    
    .. math::
        \sum_{i, \\alpha \\beta \\gamma \\delta} E_{\\alpha\\beta\\gamma\\delta}  b_{i, \\alpha} b_{i, \\beta} a'_{i, \\gamma} a'_{i, \\delta}  

    Parameters
    ----------
    field_A : jnp.array
        The field of sublattice A to calculate the energy
    field_B : jnp.array
        The field of sublattice B to calculate the energy
    parameters : jax.numpy array
        The parameters of the energy function

    Returns
    -------
    jnp.array
        The energy of the field
    """
    Exxxx = parameters[0]  ## biquadratic interaction
    Exxyy = parameters[1]
    Exyxy = parameters[2]

    energy = 0
    ## the mean-field of field_A on a B-site. So we need to average the field_A over the 8 nearest neighbors of a B-site.
    mean_field_A = mean_field_on_interwaving_sublattice(field_A)

    ## uniaxial contribution
    energy += Exxxx * jnp.sum(field_B**2 * mean_field_A**2)
    ## xxyy, xxzz, yyzz, yyxx, zzxx, zzyy contributions
    energy += Exxyy * jnp.sum(field_B**2 * jnp.roll(mean_field_A**2, 1,axis=-1))
    energy += Exxyy * jnp.sum(field_B**2 * jnp.roll(mean_field_A**2, -1,axis=-1))
    ## xyxy, xzxz, yzyz, yxyx, zxzx, zyzy contributions
    field_AB = field_A * field_B
    energy += Exyxy * jnp.sum(field_AB * jnp.roll(field_AB, 1,axis=-1))
    energy += Exyxy * jnp.sum(field_AB * jnp.roll(field_AB, -1,axis=-1))
    return energy  ## TODO: check if here a prefactor should be applied

##########################################################################################
## Energies involving spin field and all others
##########################################################################################

def short_range_biquadratic_ijii_two_sublattices(field_A, field_B, parameters):
    """
    Returns the short-range interaction of nearest neighbors for two :math:`R^3` fields defined on interwaving sublattices (A and B sublattices) with periodic boundary conditions.

    Let the lattice constant be unit length. We assume that the origin of field_B is at (0,0,0) and the origin of field_A is at (-0.5,-0.5,-0.5). 
    So (0,0,0) and (-0.5,-0.5,-0.5) will make a primitive cell. We will adopt this convention implementing the sum over i and j. 
    Note such a convention is not unique. It is just for consistency among the energy engines defined for multiferroics, to avoid confusion over the annoying amounts of jnp.roll.

    Define :math:`a'_{i}` to be the mean-field (averaged over the 8 nearest neighbors of a B-site) of field_A on the B site-i.  
    
    The energy is given by:
    
    .. math::
        \sum_{ij, \\alpha \\beta \\gamma \\delta} E_{\\alpha\\beta\\gamma\\delta}  b_{i, \\alpha} b_{j, \\beta} a'_{i, \\gamma} a'_{i, \\delta}  

    Parameters
    ----------
    field_A : jnp.array
        The field of sublattice A to calculate the energy
    field_B : jnp.array
        The field of sublattice B to calculate the energy
    parameters : jax.numpy array
        The parameters of the energy function

    Returns
    -------
    jnp.array
        The energy of the field
    """
    ## TODO: check if these parameters needed to be divided by corresponding coordination number. Likely not. 
    E1xxxx = parameters[0] 
    E1xxyy = parameters[1]
    E1xyxy = parameters[2]
    E2xxxx = parameters[3]
    E2xxyy = parameters[4]
    E2xyxy = parameters[5]
    E3xxxx = parameters[6]
    E3xxyy = parameters[7]
    E3xyxy = parameters[8]

    energy = 0
    ## the mean-field of field_A on a B-site. So we need to average the field_A over the 8 nearest neighbors of a B-site.
    mean_field_A = mean_field_on_interwaving_sublattice(field_A)
    field_B_1stnnsum = mean_field_1stnn_on_single_lattice(field_B) * 6
    field_B_2ndnnsum = mean_field_2ndnn_on_single_lattice(field_B) * 12
    field_B_3rdnnsum = mean_field_3rdnn_on_single_lattice(field_B) * 8


    ## uniaxial contribution
    energy += jnp.sum(field_B * (E1xxxx * field_B_1stnnsum + E2xxxx * field_B_2ndnnsum + E3xxxx * field_B_3rdnnsum) * mean_field_A**2)
    ## xxyy, xxzz, yyzz, yyxx, zzxx, zzyy contributions
    energy += jnp.sum(field_B * (E1xxyy * field_B_1stnnsum + E2xxyy * field_B_2ndnnsum + E3xxyy * field_B_3rdnnsum) * jnp.roll(mean_field_A**2, 1,axis=-1))
    energy += jnp.sum(field_B * (E1xxyy * field_B_1stnnsum + E2xxyy * field_B_2ndnnsum + E3xxyy * field_B_3rdnnsum) * jnp.roll(mean_field_A**2, -1,axis=-1))
    ## xyxy, xzxz, yzyz, yxyx, zxzx, zyzy contributions
    energy += jnp.sum(field_B * mean_field_A * jnp.roll((E1xyxy * field_B_1stnnsum + E2xyxy * field_B_2ndnnsum + E3xyxy * field_B_3rdnnsum) * mean_field_A, 1,axis=-1))
    energy += jnp.sum(field_B * mean_field_A * jnp.roll((E1xyxy * field_B_1stnnsum + E2xyxy * field_B_2ndnnsum + E3xyxy * field_B_3rdnnsum) * mean_field_A, -1,axis=-1))
    return energy 


def short_range_biquadratic_ijii(field_A, field_B, parameters):
    """
    Returns the short-range interaction of nearest neighbors for two :math:`R^3` fields defined on a cubic lattice with periodic boundary conditions.

    The energy is given by:
    
    .. math::
        \sum_{ij, \\alpha \\beta \\gamma \\delta} E_{\\alpha\\beta\\gamma\\delta}  b_{i, \\alpha} b_{j, \\beta} a_{i, \\gamma} a_{i, \\delta}  

    j is summed over the 1st, 2nd, 3rd nearest neighbors of i.

    Parameters
    ----------
    field_A : jnp.array
        The field a to calculate the energy
    field_B : jnp.array
        The field b to calculate the energy
    parameters : jax.numpy array
        The parameters of the energy function

    Returns
    -------
    jnp.array
        The energy of the field
    """
    ## TODO: check if these parameters needed to be divided by corresponding coordination number. Likely not. 
    E1xxxx = parameters[0] 
    E1xxyy = parameters[1]
    E1xyxy = parameters[2]
    E2xxxx = parameters[3]
    E2xxyy = parameters[4]
    E2xyxy = parameters[5]
    E3xxxx = parameters[6]
    E3xxyy = parameters[7]
    E3xyxy = parameters[8]

    energy = 0
    ## the mean-field of field_A on a B-site. So we need to average the field_A over the 8 nearest neighbors of a B-site.
    field_B_1stnnsum = mean_field_1stnn_on_single_lattice(field_B) * 6
    field_B_2ndnnsum = mean_field_2ndnn_on_single_lattice(field_B) * 12
    field_B_3rdnnsum = mean_field_3rdnn_on_single_lattice(field_B) * 8


    ## uniaxial contribution
    energy += jnp.sum(field_B * (E1xxxx * field_B_1stnnsum + E2xxxx * field_B_2ndnnsum + E3xxxx * field_B_3rdnnsum) * field_A**2)
    ## xxyy, xxzz, yyzz, yyxx, zzxx, zzyy contributions
    energy += jnp.sum(field_B * (E1xxyy * field_B_1stnnsum + E2xxyy * field_B_2ndnnsum + E3xxyy * field_B_3rdnnsum) * jnp.roll(field_A**2, 1,axis=-1))
    energy += jnp.sum(field_B * (E1xxyy * field_B_1stnnsum + E2xxyy * field_B_2ndnnsum + E3xxyy * field_B_3rdnnsum) * jnp.roll(field_A**2, -1,axis=-1))
    ## xyxy, xzxz, yzyz, yxyx, zxzx, zyzy contributions
    energy += jnp.sum(field_B * field_A * jnp.roll((E1xyxy * field_B_1stnnsum + E2xyxy * field_B_2ndnnsum + E3xyxy * field_B_3rdnnsum) * field_A, 1,axis=-1))
    energy += jnp.sum(field_B * field_A * jnp.roll((E1xyxy * field_B_1stnnsum + E2xyxy * field_B_2ndnnsum + E3xyxy * field_B_3rdnnsum) * field_A, -1,axis=-1))
    return energy 


def homo_strain_spin_interaction(global_strain, spin_field, parameters):
    """
    Returns the homogeneous strain spin interaction energy.

    The energy is given by:
    
    .. math::
        \sum_{i,j,l,\\alpha,\\beta} G_{ij, l\\alpha\\beta} \\eta_{l} m_{i, \\alpha} m_{j, \\beta}

    l is index of strain under Voigt notation.

    When i and j are 1st nearest neighbors, :math:`G_{ij, l\\alpha\\beta} = G_{1nn, l\\alpha\\beta}`
    
    When i and j are 2nd nearest neighbors, :math:`G_{ij, l\\alpha\\beta} = G_{2nn, l\\alpha\\beta}`
    
    When i and j are 3rd nearest neighbors, :math:`G_{ij, l\\alpha\\beta} = G_{3nn, l\\alpha\\beta}`

    Parameters
    ----------
    global_strain : jnp.array
        Shape=(6), the global strain of a supercell
    spin_field : jnp.array
        Shape=(nx, ny, nz, 3), the spin field
    parameters : jax.numpy array
        The parameters of the energy function containing
        G_1nn_1xx : Strain-spin interaction constant G1xx (or Gxxxx) for 1st nearest neighbors
        G_1nn_1yy : Strain-spin interaction constant G1yy (or Gxxyy) for 1st nearest neighbors
        G_1nn_4yz : Strain-spin interaction constant G4yz (or Gxyxy) for 1st nearest neighbors
        G_2nn_1xx : Strain-spin interaction constant G1xx (or Gxxxx) for 2nd nearest neighbors
        G_2nn_1yy : Strain-spin interaction constant G1yy (or Gxxyy) for 2nd nearest neighbors
        G_2nn_4yz : Strain-spin interaction constant G4yz (or Gxyxy) for 2nd nearest neighbors
        G_3nn_1xx : Strain-spin interaction constant G1xx (or Gxxxx) for 3rd nearest neighbors
        G_3nn_1yy : Strain-spin interaction constant G1yy (or Gxxyy) for 3rd nearest neighbors
        G_3nn_4yz : Strain-spin interaction constant G4yz (or Gxyxy) for 3rd nearest neighbors

    Returns
    -------
    jnp.array
        The homogeneous strain dipole interaction energy
    """
    ## in some publication these parameters are named differently: G_1nn_xxxx, G_1nn_xxyy, G_1nn_yzyz
    G_1nn_1xx, G_1nn_1yy, G_1nn_4yz = parameters[:3]
    G_2nn_1xx, G_2nn_1yy, G_2nn_4yz = parameters[3:6]
    G_3nn_1xx, G_3nn_1yy, G_3nn_4yz = parameters[6:9]
    
    gs = global_strain
    f = spin_field 
    f_1stnnsum = mean_field_1stnn_on_single_lattice(f) * 6
    f_2ndnnsum = mean_field_2ndnn_on_single_lattice(f) * 12
    f_3rdnnsum = mean_field_3rdnn_on_single_lattice(f) * 8

    
    # Calculate coef_mat directly without creating B_tensor
    coef_mat_1nn = jnp.array([
        [G_1nn_1xx*gs[0] + G_1nn_1yy*(gs[1]+gs[2]), G_1nn_4yz*gs[5], G_1nn_4yz*gs[4]],
        [G_1nn_4yz*gs[5], G_1nn_1xx*gs[1] + G_1nn_1yy*(gs[0]+gs[2]), G_1nn_4yz*gs[3]],
        [G_1nn_4yz*gs[4], G_1nn_4yz*gs[3], G_1nn_1xx*gs[2] + G_1nn_1yy*(gs[0]+gs[1])]
    ])

    coef_mat_2nn = jnp.array([
        [G_2nn_1xx*gs[0] + G_2nn_1yy*(gs[1]+gs[2]), G_2nn_4yz*gs[5], G_2nn_4yz*gs[4]],
        [G_2nn_4yz*gs[5], G_2nn_1xx*gs[1] + G_2nn_1yy*(gs[0]+gs[2]), G_2nn_4yz*gs[3]],
        [G_2nn_4yz*gs[4], G_2nn_4yz*gs[3], G_2nn_1xx*gs[2] + G_2nn_1yy*(gs[0]+gs[1])]
    ])

    coef_mat_3nn = jnp.array([
        [G_3nn_1xx*gs[0] + G_3nn_1yy*(gs[1]+gs[2]), G_3nn_4yz*gs[5], G_3nn_4yz*gs[4]],
        [G_3nn_4yz*gs[5], G_3nn_1xx*gs[1] + G_3nn_1yy*(gs[0]+gs[2]), G_3nn_4yz*gs[3]],
        [G_3nn_4yz*gs[4], G_3nn_4yz*gs[3], G_3nn_1xx*gs[2] + G_3nn_1yy*(gs[0]+gs[1])]
    ])
    
    ## 1st nearest neighbors
    energy =  coef_mat_1nn[0,0] * (f[...,0] * f_1stnnsum[...,0]).sum()  ## TODO: check if a factor 0.5 should be added for all diagonal terms
    energy +=  coef_mat_1nn[1,1] * (f[...,1] * f_1stnnsum[...,1]).sum()
    energy +=  coef_mat_1nn[2,2] * (f[...,2] * f_1stnnsum[...,2]).sum()
    energy +=  coef_mat_1nn[0,1] * (f[...,0] * f_1stnnsum[...,1]).sum()
    energy +=  coef_mat_1nn[0,2] * (f[...,0] * f_1stnnsum[...,2]).sum()
    energy +=  coef_mat_1nn[1,2] * (f[...,1] * f_1stnnsum[...,2]).sum()

    ## 2nd nearest neighbors
    energy +=  coef_mat_2nn[0,0] * (f[...,0] * f_2ndnnsum[...,0]).sum()  
    energy +=  coef_mat_2nn[1,1] * (f[...,1] * f_2ndnnsum[...,1]).sum()
    energy +=  coef_mat_2nn[2,2] * (f[...,2] * f_2ndnnsum[...,2]).sum()
    energy +=  coef_mat_2nn[0,1] * (f[...,0] * f_2ndnnsum[...,1]).sum()
    energy +=  coef_mat_2nn[0,2] * (f[...,0] * f_2ndnnsum[...,2]).sum()
    energy +=  coef_mat_2nn[1,2] * (f[...,1] * f_2ndnnsum[...,2]).sum()

    ## 3rd nearest neighbors
    energy +=  coef_mat_3nn[0,0] * (f[...,0] * f_3rdnnsum[...,0]).sum()  
    energy +=  coef_mat_3nn[1,1] * (f[...,1] * f_3rdnnsum[...,1]).sum()
    energy +=  coef_mat_3nn[2,2] * (f[...,2] * f_3rdnnsum[...,2]).sum()
    energy +=  coef_mat_3nn[0,1] * (f[...,0] * f_3rdnnsum[...,1]).sum()
    energy +=  coef_mat_3nn[0,2] * (f[...,0] * f_3rdnnsum[...,2]).sum()
    energy +=  coef_mat_3nn[1,2] * (f[...,1] * f_3rdnnsum[...,2]).sum()

    return energy



def homo_strain_spin_1stnn_interaction(global_strain, spin_field, parameters):
    """
    Returns the homogeneous strain spin interaction energy.

    The energy is given by:
    
    .. math::
        \sum_{i,j,l,\\alpha,\\beta} G_{ij, l\\alpha\\beta} \\eta_{l} m_{i, \\alpha} m_{j, \\beta}

    l is index of strain under Voigt notation.

    i and j are 1st nearest neighbors :math:`G_{ij, l\\alpha\\beta} = G_{l\\alpha\\beta}`

    Parameters
    ----------
    global_strain : jnp.array
        Shape=(6), the global strain of a supercell
    spin_field : jnp.array
        Shape=(nx, ny, nz, 3), the spin field
    parameters : jax.numpy array
        The parameters of the energy function containing
        G_1xx : Strain-spin interaction constant G1xx (or Gxxxx) for 1st nearest neighbors
        G_1yy : Strain-spin interaction constant G1yy (or Gxxyy) for 1st nearest neighbors
        G_4yz : Strain-spin interaction constant G4yz (or Gxyxy) for 1st nearest neighbors

    Returns
    -------
    jnp.array
        The homogeneous strain dipole interaction energy
    """
    ## in some publication these parameters are named differently: G_1nn_xxxx, G_1nn_xxyy, G_1nn_yzyz
    G_1xx, G_1yy, G_4yz = parameters[:3]
    gs = global_strain
    f = spin_field 
    f_1stnnsum = mean_field_1stnn_on_single_lattice(f) * 6

    ## 1st nearest neighbors, TODO: check if a factor 0.5 should be added for first three lines below
    energy =  (G_1xx*gs[0] + G_1yy*(gs[1]+gs[2])) * (f[...,0] * f_1stnnsum[...,0]).sum()  
    energy +=  (G_1xx*gs[1] + G_1yy*(gs[0]+gs[2])) * (f[...,1] * f_1stnnsum[...,1]).sum()
    energy +=  (G_1xx*gs[2] + G_1yy*(gs[0]+gs[1])) * (f[...,2] * f_1stnnsum[...,2]).sum()
    energy +=  (G_4yz*gs[5]) * (f[...,0] * f_1stnnsum[...,1]).sum()
    energy +=  (G_4yz*gs[4]) * (f[...,0] * f_1stnnsum[...,2]).sum()
    energy +=  (G_4yz*gs[3]) * (f[...,1] * f_1stnnsum[...,2]).sum()
 
    return energy

def homo_strain_spin_2ndnn_interaction(global_strain, spin_field, parameters):
    """
    Returns the homogeneous strain spin interaction energy.

    The energy is given by:
    
    .. math::
        \sum_{i,j,l,\\alpha,\\beta} G_{ij, l\\alpha\\beta} \\eta_{l} m_{i, \\alpha} m_{j, \\beta}

    l is index of strain under Voigt notation.

    i and j are 2nd nearest neighbors :math:`G_{ij, l\\alpha\\beta} = G_{l\\alpha\\beta}`

    Parameters
    ----------
    global_strain : jnp.array
        Shape=(6), the global strain of a supercell
    spin_field : jnp.array
        Shape=(nx, ny, nz, 3), the spin field
    parameters : jax.numpy array
        The parameters of the energy function containing
        G_1xx : Strain-spin interaction constant G1xx (or Gxxxx) for 2nd nearest neighbors
        G_1yy : Strain-spin interaction constant G1yy (or Gxxyy) for 2nd nearest neighbors
        G_4yz : Strain-spin interaction constant G4yz (or Gxyxy) for 2nd nearest neighbors

    Returns
    -------
    jnp.array
        The homogeneous strain dipole interaction energy
    """
    ## in some publication these parameters are named differently: G_1nn_xxxx, G_1nn_xxyy, G_1nn_yzyz
    G_1xx, G_1yy, G_4yz = parameters[:3]
    gs = global_strain
    f = spin_field 
    f_2ndnnsum = mean_field_2ndnn_on_single_lattice(f) * 12
       
    ## 2nd nearest neighbors, TODO: check if a factor 0.5 should be added for first three lines below
    energy =  (G_1xx*gs[0] + G_1yy*(gs[1]+gs[2])) * (f[...,0] * f_2ndnnsum[...,0]).sum()  
    energy +=  (G_1xx*gs[1] + G_1yy*(gs[0]+gs[2])) * (f[...,1] * f_2ndnnsum[...,1]).sum()
    energy +=  (G_1xx*gs[2] + G_1yy*(gs[0]+gs[1])) * (f[...,2] * f_2ndnnsum[...,2]).sum()
    energy +=  (G_4yz*gs[5]) * (f[...,0] * f_2ndnnsum[...,1]).sum()
    energy +=  (G_4yz*gs[4]) * (f[...,0] * f_2ndnnsum[...,2]).sum()
    energy +=  (G_4yz*gs[3]) * (f[...,1] * f_2ndnnsum[...,2]).sum()
 
    return energy

def homo_strain_spin_3rdnn_interaction(global_strain, spin_field, parameters):
    """
    Returns the homogeneous strain spin interaction energy.

    The energy is given by:
    
    .. math::
        \sum_{i,j,l,\\alpha,\\beta} G_{ij, l\\alpha\\beta} \\eta_{l} m_{i, \\alpha} m_{j, \\beta}

    l is index of strain under Voigt notation.

    i and j are 3rd nearest neighbors :math:`G_{ij, l\\alpha\\beta} = G_{l\\alpha\\beta}`

    Parameters
    ----------
    global_strain : jnp.array
        Shape=(6), the global strain of a supercell
    spin_field : jnp.array
        Shape=(nx, ny, nz, 3), the spin field
    parameters : jax.numpy array
        The parameters of the energy function containing
        G_1xx : Strain-spin interaction constant G1xx (or Gxxxx) for 3rd nearest neighbors
        G_1yy : Strain-spin interaction constant G1yy (or Gxxyy) for 3rd nearest neighbors
        G_4yz : Strain-spin interaction constant G4yz (or Gxyxy) for 3rd nearest neighbors

    Returns
    -------
    jnp.array
        The homogeneous strain dipole interaction energy
    """
    ## in some publication these parameters are named differently: G_1nn_xxxx, G_1nn_xxyy, G_1nn_yzyz
    G_1xx, G_1yy, G_4yz = parameters[:3]
    gs = global_strain
    f = spin_field 
    f_3rdnnsum = mean_field_3rdnn_on_single_lattice(f) * 8
 
    ## 3rd nearest neighbors, TODO: check if a factor 0.5 should be added for first three lines below
    energy =  (G_1xx*gs[0] + G_1yy*(gs[1]+gs[2])) * (f[...,0] * f_3rdnnsum[...,0]).sum()  
    energy +=  (G_1xx*gs[1] + G_1yy*(gs[0]+gs[2])) * (f[...,1] * f_3rdnnsum[...,1]).sum()
    energy +=  (G_1xx*gs[2] + G_1yy*(gs[0]+gs[1])) * (f[...,2] * f_3rdnnsum[...,2]).sum()
    energy +=  (G_4yz*gs[5]) * (f[...,0] * f_3rdnnsum[...,1]).sum()
    energy +=  (G_4yz*gs[4]) * (f[...,0] * f_3rdnnsum[...,2]).sum()
    energy +=  (G_4yz*gs[3]) * (f[...,1] * f_3rdnnsum[...,2]).sum()
 
    return energy


def get_inhomo_strain_spin_interaction(enable_jit=True):
    """
    Returns the inhomogeneous strain spin interaction function.

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
    def inhomo_strain_spin_interaction(local_displacement, spin_field, parameters):
        """
        Returns the inhomogeneous strain spin interaction energy.

        The energy is given by:
        
        .. math::
            \sum_{i,j,l,\\alpha,\\beta} G_{ij, l\\alpha\\beta} \\eta_{i,l} m_{i, \\alpha} m_{j, \\beta}

        l is index of local strain under Voigt notation.

        When i and j are 1st nearest neighbors, :math:`G_{ij, l\\alpha\\beta} = G_{1nn, l\\alpha\\beta}`
        
        When i and j are 2nd nearest neighbors, :math:`G_{ij, l\\alpha\\beta} = G_{2nn, l\\alpha\\beta}`
        
        When i and j are 3rd nearest neighbors, :math:`G_{ij, l\\alpha\\beta} = G_{3nn, l\\alpha\\beta}`

        Parameters
        ----------
        local_displacement : jnp.array
            Shape=(nx, ny, nz, 3), the local displacement field
        spin_field : jnp.array
            Shape=(nx, ny, nz, 3), the spin field
        parameters : jax.numpy array
            The parameters of the energy function containing
            G_1nn_1xx : Strain-spin interaction constant G1xx (or Gxxxx) for 1st nearest neighbors
            G_1nn_1yy : Strain-spin interaction constant G1yy (or Gxxyy) for 1st nearest neighbors
            G_1nn_4yz : Strain-spin interaction constant G4yz (or Gxyxy) for 1st nearest neighbors
            G_2nn_1xx : Strain-spin interaction constant G1xx (or Gxxxx) for 2nd nearest neighbors
            G_2nn_1yy : Strain-spin interaction constant G1yy (or Gxxyy) for 2nd nearest neighbors
            G_2nn_4yz : Strain-spin interaction constant G4yz (or Gxyxy) for 2nd nearest neighbors
            G_3nn_1xx : Strain-spin interaction constant G1xx (or Gxxxx) for 3rd nearest neighbors
            G_3nn_1yy : Strain-spin interaction constant G1yy (or Gxxyy) for 3rd nearest neighbors
            G_3nn_4yz : Strain-spin interaction constant G4yz (or Gxyxy) for 3rd nearest neighbors

        """
        G_1nn_1xx, G_1nn_1yy, G_1nn_4yz = parameters[:3]
        G_2nn_1xx, G_2nn_1yy, G_2nn_4yz = parameters[3:6]
        G_3nn_1xx, G_3nn_1yy, G_3nn_4yz = parameters[6:9]
        ls = get_local_strain(local_displacement)  # (l1, l2, l3, 6)
        f = spin_field
        f_1stnnsum = mean_field_1stnn_on_single_lattice(f) * 6
        f_2ndnnsum = mean_field_2ndnn_on_single_lattice(f) * 12
        f_3rdnnsum = mean_field_3rdnn_on_single_lattice(f) * 8

        energy =0 
        ## 1st nearest neighbors, TODO: check if a factor 0.5 should be added for first three lines below
        energy += ( (G_1nn_1xx * ls[...,0] + G_1nn_1yy * (ls[...,1] + ls[...,2])) * f[...,0] * f_1stnnsum[...,0] ).sum()
        energy += ( (G_1nn_1xx * ls[...,1] + G_1nn_1yy * (ls[...,0] + ls[...,2])) * f[...,1] * f_1stnnsum[...,1] ).sum()
        energy += ( (G_1nn_1xx * ls[...,2] + G_1nn_1yy * (ls[...,0] + ls[...,1])) * f[...,2] * f_1stnnsum[...,2] ).sum()
        energy += (G_1nn_4yz * ls[...,5] * f[...,0] * f_1stnnsum[...,1]).sum()
        energy += (G_1nn_4yz * ls[...,4] * f[...,0] * f_1stnnsum[...,2]).sum()
        energy += (G_1nn_4yz * ls[...,3] * f[...,1] * f_1stnnsum[...,2]).sum()

        ## 2nd nearest neighbors, TODO: check if a factor 0.5 should be added for first three lines below
        energy += ( (G_2nn_1xx * ls[...,0] + G_2nn_1yy * (ls[...,1] + ls[...,2])) * f[...,0] * f_2ndnnsum[...,0] ).sum()
        energy += ( (G_2nn_1xx * ls[...,1] + G_2nn_1yy * (ls[...,0] + ls[...,2])) * f[...,1] * f_2ndnnsum[...,1] ).sum()
        energy += ( (G_2nn_1xx * ls[...,2] + G_2nn_1yy * (ls[...,0] + ls[...,1])) * f[...,2] * f_2ndnnsum[...,2] ).sum()
        energy += (G_2nn_4yz * ls[...,5] * f[...,0] * f_2ndnnsum[...,1]).sum()
        energy += (G_2nn_4yz * ls[...,4] * f[...,0] * f_2ndnnsum[...,2]).sum()
        energy += (G_2nn_4yz * ls[...,3] * f[...,1] * f_2ndnnsum[...,2]).sum()

        ## 3rd nearest neighbors, TODO: check if a factor 0.5 should be added for first three lines below
        energy += ( (G_3nn_1xx * ls[...,0] + G_3nn_1yy * (ls[...,1] + ls[...,2])) * f[...,0] * f_3rdnnsum[...,0] ).sum()
        energy += ( (G_3nn_1xx * ls[...,1] + G_3nn_1yy * (ls[...,0] + ls[...,2])) * f[...,1] * f_3rdnnsum[...,1] ).sum()
        energy += ( (G_3nn_1xx * ls[...,2] + G_3nn_1yy * (ls[...,0] + ls[...,1])) * f[...,2] * f_3rdnnsum[...,2] ).sum()
        energy += (G_3nn_4yz * ls[...,5] * f[...,0] * f_3rdnnsum[...,1]).sum()
        energy += (G_3nn_4yz * ls[...,4] * f[...,0] * f_3rdnnsum[...,2]).sum()
        energy += (G_3nn_4yz * ls[...,3] * f[...,1] * f_3rdnnsum[...,2]).sum()
        return energy
    return inhomo_strain_spin_interaction

def DM_AFD_1stnn(AFD_field, spin_field, parameters):
    """
    Returns the Dzyaloshinskii-Moriya interaction (involving oxygen octahedral titling AFD mode) of nearest neighbors for atomistic spin field on cubic lattice with periodic boundary conditions.
    
    The energy is given by:
    
    .. math::
        \frac{1}{2}\sum_{i\sim j} L (\\omega_i - \\omega_j)\cdot (m_i \cross m_j)

    Here i and j are nearest neighbors.

    Parameters
    ----------
    AFD_field : jnp.array
        The AFD field (:math:`\\omega`) to calculate the energy
    spin_field : jnp.array
        The spin field (:math:`m`) to calculate the energy
    parameters : jax.numpy array
        The parameters of the energy function

    Returns
    -------
    jnp.array
        The energy of the field
    """

    L = parameters[0]
    energy = 0
    for axis in range(3):
        AFD_diff = AFD_field - jnp.roll(AFD_field, 1, axis=axis)
        spin_cross = jnp.cross(spin_field, jnp.roll(spin_field, 1, axis=axis))
        energy += jnp.sum(AFD_diff * spin_cross)
    return energy * L
