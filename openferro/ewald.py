"""
Functions for Ewald summation
"""
# This file is part of OpenFerro.

import numpy as np
import jax
import jax.numpy as jnp
from openferro.units import Constants


def get_dipole_dipole_ewald(latt):
    """
    Returns the function to calculate the energy of dipole-dipole interaction.
    Implemented according to Sec.5.3 of 
    "Wang, D., et al. "Ewald summation for ferroelectric perovksites with charges and dipoles." Computational Materials Science 162 (2019): 314-321."
    """
    l1, l2, l3 = latt.size
    a1, a2, a3 = latt.latt_vec
    a1 = a1[0]
    a2 = a2[1]
    a3 = a3[2]
    ref_volume = a1 * a2 * a3 * l1 * l2 * l3
    a = jnp.array([a1 , a2 , a3 ])
    b = 2 * jnp.pi / a
    bmax = jnp.max(b)
    amin = 2 * np.pi / bmax
    alpha = 5 / amin
    gcut = 2 * np.pi * alpha
    sigma = 1.0 / alpha / jnp.sqrt(2.0)   ## the ewald sigma parameter
    
    ## get coefficients
    coef_ksum = 1 / 2.0 / ref_volume / Constants.epsilon0
    coef_rsum = 1 / 2.0 / jnp.pi / Constants.epsilon0 * alpha**3 / 3.0 / jnp.sqrt(jnp.pi) 

    ## repeatation of Brillouin Zone
    n1 = int(gcut / b[0])
    n2 = int(gcut / b[1])
    n3 = int(gcut / b[2])
    
    ## reciprocal space grid for first Brillouin zone (shifted)
    G_grid_1stBZ = jnp.stack( jnp.meshgrid(
        jnp.arange(0, l1) / l1 * b[0],
        jnp.arange(0, l2) / l2 * b[1],
        jnp.arange(0, l3) / l3 * b[2],
        indexing='ij'), axis=-1)   # (l1, l2, l3, 3)

    UkGG = jnp.zeros((l1, l2, l3, 3, 3))
    for i1 in range(-n1,n1):
        for i2 in range(-n2,n2):
            for i3 in range(-n3,n3):
                G_grid = G_grid_1stBZ + jnp.array([i1*b[0], i2*b[1], i3*b[2]]).reshape(1,1,1,3) # (l1, l2, l3, 3)
                Uk_coef = jnp.exp( - 0.5 * sigma**2 * jnp.sum(G_grid**2, axis=-1) ) / jnp.sum(G_grid**2, axis=-1)   # (l1, l2, l3)
                if i1==0 and i2==0 and i3==0:
                    Uk_coef = Uk_coef.at[0,0,0].set(0.0)
                UkGG += G_grid[:,:,:,None,:] * G_grid[:,:,:,:,None] * Uk_coef[:,:,:,None,None]
    G_grid_1stBZ = None
    G_grid = None
    Uk_coef = None

    def energy_engine(field, parameters):
        Z = parameters['Z_star']
        epsilon_inf = parameters['epsilon_inf']

        ## calculate reciprocal space sum
        F_fft3 = jnp.fft.fftn(field, axes=(0,1,2))  # (l1, l2, l3, 3)
        ewald_ksum = (F_fft3.real[:,:,:,None,:] * F_fft3.real[:,:,:,:,None] * UkGG).sum()
        ewald_ksum += (F_fft3.imag[:,:,:,None,:] * F_fft3.imag[:,:,:,:,None] * UkGG).sum()
        ewald_ksum = coef_ksum * ewald_ksum

        ## calculate real space sum
        ewald_rsum = - coef_rsum * jnp.sum(field**2)
        return (ewald_ksum + ewald_rsum) * Z**2 / epsilon_inf
    return energy_engine


def get_dipole_dipole_ewald_mid_mem_usage(latt):
    """
    Returns the function to calculate the energy of dipole-dipole interaction.
    Implemented according to Sec.5.3 of 
    "Wang, D., et al. "Ewald summation for ferroelectric perovksites with charges and dipoles." Computational Materials Science 162 (2019): 314-321."
    """
    l1, l2, l3 = latt.size
    a1, a2, a3 = latt.latt_vec
    a1 = a1[0]
    a2 = a2[1]
    a3 = a3[2]
    ref_volume = a1 * a2 * a3 * l1 * l2 * l3
    a = jnp.array([a1 , a2 , a3 ])
    b = 2 * jnp.pi / a
    bmax = jnp.max(b)
    amin = 2 * np.pi / bmax
    alpha = 5 / amin
    gcut = 2 * np.pi * alpha
    sigma = 1.0 / alpha / jnp.sqrt(2.0)   ## the ewald sigma parameter
    
    ## get coefficients
    coef_ksum = 1 / 2.0 / ref_volume / Constants.epsilon0
    coef_rsum = 1 / 2.0 / jnp.pi / Constants.epsilon0 * alpha**3 / 3.0 / jnp.sqrt(jnp.pi) 

    ## get reriprocal space grid
    n1 = int(gcut / b[0])
    n2 = int(gcut / b[1])
    n3 = int(gcut / b[2])
    ng1, ng2, ng3 = l1*n1, l2*n2, l3*n3
    G_grid = jnp.stack( jnp.meshgrid(
        jnp.arange(-ng1, ng1) / l1 * b[0], 
        jnp.arange(-ng2, ng2) / l2 * b[1], 
        jnp.arange(-ng3, ng3) / l3 * b[2], 
        indexing='ij'), axis=-1)   # (2*ng1, 2*ng2, 2*ng3, 3)
    G_grid = jnp.roll(G_grid, shift=(-ng1, -ng2, -ng3), axis=(0, 1, 2))  # move gamma point to (0,0,0)
    G_grid = G_grid.reshape(2*n1, l1, 2*n2, l2, 2*n3, l3, 3)     
    G_grid = G_grid.transpose(1,3,5,0,2,4,6).reshape(l1,l2,l3,-1,3)  # (l1, l2, l3, 8*n1*n2*n3, 3)
    

    ## get coefficients for reciprocal space sum
    Uk_coef = jnp.exp( - 0.5 * sigma**2 * jnp.sum(G_grid**2, axis=-1) ) / jnp.sum(G_grid**2, axis=-1)   # (l1, l2, l3, 8*n1*n2*n3)
    Uk_coef = Uk_coef.at[0,0,0,0].set(0.0)   # mute Gamma point
    ## sum over replica of first Brillouin zone first. This reduces the memory usage by a factor of 8*n1*n2*n3/3 
    # UkGG = (G_grid[:,:,:,:,None,:] * G_grid[:,:,:,:,:,None] * Uk_coef[:,:,:,:,None,None]).sum(3)  # (l1, l2, l3, 3, 3)
    UkGG = jnp.zeros((l1, l2, l3, 3, 3))
    for i in range(Uk_coef.shape[-1]):
        UkGG += G_grid[:,:,:,i,None,:] * G_grid[:,:,:,i,:,None] * Uk_coef[:,:,:,i,None,None]
    G_grid = None
    Uk_coef = None
    
    def energy_engine(field, parameters):
        Z = parameters['Z_star']
        epsilon_inf = parameters['epsilon_inf']

        ## calculate reciprocal space sum
        F_fft3 = jnp.fft.fftn(field, axes=(0,1,2))  # (l1, l2, l3, 3)
        ewald_ksum = (F_fft3.real[:,:,:,None,:] * F_fft3.real[:,:,:,:,None] * UkGG).sum()
        ewald_ksum += (F_fft3.imag[:,:,:,None,:] * F_fft3.imag[:,:,:,:,None] * UkGG).sum()
        ewald_ksum = coef_ksum * ewald_ksum

        ## calculate real space sum
        ewald_rsum = - coef_rsum * jnp.sum(field**2)
        return (ewald_ksum + ewald_rsum) * Z**2 / epsilon_inf
    return energy_engine

def get_dipole_dipole_ewald_high_memory_usage(latt):
    """
    Returns the function to calculate the energy of dipole-dipole interaction.
    Implemented according to Sec.5.3 of 
    "Wang, D., et al. "Ewald summation for ferroelectric perovksites with charges and dipoles." Computational Materials Science 162 (2019): 314-321."
    """
    l1, l2, l3 = latt.size
    a1, a2, a3 = latt.latt_vec
    a1 = a1[0]
    a2 = a2[1]
    a3 = a3[2]
    ref_volume = a1 * a2 * a3 * l1 * l2 * l3
    a = jnp.array([a1 , a2 , a3 ])
    b = 2 * jnp.pi / a
    bmax = jnp.max(b)
    amin = 2 * np.pi / bmax
    alpha = 5 / amin
    gcut = 2 * np.pi * alpha
    sigma = 1.0 / alpha / jnp.sqrt(2.0)   ## the ewald sigma parameter
    
    ## get coefficients
    coef_ksum = 1 / 2.0 / ref_volume / Constants.epsilon0
    coef_rsum = 1 / 2.0 / jnp.pi / Constants.epsilon0 * alpha**3 / 3.0 / jnp.sqrt(jnp.pi) 

    ## get reriprocal space grid
    n1 = int(gcut / b[0])
    n2 = int(gcut / b[1])
    n3 = int(gcut / b[2])
    ng1, ng2, ng3 = l1*n1, l2*n2, l3*n3
    G_grid = jnp.stack( jnp.meshgrid(
        jnp.arange(-ng1, ng1) / l1 * b[0], 
        jnp.arange(-ng2, ng2) / l2 * b[1], 
        jnp.arange(-ng3, ng3) / l3 * b[2], 
        indexing='ij'), axis=-1)   # (2*ng1, 2*ng2, 2*ng3, 3)
    G_grid = jnp.roll(G_grid, shift=(-ng1, -ng2, -ng3), axis=(0, 1, 2))  # move gamma point to (0,0,0)
    G_grid = G_grid.reshape(2*n1, l1, 2*n2, l2, 2*n3, l3, 3)     
    G_grid = G_grid.transpose(1,3,5,0,2,4,6).reshape(l1,l2,l3,-1,3)  # (l1, l2, l3, 8*n1*n2*n3, 3)

    ## get coefficients for reciprocal space sum
    Uk_coef = jnp.exp( - 0.5 * sigma**2 * jnp.sum(G_grid**2, axis=-1) ) / jnp.sum(G_grid**2, axis=-1)   # (l1, l2, l3, 8*n1*n2*n3)
    Uk_coef = Uk_coef.at[0,0,0,0].set(0.0)   # mute Gamma point
    def energy_engine(field, parameters):
        Z = parameters['Z_star']
        epsilon_inf = parameters['epsilon_inf']

        ## calculate reciprocal space sum
        F_fft3 = jnp.fft.fftn(field, axes=(0,1,2))  # (l1, l2, l3, 3)
        Uk_squared  = jnp.sum( F_fft3.real[:,:,:,None,:] * G_grid, axis=-1)**2
        Uk_squared += jnp.sum( F_fft3.imag[:,:,:,None,:] * G_grid, axis=-1)**2   # (l1, l2, l3, 8*n1*n2*n3)
        ewald_ksum = coef_ksum * jnp.sum(Uk_coef * Uk_squared)

        ## calculate real space sum
        ewald_rsum = - coef_rsum * jnp.sum(field**2)
        return (ewald_ksum + ewald_rsum) * Z**2 / epsilon_inf
    return energy_engine

def dipole_dipole_ewald_plain(field, parameters):
    """
    For benchmarking purpose only. 
    Brute-force Ewald summation for dipole-dipole interaction
    """
    l1, l2, l3 = field.shape[0], field.shape[1], field.shape[2]
    a1 = parameters['a1']
    a2 = parameters['a2']
    a3 = parameters['a3']
    Z = parameters['Z_star']
    epsilon_inf = parameters['epsilon_inf']
    ref_volume = a1 * a2 * a3 * l1 * l2 * l3
    # if (a1[1] != 0) or (a1[2] != 0) or (a2[0] != 0) or (a2[2] != 0) or (a3[0] != 0) or (a3[1] != 0):
    #     raise NotImplementedError("Ewald summation is only implemented for orthogonal lattice vectors")
    # else:
    a = jnp.array([a1 , a2 , a3 ])
    b = 2 * jnp.pi / a
    bmax = jnp.max(b)
    amin = 2 * np.pi / bmax
    alpha = 5 / amin
    gcut = 2 * np.pi * alpha
    sigma = 1.0 / alpha / jnp.sqrt(2.0)   ## the ewald sigma parameter

    ## get coefficients
    coef_ksum = 1 / 2.0 / ref_volume / Constants.epsilon0
    coef_rsum = 1 / 2.0 / jnp.pi / Constants.epsilon0 * alpha**3 / 3.0 / jnp.sqrt(jnp.pi) 

    ## get reriprocal space grid
    n1 = int(gcut / b[0])
    n2 = int(gcut / b[1])
    n3 = int(gcut / b[2])
    ng1, ng2, ng3 = l1*n1, l2*n2, l3*n3
    G_grid = jnp.stack( jnp.meshgrid(
        jnp.arange(-ng1, ng1) / l1 * b[0], 
        jnp.arange(-ng2, ng2) / l2 * b[1], 
        jnp.arange(-ng3, ng3) / l3 * b[2], 
        indexing='ij'), axis=-1)   # (2*ng1, 2*ng2, 2*ng3, 3)
    G_grid = jnp.roll(G_grid, shift=(-ng1, -ng2, -ng3), axis=(0,1,2))  # move gamma point to (0,0,0)
    G2_grid = jnp.sum(G_grid**2, axis=-1)
    
    ## calculate reciprocal space sum
    ewald_ksum = 0.0
    ewald_rsum = 0.0
    for i1 in range(l1):
        for i2 in range(l2):
            for i3 in range(l3):
                for j1 in range(l1):
                    for j2 in range(l2):
                        for j3 in range(l3):
                            for alpha in range(3):
                                for beta in range(3):
                                    rij = - jnp.array([i1 - j1, i2 - j2, i3 - j3]) * a
                                    Q_ijabk = jnp.exp(-0.5 * sigma**2 * G2_grid) / G2_grid 
                                    Q_ijabk = Q_ijabk.at[0,0,0].set(0.0)   # mute Gamma point
                                    Q_ijabk = Q_ijabk * G_grid[..., alpha] * G_grid[..., beta]
                                    Q_ijabk = Q_ijabk * jnp.cos(jnp.sum(G_grid * rij, axis=-1))
                                    Q_ijabk = jnp.sum(Q_ijabk) * coef_ksum
                                    ewald_ksum += Q_ijabk * field[i1, i2, i3, alpha] * field[j1, j2, j3, beta]
    ## calculate real space sum
    for i1 in range(l1):
        for i2 in range(l2):
            for i3 in range(l3):
                for alpha in range(3):
                    ewald_rsum -= coef_rsum * field[i1, i2, i3, alpha]**2
    return (ewald_ksum + ewald_rsum) * Z**2 / epsilon_inf
 
