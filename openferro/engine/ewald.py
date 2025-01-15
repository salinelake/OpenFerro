"""
Functions for Ewald summation
"""
# This file is part of OpenFerro.

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from openferro.units import Constants

def get_dipole_dipole_ewald(latt, sharding=None):
    """Returns the function to calculate the energy of dipole-dipole interaction.

    Implemented according to Sec.5.3 of "Wang, D., et al. 'Ewald summation for 
    ferroelectric perovksites with charges and dipoles.' Computational Materials 
    Science 162 (2019): 314-321."

    Parameters
    ----------
    latt : Lattice
        The lattice object containing size and lattice vectors
    sharding : jax.sharding.Sharding, optional
        Sharding specification for distributed arrays

    Returns
    -------
    callable
        Function that calculates dipole-dipole interaction energy
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

    ## plain version of UkGG
    # UkGG = jnp.zeros((l1, l2, l3, 3, 3))
    # if sharding is not None:
    #     G_grid_1stBZ = jax.device_put(G_grid_1stBZ, sharding)
    #     UkGG = jax.device_put(UkGG, sharding)
    # for i1 in range(-n1,n1):
    #     for i2 in range(-n2,n2):
    #         for i3 in range(-n3,n3):
    #             G_grid = G_grid_1stBZ + jnp.array([i1*b[0], i2*b[1], i3*b[2]]).reshape(1,1,1,3) # (l1, l2, l3, 3)
    #             Uk_coef = jnp.exp( - 0.5 * sigma**2 * jnp.sum(G_grid**2, axis=-1) ) / jnp.sum(G_grid**2, axis=-1)   # (l1, l2, l3)
    #             if i1==0 and i2==0 and i3==0:
    #                 Uk_coef = Uk_coef.at[0,0,0].set(0.0)
    #             UkGG += G_grid[:,:,:,None,:] * G_grid[:,:,:,:,None] * Uk_coef[:,:,:,None,None]

    ## memory-saving version of UkGG with Voigt notation
    ## Voigt notation of a symmetric 3X3 matrix: Six elements are respectively (0,0), (1,1), (2,2), (1,2), (0,2), (0,1)-entry of a symmetric 3X3 matrix. 
    ## Slightly different from the original Voigt notation. We do not double count the fourth, fifth, and sixth elements here. 
    UkGG = jnp.zeros((l1, l2, l3, 6)) 
    if sharding is not None:
        G_grid_1stBZ = jax.device_put(G_grid_1stBZ, sharding)
        UkGG = jax.device_put(UkGG, sharding)
    for i1 in range(-n1,n1):
        for i2 in range(-n2,n2):
            for i3 in range(-n3,n3):
                G_grid = G_grid_1stBZ + jnp.array([i1*b[0], i2*b[1], i3*b[2]]).reshape(1,1,1,3) # (l1, l2, l3, 3)
                Uk_coef = jnp.exp( - 0.5 * sigma**2 * jnp.sum(G_grid**2, axis=-1) ) / jnp.sum(G_grid**2, axis=-1)   # (l1, l2, l3)
                if i1==0 and i2==0 and i3==0:
                    Uk_coef = Uk_coef.at[0,0,0].set(0.0)
                addition = jnp.stack(
                    [G_grid[...,0]**2 * Uk_coef, 
                     G_grid[...,1]**2 * Uk_coef, 
                     G_grid[...,2]**2 * Uk_coef,
                     G_grid[..., 1] * G_grid[..., 2] * Uk_coef,  # Voigt notation of entry-(1,2)
                     G_grid[..., 0] * G_grid[..., 2] * Uk_coef,  # Voigt notation of entry-(0,2)
                     G_grid[..., 0] * G_grid[..., 1] * Uk_coef],   # Voigt notation of entry-(0,1)
                    axis=-1)
                UkGG += addition
                
    G_grid_1stBZ = None
    G_grid = None
    Uk_coef = None

    ## define the computationally intensive functions to be jitted. 
    def _ewald_ksum(F_fft3):
        """Getting ewald summation over the k-space with UkGG in Voigt notation.

        Parameters
        ----------
        F_fft3 : ndarray
            Fast Fourier Transform of the field, shape=(l1, l2, l3, 3)

        Returns
        -------
        float
            Ewald summation in k-space
        """
        F_real = F_fft3.real
        F_imag = F_fft3.imag
        ewald_ksum = (F_real[..., 0]**2 + F_imag[..., 0]**2) * UkGG[..., 0]
        ewald_ksum += (F_real[..., 1]**2 + F_imag[..., 1]**2) * UkGG[..., 1]
        ewald_ksum += (F_real[..., 2]**2 + F_imag[..., 2]**2) * UkGG[..., 2]
        ewald_ksum += 2 * ((F_real[..., 0] * F_real[..., 1] + F_imag[..., 0] * F_imag[..., 1]) * UkGG[..., 5])
        ewald_ksum += 2 * ((F_real[..., 0] * F_real[..., 2] + F_imag[..., 0] * F_imag[..., 2]) * UkGG[..., 4])
        ewald_ksum += 2 * ((F_real[..., 1] * F_real[..., 2] + F_imag[..., 1] * F_imag[..., 2]) * UkGG[..., 3])
        return ewald_ksum.sum()

    ## define the computationally intensive functions to be jitted.  
    def _ewald_rsum(field):
        """Getting ewald summation over the real space.

        Parameters
        ----------
        field : ndarray
            The values of the field, shape=(l1, l2, l3, 3)

        Returns
        -------
        float
            Ewald summation in real space
        """
        return jnp.sum(field**2)

    ewald_ksum_func = jit(_ewald_ksum)
    ewald_rsum_func = jit(_ewald_rsum)

    ##  The main function of energy engine will not be jitted. jitting jnp.fft.fftn seems to lead to error.
    def energy_engine(field, parameters):
        """Calculate the energy of dipole-dipole interaction using Ewald summation.

        Parameters
        ----------
        field : ndarray
            The values of the field, shape=(l1, l2, l3, 3)
        parameters : ndarray
            Array of parameters

        Returns
        -------
        float
            The dipole-dipole interaction energy
        """
        prefactor = parameters[0]
        ## calculate reciprocal space sum. UkGG is a symmetric (l1, l2, l3, 3, 3) matrix, so we only need to calculate half of it.
        F_fft3 = jnp.fft.fftn(field, axes=(0,1,2))  # (l1, l2, l3, 3)
        ######### compute the summation over k-space
        # ewald_ksum = ewald_ksum_func(F_fft3, UkGG)
        ewald_ksum = ewald_ksum_func(F_fft3)

        ######### compute the summation over real space
        ewald_rsum = ewald_rsum_func(field)
        return (coef_ksum * ewald_ksum - coef_rsum * ewald_rsum) * prefactor
    return energy_engine


# ## Helper functions
# def _ewald_ksum(F_fft3, UkGG):  ## warning on too long constant folding because UkGG is in the argument when jitted
#     """
#     Getting ewald summation over the k-space with UkGG in Voigt notation.
#     Args:
#         F_fft3: jax.numpy array, shape=(l1, l2, l3, 3). Fast Fourier Transform of the field.
#         UkGG: jax.numpy array, shape=(l1, l2, l3, 6). UkGG in Voigt notation.
#     Returns:
#         jax.numpy array, shape=(1,)
#     """
#     F_real = F_fft3.real
#     F_imag = F_fft3.imag
#     ewald_ksum = (F_real[..., 0]**2 + F_imag[..., 0]**2) * UkGG[..., 0]
#     ewald_ksum += (F_real[..., 1]**2 + F_imag[..., 1]**2) * UkGG[..., 1]
#     ewald_ksum += (F_real[..., 2]**2 + F_imag[..., 2]**2) * UkGG[..., 2]
#     ewald_ksum += 2 * ((F_real[..., 0] * F_real[..., 1] + F_imag[..., 0] * F_imag[..., 1]) * UkGG[..., 5])
#     ewald_ksum += 2 * ((F_real[..., 0] * F_real[..., 2] + F_imag[..., 0] * F_imag[..., 2]) * UkGG[..., 4])
#     ewald_ksum += 2 * ((F_real[..., 1] * F_real[..., 2] + F_imag[..., 1] * F_imag[..., 2]) * UkGG[..., 3])
#     return ewald_ksum.sum()

# def _ewald_rsum(field):
#     """
#     Getting ewald summation over the real space.
#     Args:
#         field: jax.numpy array, shape=(l1, l2, l3, 3). The values of the field.
#     Returns:
#         jax.numpy array, shape=(1,)
#     """
#     return jnp.sum(field**2)


"""
Archived versions of Ewald summation with higher memory usage. For testing purpose only.
"""

# def get_dipole_dipole_ewald_mid_mem_usage(latt):
#     """
#     Returns the function to calculate the energy of dipole-dipole interaction.
#     Implemented according to Sec.5.3 of 
#     "Wang, D., et al. "Ewald summation for ferroelectric perovksites with charges and dipoles." Computational Materials Science 162 (2019): 314-321."
#     """
#     l1, l2, l3 = latt.size
#     a1, a2, a3 = latt.latt_vec
#     a1 = a1[0]
#     a2 = a2[1]
#     a3 = a3[2]
#     ref_volume = a1 * a2 * a3 * l1 * l2 * l3
#     a = jnp.array([a1 , a2 , a3 ])
#     b = 2 * jnp.pi / a
#     bmax = jnp.max(b)
#     amin = 2 * np.pi / bmax
#     alpha = 5 / amin
#     gcut = 2 * np.pi * alpha
#     sigma = 1.0 / alpha / jnp.sqrt(2.0)   ## the ewald sigma parameter
    
#     ## get coefficients
#     coef_ksum = 1 / 2.0 / ref_volume / Constants.epsilon0
#     coef_rsum = 1 / 2.0 / jnp.pi / Constants.epsilon0 * alpha**3 / 3.0 / jnp.sqrt(jnp.pi) 

#     ## get reriprocal space grid
#     n1 = int(gcut / b[0])
#     n2 = int(gcut / b[1])
#     n3 = int(gcut / b[2])
#     ng1, ng2, ng3 = l1*n1, l2*n2, l3*n3
#     G_grid = jnp.stack( jnp.meshgrid(
#         jnp.arange(-ng1, ng1) / l1 * b[0], 
#         jnp.arange(-ng2, ng2) / l2 * b[1], 
#         jnp.arange(-ng3, ng3) / l3 * b[2], 
#         indexing='ij'), axis=-1)   # (2*ng1, 2*ng2, 2*ng3, 3)
#     G_grid = jnp.roll(G_grid, shift=(-ng1, -ng2, -ng3), axis=(0, 1, 2))  # move gamma point to (0,0,0)
#     G_grid = G_grid.reshape(2*n1, l1, 2*n2, l2, 2*n3, l3, 3)     
#     G_grid = G_grid.transpose(1,3,5,0,2,4,6).reshape(l1,l2,l3,-1,3)  # (l1, l2, l3, 8*n1*n2*n3, 3)
    

#     ## get coefficients for reciprocal space sum
#     Uk_coef = jnp.exp( - 0.5 * sigma**2 * jnp.sum(G_grid**2, axis=-1) ) / jnp.sum(G_grid**2, axis=-1)   # (l1, l2, l3, 8*n1*n2*n3)
#     Uk_coef = Uk_coef.at[0,0,0,0].set(0.0)   # mute Gamma point
#     ## sum over replica of first Brillouin zone first. This reduces the memory usage by a factor of 8*n1*n2*n3/3 
#     # UkGG = (G_grid[:,:,:,:,None,:] * G_grid[:,:,:,:,:,None] * Uk_coef[:,:,:,:,None,None]).sum(3)  # (l1, l2, l3, 3, 3)
#     UkGG = jnp.zeros((l1, l2, l3, 3, 3))
#     for i in range(Uk_coef.shape[-1]):
#         UkGG += G_grid[:,:,:,i,None,:] * G_grid[:,:,:,i,:,None] * Uk_coef[:,:,:,i,None,None]
#     G_grid = None
#     Uk_coef = None
    
#     def energy_engine(field, parameters):
#         Z = parameters['Z_star']
#         epsilon_inf = parameters['epsilon_inf']

#         ## calculate reciprocal space sum
#         F_fft3 = jnp.fft.fftn(field, axes=(0,1,2))  # (l1, l2, l3, 3)
#         ewald_ksum = (F_fft3.real[:,:,:,None,:] * F_fft3.real[:,:,:,:,None] * UkGG).sum()
#         ewald_ksum += (F_fft3.imag[:,:,:,None,:] * F_fft3.imag[:,:,:,:,None] * UkGG).sum()
#         ewald_ksum = coef_ksum * ewald_ksum

#         ## calculate real space sum
#         ewald_rsum = - coef_rsum * jnp.sum(field**2)
#         return (ewald_ksum + ewald_rsum) * Z**2 / epsilon_inf
#     return energy_engine

# def get_dipole_dipole_ewald_high_memory_usage(latt):
#     """
#     Returns the function to calculate the energy of dipole-dipole interaction.
#     Implemented according to Sec.5.3 of 
#     "Wang, D., et al. "Ewald summation for ferroelectric perovksites with charges and dipoles." Computational Materials Science 162 (2019): 314-321."
#     """
#     l1, l2, l3 = latt.size
#     a1, a2, a3 = latt.latt_vec
#     a1 = a1[0]
#     a2 = a2[1]
#     a3 = a3[2]
#     ref_volume = a1 * a2 * a3 * l1 * l2 * l3
#     a = jnp.array([a1 , a2 , a3 ])
#     b = 2 * jnp.pi / a
#     bmax = jnp.max(b)
#     amin = 2 * np.pi / bmax
#     alpha = 5 / amin
#     gcut = 2 * np.pi * alpha
#     sigma = 1.0 / alpha / jnp.sqrt(2.0)   ## the ewald sigma parameter
    
#     ## get coefficients
#     coef_ksum = 1 / 2.0 / ref_volume / Constants.epsilon0
#     coef_rsum = 1 / 2.0 / jnp.pi / Constants.epsilon0 * alpha**3 / 3.0 / jnp.sqrt(jnp.pi) 

#     ## get reriprocal space grid
#     n1 = int(gcut / b[0])
#     n2 = int(gcut / b[1])
#     n3 = int(gcut / b[2])
#     ng1, ng2, ng3 = l1*n1, l2*n2, l3*n3
#     G_grid = jnp.stack( jnp.meshgrid(
#         jnp.arange(-ng1, ng1) / l1 * b[0], 
#         jnp.arange(-ng2, ng2) / l2 * b[1], 
#         jnp.arange(-ng3, ng3) / l3 * b[2], 
#         indexing='ij'), axis=-1)   # (2*ng1, 2*ng2, 2*ng3, 3)
#     G_grid = jnp.roll(G_grid, shift=(-ng1, -ng2, -ng3), axis=(0, 1, 2))  # move gamma point to (0,0,0)
#     G_grid = G_grid.reshape(2*n1, l1, 2*n2, l2, 2*n3, l3, 3)     
#     G_grid = G_grid.transpose(1,3,5,0,2,4,6).reshape(l1,l2,l3,-1,3)  # (l1, l2, l3, 8*n1*n2*n3, 3)

#     ## get coefficients for reciprocal space sum
#     Uk_coef = jnp.exp( - 0.5 * sigma**2 * jnp.sum(G_grid**2, axis=-1) ) / jnp.sum(G_grid**2, axis=-1)   # (l1, l2, l3, 8*n1*n2*n3)
#     Uk_coef = Uk_coef.at[0,0,0,0].set(0.0)   # mute Gamma point
#     def energy_engine(field, parameters):
#         Z = parameters['Z_star']
#         epsilon_inf = parameters['epsilon_inf']

#         ## calculate reciprocal space sum
#         F_fft3 = jnp.fft.fftn(field, axes=(0,1,2))  # (l1, l2, l3, 3)
#         Uk_squared  = jnp.sum( F_fft3.real[:,:,:,None,:] * G_grid, axis=-1)**2
#         Uk_squared += jnp.sum( F_fft3.imag[:,:,:,None,:] * G_grid, axis=-1)**2   # (l1, l2, l3, 8*n1*n2*n3)
#         ewald_ksum = coef_ksum * jnp.sum(Uk_coef * Uk_squared)

#         ## calculate real space sum
#         ewald_rsum = - coef_rsum * jnp.sum(field**2)
#         return (ewald_ksum + ewald_rsum) * Z**2 / epsilon_inf
#     return energy_engine

def dipole_dipole_ewald_plain(field, parameters):
    """Brute-force Ewald summation for dipole-dipole interaction.
    
    For benchmarking purpose only.

    Parameters
    ----------
    field : ndarray
        The field values, shape=(l1, l2, l3, 3)
    parameters : dict
        Dictionary containing:
            a1 : float
                First lattice vector
            a2 : float 
                Second lattice vector
            a3 : float
                Third lattice vector
            Z_star : float
                Born effective charge
            epsilon_inf : float
                High-frequency dielectric constant

    Returns
    -------
    float
        The dipole-dipole interaction energy
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
