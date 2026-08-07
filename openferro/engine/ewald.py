"""
Functions for Ewald summation
"""
# This file is part of OpenFerro.
from time import time as timer

import numpy as np
import jax
from jax import grad, jit
import jax.numpy as jnp

from openferro.units import Constants
 

def _canonical_float_dtype(dtype):
    return jnp.asarray(1.0).dtype if dtype is None else jnp.dtype(dtype)



def _dipole_dipole_ewald_setup(latt, dtype=None):
    dtype = _canonical_float_dtype(dtype)
    if getattr(latt, "dim", None) != 3:
        raise ValueError("Dipole Ewald summation requires a three-dimensional lattice.")

    l1, l2, l3 = (int(value) for value in np.asarray(latt.size).tolist())
    if min(l1, l2, l3) <= 0:
        raise ValueError("Dipole Ewald lattice sizes must be positive integers.")

    latt_vec = np.asarray(latt.latt_vec, dtype=np.float64)
    if latt_vec.shape != (3, 3) or not np.all(np.isfinite(latt_vec)):
        raise ValueError("Dipole Ewald lattice vectors must be a finite 3x3 array.")

    diagonal = np.diag(latt_vec)
    if np.any(diagonal < 0.0) or not np.allclose(
        latt_vec, np.diag(diagonal), rtol=0.0, atol=1e-12
    ):
        raise NotImplementedError(
            "Dipole Ewald summation currently supports only positive, axis-aligned "
            "orthogonal primitive vectors; rotated and skew cells are unsupported."
        )
    if np.any(diagonal == 0.0):
        raise ValueError("Dipole Ewald lattice vectors must be nondegenerate.")

    a1, a2, a3 = (float(value) for value in diagonal)

    ref_volume = a1 * a2 * a3 * l1 * l2 * l3
    a = np.array([a1, a2, a3], dtype=np.float64)
    b_np = 2 * np.pi / a
    bmax = np.max(b_np)
    amin = 2 * np.pi / bmax
    alpha = 5 / amin
    gcut = 2 * np.pi * alpha
    sigma = 1.0 / alpha / np.sqrt(2.0)

    coef_ksum = 1 / 2.0 / ref_volume / Constants.epsilon0
    coef_rsum = 1 / 2.0 / np.pi / Constants.epsilon0 * alpha**3 / 3.0 / np.sqrt(np.pi)

    n1 = int(gcut / b_np[0])
    n2 = int(gcut / b_np[1])
    n3 = int(gcut / b_np[2])

    return {
        "shape": (l1, l2, l3),
        "b": jnp.asarray(b_np, dtype=dtype),
        "sigma": jnp.asarray(sigma, dtype=dtype),
        "coef_ksum": jnp.asarray(coef_ksum, dtype=dtype),
        "coef_rsum": jnp.asarray(coef_rsum, dtype=dtype),
        "replicas": (n1, n2, n3),
        "dtype": dtype,
    }


def get_UkGG(l1, l2, l3, n1, n2, n3, b, sigma, dtype=None, sharding=None):
    """Get the reciprocal-space Ewald kernel in Voigt notation.

    Voigt notation stores the six independent entries of the symmetric
    3x3 kernel in the order xx, yy, zz, yz, xz, xy.

    Parameters
    ----------
    l1, l2, l3 : int
        Lattice dimensions.
    n1, n2, n3 : int
        Reciprocal-cell replica counts in each direction.
    b : array_like
        Reciprocal lattice vector magnitudes.
    sigma : float
        Ewald Gaussian width.
    dtype : dtype, optional
        Floating-point dtype used for the stored kernel.
    sharding : jax.sharding.Sharding, optional
        Sharding used for the returned kernel.

    Returns
    -------
    jax.Array
        Kernel with shape ``(l1, l2, l3, 6)``.
    """
    dtype = _canonical_float_dtype(dtype)
    b = jnp.asarray(b, dtype=dtype)
    sigma = jnp.asarray(sigma, dtype=dtype)
    l1, l2, l3 = int(l1), int(l2), int(l3)

    G_grid_1stBZ = jnp.stack(jnp.meshgrid(
        jnp.arange(0, l1, dtype=dtype) / l1 * b[0],
        jnp.arange(0, l2, dtype=dtype) / l2 * b[1],
        jnp.arange(0, l3, dtype=dtype) / l3 * b[2],
        indexing='ij'), axis=-1)

    if sharding is not None:
        G_grid_1stBZ = jax.device_put(G_grid_1stBZ, sharding)

    def _get_Uk_coef(G_grid_1stBZ, offset):
        G_grid = G_grid_1stBZ + offset.reshape(1, 1, 1, 3)
        G2 = jnp.sum(G_grid**2, axis=-1)
        Uk_coef = jnp.where(
            G2 > 0.0,
            jnp.exp(-0.5 * sigma**2 * G2) / G2,
            jnp.zeros_like(G2),
        )
        return G_grid, Uk_coef

    def _modify_UkGG(UkGG, G_grid, Uk_coef):
        addition = jnp.stack(
            [G_grid[..., 0]**2 * Uk_coef,
             G_grid[..., 1]**2 * Uk_coef,
             G_grid[..., 2]**2 * Uk_coef,
             G_grid[..., 1] * G_grid[..., 2] * Uk_coef,
             G_grid[..., 0] * G_grid[..., 2] * Uk_coef,
             G_grid[..., 0] * G_grid[..., 1] * Uk_coef],
            axis=-1)
        return UkGG + addition

    get_Uk_coef = jit(_get_Uk_coef)
    modify_UkGG = jit(_modify_UkGG)
    UkGG = jnp.zeros((l1, l2, l3, 6), dtype=dtype)
    if sharding is not None:
        UkGG = jax.device_put(UkGG, sharding)

    for i1 in range(-n1, n1):
        for i2 in range(-n2, n2):
            for i3 in range(-n3, n3):
                offset = jnp.asarray([i1 * b[0], i2 * b[1], i3 * b[2]], dtype=dtype)
                G_grid, Uk_coef = get_Uk_coef(G_grid_1stBZ, offset)
                UkGG = modify_UkGG(UkGG, G_grid, Uk_coef)
    return UkGG


def apply_ewald_kernel_fft(field_fft, UkGG):
    """Apply the Voigt-form Ewald kernel to a Fourier-space vector field."""
    field_fft_x = field_fft[..., 0]
    field_fft_y = field_fft[..., 1]
    field_fft_z = field_fft[..., 2]
    kernel_field_fft = jnp.stack([
        UkGG[..., 0] * field_fft_x + UkGG[..., 5] * field_fft_y + UkGG[..., 4] * field_fft_z,
        UkGG[..., 5] * field_fft_x + UkGG[..., 1] * field_fft_y + UkGG[..., 3] * field_fft_z,
        UkGG[..., 4] * field_fft_x + UkGG[..., 3] * field_fft_y + UkGG[..., 2] * field_fft_z,
    ], axis=-1)
    return kernel_field_fft


def calc_ewald_reciprocal_sum(field_fft, UkGG):
    """Calculate ``sum_k conj(F_k) dot UkGG_k dot F_k``."""
    kernel_field_fft = apply_ewald_kernel_fft(field_fft, UkGG)
    return jnp.real(jnp.sum(jnp.conj(field_fft) * kernel_field_fft))


def get_dipole_dipole_ewald(latt, dtype=None, sharding=None):
    """Returns the function to calculate the energy of dipole-dipole interaction.

    Implemented according to Sec.5.3 of "Wang, D., et al. 'Ewald summation for 
    ferroelectric perovksites with charges and dipoles.' Computational Materials 
    Science 162 (2019): 314-321."

    Parameters
    ----------
    latt : Lattice
        The lattice object containing size and lattice vectors
    dtype : dtype, optional
        Floating-point dtype used for precomputed Ewald coefficients.
    sharding : jax.sharding.Sharding, optional
        Sharding specification for distributed arrays

    Returns
    -------
    callable
        Function that calculates dipole-dipole interaction energy
    """
    setup = _dipole_dipole_ewald_setup(latt, dtype=dtype)
    l1, l2, l3 = setup["shape"]
    n1, n2, n3 = setup["replicas"]

    UkGG = get_UkGG(
        l1, l2, l3, n1, n2, n3, setup["b"], setup["sigma"],
        dtype=setup["dtype"], sharding=sharding,
    )
    coef_ksum = setup["coef_ksum"]
    coef_rsum = setup["coef_rsum"]

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
        if field.shape != (l1, l2, l3, 3):
            raise ValueError(
                "Dipole field must have shape "
                f"{(l1, l2, l3, 3)} for this Ewald engine."
            )
        parameters = jnp.asarray(parameters)
        if parameters.shape != (1,):
            raise ValueError("Dipole Ewald parameters must have shape (1,).")
        prefactor = jnp.asarray(parameters[0], dtype=setup["dtype"])
        field_fft = jnp.fft.fftn(field, axes=(0,1,2))
        ewald_ksum = calc_ewald_reciprocal_sum(field_fft, UkGG)
        ewald_rsum = jnp.sum(field**2)
        energy = (coef_ksum * ewald_ksum - coef_rsum * ewald_rsum) * prefactor
        return energy
    return energy_engine
 


def estimate_dipole_dipole_ewald_memory(latt, dtype=None):
    """Estimate tracked Ewald array sizes in bytes.

    The estimate covers arrays that are explicit in OpenFerro's Ewald path.
    Backend FFT workspaces, compiler temporaries, and autodiff residuals are
    not included because they are backend- and version-dependent.
    """
    dtype = _canonical_float_dtype(dtype)
    l1, l2, l3 = (int(value) for value in np.asarray(latt.size).tolist())
    nsites = int(l1 * l2 * l3)
    float_bytes = np.dtype(dtype).itemsize
    complex_bytes = np.dtype(np.complex64 if float_bytes <= 4 else np.complex128).itemsize
    arrays = {
        "field": nsites * 3 * float_bytes,
        "UkGG": nsites * 6 * float_bytes,
        "field_fft": nsites * 3 * complex_bytes,
        "kernel_field_fft": nsites * 3 * complex_bytes,
    }
    return {
        "shape": (l1, l2, l3),
        "nsites": nsites,
        "dtype": str(dtype),
        "arrays": arrays,
        "tracked_total": sum(arrays.values()),
        "notes": (
            "FFT workspaces, compiler temporaries, and autodiff residuals are "
            "not included in this lower-bound estimate."
        ),
    }


def benchmark_dipole_dipole_ewald(
    latt,
    field=None,
    prefactor=1.0,
    repeat=3,
    seed=0,
    dtype=None,
    sharding=None,
    include_force=False,
):
    """Run a lightweight Ewald energy benchmark.

    Set ``include_force=True`` to benchmark the current autodiff force path.
    This helper is intended for small profiling runs and does not implement an
    explicit force engine.
    """
    dtype = _canonical_float_dtype(dtype)
    l1, l2, l3 = (int(value) for value in np.asarray(latt.size).tolist())
    if field is None:
        key = jax.random.PRNGKey(seed)
        field = jax.random.normal(key, (l1, l2, l3, 3), dtype=dtype)
    else:
        field = jnp.asarray(field, dtype=dtype)
    if sharding is not None:
        field = jax.device_put(field, sharding)

    energy_engine = jit(get_dipole_dipole_ewald(latt, dtype=dtype, sharding=sharding))
    parameters = jnp.asarray([prefactor], dtype=dtype)

    t0 = timer()
    energy = energy_engine(field, parameters)
    jax.block_until_ready(energy)
    compile_and_first_eval_seconds = timer() - t0

    t0 = timer()
    for _ in range(repeat):
        energy = energy_engine(field, parameters)
    jax.block_until_ready(energy)
    energy_seconds = (timer() - t0) / repeat

    result = {
        "shape": (l1, l2, l3),
        "dtype": str(dtype),
        "repeat": repeat,
        "compile_and_first_eval_seconds": compile_and_first_eval_seconds,
        "energy_seconds": energy_seconds,
        "energy": float(energy),
        "memory_estimate": estimate_dipole_dipole_ewald_memory(latt, dtype=dtype),
    }

    if include_force:
        force_engine = jit(grad(energy_engine, argnums=0))
        t0 = timer()
        force = force_engine(field, parameters)
        jax.block_until_ready(force)
        result["force_compile_and_first_eval_seconds"] = timer() - t0

        t0 = timer()
        for _ in range(repeat):
            force = force_engine(field, parameters)
        jax.block_until_ready(force)
        result["force_seconds"] = (timer() - t0) / repeat

    return result

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
