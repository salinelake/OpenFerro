"""
Functions for Ewald summation
"""
# This file is part of OpenFerro.

import numpy as np
import jax
import jax.numpy as jnp
from openferro.units import Constants
 
def dipole_dipole_ewald(field, parameters):
    """
    FFT realization of Ewald summation for dipole-dipole interaction.
    lattice vectors are assumed to be orthogonal.
    """
    l1, l2, l3 = field.shape[0], field.shape[1], field.shape[2]
    # a1, a2, a3 = latt_vec
    a1 = parameters['a1']
    a2 = parameters['a2']
    a3 = parameters['a3']
    # if (a1[1] != 0) or (a1[2] != 0) or (a2[0] != 0) or (a2[2] != 0) or (a3[0] != 0) or (a3[1] != 0):
    #     raise NotImplementedError("Ewald summation is only implemented for orthogonal lattice vectors")
    a = jnp.array([a1 , a2 , a3 ])
    b = 2 * jnp.pi / a
    bmax = jnp.max(b)
    amin = 2 * np.pi / bmax
    tol = 1.0e-12
    alpha = jnp.sqrt(-jnp.log(tol)) / amin
    sigma = 1.0 / alpha / jnp.sqrt(2.0)   ## the ewald sigma parameter

    ## k-space sum
    coef_ksum = 1 / 2.0 / latt.ref_volume / Constants.epsilon0
    G_grid = jnp.stack( jnp.meshgrid(
        jnp.arange(l1) / l1 * b[0], 
        jnp.arange(l2) / l2 * b[1], 
        jnp.arange(l3) / l3 * b[2], 
        indexing='ij'), axis=-1)   # (l1, l2, l3, 3)

    F_fft3 = jnp.fft.fftn(field, axes=(0,1,2))
    ewald_ksum = jnp.exp( - 0.5 * sigma**2 * jnp.sum(G_grid**2, axis=-1) ) / jnp.sum(G_grid**2, axis=-1)   # (l1, l2, l3)
    ewald_ksum = ewald_ksum.at[0,0,0].set(0.0)   # mute Gamma point
    Uk_squred = jnp.sum(F_fft3.real * G_grid, axis=-1)**2
    Uk_squred += jnp.sum( F_fft3.imag * G_grid , axis=-1)**2   # (l1, l2, l3)
    ewald_ksum = ewald_ksum * Uk_squred
    ewald_ksum = coef_ksum * jnp.sum(ewald_ksum, axis=(0,1,2))

    ## real-space sum
    coef_rsum = 1 / 2.0 / jnp.pi / Constants.epsilon0 * alpha**3 / 3.0 / jnp.sqrt(jnp.pi) 
    ewald_rsum = - coef_rsum * jnp.sum(field**2)
    return ewald_ksum + ewald_rsum


def dipole_dipole_ewald_slow(field, latt_vec):
    """
    For benchmarking purpose only. 
    Brute-force Ewald summation for dipole-dipole interaction
    """
    l1, l2, l3 = field.shape[0], field.shape[1], field.shape[2]
    a1, a2, a3 = latt_vec
    if (a1[1] != 0) or (a1[2] != 0) or (a2[0] != 0) or (a2[2] != 0) or (a3[0] != 0) or (a3[1] != 0):
        raise NotImplementedError("Ewald summation is only implemented for orthogonal lattice vectors")
    else:
        a = jnp.array([a1[0], a2[1], a3[2]])
        b = 2 * jnp.pi / a
        bmax = jnp.max(b)
        amin = 2 * np.pi / bmax
    tol = 1.0e-12
    alpha = jnp.sqrt(-jnp.log(tol)) / amin
    sigma = 1.0 / alpha / jnp.sqrt(2.0)   ## the ewald sigma parameter
    ## get k-grid
    k_grid = np.zeros((l1, l2, l3, 3))
    for i1 in range(l1):
        for i2 in range(l2):
            for i3 in range(l3):
                k_grid[i1, i2, i3] = np.array([i1 / l1 * b[0], i2 / l2 * b[1], i3 / l3 * b[2]])
    k_grid = jnp.array(k_grid)
    k_grid = k_grid.reshape(-1, 3)
    k2_grid = jnp.sum(k_grid**2, axis=-1)
    
    ## get coefficients
    coef_ksum = 1 / 2.0 / latt.ref_volume / Constants.epsilon0
    coef_rsum = 1 / 2.0 / jnp.pi / Constants.epsilon0 * alpha**3 / 3.0 / jnp.sqrt(jnp.pi) 

    ###
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
                                    Q_ijabk = jnp.exp(-0.5 * sigma**2 * k2_grid) / k2_grid 
                                    Q_ijabk = Q_ijabk.at[0].set(0.0)   # mute Gamma point
                                    Q_ijabk = Q_ijabk * k_grid[..., alpha] * k_grid[..., beta]
                                    Q_ijabk = Q_ijabk * jnp.cos(jnp.sum(k_grid * rij, axis=-1))
                                    Q_ijabk = jnp.sum(Q_ijabk) * coef_ksum
                                    ewald_ksum += Q_ijabk * field[i1, i2, i3, alpha] * field[j1, j2, j3, beta]
    for i1 in range(l1):
        for i2 in range(l2):
            for i3 in range(l3):
                for alpha in range(3):
                    ewald_rsum -= coef_rsum * field[i1, i2, i3, alpha]**2
    return ewald_ksum + ewald_rsum

if __name__ == "__main__":
    from openferro.lattice import BravaisLattice3D
    from time import time
    from jax import jit, grad
    l1,l2,l3 = 2,2,2
    latt = BravaisLattice3D(l1, l2, l3)
    latt_vec = latt.latt_vec
    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (l1, l2, l3, 3))
    paras = {'a1': latt_vec[0][0], 'a2': latt_vec[1][1], 'a3': latt_vec[2][2]}
    ## check dipole-dipole interaction energy calculation
    t0 = time()
    E1 = dipole_dipole_ewald_slow(field,  latt_vec)
    print("Time for slow method: ", time() - t0)
    t0 = time()
    E2 = dipole_dipole_ewald(field,  paras)
    print("Time for fast method: ", time() - t0)
    print('E1={}eV,E2={}eV. They should be the same'.format(E1,E2))

    ## check dipole-dipole interaction force calculation
    grad_slow = grad( dipole_dipole_ewald_slow  )
    grad_fast =  jit(grad( jit(dipole_dipole_ewald) ))
    t0 = time()
    force = grad_slow(field, latt_vec)
    print("Time for slow gradient method: ", time() - t0)
    t0 = time()
    force = grad_fast(field, paras)
    print("Time for fast gradient method: ", time() - t0)

    ## scaling
    l1_list = np.arange(1, 10) * 20
    t_list = []
    for l1 in l1_list:
        l2 = l1
        l3 = l1
        latt = BravaisLattice3D(l1, l2, l3)
        latt_vec = latt.latt_vec
        paras = {'a1': latt_vec[0][0], 'a2': latt_vec[1][1], 'a3': latt_vec[2][2]}
        field = jax.random.normal(key, (l1, l2, l3, 3))
        t0 = time()
        E = grad_fast(field,  paras)
        t_list.append(time() - t0)
    print("force scaling test: ", t_list)

    t_list = []
    energy_fast = jit(dipole_dipole_ewald)
    for l1 in l1_list:
        l2 = l1
        l3 = l1
        latt = BravaisLattice3D(l1, l2, l3)
        latt_vec = latt.latt_vec
        paras = {'a1': latt_vec[0][0], 'a2': latt_vec[1][1], 'a3': latt_vec[2][2]}
        field = jax.random.normal(key, (l1, l2, l3, 3))
        t0 = time()
        E = energy_fast(field,  paras)
        t_list.append(time() - t0)
    print("energy Scaling test: ", t_list)