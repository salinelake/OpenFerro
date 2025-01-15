"""
Utility functions.
"""
# This file is part of OpenFerro.

import numpy as np
import jax
import jax.numpy as jnp

def SO3_rotation(B: jnp.ndarray, dt: float):
    """
    Calculate batched rotation matrix for proper rotation around magnetic field axis.

    Parameters
    ----------
    B : jnp.ndarray
        The magnetic field with shape (*, 3)
    dt : float
        The time step

    Returns
    -------
    jnp.ndarray
        The rotation matrix with shape (*, 3, 3)

    Notes
    -----
    Returns the batched rotation matrix (shape=(*, 3, 3)) of a proper rotation of a vector 
    by an angle theta (shape=(*,)) around the axes u (shape=(*, 3)).
    See https://en.wikipedia.org/wiki/Rotation_matrix for details.

    The angle and axes are specified by the magnetic field B and time step dt as:
    u = B / jnp.linalg.norm(B, axis=-1)[..., None]
    theta = jnp.linalg.norm(B, axis=-1) * dt
    """
    theta = jnp.linalg.norm(B, axis=-1)
    b = B / theta[..., None]
    bx = b[..., 0]
    by = b[..., 1]
    bz = b[..., 2]
    theta = theta * dt
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    u = 1 - cos_theta
    ## initialize the rotation matrix
    rotation_matrix = jnp.stack([
        bx*bx*u + cos_theta, bx*by*u-bz*sin_theta, bx*bz*u+by*sin_theta,
        bx*by*u+bz*sin_theta, by*by*u + cos_theta, by*bz*u-bx*sin_theta,
        bx*bz*u-by*sin_theta, by*bz*u+bx*sin_theta, bz*bz*u + cos_theta
        ], axis=-1)
    rotation_matrix = rotation_matrix.reshape(B.shape[:-1] + (3, 3))
    return rotation_matrix
