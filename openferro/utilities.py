import numpy as np
import jax
import jax.numpy as jnp


def SO3_rotation(B: jnp.ndarray, dt: float):
    """
    Return the batched rotation matrix (shape=(*, 3, 3)) of a proper rotation of a vector by an angle theta (shape=(*,)) around the axes u (shape=(*, 3)).
    See https://en.wikipedia.org/wiki/Rotation_matrix for details.
    
    The angle and the axes are specified by the magnetic field B and the time step dt, given as 
    u = B / jnp.linalg.norm(B, axis=-1)[..., None]
    theta = jnp.linalg.norm(B, axis=-1) * dt
    Args:
        B: the magnetic field (shape=(*, 3))
        dt : the time step
    Returns:

    """
    theta = jnp.linalg.norm(B, axis=-1)
    u = B / theta[..., None]
    theta = theta * dt
    rotation_matrix = jnp.cos(theta)[..., None, None] * jnp.eye(3)
    rotation_matrix += (1-jnp.cos(theta))[..., None, None] * u[..., None, :] * u[..., :, None]
    rotation_matrix += jnp.sin(theta)[..., None, None] * jnp.array(
        [[0, -u[..., 2], u[..., 1]], 
         [u[..., 2], 0, -u[..., 0]], 
         [-u[..., 1], u[..., 0], 0]])
    return rotation_matrix
