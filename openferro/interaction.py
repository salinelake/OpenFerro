import numpy as np
import jax.numpy as jnp

@jit
def self_energy_onsite(field, parameters):
    """
    Returns the isotropic self-energy of a field.
    See [Zhong, W., David Vanderbilt, and K. M. Rabe. Physical Review B 52.9 (1995): 6301.] for the meaning of the parameters.
    """
    k2 = parameters['k2']
    alpha = parameters['alpha']
    gamma = parameters['gamma']
    energy = k2 * jnp.sum(field**2)
    energy += alpha * jnp.sum(field**4)
    energy += gamma * jnp.sum(field**6)
    
    return energy