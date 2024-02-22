"""
Classes which define the fields on the lattice.
"""
# This file is part of OpenFerro.

import numpy as np
import jax
import jax.numpy as jnp
from openferro.units import Constants

class Field:
    """
    A class to define a field on a lattice. 
    Each lattice site is associated with a value x. The corresponding local field is a function f(x).
    For flexible R^n vector field, x is the same as the field. f is the identity function. 
    For SO(3) fields, x is the spherical coordinates (theta, phi).  f maps (theta, phi) to (x, y, z) in Cartesian coordinates.
    """
    def __init__(self, lattice, name ):
        self.lattice = lattice
        self.dim = lattice.size.size
        self.name = name
        self._values = None
        self._mass = None
        self._velocity = None
        self._force = None

    ## setter and getter methods for values that encode the field
    def set_values(self, values):
        self._values = values.copy()
    def get_values(self):
        return self._values
    def set_local_value(self, loc, value):
        assert type(loc) is tuple and len(loc) == self.dim, "Location must be a tuple of length equal to the dimension of the lattice"
        self._values[loc] = value
    def get_mean(self):
        return jnp.mean(self._values, axis=[i for i in range(self.dim)])
    def get_variance(self):
        return jnp.var(self._values, axis=[i for i in range(self.dim)])

    ## setter and getter methods for getting the field in R^n
    def value2Rn(self, value):
        return value
    def get_Euclidean_value(self, i, j, k):
        return self.value2Rn(self._values[i, j, k])
    def get_Euclidean_values(self):
        return self.value2Rn(self._values)

    ## setter and getter methods for mass
    def set_mass(self, mass):
        if self._values is None:
            raise ValueError("Field has no values. Set values before setting mass.")
        else:
            assert jnp.min(mass) >= 0.0, "Mass must be non-negative"
            self._mass = jnp.zeros(self.lattice.size) + mass
            self._mass = self._mass[..., None]
    def get_mass(self):
        if self._mass is None:
            raise ValueError("Mass is not set")
        else:
            return self._mass

    ## setter and getter methods for velocity
    def init_velocity(self, mode='zero',  temperature=None):
        if self._values is None:
            raise ValueError("Field has no values. Set values before setting velocity.")
        else:
            if mode == 'zero':
                self._velocity = jnp.zeros_like(self._values)
            elif mode == 'gaussian':
                key = jax.random.PRNGKey(np.random.randint(0, 1000000))
                self._velocity = jax.random.normal(key, self._values.shape) * jnp.sqrt(1 / self._mass * Constants.kb * temperature)
    def get_velocity(self):
        if self._velocity is None:
            raise ValueError("Velocity is not set")
        else:
            return self._velocity
    def set_velocity(self, velocity):
        if self._values is None:
            raise ValueError("Field has no values. Set values before setting velocity.")
        else:
            assert velocity.shape == self._values.shape
            self._velocity = velocity
    ## setter and getter methods for forces
    def zero_force(self):
        if self._values is None:
            raise ValueError("Field has no values. Set values before zeroing forces.")
        else:
            self._force = jnp.zeros_like(self._values)
    def set_force(self, force):
        if self._values is None:
            raise ValueError("Field has no values. Set values before setting forces.")
        else:
            assert force.shape == self._values.shape
            self._force = force
    def accumulate_force(self, force):
        if self._force is None:
            raise ValueError("Gradients do not exist. Set or zero forces before accumulating.")
        else:
            assert force.shape == self._force.shape
            self._force += force
    def get_force(self):
        if self._force is None:
            raise ValueError("Force do not exist")
        else:
            return self._force


class FieldRn(Field):
    """
    R^n field. Values are stored as n-dimensional vectors. 
    Example: flexible dipole moment, strain field, electric field, etc.
    """
    def __init__(self, lattice, name, dim, unit=None):
        super().__init__(lattice, name)
        self._values = jnp.zeros((lattice.size[0], lattice.size[1], lattice.size[2], dim))
        self.unit = unit

class FieldScalar(FieldRn):
    """
    Scalar field. Values are stored as scalars. 
    Example: mass, density, any onsite constant, etc.
    """
    def __init__(self, lattice, name, unit=None):
        super().__init__(lattice, name, dim=1, unit=unit)

class FieldSO3(FieldRn):
    """
    Unitless SO(3) field. Values are stored as (theta, phi) pairs in sperical coordinates. 
    Example: atomistic spin, molecular orientation, etc.
    """
    def __init__(self, lattice, name ):
        super().__init__(lattice, name, dim=2, unit=unit)
    def value2Rn(self, value):
        theta = self._values[..., [0]]
        phi = self._values[..., [1]]
        x = jnp.sin(theta) * jnp.cos(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(theta)
        return jnp.concatenate((x, y, z), axis=-1)
 