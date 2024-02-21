"""
Classes which define the fields on the lattice.
"""
# This file is part of OpenFerro.

import numpy as np
import jax.numpy as jnp

class Field:
    """
    A class to define a field on a lattice. 
    Each lattice site is associated with a value x. The corresponding local field is a function f(x).
    For flexible R^n vector field, x is the same as the field. f is the identity function. 
    For SO(3) fields, x is the spherical coordinates (theta, phi).  f maps (theta, phi) to (x, y, z) in Cartesian coordinates.
    A gradient field is used to store the gradients on x.
    """
    def __init__(self, lattice, name ):
        self.lattice = lattice
        self.name = name
        self._values = None
        self._grads = None
    def __repr__(self):
        return f"Field {self.name} with value {self.values}"
    def set_values(self, values):
        self._values = values
    def set_local_value(self, i, j, k, value):
        self._values[i, j, k] = value
    def get_values(self):
        return self._values
    def value2field(self, value):
        pass
    def get_field(self):
        return value2field(self._values)
    def get_local_field(self, i, j, k):
        return value2field(self._values[i, j, k])
    def get_field_mean(self):
        pass
    def get_field_variance(self):
        pass
    ## setter and getter methods for gradients
    def zero_grad(self):
        if self._values is None:
            raise ValueError("Field has no values. Set values before zeroing gradients.")
        else:
            self._grads = jnp.zeros_like(self._values)
    def accumulate_grad(self, grad):
        if self._grads is None:
            raise ValueError("Gradients have not been zeroed. Zero gradients before accumulating.")
        else:
            self._grads += grad
    def get_grads(self):
        if self._grads is None:
            raise ValueError("Gradients do not exist")
        else:
            return self._grads


class FieldRn(Field):
    """
    R^n field. Values are stored as n-dimensional vectors. 
    Example: flexible dipole moment, strain field, electric field, etc.
    """
    def __init__(self, lattice, name, dim, unit=None):
        super().__init__(lattice, name)
        self._values = jnp.zeros((lattice.size[0], lattice.size[1], lattice.size[2], dim))
        self.unit = unit
    def value2field(self, value):
        return value
    def get_field_mean(self):
        return jnp.mean(self._values, axis=[i for i in range(dim)])
    def get_field_variance(self):
        return jnp.var(self._values, axis=[i for i in range(dim)])

class FieldScalar(FieldRn):
    """
    Scalar field. Values are stored as scalars. 
    Example: mass, density, any onsite constant, etc.
    """
    def __init__(self, lattice, name, unit=None):
        super().__init__(lattice, name, dim=1, unit=unit)

class FieldSO3(Field):
    """
    Unitless SO(3) field. Values are stored as (theta, phi) pairs in sperical coordinates. 
    Example: atomistic spin, molecular orientation, etc.
    """
    def __init__(self, lattice, name):
        super().__init__(lattice, name)
        self._values = jnp.zeros((lattice.size[0], lattice.size[1], lattice.size[2], 2))
    def value2field(self, value):
        theta = self._values[..., [0]]
        phi = self._values[..., [1]]
        x = jnp.sin(theta) * jnp.cos(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(theta)
        return jnp.concatenate((x, y, z), axis=-1)
    def get_field_mean(self):
        return jnp.mean(self.value2field(self._values), axis=[i for i in range(3)])
    def get_field_variance(self):
        return jnp.var(self.value2field(self._values), axis=[i for i in range(3)])

