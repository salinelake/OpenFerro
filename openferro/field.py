"""
Classes which define the fields on the lattice.
"""
# This file is part of OpenFerro.

import numpy as np
import jax.numpy as jnp

class Field:
    def __init__(self, lattice, name ):
        self.lattice = lattice
        self.name = name
        self._values = None
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
