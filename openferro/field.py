"""
Classes which define the fields on the lattice.
"""
# This file is part of OpenFerro.

from time import time as timer
import numpy as np
import jax
import jax.numpy as jnp
from openferro.units import Constants
from openferro.lattice import BravaisLattice3D
from openferro.parallelism import DeviceMesh
from openferro.integrator import *

class Field:
    """
    Template class to define a field on a lattice. 
    """
    def __init__(self, lattice, name: str):
        """
        Initialize a field.
        Args:
            lattice: BravaisLattice3D object.
            name: str, name of the field.
        """
        self.lattice = lattice
        self.name = name
        self._values = None
        self._mass = None
        self._velocity = None
        self._force = None
        self._sharding = None
        self.integrator_class = None

    """
    These methods are used to handle the values of the field.
    """

    def set_values(self, values):
        self._values = values
        return

    def get_values(self):
        """
        Get the values of the field.
        """
        if self._values is None:
            raise ValueError("Field has no values. Set values before getting them.")
        else:
            return self._values

    @property
    def size(self):
        return self.get_values().size

    """
    These methods are used to handle the mass of the field.
    """
    def set_mass(self, mass):
        if self._values is None:
            raise ValueError("Set field values before setting mass.")
        else:
            assert jnp.min(mass) >= 0.0, "Mass must be non-negative"
            if jnp.isscalar(mass):
                self._mass = jnp.zeros_like(self._values[..., 0]) + mass
                self._mass = self._mass[..., None]
            elif mass.shape == self._values[..., 0].shape:
                self._mass = mass[..., None]
            else:
                raise ValueError("Mass must be a scalar or an array of the same size as the all but the last dimension of the field values.")
            
    def get_mass(self):
        if self._mass is None:
            raise ValueError("Mass is not set")
        else:
            return self._mass

    """
    These methods are used to handle the velocity of the field.
    """

    def set_velocity(self, velocity):
        if self._values is None:
            raise ValueError("Field has no values. Set values before setting velocity.")
        else:
            self.compare_shape(velocity, self._values)
            self._velocity = velocity 

    def get_velocity(self):
        if self._velocity is None:
            raise ValueError("Velocity is not set")
        else:
            return self._velocity

    def init_velocity(self, mode='zero',  temperature=None):
        if self._values is None:
            raise ValueError("Set field values before initializing velocity.")
        else:
            if mode == 'zero':
                self._velocity = jnp.zeros_like(self._values)
            elif mode == 'gaussian':
                key = jax.random.PRNGKey(np.random.randint(0, 1000000))
                self._velocity = jax.random.normal(key, self._values.shape) * jnp.sqrt(1 / self._mass * Constants.kb * temperature)
            if self._sharding is not None:
                self._velocity = jax.device_put(self._velocity, self._sharding)
    
    """
    These methods are used to handle the force of the field.
    """

    def set_force(self, force):
        if self._values is None:
            raise ValueError("Field has no values. Set values before setting forces.")
        else:
            self.compare_shape(force, self._values)
            self._force = force

    def get_force(self):
        if self._force is None:
            raise ValueError("Force do not exist")
        else:
            return self._force

    def zero_force(self):
        if self._values is None:
            raise ValueError("Field has no values. Set values before zeroing forces.")
        else:
            self._force = jnp.zeros_like(self._values)

    def accumulate_force(self, force):
        if self._force is None:
            raise ValueError("Gradients do not exist. Set or zero forces before accumulating.")
        else:
            self.compare_shape(force, self._force)
            self._force += force

    """
    These methods are used to handle the energy of the field.
    """
    def get_kinetic_energy(self):
        if self._velocity is None:
            raise ValueError("Velocity is not set")
        elif self._mass is None:
            raise ValueError("Mass is not set")
        else:
            return 0.5 * jnp.sum(self._mass * jnp.square(self._velocity))

    def get_temperature(self):
        if self._velocity is None:
            raise ValueError("Velocity is not set")
        elif self._mass is None:
            raise ValueError("Mass is not set")
        return jnp.mean(self._mass * jnp.square(self._velocity)) / Constants.kb

    """
    Utility methods
    """
    def compare_shape(self, x, y):
        if x.shape != y.shape:
            raise ValueError("The two arrays to be compared have different shapes.")

    def compare_sharding(self, x, y):
        if x.sharding != y.sharding:
            raise ValueError("The two arrays to be compared has different sharding patterns.")

    def to_multi_devs(self, mesh: DeviceMesh):
        sharding = mesh.partition_sharding()
        if self._values is None:
            raise ValueError("Field has no values. Set values before put the array to multiple devices.")
        else:
            self._values = jax.device_put(self._values, sharding)
            self._sharding = sharding
        if self._mass is not None:
            self._mass = jax.device_put(self._mass, sharding)
        if self._velocity is not None:
            self._velocity = jax.device_put(self._velocity, sharding)
        if self._force is not None:
            self._force = jax.device_put(self._force, sharding)

    """
    These methods are used to handle the time evolution of the field.
    """
    def step(self, dt):
        pass

class FieldRn(Field):
    """
    R^n field. Values are stored as n-dimensional vectors. 
    """
    def __init__(self, lattice, name, dim, unit=None):
        super().__init__(lattice, name)
        self.fdim = dim
        self.ldim = lattice.dim
        self.shape = [lattice.size[i] for i in range(self.ldim)] + [self.fdim]
        self._values = jnp.zeros(self.shape)
        self.unit = unit
        self.integrator_class = {'optimization': GradientDescentIntegrator,
                                 'adiabatic': LeapFrogIntegrator, 
                                 'isothermal': LangevinIntegrator}

    @property
    def mean(self):
        """
        Calculate the average of the field over the lattice.
        """
        return jnp.mean(self.get_values(), axis=[i for i in range(self.ldim)])

    @property
    def var(self):
        """
        Calculate the variance of the field over the lattice.
        """
        return jnp.var(self.get_values(), axis=[i for i in range(self.ldim)])

    def set_local_value(self, loc, value):
        """
        Set the value of the field at a given location. 
        """
        assert type(loc) is tuple and len(loc) == self.ldim, "Location must be a tuple of length equal to the dimension of the lattice"
        self._values[loc] = value
        return


class FieldScalar(FieldRn):
    """
    Scalar field. Values are stored as scalars. 
    Example: mass, density, any onsite constant, etc.
    """
    def __init__(self, lattice, name, unit=None):
        super().__init__(lattice, name, dim=1, unit=unit)

class FieldR3(FieldRn):
    """
    3D vector field. Values are stored as 3-dimensional vectors. 
    Example: flexible dipole moment.
    """
    def __init__(self, lattice, name, unit=None):
        super().__init__(lattice, name, dim=3, unit=unit)

class FieldSO3(FieldRn):
    """
    3D vector field with fixed magnitude and flexible orientation. Values are stored as 3-dimensional vectors. 
    Example: rigid atomistic spin, molecular orientation, etc.
    """
    def __init__(self, lattice, name, unit=None):
        super().__init__(lattice, name, dim=3, unit=unit)
        self._magnitude = jnp.ones(self.shape[:-1])
        self.integrator_class = {'optimization': LLIntegrator,
                                 'adiabatic': ConservativeLLIntegrator,
                                 'isothermal': LLLangevinIntegrator}

    def set_magnitude(self, magnitude):
        if self._values is None:
            raise ValueError("Field has no values. Set values before setting magnitude.")
        elif jnp.isscalar(magnitude):
            self._magnitude = jnp.ones(self.shape[:-1]) * magnitude
        elif magnitude.shape == self._values.shape[:-1]:
            self._magnitude = magnitude
        else:
            raise ValueError("Magnitude must be a scalar or an array of the same size as the all but the last dimension of the field values.")

    def get_magnitude(self):
        if self._magnitude is None:
            raise ValueError("Magnitude is not set")
        else:
            return self._magnitude
    
    def normalize(self):
        if self._values is None:
            raise ValueError("Field has no values. Set values before normalizing.")
        elif self._magnitude is None:
            raise ValueError("Magnitude is not set. Set magnitude before normalizing.")
        else:
            self._values = self._values / jnp.linalg.norm(self._values, axis=-1, keepdims=True) * self._magnitude[..., None]
        return
    
    def to_multi_devs(self, mesh: DeviceMesh):
        sharding = mesh.replicate_sharding()
        if self._values is None:
            raise ValueError("Field has no values. Set values before put the array to multiple devices.")
        else:
            self._values = jax.device_put(self._values, sharding)
            self._sharding = sharding
        if self._magnitude is not None:
            self._magnitude = jax.device_put(self._magnitude, sharding)
        if self._mass is not None:
            self._mass = jax.device_put(self._mass, sharding)
        if self._velocity is not None:
            self._velocity = jax.device_put(self._velocity, sharding)
        if self._force is not None:
            self._force = jax.device_put(self._force, sharding)
        return

class LocalStrain3D(FieldRn):
    """
    Strain field on 3D lattice are separated into local contribution (local strain field) and global contribution (homogeneous strain associated to the supercell). 
    The local strain field is encoded by the local displacement vector v_i(R)/a_i (a_i: the lattice vector) associated with each lattice site at R.
    """
    def __init__(self, lattice, name):
        super().__init__(lattice, name, dim=3)

    @staticmethod
    def get_local_strain(values):
        '''
        Calculate the local strain field from the local displacement field.
        '''
        padded_values = jnp.pad(values, ((1, 1), (1, 1), (1, 1), (0, 0)), mode='wrap')
        grad_0, grad_1, grad_2 = jnp.gradient(padded_values, axis=(0, 1, 2))
        grad_0 = grad_0[1:-1, 1:-1, 1:-1]
        grad_1 = grad_1[1:-1, 1:-1, 1:-1]
        grad_2 = grad_2[1:-1, 1:-1, 1:-1]

        eta_1 = grad_0[..., 0]   # eta_xx
        eta_2 = grad_1[..., 1]   # eta_yy
        eta_3 = grad_2[..., 2]   # eta_zz
        eta_4 = (grad_1[..., 2] + grad_2[..., 1]) / 2   # eta_yz
        eta_5 = (grad_0[..., 2] + grad_2[..., 0]) / 2   # eta_xz
        eta_6 = (grad_0[..., 1] + grad_1[..., 0]) / 2   # eta_xy
        local_strain = jnp.stack([eta_1, eta_2, eta_3, eta_4, eta_5, eta_6], axis=-1)  # (l1, l2, l3, 6)
        return local_strain


class GlobalStrain(Field):
    """
    The homogeneous strain is represented by the strain tensor with Voigt convention, which is a 6-dimensional vector.
    """
    def __init__(self, lattice, name):
        super().__init__(lattice, name)
        self._values = jnp.zeros((6))
        self.integrator_class = {'optimization': GradientDescentIntegrator,
                                 'adiabatic': LeapFrogIntegrator,
                                 'isothermal': LangevinIntegrator}

    def to_multi_devs(self, mesh: DeviceMesh):
        sharding = mesh.replicate_sharding()
        if self._values is None:
            raise ValueError("Field has no values. Set values before put the array to multiple devices.")
        else:
            self._values = jax.device_put(self._values, sharding)
            self._sharding = sharding
        if self._mass is not None:
            self._mass = jax.device_put(self._mass, sharding)
        if self._velocity is not None:
            self._velocity = jax.device_put(self._velocity, sharding)
        if self._force is not None:
            self._force = jax.device_put(self._force, sharding)

    def get_excess_stress(self):
        return self.get_force() / self.lattice.ref_volume / Constants.bar 