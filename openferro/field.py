"""
Classes which define the fields on the lattice.
"""
# This file is part of OpenFerro.

import numpy as np
import jax
import jax.numpy as jnp
from openferro.units import Constants
from openferro.parallelism import DeviceMesh, _random_normal
from openferro.integrator.llg import *
from openferro.integrator.md import *

class Field:
    """
    Template class to define a field on a lattice.
    """
    def __init__(self, lattice, ID: str):
        """
        Initialize a field.

        Parameters
        ----------
        lattice : BravaisLattice3D
            Lattice object
        ID : str
            ID of the field
        """
        self.lattice = lattice
        self.ID = ID
        self._values = None
        self._mass = None
        self._velocity = None
        self._force = None
        self._sharding = None
        self.integrator = None
        self.integrator_class = None


    """
    These methods are used to handle the values of the field.
    """

    def set_values(self, values):
        self._values = values
        return

    def _prepare_real_values(self, values, expected_shape, name):
        """Convert and validate values for a real-valued field."""
        try:
            values = jnp.asarray(values)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric array-like values.") from exc

        if values.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}.")
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise ValueError(f"{name} must be real-valued.")
        if not jnp.issubdtype(values.dtype, jnp.floating):
            if not (
                jnp.issubdtype(values.dtype, jnp.integer)
                or jnp.issubdtype(values.dtype, jnp.bool_)
            ):
                raise TypeError(f"{name} must be numeric array-like values.")
            target_dtype = (
                self._values.dtype
                if self._values is not None
                and jnp.issubdtype(self._values.dtype, jnp.floating)
                else jnp.asarray(0.0).dtype
            )
            values = values.astype(target_dtype)
        if bool(jax.device_get(jnp.any(~jnp.isfinite(values)))):
            raise ValueError(f"{name} must contain only finite values.")
        return values

    def _set_values_from_integrator(self, values):
        """Store a built-in integrator result without a host synchronization."""
        values = jnp.asarray(values)
        self.compare_shape(values, self._values)
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise ValueError("Integrator output must be real floating-point values.")
        if self._sharding is not None and values.sharding != self._sharding:
            raise ValueError("Integrator output must preserve field sharding.")
        self._values = values

    def get_values(self):
        """
        Get the values of the field.

        Returns
        -------
        array_like
            Values of the field

        Raises
        ------
        ValueError
            If field has no values set
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
        mass = jnp.asarray(mass)
        if mass.ndim == 0:
            mass = jnp.zeros_like(self._values[..., 0]) + mass
        elif mass.shape != self._values.shape[:-1]:
            raise ValueError(
                "Mass must be a scalar or an array matching the lattice shape."
            )
        invalid = jnp.any((mass <= 0.0) | ~jnp.isfinite(mass))
        if bool(jax.device_get(invalid)):
            raise ValueError("Mass must be finite and strictly positive.")
        self._mass = mass[..., None]
        if self._sharding is not None:
            self._mass = jax.device_put(self._mass, self._sharding)
            
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
            velocity = jnp.asarray(velocity)
            self.compare_shape(velocity, self._values)
            if bool(jax.device_get(jnp.any(~jnp.isfinite(velocity)))):
                raise ValueError("Velocity must contain only finite values.")
            self._velocity = velocity
            if self._sharding is not None:
                self._velocity = jax.device_put(self._velocity, self._sharding)

    def _set_velocity_from_integrator(self, velocity):
        """Store a velocity produced from field arrays by a built-in integrator."""
        self.compare_shape(velocity, self._values)
        if self._sharding is not None and velocity.sharding != self._sharding:
            raise ValueError("Integrator output must preserve field sharding.")
        self._velocity = velocity

    def get_velocity(self):
        if self._velocity is None:
            raise ValueError("Velocity is not set")
        else:
            return self._velocity

    def init_velocity(self, mode='zero', temperature=None, seed=42, key=None):
        """Initialize stored velocities.

        Leapfrog and LFMiddle integrators interpret these values as half-step
        velocities. A Gaussian draw is already stationary for that momentum
        variable and requires no deterministic half-kick.
        """
        if self._values is None:
            raise ValueError("Set field values before initializing velocity.")
        if mode == 'zero':
            self._velocity = jnp.zeros_like(self._values)
        elif mode == 'gaussian':
            if temperature is None:
                raise ValueError("Temperature is required for Gaussian velocities.")
            if (
                not np.isscalar(temperature)
                or not np.isfinite(temperature)
                or temperature < 0
            ):
                raise ValueError("Temperature must be finite and non-negative.")
            if self._mass is None:
                raise ValueError("Mass must be set before initializing Gaussian velocities.")
            random_key = (
                jax.random.PRNGKey(seed)
                if key is None
                else key
            )
            self._velocity = (
                _random_normal(
                    random_key,
                    self._values.shape,
                    dtype=self._values.dtype,
                    sharding=self._sharding,
                )
                * jnp.sqrt(Constants.kb * temperature / self._mass)
            )
        else:
            raise ValueError(f"Unknown velocity initialization mode {mode!r}.")
    
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
            # raise ValueError("Velocity is not set")
            return 0.0
        elif self._mass is None:
            # raise ValueError("Mass is not set")
            return 0.0
        else:
            return 0.5 * jnp.sum(self._mass * jnp.square(self._velocity))

    def get_temperature(self):
        if self._velocity is None:
            # raise ValueError("Velocity is not set")
            return 0.0
        elif self._mass is None:
            # raise ValueError("Mass is not set")
            return 0.0
        else:
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
    These methods are used to handle the integrator of the field.
    """
    def set_integrator(self, integrator_class, dt, **kwargs):
        """
        Set the integrator according to the given integrator class. Set the time step.
        To be implemented by the subclasses.

        Parameters
        ----------
        integrator_class : str
            Class of integrator to use
        dt : float
            Time step
        **kwargs
            Additional arguments passed to integrator
        """
        pass

    def set_custom_integrator(self, integrator):
        self.integrator = integrator

class FieldRn(Field):
    """
    R^n field on a lattice. Values are stored as n-dimensional vectors.
    """
    def __init__(self, lattice, ID, dim, unit=None):
        super().__init__(lattice, ID)
        self.fdim = dim
        self.ldim = lattice.dim
        self.shape = [lattice.size[i] for i in range(self.ldim)] + [self.fdim]
        self._values = jnp.zeros(self.shape)
        self.unit = unit
        self.integrator_class = {'optimization': GradientDescentIntegrator,
                                 'adiabatic': LeapFrogIntegrator, 
                                 'isothermal': LangevinIntegrator}

    def set_values(self, values):
        """Set finite, real field values with the declared lattice shape.

        Integer and Boolean inputs are converted to the field's current
        floating-point dtype. Floating-point inputs retain their dtype.

        Parameters
        ----------
        values : array-like
            Values with shape ``(*lattice.size, field_dimension)``.

        Raises
        ------
        TypeError
            If values are not numeric.
        ValueError
            If values have the wrong shape, are complex, or are non-finite.
        """
        expected_shape = tuple(int(extent) for extent in self.shape)
        values = self._prepare_real_values(
            values, expected_shape, "Field values"
        )
        if self._sharding is not None:
            values = jax.device_put(values, self._sharding)
        self._values = values

    @property
    def mean(self):
        """
        Calculate the average of the field over the lattice.

        Returns
        -------
        array_like
            Mean value of field
        """
        return jnp.mean(self.get_values(), axis=[i for i in range(self.ldim)])

    @property
    def var(self):
        """
        Calculate the variance of the field over the lattice.

        Returns
        -------
        array_like
            Variance of field
        """
        return jnp.var(self.get_values(), axis=[i for i in range(self.ldim)])

    def set_local_value(self, loc, value):
        """
        Set the value of the field at a given location.

        Parameters
        ----------
        loc : tuple
            Location tuple with length equal to lattice dimension
        value : array_like
            Value to set at location
        """
        if not isinstance(loc, tuple) or len(loc) != self.ldim:
            raise ValueError(
                "Location must be a tuple with one integer index per lattice dimension."
            )
        if not all(isinstance(index, (int, np.integer)) for index in loc):
            raise TypeError("Location indices must be integers.")
        for index, extent in zip(loc, self._values.shape[:-1]):
            if index < -extent or index >= extent:
                raise IndexError(f"Field location {loc} is outside shape {self._values.shape[:-1]}.")

        try:
            value = jnp.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Local field value must be numeric array-like values."
            ) from exc
        if value.ndim == 0 and self.fdim == 1:
            value = value.reshape((1,))
        value = self._prepare_real_values(
            value, (self.fdim,), "Local field value"
        )

        values = self._values.at[loc].set(value)
        if self._sharding is not None:
            values = jax.device_put(values, self._sharding)
        self._values = values
        return

    def set_integrator(self, integrator_class, dt, temp=None, tau=None):
        """
        Set the integrator according to the given integrator class. Set the time step.

        Parameters
        ----------
        integrator_class : str
            Integrator class
        dt : float
            Time step
        temp : float, optional
            Temperature for isothermal integrator
        tau : float, optional
            Relaxation time for Langevin integrator

        Raises
        ------
        ValueError
            If integrator class not supported or required parameters missing
        """
        if integrator_class not in self.integrator_class:
            raise ValueError(f"Integrator class {integrator_class} is not supported for this field.")
        else:
            if integrator_class == 'isothermal':
                if temp is None or tau is None:
                    raise ValueError("Temperature and relaxation time must be specified for the isothermal integrator.")
                else:
                    integrator = self.integrator_class[integrator_class](dt, temp, tau)
            else:
                integrator = self.integrator_class[integrator_class](dt)
            self.integrator = integrator
        return


class MaskedFieldRn(FieldRn):
    """Euclidean lattice field constrained to fixed active sites.

    Values, velocities, and assembled forces are stored as full lattice arrays,
    but entries outside ``active_mask`` are projected to exact zero at every
    public and built-in-integrator commit point.  An optional small basis may
    additionally remove global linear modes from each field component.
    Masses remain finite and strictly positive on all lattice sites.

    The inherited :attr:`FieldRn.mean` and :attr:`FieldRn.var` retain their
    ordinary padded-lattice meaning.  Use :attr:`n_active_sites` and
    :attr:`active_dof` when an active-site thermodynamic count is required.

    Parameters
    ----------
    lattice : BravaisLattice3D
        Lattice on which the field is stored.
    ID : str
        Field identifier.
    dim : int
        Positive number of components per lattice site.
    active_mask : array-like of bool
        Boolean array with shape ``tuple(lattice.size)``.  At least one site
        must be active.
    constraint_basis : array-like, optional
        Full-lattice array with shape ``(*lattice.size, n_basis)``.  Its
        active-site columns must be finite and linearly independent.  Each
        field component is projected orthogonally to these columns.
    unit : optional
        Optional field unit metadata.
    """

    def __init__(
        self,
        lattice,
        ID,
        dim,
        active_mask,
        constraint_basis=None,
        unit=None,
    ):
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim <= 0:
            raise ValueError("Masked field dimension must be a positive integer.")

        try:
            host_mask = np.asarray(jax.device_get(active_mask))
        except (TypeError, ValueError) as exc:
            raise TypeError("active_mask must be a Boolean array.") from exc
        expected_shape = tuple(int(extent) for extent in lattice.size)
        if host_mask.shape != expected_shape:
            raise ValueError(f"active_mask must have shape {expected_shape}.")
        if host_mask.dtype != np.bool_:
            raise TypeError("active_mask must have Boolean dtype.")
        n_active_sites = int(host_mask.sum())
        if n_active_sites == 0:
            raise ValueError("active_mask must contain at least one active site.")

        super().__init__(lattice, ID, int(dim), unit=unit)
        self._active_mask = jnp.asarray(host_mask)
        self._n_active_sites = n_active_sites
        self._constraint_basis = None
        self._constraint_gram_inverse = None
        self._constraint_rank = 0

        if constraint_basis is not None:
            try:
                host_basis = np.asarray(jax.device_get(constraint_basis))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "constraint_basis must be a finite real array."
                ) from exc
            if host_basis.ndim != len(expected_shape) + 1:
                raise ValueError(
                    "constraint_basis must have shape "
                    f"{expected_shape + ('n_basis',)}."
                )
            if host_basis.shape[:-1] != expected_shape or host_basis.shape[-1] == 0:
                raise ValueError(
                    "constraint_basis must match the lattice shape and contain "
                    "at least one basis function."
                )
            if (
                not np.issubdtype(host_basis.dtype, np.number)
                or np.issubdtype(host_basis.dtype, np.complexfloating)
            ):
                raise TypeError("constraint_basis must be a finite real array.")
            host_basis = host_basis.astype(float, copy=False)
            if not np.all(np.isfinite(host_basis)):
                raise ValueError("constraint_basis must contain only finite values.")

            active_basis = host_basis[host_mask]
            constraint_rank = int(np.linalg.matrix_rank(active_basis))
            n_basis = int(active_basis.shape[1])
            if constraint_rank != n_basis:
                raise ValueError(
                    "constraint_basis columns must be linearly independent on "
                    "active sites."
                )
            if n_basis >= n_active_sites:
                raise ValueError(
                    "constraint_basis must leave at least one unconstrained "
                    "site mode."
                )

            masked_basis = np.where(host_mask[..., None], host_basis, 0.0)
            gram_inverse = np.linalg.inv(active_basis.T @ active_basis)
            dtype = self._values.dtype
            self._constraint_basis = jnp.asarray(masked_basis, dtype=dtype)
            self._constraint_gram_inverse = jnp.asarray(
                gram_inverse, dtype=dtype
            )
            self._constraint_rank = constraint_rank

    @property
    def active_mask(self):
        """Boolean lattice-shaped mask defining the active sites."""
        return self._active_mask

    @property
    def n_active_sites(self):
        """Number of active lattice sites as a host integer."""
        return self._n_active_sites

    @property
    def constraint_basis(self):
        """Optional full-lattice basis removed from every field component."""
        return self._constraint_basis

    @property
    def constraint_rank(self):
        """Number of independent basis functions removed per component."""
        return self._constraint_rank

    @property
    def active_dof(self):
        """Number of unconstrained active Euclidean degrees of freedom."""
        return (self._n_active_sites - self._constraint_rank) * self.fdim

    def _project_active(self, array):
        """Apply the fixed occupancy and optional linear constraints."""
        array = jnp.asarray(array)
        expected_shape = tuple(int(extent) for extent in self.shape)
        if array.shape != expected_shape:
            raise ValueError(f"Constrained array must have shape {expected_shape}.")
        if jnp.issubdtype(array.dtype, jnp.complexfloating):
            raise ValueError("Constrained arrays must be real-valued.")
        if not jnp.issubdtype(array.dtype, jnp.floating):
            array = array.astype(self._values.dtype)
        projected = jnp.where(
            self._active_mask[..., None], array, jnp.zeros((), dtype=array.dtype)
        )
        if self._constraint_basis is not None:
            basis = self._constraint_basis.astype(array.dtype)
            gram_inverse = self._constraint_gram_inverse.astype(array.dtype)
            overlaps = jnp.einsum(
                "...k,...d->kd", basis, projected, precision="highest"
            )
            coefficients = gram_inverse @ overlaps
            fitted = jnp.einsum(
                "...k,kd->...d", basis, coefficients, precision="highest"
            )
            projected = jnp.where(
                self._active_mask[..., None],
                projected - fitted,
                jnp.zeros((), dtype=array.dtype),
            )
        if self._sharding is not None:
            projected = jax.device_put(projected, self._sharding)
        return projected

    def set_values(self, values):
        """Set field values and clamp inactive sites to exact zero."""
        expected_shape = tuple(int(extent) for extent in self.shape)
        values = self._prepare_real_values(values, expected_shape, "Field values")
        super().set_values(self._project_active(values))

    def set_local_value(self, loc, value):
        """Set one local value and reapply the fixed active-site constraint."""
        super().set_local_value(loc, value)
        self.set_values(self._values)

    def set_mass(self, mass):
        """Set mass, requiring uniform active mass for linear constraints."""
        if self._constraint_basis is not None:
            host_mass = np.asarray(jax.device_get(mass))
            expected_shape = tuple(int(extent) for extent in self.shape[:-1])
            if host_mass.shape == expected_shape:
                host_mask = np.asarray(jax.device_get(self._active_mask))
                active_mass = host_mass[host_mask]
                if (
                    np.all(np.isfinite(active_mass))
                    and np.all(active_mass > 0.0)
                    and not np.allclose(
                        active_mass,
                        active_mass[0],
                        rtol=1.0e-12,
                        atol=0.0,
                    )
                ):
                    raise ValueError(
                        "Fields with constraint_basis require uniform mass on "
                        "active sites."
                    )
        super().set_mass(mass)

    def set_velocity(self, velocity):
        """Set finite velocities and clamp inactive sites to exact zero."""
        if self._values is None:
            raise ValueError("Field has no values. Set values before setting velocity.")
        velocity = jnp.asarray(velocity)
        self.compare_shape(velocity, self._values)
        if bool(jax.device_get(jnp.any(~jnp.isfinite(velocity)))):
            raise ValueError("Velocity must contain only finite values.")
        super().set_velocity(self._project_active(velocity))

    def init_velocity(self, mode='zero', temperature=None, seed=42, key=None):
        """Initialize velocities and discard all inactive-site components."""
        super().init_velocity(mode=mode, temperature=temperature, seed=seed, key=key)
        self.set_velocity(self._velocity)

    def set_force(self, force):
        """Store an assembled force projected onto the active sites."""
        super().set_force(self._project_active(force))

    def accumulate_force(self, force):
        """Accumulate only the active components of a force contribution."""
        super().accumulate_force(self._project_active(force))

    def _set_values_from_integrator(self, values):
        """Commit projected integrator values without host-value validation."""
        super()._set_values_from_integrator(self._project_active(values))

    def _set_velocity_from_integrator(self, velocity):
        """Commit projected integrator velocities without a host round trip."""
        super()._set_velocity_from_integrator(self._project_active(velocity))

    def get_temperature(self):
        """Return kinetic temperature using only active degrees of freedom."""
        if self._velocity is None or self._mass is None:
            return 0.0
        return 2.0 * self.get_kinetic_energy() / (Constants.kb * self.active_dof)

    def to_multi_devs(self, mesh: DeviceMesh):
        """Shard state, occupancy, and optional constraint data."""
        super().to_multi_devs(mesh)
        self._active_mask = jax.device_put(self._active_mask, self._sharding)
        if self._constraint_basis is not None:
            self._constraint_basis = jax.device_put(
                self._constraint_basis, self._sharding
            )
            self._constraint_gram_inverse = jax.device_put(
                self._constraint_gram_inverse, mesh.replicate_sharding()
            )


class FieldScalar(FieldRn):
    """
    Scalar field. Values are stored as scalars.
    Example: mass, density, any onsite constant, etc.
    """
    def __init__(self, lattice, ID, unit=None):
        super().__init__(lattice, ID, dim=1, unit=unit)

class FieldR3(FieldRn):
    """
    3D vector field. Values are stored as 3-dimensional vectors.
    Example: flexible dipole moment.
    """
    def __init__(self, lattice, ID, unit=None):
        super().__init__(lattice, ID, dim=3, unit=unit)

class FieldSO3(FieldRn):
    """
    3D vector field with fixed magnitude and flexible orientation. Values are stored as 3-dimensional vectors.
    Example: rigid atomistic spin, molecular orientation, etc.
    """
    def __init__(self, lattice, ID, unit=None):
        super().__init__(lattice, ID, dim=3, unit=unit)
        self._magnitude = jnp.ones(self.shape[:-1])
        self._values = self._values.at[..., 2].set(self._magnitude)
        self.integrator_class = {'optimization': LLSIBIntegrator,
                                 'adiabatic': ConservativeLLSIBIntegrator,
                                 'isothermal': LLSIBLangevinIntegrator}

    def _normalize_values(self, values):
        norms = jnp.linalg.norm(values, axis=-1, keepdims=True)
        invalid = jnp.any((norms <= 0.0) | ~jnp.isfinite(norms))
        if bool(jax.device_get(invalid)):
            raise ValueError("SO(3) field vectors must be finite and nonzero.")
        return values / norms * self._magnitude[..., None]

    def set_values(self, values):
        """Set orientations and normalize them to the configured magnitude."""
        values = jnp.asarray(values)
        expected_shape = tuple(int(extent) for extent in self.shape)
        if values.shape != expected_shape:
            raise ValueError(f"SO(3) field values must have shape {expected_shape}.")
        values = self._normalize_values(values)
        if self._sharding is not None:
            values = jax.device_put(values, self._sharding)
        self._values = values

    def _set_values_for_force_evaluation(self, values):
        """Set a finite unconstrained SIB stage value without normalization."""
        values = jnp.asarray(values)
        expected_shape = tuple(int(extent) for extent in self.shape)
        if values.shape != expected_shape:
            raise ValueError(f"SO(3) stage values must have shape {expected_shape}.")
        if bool(jax.device_get(jnp.any(~jnp.isfinite(values)))):
            raise ValueError("SO(3) stage values must be finite.")
        if self._sharding is not None:
            values = jax.device_put(values, self._sharding)
        self._values = values

    def set_local_value(self, loc, value):
        """Set one orientation and preserve the field magnitude invariant."""
        value = jnp.asarray(value)
        if value.shape != (3,):
            raise ValueError("Local SO(3) field value must have shape (3,).")
        invalid = jnp.any(~jnp.isfinite(value)) | (jnp.linalg.norm(value) <= 0.0)
        if bool(jax.device_get(invalid)):
            raise ValueError("SO(3) field vectors must be finite and nonzero.")
        super().set_local_value(loc, value)
        self.normalize()

    def set_magnitude(self, magnitude):
        if self._values is None:
            raise ValueError("Field has no values. Set values before setting magnitude.")

        magnitude = jnp.asarray(magnitude)
        if magnitude.ndim == 0:
            magnitude = jnp.ones(self.shape[:-1]) * magnitude
        elif magnitude.shape != self._values.shape[:-1]:
            raise ValueError(
                "Magnitude must be a scalar or an array matching the lattice shape."
            )
        invalid = jnp.any((magnitude <= 0.0) | ~jnp.isfinite(magnitude))
        if bool(jax.device_get(invalid)):
            raise ValueError("SO(3) field magnitude must be finite and strictly positive.")
        if self._sharding is not None:
            magnitude = jax.device_put(magnitude, self._sharding)
        self._magnitude = magnitude
        self.normalize()

    def get_magnitude(self):
        if self._magnitude is None:
            raise ValueError("Magnitude is not set")
        else:
            return self._magnitude

    def perturb(self, sigma, seed=42):
        key = jax.random.PRNGKey(seed)
        norms = jnp.linalg.norm(self._values, axis=-1, keepdims=True)
        invalid = jnp.any((norms <= 0.0) | ~jnp.isfinite(norms))
        if bool(jax.device_get(invalid)):
            raise ValueError("SO(3) field vectors must be finite and nonzero.")
        self._values = self._values / norms
        self._values += jax.random.normal(key, self._values.shape) * sigma
        self.normalize()
        return

    def normalize(self):
        if self._values is None:
            raise ValueError("Field has no values. Set values before normalizing.")
        elif self._magnitude is None:
            raise ValueError("Magnitude is not set. Set magnitude before normalizing.")
        else:
            values = self._normalize_values(self._values)
            if self._sharding is not None:
                values = jax.device_put(values, self._sharding)
            self._values = values
        return
    
    def init_velocity(self, mode='zero', temperature=None, seed=42, key=None):
        del mode, temperature, seed, key
        return

    def to_multi_devs(self, mesh: DeviceMesh):
        sharding = mesh.partition_sharding()
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


    def set_integrator(self, integrator_class, dt, temp=None, alpha=None):
        """
        Set the integrator according to the given integrator class. Set the time step.

        Parameters
        ----------
        integrator_class : str
            Integrator class
        dt : float
            Time step
        temp : float, optional
            Temperature for isothermal integrator
        alpha : float, optional
            Gilbert damping constant for Landau-Lifshitz equation of motion

        Raises
        ------
        ValueError
            If integrator class not supported or required parameters missing
        """
        if integrator_class not in self.integrator_class:
            raise ValueError(f"Integrator class {integrator_class} is not supported for this field.")
        else:
            if integrator_class == 'adiabatic':
                integrator = self.integrator_class[integrator_class](dt)
            elif integrator_class == 'optimization':
                if alpha is None:
                    raise ValueError("Gilbert damping constant must be specified for the optimization integrator.")
                else:
                    integrator = self.integrator_class[integrator_class](dt, alpha)
            elif integrator_class == 'isothermal':
                if alpha is None or temp is None:
                    raise ValueError("Gilbert damping constant and temperature must be specified for the isothermal integrator.")
                else:
                    integrator = self.integrator_class[integrator_class](dt, temp, alpha)
            self.integrator = integrator
        return
        
class LocalStrain3D(FieldRn):
    """
    Strain field on 3D lattice are separated into local contribution (local strain field) and global contribution (homogeneous strain associated to the supercell). 
    The local strain field is encoded by the local displacement vector v_i(R)/a_i (a_i: the lattice vector) associated with each lattice site at R.
    """
    def __init__(self, lattice, ID):
        super().__init__(lattice, ID, dim=3)

    @staticmethod
    def get_local_strain_symmetric(values):
        """
        Calculate the local strain field from the local displacement field.

        Parameters
        ----------
        values : array_like
            Local displacement field values

        Returns
        -------
        array_like
            Local strain field with shape (l1, l2, l3, 6)
        """
        padded_values = jnp.pad(values, ((1, 1), (1, 1), (1, 1), (0, 0)), mode='wrap') ## pad x,y,z axis with periodic boundary condition
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

    @staticmethod
    def get_local_strain(values):
        """
        Calculate the local strain field from the local displacement field.
        Implemented according to Physical Review B 52.9 (1995): 6301.

        Parameters
        ----------
        values : array_like
            Local displacement field values

        Returns
        -------
        array_like
            Local strain field with shape (l1, l2, l3, 6)
        """
        eta_1 = jnp.roll(values[..., 0], 1, axis=0) - values[..., 0]  # vx(R-dx) - vx(R)
        eta_1 = eta_1 + jnp.roll(eta_1, 1, axis=1) + jnp.roll(eta_1, 1, axis=2) + jnp.roll(jnp.roll(eta_1, 1, axis=1), 1, axis=2)
        eta_1 = eta_1 / 4.0

        eta_2 = jnp.roll(values[..., 1], 1, axis=1) - values[..., 1]  # vy(R-dy) - vy(R)
        eta_2 = eta_2 + jnp.roll(eta_2, 1, axis=0) + jnp.roll(eta_2, 1, axis=2) + jnp.roll(jnp.roll(eta_2, 1, axis=0), 1, axis=2)
        eta_2 = eta_2 / 4.0

        eta_3 = jnp.roll(values[..., 2], 1, axis=2) - values[..., 2]  # vz(R-dz) - vz(R)
        eta_3 = eta_3 + jnp.roll(eta_3, 1, axis=0) + jnp.roll(eta_3, 1, axis=1) + jnp.roll(jnp.roll(eta_3, 1, axis=0), 1, axis=1)
        eta_3 = eta_3 / 4.0

        eta_xy = jnp.roll(values[..., 1], 1, axis=0) - values[..., 1]   # vy(R-dx) - vy(R)
        eta_xy = eta_xy + jnp.roll(eta_xy, 1, axis=1) + jnp.roll(eta_xy, 1, axis=2) + jnp.roll(jnp.roll(eta_xy, 1, axis=1), 1, axis=2)
        
        eta_yx = jnp.roll(values[..., 0], 1, axis=1) - values[..., 0]  # vx(R-dy) - vx(R)
        eta_yx = eta_yx + jnp.roll(eta_yx, 1, axis=0) + jnp.roll(eta_yx, 1, axis=2) + jnp.roll(jnp.roll(eta_yx, 1, axis=0), 1, axis=2)

        eta_yz = jnp.roll(values[..., 2], 1, axis=1) - values[..., 2]   # vz(R-dy) - vz(R)
        eta_yz = eta_yz + jnp.roll(eta_yz, 1, axis=0) + jnp.roll(eta_yz, 1, axis=2) + jnp.roll(jnp.roll(eta_yz, 1, axis=0), 1, axis=2)

        eta_zy = jnp.roll(values[..., 1], 1, axis=2) - values[..., 1]   # vy(R-dz) - vy(R)
        eta_zy = eta_zy + jnp.roll(eta_zy, 1, axis=0) + jnp.roll(eta_zy, 1, axis=1) + jnp.roll(jnp.roll(eta_zy, 1, axis=0), 1, axis=1)

        eta_zx = jnp.roll(values[..., 0], 1, axis=2) - values[..., 0]   # vx(R-dz) - vx(R)
        eta_zx = eta_zx + jnp.roll(eta_zx, 1, axis=0) + jnp.roll(eta_zx, 1, axis=1) + jnp.roll(jnp.roll(eta_zx, 1, axis=0), 1, axis=1)

        eta_xz = jnp.roll(values[..., 2], 1, axis=0) - values[..., 2]   # vz(R-dx) - vz(R)
        eta_xz = eta_xz + jnp.roll(eta_xz, 1, axis=1) + jnp.roll(eta_xz, 1, axis=2) + jnp.roll(jnp.roll(eta_xz, 1, axis=1), 1, axis=2)

        eta_4 = (eta_yz + eta_zy) / 4.0
        eta_5 = (eta_xz + eta_zx) / 4.0
        eta_6 = (eta_xy + eta_yx) / 4.0

        local_strain = jnp.stack([eta_1, eta_2, eta_3, eta_4, eta_5, eta_6], axis=-1)  # (l1, l2, l3, 6)
        return local_strain


class GlobalStrain(Field):
    """
    Homogeneous strain in engineering Voigt notation.

    Values are ordered as ``(e_xx, e_yy, e_zz, 2e_yz, 2e_xz, 2e_xy)``.
    Pressure and volume reporting use ``det(I + epsilon)`` by default. The
    first-order trace volume remains available as an explicit compatibility
    mode when the field is added to a system.
    """
    def __init__(self, lattice, ID):
        super().__init__(lattice, ID)
        self._values = jnp.zeros((6))
        self.integrator_class = {'optimization': GradientDescentIntegrator_Strain,
                                 'adiabatic': LeapFrogIntegrator_Strain,
                                 'isothermal': LangevinIntegrator_Strain}

    def set_values(self, values):
        """Set the finite, real six-component engineering strain.

        Integer and Boolean inputs are converted to the field's current
        floating-point dtype. Floating-point inputs retain their dtype.

        Parameters
        ----------
        values : array-like
            Engineering Voigt strain with shape ``(6,)``.

        Raises
        ------
        TypeError
            If values are not numeric.
        ValueError
            If values do not have shape ``(6,)``, are complex, or are
            non-finite.
        """
        values = self._prepare_real_values(
            values, (6,), "Global strain values"
        )
        if self._sharding is not None:
            values = jax.device_put(values, self._sharding)
        self._values = values

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

    def get_excess_stress(self, reference_volume=None):
        """Return generalized excess stress normalized by a reference volume.

        This is a nominal/reference-volume measure, not a finite-strain Cauchy
        stress.  Omitting ``reference_volume`` preserves the historical lattice
        reference-volume normalization.
        """
        if reference_volume is None:
            reference_volume = self.lattice.ref_volume
        return self.get_force() / reference_volume / Constants.bar

    def set_integrator(self, integrator_class, dt, temp=None, tau=None, freeze_x=False, freeze_y=False, freeze_z=False):
        """
        Set the integrator according to the given integrator class. Set the time step.

        Parameters
        ----------
        integrator_class : str
            Integrator class
        dt : float
            Time step
        temp : float, optional
            Temperature for isothermal integrator
        tau : float, optional
            Relaxation time for Langevin integrator
        freeze_x : bool, optional
            Whether to freeze x-component of strain, by default False
        freeze_y : bool, optional
            Whether to freeze y-component of strain, by default False
        freeze_z : bool, optional
            Whether to freeze z-component of strain, by default False

        Raises
        ------
        ValueError
            If integrator class not supported or required parameters missing
        """
        if integrator_class not in self.integrator_class:
            raise ValueError(f"Integrator class {integrator_class} is not supported for this field.")
        else:
            if integrator_class == 'isothermal':
                if temp is None or tau is None:
                    raise ValueError("Temperature and relaxation time must be specified for the isothermal integrator.")
                else:
                    integrator = self.integrator_class[integrator_class](dt, temp, tau, freeze_x, freeze_y, freeze_z)
            else:
                integrator = self.integrator_class[integrator_class](dt, freeze_x, freeze_y, freeze_z)
            self.integrator = integrator
        return
