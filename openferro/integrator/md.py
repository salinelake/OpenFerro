"""
Integrators for unconstrained molecular dynamics.

This file is part of OpenFerro.

"""

import math
import numbers

import jax
from jax import jit
import jax.numpy as jnp

from openferro.integrator.base import Integrator
from openferro.parallelism import _random_normal
from openferro.units import Constants


def _validate_finite_scalar(name, value, *, minimum=0.0, strict=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or (value <= minimum if strict else value < minimum)
    ):
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be finite and {relation} {minimum}.")


class GradientDescentIntegrator(Integrator):
    """
    Gradient descent integrator.

    Parameters
    ----------
    dt : float
        Time step size
    """
    def _step_x(self, x, f, m, dt):
        return x + f / m * dt

    def __init__(self, dt):
        super().__init__(dt)
        self.step_x = jit(self._step_x)
    
    def step(self, field):
        """
        Update the field by one time step.

        Parameters
        ----------
        field : Field
            The field to be updated

        Returns
        -------
        Field
            The updated field
        """
        x0 = field.get_values()
        f0 = field.get_force()
        x0 = self.step_x(x0, f0, field.get_mass(), self.dt)
        field._set_values_from_integrator(x0)
        return field

class GradientDescentIntegrator_Strain(GradientDescentIntegrator):
    """
    Gradient descent integrator for global strain.

    Parameters
    ----------
    dt : float
        Time step size
    freeze_x : bool, optional
        Whether to freeze motion in x direction
    freeze_y : bool, optional  
        Whether to freeze motion in y direction
    freeze_z : bool, optional
        Whether to freeze motion in z direction
    """
    def __init__(self, dt, freeze_x=False, freeze_y=False, freeze_z=False):
        super().__init__(dt)
        if (not freeze_x) and (not freeze_y) and (not freeze_z):
            self.mask = jnp.ones((6,))
        else:
            self.mask = jnp.array([int(not freeze_x), int(not freeze_y), int(not freeze_z), 0, 0, 0])
        def _step_x(x, f, m, dt):
            return x + f / m * dt * self.mask
        self.step_x = jit(_step_x)
    
class LeapFrogIntegrator(Integrator):
    """
    Leapfrog integrator with velocities stored at half time steps.

    At state ``(x_n, v_(n-1/2))``, one step applies a full kick followed by a
    full drift and stores ``(x_(n+1), v_(n+1/2))``.

    Parameters
    ----------
    dt : float
        Time step size
    """
    velocity_time_offset = -0.5

    def _step_xp(self, x, v, f, m, dt):
        v += f / m * dt
        x += v * dt
        return x, v

    def __init__(self, dt):
        super().__init__(dt)
        self.step_xp = jit(self._step_xp)

    def step(self, field):
        """
        Update the field by one time step.

        Parameters
        ----------
        field : Field
            The field to be updated

        Returns
        -------
        Field
            The updated field
        """
        x0 = field.get_values()
        v0 = field.get_velocity()
        x0, v0 = self.step_xp(x0, v0, field.get_force(), field.get_mass(), self.dt)
        field._set_values_from_integrator(x0)
        field._set_velocity_from_integrator(v0)
        return field

class LeapFrogIntegrator_Strain(LeapFrogIntegrator):
    """
    Leapfrog integrator for global strain.

    Parameters
    ----------
    dt : float
        Time step size
    freeze_x : bool, optional
        Whether to freeze motion in x direction
    freeze_y : bool, optional
        Whether to freeze motion in y direction
    freeze_z : bool, optional
        Whether to freeze motion in z direction
    """
    def __init__(self, dt, freeze_x=False, freeze_y=False, freeze_z=False):
        super().__init__(dt)
        if (not freeze_x) and (not freeze_y) and (not freeze_z):
            self.mask = jnp.ones((6,))
        else:
            self.mask = jnp.array([int(not freeze_x), int(not freeze_y), int(not freeze_z), 0, 0, 0])
        def _step_xp(x, v, f, m, dt):
            v += f / m * dt
            v *= self.mask
            x += v * dt
            return x, v
        self.step_xp = jit(_step_xp)

class LangevinIntegrator(Integrator):
    """
    LFMiddle Langevin integrator with half-step velocities.

    Following J. Phys. Chem. A 123, 6056-6079 (2019), the stored state is
    ``(x_n, v_(n-1/2))`` and a step applies ``B-A-O-A``: a full force kick,
    half drift, exact Ornstein-Uhlenbeck thermostat, and half drift. The final
    kick of velocity-Verlet middle is merged with the next step's first kick.

    Parameters
    ----------
    dt : float
        Time step size
    temp : float
        Temperature
    tau : float
        Relaxation time
    """
    velocity_time_offset = -0.5

    def _step_lfmiddle(self, x, v, f, m, gaussian, dt, kbT, z1, z2):
        v += f / m * dt
        x += 0.5 * v * dt
        noise = gaussian * jnp.sqrt(kbT / m)
        v = z1 * v + z2 * noise
        x += 0.5 * v * dt
        return x, v

    def __init__(self, dt, temp, tau):
        super().__init__(dt)
        _validate_finite_scalar("Temperature", temp)
        _validate_finite_scalar("Relaxation time tau", tau, strict=True)
        self.temp = float(temp)
        self.kbT = Constants.kb * self.temp
        self.tau = float(tau)
        self.gamma = 1.0 / self.tau
        self.z1 = jnp.exp(-self.dt * self.gamma)
        self.z2 = jnp.sqrt(1 - jnp.exp(-2 * self.dt * self.gamma))
        self.step_lfmiddle = jit(self._step_lfmiddle)
    
    def get_noise(self, key, field):
        """
        Generate random noise for the Langevin dynamics.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key
        field : Field
            The field to generate noise for

        Returns
        -------
        jax.Array
            Random noise array
        """
        velocity = field.get_velocity()
        return _random_normal(
            key,
            velocity.shape,
            dtype=velocity.dtype,
            sharding=field._sharding,
        )
        
    def step(self, key, field):
        """
        Update the field by one time step.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key
        field : Field
            The field to be updated

        Returns
        -------
        Field
            The updated field
        """
        dt = self.dt
        mass = field.get_mass()
        force = field.get_force()
        v0 = field.get_velocity()
        x0 = field.get_values()
        gaussian = self.get_noise(key, field)
        x0, v0 = self.step_lfmiddle(
            x0, v0, force, mass, gaussian, dt, self.kbT, self.z1, self.z2
        )
        field._set_values_from_integrator(x0)
        field._set_velocity_from_integrator(v0)
        return field

class LangevinIntegrator_Strain(LangevinIntegrator):
    """
    Langevin integrator for global strain.

    Parameters
    ----------
    dt : float
        Time step size
    temp : float
        Temperature
    tau : float
        Relaxation time
    freeze_x : bool, optional
        Whether to freeze motion in x direction
    freeze_y : bool, optional
        Whether to freeze motion in y direction
    freeze_z : bool, optional
        Whether to freeze motion in z direction
    """
    def __init__(self, dt, temp, tau, freeze_x=False, freeze_y=False, freeze_z=False):
        super().__init__(dt, temp, tau)
        if (not freeze_x) and (not freeze_y) and (not freeze_z):
            self.mask = jnp.ones((6,))
        else:
            self.mask = jnp.array([int(not freeze_x), int(not freeze_y), int(not freeze_z), 0, 0, 0])
        def _step_lfmiddle(x, v, f, m, gaussian, dt, kbT, z1, z2):
            v += f / m * dt
            v *= self.mask
            x += 0.5 * v * dt
            noise = gaussian * jnp.sqrt(kbT / m)
            v = z1 * v + z2 * noise
            v *= self.mask
            x += 0.5 * v * dt
            return x, v

        self.step_lfmiddle = jit(_step_lfmiddle)
    
class OverdampedLangevinIntegrator(Integrator):
    """
    Overdamped Langevin integrator.

    Parameters
    ----------
    dt : float
        Time step size
    temp : float
        Temperature
    tau : float
        Relaxation time
    """
    def __init__(self, dt, temp, tau):
        super().__init__(dt)
        raise NotImplementedError("Overdamped Langevin integrator is not implemented yet.")
