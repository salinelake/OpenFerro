"""
Integrators for unconstrained molecular dynamics.

This file is part of OpenFerro.

"""

import jax
from jax import jit
import jax.numpy as jnp
from openferro.units import Constants
from .base import Integrator


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
        field.set_values(x0)
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
    Leapfrog integrator.

    Parameters
    ----------
    dt : float
        Time step size
    """
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
        field.set_values(x0)
        field.set_velocity(v0)
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
    Langevin integrator as in J. Phys. Chem. A 2019, 123, 28, 6056-6079.
    ABOBA scheme: exp(i L dt) = exp(i Lx dt/2)exp(i Lt dt)exp(i Lx dt/2)exp(i Lp dt)
    
    Lx/2: half-step position update (_step_x)
    Lt: velocity update from damping and noise (_step_t)
    Lp: full-step velocity update (_step_p)

    Parameters
    ----------
    dt : float
        Time step size
    temp : float
        Temperature
    tau : float
        Relaxation time
    """
    def _step_p(self, v, f, m, dt):
        v += f / m * dt
        return v

    def _step_x(self, x, v, dt):
        x += 0.5 * v * dt
        return x

    def _step_t(self, v, noise, z1, z2):
        v = z1 * v + z2 * noise
        return v

    def __init__(self, dt, temp, tau):
        super().__init__(dt)
        self.temp = temp
        self.kbT = Constants.kb * temp
        self.tau = tau
        self.gamma = 1.0 / tau
        self.z1 = jnp.exp(-dt * self.gamma)
        self.z2 = jnp.sqrt(1 - jnp.exp(-2 * dt * self.gamma))
        self.step_p = jit(self._step_p)
        self.step_x = jit(self._step_x)
        self.step_t = jit(self._step_t)
    
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
        gaussian = jax.random.normal(key, field.get_velocity().shape) 
        if field._sharding != gaussian.sharding:
            gaussian = jax.device_put(gaussian, field._sharding)
        return gaussian
        
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
        v0 = self.step_p(v0, force, mass, dt)
        x0 = self.step_x(x0, v0, dt)
        gaussian = self.get_noise(key, field) 
        gaussian *= (self.kbT/ mass)**0.5
        v0 = self.step_t(v0, gaussian, self.z1, self.z2)
        x0 = self.step_x(x0, v0, dt)
        field.set_values(x0)
        field.set_velocity(v0)
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
        def _step_p(v, f, m, dt):
            v += f / m * dt
            v *= self.mask
            return v

        def _step_x(x, v, dt):
            x += 0.5 * v * dt
            return x

        def _step_t(v, noise, z1, z2):
            v = z1 * v + z2 * noise
            v *= self.mask
            return v 
        self.step_p = jit(_step_p)
        self.step_x = jit(_step_x)
        self.step_t = jit(_step_t)
    
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
