"""
Methods for time integration.
"""
# This file is part of OpenFerro.
import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from openferro.units import Constants
from openferro.utilities import SO3_rotation


class Integrator:
    """
    Base class for integrators.
    """
    def __init__(self, dt):
        self.dt = dt
    def step(self, field):
        pass


"""
Integrators for unconstrained molecular dynamics
"""

class GradientDescentIntegrator(Integrator):
    """
    Gradient descent integrator.
    """
    def _step_x(self, x, f, m, dt):
        return x + f / m * dt

    def __init__(self, dt):
        super().__init__(dt)
        self.step_x = jit(self._step_x)
    
    def step(self, field):
        x0 = field.get_values()
        f0 = field.get_force()
        x0 = self.step_x(x0, f0, field.get_mass(), self.dt)
        field.set_values(x0)
        return field

class LeapFrogIntegrator(Integrator):
    """
    Leapfrog integrator.
    """
    def _step_xp(self, x, v, f, m, dt):
        v += f / m * dt
        x += v * dt
        return x, v

    def __init__(self, dt):
        super().__init__(dt)
        self.step_xp = jit(self._step_xp)

    def step(self, field):
        x0 = field.get_values()
        v0 = field.get_velocity()
        x0, v0 = self.step(x0, v0, field.get_force(), field.get_mass(), self.dt)
        field.set_values(x0)
        field.set_velocity(v0)
        return field
    

class LangevinIntegrator(Integrator):
    """
    Langevin integrator as in J. Phys. Chem. A 2019, 123, 28, 6056-6079. 
    ABOBA scheme: exp(i L dt) = exp(i Lx dt/2)exp(i Lt dt)exp(i Lx dt/2)exp(i Lp dt)
    Lx/2: half-step position update (_step_x)
    Lt: velocity update from damping and noise (_step_t)
    Lp: full-step velocity update (_step_p)
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
        gaussian = jax.random.normal(key, field.get_velocity().shape) 
        if field._sharding != gaussian.sharding:
            gaussian = jax.device_put(gaussian, field._sharding)
        return gaussian
        
    def step(self, key, field):
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

class OverdampedLangevinIntegrator(Integrator):
    """
    Overdamped Langevin integrator.
    """
    def __init__(self, dt, temp, tau):
        super().__init__(dt)
        raise NotImplementedError("Overdamped Langevin integrator is not implemented yet.")

"""
Integrators for Landau-Lifshitz equations of motion. Fields are constrained to have fixed magnitude. 
"""

class ConservativeLLIntegrator(Integrator):
    """
    Adiabatic spin precession, i.e. Landau-Lifshitz equation of motion without dissipative damping term.
    The equation of motion is:
    dM/dt = -gamma M x B
    """
    def _step_x(self, M, B, gamma, dt):
        """
        Update the orientation of the magnetization M by rotating it around the magnetic field B
        Args:
            M: the magnetization (shape=(*, 3))
            B: the magnetic field (shape=(*, 3))
            gamma: the gyromagnetic ratio
            dt: the time step
        Returns:
            the updated magnetization (shape=(*, 3))
        """
        R = SO3_rotation(B, - gamma * dt)
        return R @ M

    def __init__(self, dt, gamma):
        super().__init__(dt)
        self.gamma = gamma
        self.step_x = jit(self._step_x)

    def step(self, field):
        M = field.get_values()
        B = field.get_force()
        M = self.step_x(M, B, self.gamma, self.dt)
        field.set_values(M)
        return field

class LLIntegrator(ConservativeLLIntegrator):
    """
    Landau-Lifshitz equation of motion.
    See Eriksson, Olle, et al. Atomistic spin dynamics: foundations and applications. Oxford university press, 2017, Sec.7.4.5 for details.
    The equation of motion is:
    dM/dt = -gamma M x B - (gamma * alpha / |M|) * M x (M x B)
    """
    def _step_b(self, M, B, alpha, Ms):
        """
        Update the magnetic field B by adding the term ( alpha / |M|) * (M x B)
        Args:
            M: the magnetization (shape=(*, 3))
            B: the magnetic field (shape=(*, 3))
            alpha: the Gilbert damping constant
            Ms: the magnitude of the magnetization (shape=(*,))
        Returns:
            the updated magnetic field (shape=(*, 3))
        """
        return B + alpha / Ms[..., None] * jnp.cross(M, B)
 
    def __init__(self, dt, gamma, alpha):
        super().__init__(dt, gamma)
        self.alpha = alpha
        self.step_b = jit(self._step_b)

    def step(self, field):
        """
        A simplified Heun scheme. Less accurate but requires only one force evaluation.
        """
        M = field.get_values()
        B = field.get_force()
        Ms = field.get_magnitude()
        B_tmp = self.step_b(M, B, self.alpha, Ms)
        M_tmp = self.step_x(M, B_tmp, self.gamma, self.dt)
        B = self.step_b((M+M_tmp)/2, B, self.alpha, Ms)
        M = self.step_x(M, B, self.gamma, self.dt)
        field.set_values(M)
        field.set_force(B)
        return field

class LLLangevinIntegrator(LLIntegrator):
    """
    Stochastic Landau-Lifshitz equation of motion.
    See Eriksson, Olle, et al. Atomistic spin dynamics: foundations and applications. Oxford university press, 2017, Sec.7.4.5 for details.
    The equation of motion is:
    dM/dt = -gamma M x (B + b) - (gamma * alpha / |M|) * M x (M x (B + b))
    b is the stochastic force <b_i_alpha(t) b_j_beta(s)> = 2 * D * delta_ij * delta_alpha_beta * delta(t-s)
    A steady Boltzmann state requires D= alpha/(1+alpha^2) * kbT / gamma / |m|
    """
 
    def __init__(self, dt, gamma, alpha, temp):
        super().__init__(dt, gamma)
        self.alpha = alpha
        self.kbT = Constants.kb * temp
        self.D_base = alpha/(1+alpha**2) * self.kbT / gamma  

    def get_noise(self, key, field):
        gaussian = jax.random.normal(key, field.get_values().shape) 
        if field._sharding != gaussian.sharding:
            gaussian = jax.device_put(gaussian, field._sharding)
        return gaussian    
    
    def step(self, key, field):
        """
        A simplified Heun scheme. Less accurate but requires only one force evaluation.
        """
        gaussian = self.get_noise(key, field)
        gaussian *= (2 * self.D_base / field.get_magnitude()[..., None] * self.dt)**0.5
        Ms = field.get_magnitude()
        M = field.get_values()
        B = field.get_force() + gaussian
        B_tmp = self.step_b(M, B, self.alpha, Ms)
        M_tmp = self.step_x(M, B_tmp, self.gamma, self.dt)
        B = self.step_b((M+M_tmp)/2, B, self.alpha, Ms)
        M = self.step_x(M, B, self.gamma, self.dt)
        field.set_values(M)
        field.set_force(B)
        return field