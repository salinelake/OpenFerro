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
import warnings
import logging

class Integrator:
    """
    Base class for integrators.
    """
    def __init__(self, dt):
        self.dt = dt

    def step(self, field, force_updater=None):
        """
        Update the field by one time step. In most cases, the force will be updated for all fields in one setting, before any integrator is called.
        So the force_updater is not necessary in most cases. However, for some implicit integrators, the force_updater is needed. 
        Args:
            field: the field to be updated
            force_updater: a function that updates the force of all field. 
        """
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
        """
        Update the field by one time step. 
        """
        x0 = field.get_values()
        f0 = field.get_force()
        x0 = self.step_x(x0, f0, field.get_mass(), self.dt)
        field.set_values(x0)
        return field

class GradientDescentIntegrator_Strain(GradientDescentIntegrator):
    """
    Gradient descent integrator for global strain.
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
        """
        Update the field by one time step. 
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
    """
    def __init__(self, dt, temp, tau):
        super().__init__(dt)
        raise NotImplementedError("Overdamped Langevin integrator is not implemented yet.")

"""
Integrators for Landau-Lifshitz equations of motion. Fields are constrained to have fixed magnitude. 
"""

class ConservativeLLIntegrator(Integrator):
    """
    [For testing purposes: naive geometric integrator that may cause increasement in energy.]
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
        R = SO3_rotation(B, gamma * dt)
        return (R * M[..., None, :]).sum(-1)

    def __init__(self, dt, gamma=None):
        super().__init__(dt)
        """
        Args:
            dt: the time step
            gamma: the gyromagnetic ratio
        """
        if gamma is None:
            self.gamma = Constants.electron_gyro_ratio
        else:
            self.gamma = gamma
        self.step_x = jit(self._step_x)

    def step(self, field, force_updater=None):
        """
        Update the field by one time step. 
        """
        M = field.get_values()
        B = field.get_force()
        M = self.step_x(M, B, self.gamma, self.dt)
        field.set_values(M)
        return field

class LLIntegrator(ConservativeLLIntegrator):
    """
    [For testing purposes: naive geometric integrator.]
    Landau-Lifshitz equation of motion.
    See Eriksson, Olle, et al. Atomistic spin dynamics: foundations and applications. Oxford university press, 2017, Sec.7.4.5 for details.
    The equation of motion is:
    dM/dt = -gamma M x B - (gamma * alpha / |M|) * M x (M x B)
    Here gamma=(gyromagnetic ratio)/ (1+alpha^2) is the renormalized gyromagnetic ratio for simulating LLG equation in Landau-Lifshitz form.
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
 
    def __init__(self, dt, alpha, gamma=None):
        super().__init__(dt, gamma)
        """
        Args:
            dt: the time step
            alpha: the Gilbert damping constant
            gamma: the gyromagnetic ratio
        """
        ## get the renormalized gyromagnetic ratio for simulating LLG equation in Landau-Lifshitz form
        if gamma is None:
            self.gamma = Constants.electron_gyro_ratio / (1 + alpha**2)
        else:
            self.gamma = gamma / (1 + alpha**2)
        self.alpha = alpha
        self.step_b = jit(self._step_b)

    def step(self, field, force_updater=None):
        """
        Update the field by one time step. 
        """
        Ms = field.get_magnitude()
        M = field.get_values()
        B = field.get_force()
        B = self.step_b(M, B, self.alpha, Ms)
        M = self.step_x(M, B, self.gamma, self.dt)
        field.set_values(M)
        # field.set_force(B)
        return field

class LLLangevinIntegrator(LLIntegrator):
    """
    [For testing purposes: naive geometric integrator.]
    Stochastic Landau-Lifshitz equation of motion.
    See Eriksson, Olle, et al. Atomistic spin dynamics: foundations and applications. Oxford university press, 2017, Sec.7.4.5 for details.
    The equation of motion is:
    dM/dt = -gamma M x (B + b) - (gamma * alpha / |M|) * M x (M x (B + b))
    b is the stochastic force <b_i_alpha(t) b_j_beta(s)> = 2 * D * delta_ij * delta_alpha_beta * delta(t-s)
    A steady Boltzmann state requires D= alpha/(1+alpha^2) * kbT / gamma / |m|
    """
 
    def __init__(self, dt, temp, alpha, gamma=None):
        super().__init__(dt, alpha, gamma)
        """
        Args:
            dt: the time step
            temp: the temperature
            alpha: the Gilbert damping constant
            gamma: the gyromagnetic ratio
        """
        self.kbT = Constants.kb * temp
        self.D_base = self.alpha/(1+self.alpha**2) * self.kbT / self.gamma  

    def get_noise(self, key, field):
        gaussian = jax.random.normal(key, field.get_values().shape) 
        if field._sharding != gaussian.sharding:
            gaussian = jax.device_put(gaussian, field._sharding)
        return gaussian    
    
    def step(self, key, field, force_updater=None):
        """
        Update the field by one time step. 
        """
        Ms = field.get_magnitude()
        M = field.get_values()
        gaussian = self.get_noise(key, field)
        gaussian *= (2 * self.D_base / Ms[..., None] / self.dt)**0.5
        B = field.get_force() + gaussian
        B = self.step_b(M, B, self.alpha, Ms)
        M = self.step_x(M, B, self.gamma, self.dt)
        field.set_values(M)
        return field

class ConservativeLLSIBIntegrator(Integrator):
    """
    [Semi-implicit B (SIB) scheme. (J. Phys.: Condens. Matter 22 (2010) 176001) ]
    Adiabatic spin precession, i.e. Landau-Lifshitz equation of motion without dissipative damping term.
    The equation of motion is:
        dM/dt = -gamma M x B
    Let M[i] be the spin configuration at time step i, Y[i] be the auxiliary spin configuration at time step i+1, B(M) be the magnetic field of configuration M.
    The SIB scheme is:  
        (step 1) Y[i] = M[i] - dt * gamma * (M[i]+Y[i])/2 x B(M[i])
        (step 2) M[i+1] = M[i] - dt * gamma * (M[i]+M[i+1])/2 x B((M[i] + Y[i])/2)
    Both equations are implicit, and can be solved iteratively through fixed-point iterations.
    """
    def _update_x(self, M, Ms, Y, B, step_size):
        """
        One iteration of (step 1) or (step 2) in the SIB scheme. Will be called iteratively through fixed-point iterations.
        """
        Y_new = M - step_size * jnp.cross((M + Y)/2, B)
        L1_diff = jnp.linalg.norm(Y_new - Y, axis=-1) 
        normalized_diff_avg = (L1_diff/ Ms).mean()
        return Y_new, normalized_diff_avg
    
    def __init__(self, dt, gamma=None, max_iter=10, tol=1e-6):
        super().__init__(dt)
        """
        Args:
            dt: the time step
            gamma: the gyromagnetic ratio
            max_iter: the maximum number of iterations for fixed-point iterations
            tol: the tolerance for convergence. Convergence is declared if Average[|M_new - M_old|/Ms] < tol
        """
        if gamma is None:
            self.gamma = Constants.electron_gyro_ratio
        else:
            self.gamma = gamma
        self.max_iter = max_iter  
        self.tol = tol
        self.update_x = jit(self._update_x)
    
    def step(self, field, force_updater):
        """
        Iteratively solve the implicit equations (step 1) and (step 2) in the SIB scheme.
        Args:
            field: the field to be updated
            force_updater: a function that updates the force of all fields
        Returns:
            the updated spin configuration (shape=(*, 3))
        """
        force_updater()
        M = field.get_values()
        Ms = field.get_magnitude()
        B = field.get_force()
        Y = M.copy()
        ## step 1
        for i in range(self.max_iter):
            Y, normalized_diff_avg = self.update_x(M, Ms, Y, B, self.dt * self.gamma)
            if normalized_diff_avg < self.tol:
                break
            if i == self.max_iter - 1:
                logging.warning("SIB integrator for field '{}' does not converge: fixed-point iterations for step 1 exceed {}. The current tolerance for convergence is {} (in terms of |M_new - M_old|/Ms, averaged over lattice).".format(field.ID, self.max_iter, self.tol))
                logging.warning("If this warning happens frequently, consider decreasing the time step.")
        ## set the current configuration as the auxiliary configuration Y, then update the force
        field.set_values( (Y + M)/2 )
        force_updater()
        B = field.get_force()
        ## step 2
        for i in range(self.max_iter):
            Y, normalized_diff_avg = self.update_x(M, Ms, Y, B, self.dt * self.gamma)
            if normalized_diff_avg < self.tol:
                break
            if i == self.max_iter - 1:
                logging.warning("SIB integrator for field '{}' does not converge: fixed-point iterations for step 2 exceed {}. The current tolerance for convergence is {} (in terms of |M_new - M_old|/Ms, averaged over lattice).".format(field.ID, self.max_iter, self.tol))
                logging.warning("If this warning happens frequently, consider decreasing the time step dt.")
        field.set_values(Y)
        return field

class LLSIBIntegrator(Integrator):
    """
    Landau-Lifshitz equation of motion.
    See Eriksson, Olle, et al. Atomistic spin dynamics: foundations and applications. Oxford university press, 2017, Sec.7.4.5 for details.
    The equation of motion is:
        dM/dt = -gamma M x B - (gamma * alpha / |M|) * M x (M x B)
    Here gamma=(gyromagnetic ratio)/ (1+alpha^2) is the renormalized gyromagnetic ratio for simulating LLG equation in Landau-Lifshitz form.
    Let M[i] be the spin configuration at time step i, Y[i] be the auxiliary spin configuration at time step i+1.
    Let B(M) be the effective magnetic field that includes also the dissipative damping term.
    The SIB scheme is:  
        (step 1) Y[i] = M[i] - dt * gamma * (M[i]+Y[i])/2 x B(M[i])
        (step 2) M[i+1] = M[i] - dt * gamma * (M[i]+M[i+1])/2 x B((M[i] + Y[i])/2)
    Both equations are implicit, and can be solved iteratively through fixed-point iterations.
    """
    def _update_x(self, M, Ms, Y, B, step_size):
        """
        One iteration of (step 1) or (step 2) in the SIB scheme. Will be called iteratively through fixed-point iterations.
        """
        Y_new = M - step_size * jnp.cross((M + Y)/2.0, B)
        L1_diff = jnp.linalg.norm(Y_new - Y, axis=-1) 
        normalized_diff_avg = (L1_diff/ Ms).mean()
        return Y_new, normalized_diff_avg
    
    def _update_b(self, M, B, alpha, Ms):
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
        return B + alpha * jnp.cross(M, B) / Ms[..., None]
 
    def __init__(self, dt, alpha, gamma=None, max_iter=10, tol=1e-6):
        super().__init__(dt)
        """
        Args:
            dt: the time step
            alpha: the Gilbert damping constant
            gamma: the gyromagnetic ratio
            max_iter: the maximum number of iterations for fixed-point iterations
            tol: the tolerance for convergence. Convergence is declared if Average[|M_new - M_old|/Ms averaged over lattice] < tol
        """
        ## get the renormalized gyromagnetic ratio for simulating LLG equation in Landau-Lifshitz form
        self.alpha = alpha
        if gamma is None:
            self.gamma = Constants.electron_gyro_ratio / (1 + alpha**2)
        else:
            self.gamma = gamma / (1 + alpha**2)
        self.max_iter = max_iter  
        self.tol = tol
        self.update_x = jit(self._update_x)
        self.update_b = jit(self._update_b)

    def step(self, field, force_updater):
        """
        Iteratively solve the implicit equations (step 1) and (step 2) in the SIB scheme.
        Args:
            field: the field to be updated
            force_updater: a function that updates the force of all fields
        Returns:
            the updated spin configuration (shape=(*, 3))
        """
        force_updater()
        M = field.get_values()
        Ms = field.get_magnitude()
        B = field.get_force()
        B = self.update_b(M, B, self.alpha, Ms)
        Y = M.copy()
        ## step 1
        for i in range(self.max_iter):
            Y, normalized_diff_avg = self.update_x(M, Ms, Y, B, self.dt * self.gamma)
            if normalized_diff_avg < self.tol:
                break
            if i == self.max_iter - 1:
                logging.warning("SIB integrator for field '{}' does not converge: fixed-point iterations for step 1 exceed {}. The current tolerance for convergence is {} (in terms of |M_new - M_old|/Ms, averaged over lattice).".format(field.ID, self.max_iter, self.tol))
                logging.warning("If this warning happens frequently, consider decreasing the time step.")
        ## set the current configuration as the auxiliary configuration Y, then update the force
        field.set_values( (Y + M)/2 )
        force_updater()
        B = field.get_force()
        B = self.update_b(M, B, self.alpha, Ms)
        ## step 2
        for i in range(self.max_iter):
            Y, normalized_diff_avg = self.update_x(M, Ms, Y, B, self.dt * self.gamma)
            if normalized_diff_avg < self.tol:
                break
            if i == self.max_iter - 1:
                logging.warning("SIB integrator for field '{}' does not converge: fixed-point iterations for step 2 exceed {}. The current tolerance for convergence is {} (in terms of |M_new - M_old|/Ms, averaged over lattice).".format(field.ID, self.max_iter, self.tol))
                logging.warning("If this warning happens frequently, consider decreasing the time step.")
        field.set_values(Y)
        return field

class LLSIBLangevinIntegrator(LLSIBIntegrator):
    """
    Stochastic Landau-Lifshitz equation of motion.
    See Eriksson, Olle, et al. Atomistic spin dynamics: foundations and applications. Oxford university press, 2017, Sec.7.4.5 for details.
    The equation of motion is:
        dM/dt = -gamma M x (B + b) - (gamma * alpha / |M|) * M x (M x (B + b))
    Here gamma=(gyromagnetic ratio)/ (1+alpha^2) is the renormalized gyromagnetic ratio for simulating LLG equation in Landau-Lifshitz form.
    b is the stochastic force <b_i_alpha(t) b_j_beta(s)> = 2 * D * delta_ij * delta_alpha_beta * delta(t-s)
    A steady Boltzmann state requires D= alpha/(1+alpha^2) * kbT / gamma / |m|
    Let M[i] be the spin configuration at time step i, Y[i] be the auxiliary spin configuration at time step i+1.
    Let B(M[i]) be the effective magnetic field including also the dissipative damping term and the stochastic force.
    The SIB scheme is:  
        (step 1) Y[i] = M[i] - dt * gamma * (M[i]+Y[i])/2 x B(M[i])
        (step 2) M[i+1] = M[i] - dt * gamma * (M[i]+M[i+1])/2 x B((M[i] + Y[i])/2)
    Both equations are implicit, and can be solved iteratively through fixed-point iterations.
    """
 
    def __init__(self, dt, temp, alpha, gamma=None, max_iter=10, tol=1e-6):
        super().__init__(dt, alpha, gamma, max_iter, tol)
        """
        Args:
            dt: the time step
            temp: the temperature
            alpha: the Gilbert damping constant
            gamma: the gyromagnetic ratio
        """
        self.kbT = Constants.kb * temp
        self.D_base = self.alpha/(1+self.alpha**2) * self.kbT / self.gamma  
        self.noise_max = jnp.sqrt(2* jnp.abs(jnp.log(self.dt)))
    def get_noise(self, key, field):
        gaussian = jax.random.normal(key, field.get_values().shape) 
        if field._sharding != gaussian.sharding:
            gaussian = jax.device_put(gaussian, field._sharding)
        return gaussian    
    
    def step(self, key, field, force_updater=None):
        """
        Iteratively solve the implicit equations (step 1) and (step 2) in the SIB scheme.
        Args:
            field: the field to be updated
            force_updater: a function that updates the force of all fields
        Returns:
            the updated spin configuration (shape=(*, 3))
        """
        M = field.get_values()
        Ms = field.get_magnitude()
        ## get the effective magnetic field for step 1
        force_updater()
        B = field.get_force()
        gaussian = self.get_noise(key, field)
        # gaussian = jnp.clip(gaussian, -self.noise_max, self.noise_max)
        gaussian *= (2 * self.D_base / Ms[..., None] / self.dt)**0.5
        B += gaussian
        B = self.update_b(M, B, self.alpha, Ms)
        Y = M.copy()
        ## step 1
        for i in range(self.max_iter):
            Y, normalized_diff_avg = self.update_x(M, Ms, Y, B, self.dt * self.gamma)
            if normalized_diff_avg < self.tol:
                break
            if i == self.max_iter - 1:
                logging.warning("SIB integrator for field '{}' does not converge: fixed-point iterations for step 1 exceed {}. The current tolerance for convergence is {} (in terms of |M_new - M_old|/Ms, averaged over lattice).".format(field.ID, self.max_iter, self.tol))
                logging.warning("If this warning happens frequently, consider decreasing the time step.")
        ## set the current configuration as the auxiliary configuration Y
        field.set_values( (Y + M)/2 )
        ## get the effective magnetic field for step 2
        force_updater()
        B = field.get_force()
        B += gaussian
        B = self.update_b(field.get_values(), B, self.alpha, Ms)
        ## step 2
        for i in range(self.max_iter):
            Y, normalized_diff_avg = self.update_x(M, Ms, Y, B, self.dt * self.gamma)
            if normalized_diff_avg < self.tol:
                break
            if i == self.max_iter - 1:
                logging.warning("SIB integrator for field '{}' does not converge: fixed-point iterations for step 2 exceed {}. The current tolerance for convergence is {} (in terms of |M_new - M_old|/Ms, averaged over lattice).".format(field.ID, self.max_iter, self.tol))
                logging.warning("If this warning happens frequently, consider decreasing the time step.")
        field.set_values(Y)
        return field
