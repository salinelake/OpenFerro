"""
Integrators for Landau-Lifshitz equations of motion. Fields are constrained to have fixed magnitude. 

This file is part of OpenFerro.
"""

import jax
from jax import jit
import jax.numpy as jnp
from openferro.units import Constants
from openferro.utilities import SO3_rotation
import logging
from openferro.integrator.base import Integrator


class ConservativeLLIntegrator(Integrator):
    """
    [For testing purposes: naive geometric integrator that may cause increasement in energy.]
    Adiabatic spin precession, i.e. Landau-Lifshitz equation of motion without dissipative damping term.
    
    The equation of motion is:
    dM/dt = -gamma M x B

    Parameters
    ----------
    dt : float
        Time step size
    gamma : float, optional
        Gyromagnetic ratio. If None, uses electron gyromagnetic ratio
    """
    def _step_x(self, M, B, gamma, dt):
        """
        Update the orientation of the magnetization M by rotating it around the magnetic field B

        Parameters
        ----------
        M : jax.Array
            The magnetization (shape=(\*, 3))
        B : jax.Array
            The magnetic field (shape=(\*, 3))
        gamma : float
            The gyromagnetic ratio
        dt : float
            The time step

        Returns
        -------
        jax.Array
            The updated magnetization (shape=(\*, 3))
        """
        R = SO3_rotation(B, gamma * dt)
        return (R * M[..., None, :]).sum(-1)

    def __init__(self, dt, gamma=None):
        super().__init__(dt)
        if gamma is None:
            self.gamma = Constants.electron_gyro_ratio
        else:
            self.gamma = gamma
        self.step_x = jit(self._step_x)

    def step(self, field, force_updater=None):
        """
        Update the field by one time step.

        Parameters
        ----------
        field : Field
            The field to be updated
        force_updater : callable, optional
            A function that updates the force of fields

        Returns
        -------
        Field
            The updated field
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
    
    The equation of motion is

    dM/dt = -gamma M x B - (gamma * alpha / \|M\|) * M x (M x B)
    
    Here gamma=(gyromagnetic ratio)/ (1+alpha^2) is the renormalized gyromagnetic ratio for simulating LLG equation in Landau-Lifshitz form.

    Parameters
    ----------
    dt : float
        Time step size
    alpha : float
        Gilbert damping constant
    gamma : float, optional
        Gyromagnetic ratio. If None, uses electron gyromagnetic ratio
    """
    def _step_b(self, M, B, alpha, Ms):
        """
        Update the magnetic field B by adding the term ( alpha / \|M\|) * (M x B)

        Parameters
        ----------
        M : jax.Array
            The magnetization (shape=(\*, 3))
        B : jax.Array
            The magnetic field (shape=(\*, 3))
        alpha : float
            The Gilbert damping constant
        Ms : jax.Array
            The magnitude of the magnetization (shape=(\*,))

        Returns
        -------
        jax.Array
            The updated magnetic field (shape=(\*, 3))
        """
        return B + alpha / Ms[..., None] * jnp.cross(M, B)
 
    def __init__(self, dt, alpha, gamma=None):
        super().__init__(dt, gamma)
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

        Parameters
        ----------
        field : Field
            The field to be updated
        force_updater : callable, optional
            A function that updates the force of fields

        Returns
        -------
        Field
            The updated field
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
    dM/dt = -gamma M x (B + b) - (gamma * alpha / \|M\|) * M x (M x (B + b))
    
    b is the stochastic force <b_i_alpha(t) b_j_beta(s)> = 2 * D * delta_ij * delta_alpha_beta * delta(t-s)
    A steady Boltzmann state requires D= alpha/(1+alpha^2) * kbT / gamma / \|m\|

    Parameters
    ----------
    dt : float
        Time step size
    temp : float
        Temperature
    alpha : float
        Gilbert damping constant
    gamma : float, optional
        Gyromagnetic ratio. If None, uses electron gyromagnetic ratio
    """
 
    def __init__(self, dt, temp, alpha, gamma=None):
        super().__init__(dt, alpha, gamma)
        self.kbT = Constants.kb * temp
        self.D_base = self.alpha/(1+self.alpha**2) * self.kbT / self.gamma  

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
        gaussian = jax.random.normal(key, field.get_values().shape) 
        if field._sharding != gaussian.sharding:
            gaussian = jax.device_put(gaussian, field._sharding)
        return gaussian    
    
    def step(self, key, field, force_updater=None):
        """
        Update the field by one time step.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key
        field : Field
            The field to be updated
        force_updater : callable, optional
            A function that updates the force of fields

        Returns
        -------
        Field
            The updated field
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
    
    The equation of motion is

    dM/dt = -gamma M x B
        
    Let M[i] be the spin configuration at time step i, Y[i] be the auxiliary spin configuration at time step i+1, B(M) be the magnetic field of configuration M.
    The SIB scheme is

    (step 1) Y[i] = M[i] - dt * gamma * (M[i]+Y[i])/2 x B(M[i])
    
    (step 2) M[i+1] = M[i] - dt * gamma * (M[i]+M[i+1])/2 x B((M[i] + Y[i])/2)

    Both equations are implicit, and can be solved iteratively through fixed-point iterations.

    Parameters
    ----------
    dt : float
        Time step size
    gamma : float, optional
        Gyromagnetic ratio. If None, uses electron gyromagnetic ratio
    max_iter : int, optional
        Maximum number of iterations for fixed-point iterations
    tol : float, optional
        Tolerance for convergence. Convergence is declared if Average[\|M_new - M_old\|/Ms] < tol
    """
    def _update_x(self, M, Ms, Y, B, step_size):
        """
        One iteration of (step 1) or (step 2) in the SIB scheme. Will be called iteratively through fixed-point iterations.

        Parameters
        ----------
        M : jax.Array
            Current magnetization
        Ms : jax.Array
            Magnitude of magnetization
        Y : jax.Array
            Previous iteration result
        B : jax.Array
            Magnetic field
        step_size : float
            Time step size

        Returns
        -------
        tuple
            (Updated Y, normalized difference average)
        """
        Y_new = M - step_size * jnp.cross((M + Y)/2, B)
        L1_diff = jnp.linalg.norm(Y_new - Y, axis=-1) 
        normalized_diff_avg = (L1_diff/ Ms).mean()
        return Y_new, normalized_diff_avg
    
    def __init__(self, dt, gamma=None, max_iter=10, tol=1e-6):
        super().__init__(dt)
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

        Parameters
        ----------
        field : Field
            The field to be updated
        force_updater : callable
            A function that updates the force of fields

        Returns
        -------
        Field
            The updated field
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
    
    The equation of motion is
    
    dM/dt = -gamma M x B - (gamma * alpha / \|M\|) * M x (M x B)
        
    Here gamma=(gyromagnetic ratio)/ (1+alpha^2) is the renormalized gyromagnetic ratio for simulating LLG equation in Landau-Lifshitz form.
    Let M[i] be the spin configuration at time step i, Y[i] be the auxiliary spin configuration at time step i+1.
    Let B(M) be the effective magnetic field that includes also the dissipative damping term.
    The SIB scheme is

    (step 1) Y[i] = M[i] - dt * gamma * (M[i]+Y[i])/2 x B(M[i])
    
    (step 2) M[i+1] = M[i] - dt * gamma * (M[i]+M[i+1])/2 x B((M[i] + Y[i])/2)
    
    Both equations are implicit, and can be solved iteratively through fixed-point iterations.

    Parameters
    ----------
    dt : float
        Time step size
    alpha : float
        Gilbert damping constant
    gamma : float, optional
        Gyromagnetic ratio. If None, uses electron gyromagnetic ratio
    max_iter : int, optional
        Maximum number of iterations for fixed-point iterations
    tol : float, optional
        Tolerance for convergence. Convergence is declared if Average[\|M_new - M_old\|/Ms averaged over lattice] < tol
    """
    def _update_x(self, M, Ms, Y, B, step_size):
        """
        One iteration of (step 1) or (step 2) in the SIB scheme. Will be called iteratively through fixed-point iterations.

        Parameters
        ----------
        M : jax.Array
            Current magnetization
        Ms : jax.Array
            Magnitude of magnetization
        Y : jax.Array
            Previous iteration result
        B : jax.Array
            Magnetic field
        step_size : float
            Time step size

        Returns
        -------
        tuple
            (Updated Y, normalized difference average)
        """
        Y_new = M - step_size * jnp.cross((M + Y)/2.0, B)
        L1_diff = jnp.linalg.norm(Y_new - Y, axis=-1) 
        normalized_diff_avg = (L1_diff/ Ms).mean()
        return Y_new, normalized_diff_avg
    
    def _update_b(self, M, B, alpha, Ms):
        """
        Update the magnetic field B by adding the term ( alpha / \|M\|) * (M x B)

        Parameters
        ----------
        M : jax.Array
            The magnetization (shape=(\*, 3))
        B : jax.Array
            The magnetic field (shape=(\*, 3))
        alpha : float
            The Gilbert damping constant
        Ms : jax.Array
            The magnitude of the magnetization (shape=(\*,))

        Returns
        -------
        jax.Array
            The updated magnetic field (shape=(\*, 3))
        """
        return B + alpha * jnp.cross(M, B) / Ms[..., None]
 
    def __init__(self, dt, alpha, gamma=None, max_iter=10, tol=1e-6):
        super().__init__(dt)
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

        Parameters
        ----------
        field : Field
            The field to be updated
        force_updater : callable
            A function that updates the force of fields

        Returns
        -------
        Field
            The updated field
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
    
    The equation of motion is

    dM/dt = -gamma M x (B + b) - (gamma * alpha / \|M\|) * M x (M x (B + b))
        
    Here gamma=(gyromagnetic ratio)/ (1+alpha^2) is the renormalized gyromagnetic ratio for simulating LLG equation in Landau-Lifshitz form.
    b is the stochastic force <b_i_alpha(t) b_j_beta(s)> = 2 * D * delta_ij * delta_alpha_beta * delta(t-s)
    A steady Boltzmann state requires D= alpha/(1+alpha^2) * kbT / gamma / \|m\|

    Let M[i] be the spin configuration at time step i, Y[i] be the auxiliary spin configuration at time step i+1.
    Let B(M[i]) be the effective magnetic field including also the dissipative damping term and the stochastic force.
    The SIB scheme is
    
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

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key
        field : Field
            The field to be updated
        force_updater : callable, optional
            A function that updates the force of fields

        Returns
        -------
        Field
            The updated field with new spin configuration (shape=(\*, 3))
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
