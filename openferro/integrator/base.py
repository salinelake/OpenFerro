"""
Base class for integrators.

This file is part of OpenFerro.
"""
import jax
from jax import jit
import jax.numpy as jnp
import logging


class Integrator:
    """
    Base class for integrators.

    Parameters
    ----------
    dt : float
        Time step size
    """
    def __init__(self, dt):
        self.dt = dt

    def step(self, field, force_updater=None):
        """
        Update the field by one time step.

        In most cases, the force will be updated for all fields in one setting, before any integrator is called.
        So the force_updater is not necessary in most cases. However, for some implicit integrators, the force_updater is needed.

        Parameters
        ----------
        field : Field
            The field to be updated
        force_updater : callable, optional
            A function that updates the force of all fields

        Returns
        -------
        Field
            The updated field
        """
        pass
