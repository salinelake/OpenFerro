"""
Base class for integrators.

This file is part of OpenFerro.
"""

import math
import numbers


class Integrator:
    """
    Base class for integrators.

    Parameters
    ----------
    dt : float
        Time step size
    """
    def __init__(self, dt):
        if (
            isinstance(dt, bool)
            or not isinstance(dt, numbers.Real)
            or not math.isfinite(float(dt))
            or dt <= 0.0
        ):
            raise ValueError("Integrator time step dt must be finite and positive.")
        self.dt = float(dt)

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
            A function that updates the force of fields

        Returns
        -------
        Field
            The updated field
        """
        pass
