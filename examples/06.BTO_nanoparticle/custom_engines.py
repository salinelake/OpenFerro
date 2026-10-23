"""Electrostatic boundary and direct-reference helpers for Phase A."""

import numpy as np
import jax
import jax.numpy as jnp
from geometry import radial_surface_shell_mask

from openferro.units import Constants

def surface_term_energy(local_modes, parameters):
    r"""Return the spherical Ewald boundary energy.

    ``parameters`` has order ``(Z_star**2 / epsilon_inf, V_box)``.  The field
    is already constrained, so the sum may include the full zero-padded box.

    .. math::

       \Delta H = \frac{Z_*^2}{\epsilon_\infty}
       \frac{|\sum_i u_i|^2}{6\epsilon_0 V_{\rm box}}.
    """
    if local_modes.ndim != 4 or local_modes.shape[-1] != 3:
        raise ValueError("local_modes must have shape (l1, l2, l3, 3).")
    parameters = jnp.asarray(parameters)
    if parameters.shape != (2,):
        raise ValueError("surface-term parameters must have shape (2,).")
    dipole_prefactor, box_volume = parameters
    total_local_mode = jnp.sum(local_modes, axis=(0, 1, 2))
    return (
        dipole_prefactor
        * jnp.dot(total_local_mode, total_local_mode)
        / (6.0 * Constants.epsilon0 * box_volume)
    )



def make_surface_k2_hardening_potential(
    geometry,
    shell_thickness_cells,
    k2_shift,
):
    """Build a positive quadratic onsite correction in a radial shell.

    The bulk onsite interaction already contributes ``k2 * |u_i|**2``.
    This correction adds ``k2_shift * |u_i|**2`` to cells in the shell, where
    a positive ``k2_shift`` suppresses the local soft mode.

    Parameters
    ----------
    geometry : ParticleGeometry
        Spherical particle geometry.
    shell_thickness_cells : float
        Shell thickness in reference-lattice cell units.
    k2_shift : float
        Positive quadratic onsite shift in eV/Angstrom^2.

    Returns
    -------
    callable
        Energy engine with signature ``(local_modes, parameters) -> scalar``.
    """
    if (
        not np.isscalar(k2_shift)
        or isinstance(k2_shift, (bool, np.bool_))
        or not np.isfinite(float(k2_shift))
    ):
        raise ValueError("k2_shift must be a finite scalar.")
    k2_shift = float(k2_shift)
    if k2_shift <= 0.0:
        raise ValueError("k2_shift must be strictly positive.")

    shell = radial_surface_shell_mask(geometry, shell_thickness_cells)
    shell_jax = jnp.asarray(shell)

    def surface_k2_hardening_potential(local_modes, parameters):
        if local_modes.ndim != 4 or local_modes.shape[-1] != 3:
            raise ValueError("local_modes must have shape (l1, l2, l3, 3).")
        u2 = jnp.sum(local_modes * local_modes, axis=-1)
        return k2_shift * jnp.sum(u2 * shell_jax)

    return surface_k2_hardening_potential

