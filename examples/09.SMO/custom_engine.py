import jax.numpy as jnp


def _polyval_ascending_jax(value, coefficients):
    result = jnp.zeros_like(value)
    for coefficient in coefficients[::-1]:
        result = result * value + coefficient
    return result


def smo_onsite_energy(local_mode, parameters):
    """Return the complete SrMnO3 onsite polynomial from the 2018 supplement.

    With ``u2 = ux^2 + uy^2 + uz^2``, the terms are
    ``kappa2*u2 + alpha*u2^2 + gamma*(ux^2 uy^2 + ...) + k1*u2^3``;
    ``k2`` multiplies ``sum_alpha u_alpha^4 (u_beta^2+u_gamma^2)``;
    ``k3`` multiplies ``ux^2 uy^2 uz^2``; and ``k4`` multiplies ``u2^4``.
    """
    kappa2, alpha, gamma, sixth_k1, sixth_k2, sixth_k3, eighth_k4 = parameters
    components2 = local_mode**2
    ux2, uy2, uz2 = components2[..., 0], components2[..., 1], components2[..., 2]
    norm2 = ux2 + uy2 + uz2
    pair_quartic = ux2 * uy2 + uy2 * uz2 + uz2 * ux2
    mixed_sixth = (
        ux2**2 * (uy2 + uz2)
        + uy2**2 * (ux2 + uz2)
        + uz2**2 * (ux2 + uy2)
    )
    onsite = (
        kappa2 * norm2
        + alpha * norm2**2
        + gamma * pair_quartic
        + sixth_k1 * norm2**3
        + sixth_k2 * mixed_sixth
        + sixth_k3 * ux2 * uy2 * uz2
        + eighth_k4 * norm2**4
    )
    return jnp.sum(onsite)


def anisotropic_first_neighbor_spin_energy(spin, parameters):
    """Return tetragonal first-neighbor exchange with unique bond counting."""
    J_ip, J_op = parameters
    energy = 0.0
    for axis, coupling in ((0, J_ip), (1, J_ip), (2, J_op)):
        neighbor = jnp.roll(spin, 1, axis=axis)
        energy -= coupling * jnp.sum(spin * neighbor)
    return energy


def isotropic_second_neighbor_spin_energy(spin, parameters):
    """Return the six unique simple-cubic second-neighbor bond directions."""
    (J2,) = parameters
    energy = 0.0
    for axes, shifts in (
        ((0, 1), (1, 1)),
        ((0, 1), (-1, 1)),
        ((0, 2), (1, 1)),
        ((0, 2), (-1, 1)),
        ((1, 2), (1, 1)),
        ((1, 2), (-1, 1)),
    ):
        neighbor = jnp.roll(spin, shifts, axis=axes)
        energy -= J2 * jnp.sum(spin * neighbor)
    return energy


def reference_matched_first_neighbor_spin_local_energy(
    local_mode, spin, parameters
):
    """Return bond-local first-neighbor exchange relative to G-type order.

    Parameter order is ``u_eq``, four in-plane coefficients, then four
    out-of-plane coefficients.  Coefficients are already converted to eV.
    The ``dot + 1`` reference subtraction makes both this energy and its
    local-mode force vanish for exact G-type spins.
    """
    u_eq = parameters[0]
    ip_coefficients = parameters[1:5]
    op_coefficients = parameters[5:9]
    norm2 = jnp.sum(local_mode**2, axis=-1)
    energy = 0.0
    for axis in range(3):
        neighbor_mode2 = jnp.roll(norm2, 1, axis=axis)
        neighbor_spin = jnp.roll(spin, 1, axis=axis)
        bond_s = (norm2 + neighbor_mode2) / (2.0 * u_eq**2)
        coefficients = ip_coefficients if axis < 2 else op_coefficients
        bond_J = _polyval_ascending_jax(bond_s, coefficients)
        spin_product = jnp.sum(spin * neighbor_spin, axis=-1)
        energy -= jnp.sum(bond_J * (spin_product + 1.0))
    return energy


def reference_matched_anisotropic_second_neighbor_spin_energy(spin, parameters):
    """Return relaxed-FE second-neighbor exchange relative to G-type order.

    Parameters are the [110], [1 -1 0], and common xz/yz couplings in eV.
    Each of the six positive-half-shell directions is counted once. G-type
    second neighbors are parallel, hence the ``dot - 1`` subtraction.
    """
    J2_ip_1, J2_ip_2, J2_op = parameters
    energy = 0.0
    directions = (
        ((0, 1), (1, 1), J2_ip_1),
        ((0, 1), (-1, 1), J2_ip_2),
        ((0, 2), (1, 1), J2_op),
        ((0, 2), (-1, 1), J2_op),
        ((1, 2), (1, 1), J2_op),
        ((1, 2), (-1, 1), J2_op),
    )
    for axes, shifts, coupling in directions:
        neighbor = jnp.roll(spin, shifts, axis=axes)
        spin_product = jnp.sum(spin * neighbor, axis=-1)
        energy -= coupling * jnp.sum(spin_product - 1.0)
    return energy
