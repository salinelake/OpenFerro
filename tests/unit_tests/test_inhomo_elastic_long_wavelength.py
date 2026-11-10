"""Long-wavelength limit of the local (inhomogeneous) cubic elastic energy.

A plane wave ``v(R) = U cos(k.R)`` with ``k = 2 pi m / L`` has, at long
wavelength, the continuum energy per cell ``1/2 eps:C:eps`` with
``eps = sym(grad v)``. The lattice energy from ``inhomo_elastic_energy`` must
reproduce it for every propagation direction, not only along a cubic axis.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from openferro.engine.elastic import inhomo_elastic_energy

B11, B12, B44 = 5.0, 2.0, 2.5
PARAMETERS = (B11, B12, B44)


def _grid(shape):
    axes = [np.arange(n) for n in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).astype(np.float64)


def _plane_wave(shape, mode, polarization):
    """Return ``v(R) = U cos(k.R)`` with ``k = 2 pi mode / shape`` (lattice units)."""
    mode = np.asarray(mode, dtype=np.float64)
    k = 2 * np.pi * mode / np.asarray(shape, dtype=np.float64)
    U = np.asarray(polarization, dtype=np.float64)
    phase = _grid(shape) @ k
    return np.cos(phase)[..., None] * U, k, U


def _continuum_energy(shape, k, U):
    """Continuum sum of 1/2 eps:C:eps over the cells of a plane wave."""
    phase = _grid(shape) @ k
    grad = -np.einsum("a,b,...->...ab", U, k, np.sin(phase))  # d_b v_a
    e = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    e11, e22, e33 = e[..., 0, 0], e[..., 1, 1], e[..., 2, 2]
    g23, g13, g12 = 2 * e[..., 1, 2], 2 * e[..., 0, 2], 2 * e[..., 0, 1]
    energy = 0.5 * B11 * (e11**2 + e22**2 + e33**2)
    energy += B12 * (e11 * e22 + e22 * e33 + e11 * e33)
    energy += 0.5 * B44 * (g23**2 + g13**2 + g12**2)
    return energy.sum()


def _ratio(shape, mode, polarization):
    v, k, U = _plane_wave(shape, mode, polarization)
    lattice = float(inhomo_elastic_energy(jnp.asarray(v), PARAMETERS))
    return lattice / _continuum_energy(shape, k, U)


# (name, mode, polarization) on a 64 x 64 x 8 grid, polarization scaled to unit norm.
CASES_LX64 = [
    ("L along x", (1, 0, 0), (1.0, 0.0, 0.0)),
    ("T along x, pol y", (1, 0, 0), (0.0, 1.0, 0.0)),
    ("L along [110]", (1, 1, 0), (1.0, 1.0, 0.0) / np.sqrt(2)),
    ("T along [110], pol [1-10]", (1, 1, 0), (1.0, -1.0, 0.0) / np.sqrt(2)),
]


@pytest.mark.parametrize("name,mode,pol", CASES_LX64)
def test_plane_wave_energy_matches_continuum_at_L64(name, mode, pol):
    ratio = _ratio((64, 64, 8), mode, pol)
    assert abs(ratio - 1.0) < 1e-2, f"{name}: E_lattice/E_continuum = {ratio:.5f}"


def test_longitudinal_111_matches_continuum_at_L64():
    # Cubic grid so that a [111] wave is periodic in every direction.
    ratio = _ratio((64, 64, 64), (1, 1, 1), (1.0, 1.0, 1.0) / np.sqrt(3))
    assert abs(ratio - 1.0) < 1e-2, f"L along [111]: ratio = {ratio:.5f}"


@pytest.mark.parametrize("name,mode,pol", CASES_LX64[2:])
def test_error_shrinks_quadratically_with_wavelength(name, mode, pol):
    # The leading correction is O(k^2) = O((2 pi / L)^2), so doubling L
    # should reduce the error by about 4.
    err_32 = abs(_ratio((32, 32, 8), mode, pol) - 1.0)
    err_64 = abs(_ratio((64, 64, 8), mode, pol) - 1.0)
    assert err_64 > 0.0
    assert 3.0 < err_32 / err_64 < 5.0, f"{name}: errors {err_32:.3e}, {err_64:.3e}"


def test_checkerboard_displacement_costs_positive_energy():
    # Zone-boundary mode k = pi along each axis: strongly non-long-wavelength,
    # but the energy must still be positive and nonzero.
    shape = (8, 8, 8)
    signs = (-1.0) ** np.sum(_grid(shape), axis=-1)
    v = signs[..., None] * np.array([1.0, 0.0, 0.0]) * 1e-3
    energy = float(inhomo_elastic_energy(jnp.asarray(v), PARAMETERS))
    assert energy > 0.0
    # Only the xx, zx and yx combinations contribute; per site D v = 2 v.
    n_sites = np.prod(shape)
    expected = (2 * B11 + 4 * B44) * n_sites * (1e-3) ** 2
    np.testing.assert_allclose(energy, expected, rtol=1e-10)
