from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


ArrayEnergy = Callable[[jax.Array], jax.Array]


def central_difference_force(
    energy: ArrayEnergy,
    values,
    *,
    step: float = 1.0e-5,
) -> np.ndarray:
    """Return ``-dE/dx`` from independent componentwise central differences."""
    reference = np.asarray(values, dtype=np.float64)
    force = np.empty_like(reference)

    for index in np.ndindex(reference.shape):
        plus = reference.copy()
        minus = reference.copy()
        plus[index] += step
        minus[index] -= step
        e_plus = float(jax.device_get(energy(jnp.asarray(plus))))
        e_minus = float(jax.device_get(energy(jnp.asarray(minus))))
        force[index] = -(e_plus - e_minus) / (2.0 * step)

    return force


def assert_force_matches_finite_difference(
    energy: ArrayEnergy,
    values,
    *,
    step: float = 1.0e-5,
    rtol: float = 2.0e-6,
    atol: float = 2.0e-7,
) -> None:
    """Compare autodiff and finite-difference forces with useful diagnostics."""
    values = jnp.asarray(values, dtype=jnp.float64)
    autodiff = np.asarray(jax.device_get(-jax.grad(energy)(values)))
    finite_difference = central_difference_force(energy, values, step=step)

    error = np.abs(autodiff - finite_difference)
    tolerance = atol + rtol * np.abs(finite_difference)
    if np.any(error > tolerance):
        index = np.unravel_index(np.argmax(error / tolerance), error.shape)
        raise AssertionError(
            "force mismatch at component "
            f"{index}: autodiff={autodiff[index]:.16g}, "
            f"finite_difference={finite_difference[index]:.16g}, "
            f"abs_error={error[index]:.3g}, tolerance={tolerance[index]:.3g}"
        )


def unique_periodic_bond_sum(
    field,
    displacements: Sequence[tuple[int, int, int]],
) -> float:
    """Enumerate a directed half-shell without using production JAX rollers."""
    values = np.asarray(field)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("field must have shape (l1, l2, l3, 3).")

    total = 0.0
    lattice_shape = values.shape[:3]
    for site in np.ndindex(lattice_shape):
        for displacement in displacements:
            neighbor = tuple(
                (site[axis] - displacement[axis]) % lattice_shape[axis]
                for axis in range(3)
            )
            total += float(np.dot(values[site], values[neighbor]))
    return total


def assert_eager_jit_parity(
    function: Callable,
    *args,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> None:
    eager = jax.device_get(function(*args))
    compiled = jax.device_get(jax.jit(function)(*args))
    np.testing.assert_allclose(compiled, eager, rtol=rtol, atol=atol)


def assert_float32_float64_parity(
    function: Callable,
    *args,
    rtol: float = 2.0e-5,
    atol: float = 2.0e-6,
) -> None:
    args32 = tuple(jnp.asarray(arg, dtype=jnp.float32) for arg in args)
    args64 = tuple(jnp.asarray(arg, dtype=jnp.float64) for arg in args)
    result32 = jax.device_get(function(*args32))
    result64 = jax.device_get(function(*args64))
    np.testing.assert_allclose(result32, result64, rtol=rtol, atol=atol)
