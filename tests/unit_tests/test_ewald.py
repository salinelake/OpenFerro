import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openferro.lattice import BravaisLattice3D
from openferro.engine.ewald import (
    apply_ewald_kernel_fft,
    build_dipole_dipole_ewald,
    calc_ewald_reciprocal_sum,
    dipole_dipole_ewald_plain,
    estimate_dipole_dipole_ewald_memory,
    get_UkGG,
)
from openferro.parallelism import DeviceMesh
from openferro.system import System
from jax import jit
from scientific_helpers import (
    assert_eager_jit_parity,
    assert_force_matches_finite_difference,
)


pytestmark = pytest.mark.scientific


@pytest.fixture(scope="module")
def reference_engines():
    return {
        shape: build_dipole_dipole_ewald(
            BravaisLattice3D(*shape), dtype=jnp.float64
        )
        for shape in ((2, 2, 2), (3, 2, 2))
    }


@pytest.mark.parametrize("shape", [(2, 2, 2), (3, 2, 2)])
@pytest.mark.parametrize("configuration", ["uniform", "nonsymmetric"])
def test_ewald_matches_direct_reference(reference_engines, shape, configuration):
    nvalues = int(np.prod(shape) * 3)
    if configuration == "uniform":
        field = jnp.ones(shape + (3,)) * jnp.array([0.2, -0.1, 0.3])
    else:
        field = jnp.sin(jnp.arange(nvalues).reshape(shape + (3,)) * 0.37)
    parameters = {
        "a1": 1.0,
        "a2": 1.0,
        "a3": 1.0,
        "Z_star": 1.0,
        "epsilon_inf": 1.0,
    }

    direct = dipole_dipole_ewald_plain(field, parameters)
    engine, UkGG = reference_engines[shape]
    fft = engine(field, UkGG, jnp.array([1.0]))
    np.testing.assert_allclose(fft, direct, rtol=2.0e-11, atol=2.0e-11)


def test_ewald_zero_prefactor_scaling_and_cubic_symmetry(reference_engines):
    shape = (2, 2, 2)
    field = jnp.sin(jnp.arange(24).reshape(shape + (3,)) * 0.31)
    engine, UkGG = reference_engines[shape]
    permuted = jnp.transpose(field, (1, 0, 2, 3))[..., jnp.array([1, 0, 2])]

    np.testing.assert_allclose(
        engine(jnp.zeros_like(field), UkGG, jnp.array([1.0])), 0.0
    )
    np.testing.assert_allclose(
        engine(field, UkGG, jnp.array([2.5])),
        2.5 * engine(field, UkGG, jnp.array([1.0])),
    )
    np.testing.assert_allclose(
        engine(permuted, UkGG, jnp.array([1.0])),
        engine(field, UkGG, jnp.array([1.0])),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_ewald_force_jit_and_dtype_parity(reference_engines):
    shape = (2, 2, 2)
    field64 = jnp.sin(jnp.arange(24).reshape(shape + (3,)) * 0.23) * 0.1
    parameters64 = jnp.array([1.73], dtype=jnp.float64)
    engine64, UkGG64 = reference_engines[shape]

    assert_force_matches_finite_difference(
        lambda value: engine64(value, UkGG64, parameters64),
        field64,
        rtol=3.0e-6,
        atol=3.0e-7,
    )
    assert_eager_jit_parity(engine64, field64, UkGG64, parameters64)

    engine32, UkGG32 = build_dipole_dipole_ewald(
        BravaisLattice3D(*shape), dtype=jnp.float32
    )
    energy32 = engine32(
        field64.astype(jnp.float32), UkGG32, parameters64.astype(jnp.float32)
    )
    energy64 = engine64(field64, UkGG64, parameters64)
    # Float32 builds the reciprocal kernel through many accumulated replicas.
    np.testing.assert_allclose(energy32, energy64, rtol=3.0e-4, atol=3.0e-6)


def test_ewald_engine_rejects_incompatible_shapes(reference_engines):
    engine, UkGG = reference_engines[(2, 2, 2)]
    with pytest.raises(ValueError, match=r"shape \(2, 2, 2, 3\)"):
        engine(jnp.zeros((2, 2, 3, 3)), UkGG, jnp.array([1.0]))
    with pytest.raises(ValueError, match=r"parameters.*shape \(1,\)"):
        engine(jnp.zeros((2, 2, 2, 3)), UkGG, jnp.array([1.0, 2.0]))


def test_system_passes_ewald_kernel_to_energy_and_force():
    system = System(BravaisLattice3D(2, 2, 2))
    field = system.add_field(
        "dipole",
        ftype="Rn",
        dim=3,
        value=jnp.asarray([0.2, -0.1, 0.3]),
        mass=None,
    )
    interaction = system.add_dipole_dipole_interaction(
        "ewald", "dipole", prefactor=1.7
    )

    energy = system.calc_energy_by_ID("ewald")
    force = system.calc_force_by_ID("ewald")

    assert interaction.engine_data.shape == (2, 2, 2, 6)
    assert bool(jnp.isfinite(energy))
    assert force.shape == field.get_values().shape
    assert bool(jnp.all(jnp.isfinite(force)))


def test_ewald_reciprocal_sum_matches_component_form():
    key = jax.random.PRNGKey(1)
    field_fft = jax.random.normal(key, (2, 2, 2, 3)) + 1j * jax.random.normal(
        key, (2, 2, 2, 3)
    )
    UkGG = jnp.arange(2 * 2 * 2 * 6, dtype=field_fft.real.dtype).reshape(
        (2, 2, 2, 6)
    ) / 10.0

    kernel_field_fft = apply_ewald_kernel_fft(field_fft, UkGG)
    reciprocal_sum = calc_ewald_reciprocal_sum(field_fft, UkGG)

    manual = (
        (field_fft.real[..., 0] ** 2 + field_fft.imag[..., 0] ** 2)
        * UkGG[..., 0]
    ).sum()
    manual += (
        (field_fft.real[..., 1] ** 2 + field_fft.imag[..., 1] ** 2)
        * UkGG[..., 1]
    ).sum()
    manual += (
        (field_fft.real[..., 2] ** 2 + field_fft.imag[..., 2] ** 2)
        * UkGG[..., 2]
    ).sum()
    manual += 2 * (
        (
            field_fft.real[..., 0] * field_fft.real[..., 1]
            + field_fft.imag[..., 0] * field_fft.imag[..., 1]
        )
        * UkGG[..., 5]
    ).sum()
    manual += 2 * (
        (
            field_fft.real[..., 0] * field_fft.real[..., 2]
            + field_fft.imag[..., 0] * field_fft.imag[..., 2]
        )
        * UkGG[..., 4]
    ).sum()
    manual += 2 * (
        (
            field_fft.real[..., 1] * field_fft.real[..., 2]
            + field_fft.imag[..., 1] * field_fft.imag[..., 2]
        )
        * UkGG[..., 3]
    ).sum()

    assert kernel_field_fft.shape == field_fft.shape
    assert jnp.allclose(reciprocal_sum, manual)


def test_ewald_memory_estimate():
    latt = BravaisLattice3D(3, 2, 2)
    estimate = estimate_dipole_dipole_ewald_memory(latt, dtype=jnp.float32)

    assert estimate["shape"] == (3, 2, 2)
    assert estimate["nsites"] == 12
    assert estimate["arrays"]["UkGG"] == 12 * 6 * 4
    assert estimate["arrays"]["field_fft"] == 12 * 3 * 8
    assert estimate["tracked_total"] == sum(estimate["arrays"].values())


def test_ewald_coefficients_dtype_and_sharding():
    latt = BravaisLattice3D(2, 2, 2)
    mesh = DeviceMesh(devices=jax.devices()[:1], num_rows=1, num_cols=1)
    sharding = mesh.partition_sharding()
    b = jnp.ones(3, dtype=jnp.float32)
    UkGG = get_UkGG(2, 2, 2, 1, 1, 1, b, 1.0, dtype=jnp.float32, sharding=sharding)

    assert UkGG.shape == (2, 2, 2, 6)
    assert UkGG.dtype == jnp.float32
    assert UkGG.sharding == sharding

    field = jnp.ones((2, 2, 2, 3), dtype=jnp.float32)
    engine, sharded_UkGG = build_dipole_dipole_ewald(
        latt, dtype=field.dtype, sharding=sharding
    )
    engine = jit(engine)
    sharded_field = jax.device_put(field, sharding)
    energy = engine(
        sharded_field, sharded_UkGG, jnp.array([1.0], dtype=field.dtype)
    )

    assert energy.dtype == field.dtype
    assert sharded_UkGG.sharding == sharding
 
# ## check dipole-dipole interaction force calculation
# grad_slow = grad( dipole_dipole_ewald_plain  )
# grad_fast =  jit(grad( jit(dipole_dipole_ewald) ))
# t0 = time()
# force = grad_slow(field, latt_vec)
# print("Time for slow gradient method: ", time() - t0)
# t0 = time()
# force = grad_fast(field, paras)
# print("Time for fast gradient method: ", time() - t0)

# ## scaling
# l1_list = np.arange(1, 10) * 20
# t_list = []
# for l1 in l1_list:
#     l2 = l1
#     l3 = l1
#     latt = BravaisLattice3D(l1, l2, l3)
#     latt_vec = latt.latt_vec
#     paras = {'a1': latt_vec[0][0], 'a2': latt_vec[1][1], 'a3': latt_vec[2][2]}
#     field = jax.random.normal(key, (l1, l2, l3, 3))
#     t0 = time()
#     E = grad_fast(field,  paras)
#     t_list.append(time() - t0)
# print("force scaling test: ", t_list)

# t_list = []
# energy_fast = jit(dipole_dipole_ewald)
# for l1 in l1_list:
#     l2 = l1
#     l3 = l1
#     latt = BravaisLattice3D(l1, l2, l3)
#     latt_vec = latt.latt_vec
#     paras = {'a1': latt_vec[0][0], 'a2': latt_vec[1][1], 'a3': latt_vec[2][2]}
#     field = jax.random.normal(key, (l1, l2, l3, 3))
#     t0 = time()
#     E = energy_fast(field,  paras)
#     t_list.append(time() - t0)
# print("energy Scaling test: ", t_list)

if __name__ == "__main__":
    test_ewald()
    test_ewald_reciprocal_sum_matches_component_form()
    test_ewald_memory_estimate()
    test_ewald_coefficients_dtype_and_sharding()
