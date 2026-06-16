import jax
import jax.numpy as jnp
from openferro.lattice import BravaisLattice3D
from openferro.engine.ewald import (
    apply_ewald_kernel_fft,
    calc_ewald_reciprocal_sum,
    dipole_dipole_ewald_plain,
    estimate_dipole_dipole_ewald_memory,
    get_UkGG,
    get_dipole_dipole_ewald,
)
from openferro.parallelism import DeviceMesh
from jax import jit


def test_ewald():
    l1,l2,l3 = 3,2,2
    latt = BravaisLattice3D(l1, l2, l3)
    latt_vec = latt.latt_vec
    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (l1, l2, l3, 3))
    paras = {'a1': latt_vec[0][0], 'a2': latt_vec[1][1], 'a3': latt_vec[2][2], 'Z_star': 1.0, 'epsilon_inf': 1.0}
    ## dipole-dipole interaction energy from exact calculation
    E1 = dipole_dipole_ewald_plain(field,  paras)
    ## dipole-dipole interaction energy from approximate Ewald summation
    dipole_dipole_ewald_engine = jit(get_dipole_dipole_ewald(latt))
    E2 = dipole_dipole_ewald_engine(field,  [paras['Z_star'] ** 2/ paras['epsilon_inf']])
    print('Plain Ewald summation: ', E1)
    print('Fast Ewald summation: ', E2)
    assert abs(E1 - E2) <  (abs(E1) / 100)


def test_ewald_reciprocal_sum_matches_component_form():
    key = jax.random.PRNGKey(1)
    field_fft = jax.random.normal(key, (2, 2, 2, 3)) + 1j * jax.random.normal(key, (2, 2, 2, 3))
    UkGG = jnp.arange(2 * 2 * 2 * 6, dtype=field_fft.real.dtype).reshape((2, 2, 2, 6)) / 10.0

    kernel_field_fft = apply_ewald_kernel_fft(field_fft, UkGG)
    reciprocal_sum = calc_ewald_reciprocal_sum(field_fft, UkGG)

    manual = ((field_fft.real[..., 0]**2 + field_fft.imag[..., 0]**2) * UkGG[..., 0]).sum()
    manual += ((field_fft.real[..., 1]**2 + field_fft.imag[..., 1]**2) * UkGG[..., 1]).sum()
    manual += ((field_fft.real[..., 2]**2 + field_fft.imag[..., 2]**2) * UkGG[..., 2]).sum()
    manual += 2 * ((field_fft.real[..., 0] * field_fft.real[..., 1] + field_fft.imag[..., 0] * field_fft.imag[..., 1]) * UkGG[..., 5]).sum()
    manual += 2 * ((field_fft.real[..., 0] * field_fft.real[..., 2] + field_fft.imag[..., 0] * field_fft.imag[..., 2]) * UkGG[..., 4]).sum()
    manual += 2 * ((field_fft.real[..., 1] * field_fft.real[..., 2] + field_fft.imag[..., 1] * field_fft.imag[..., 2]) * UkGG[..., 3]).sum()

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
    engine = jit(get_dipole_dipole_ewald(latt, dtype=field.dtype, sharding=sharding))
    sharded_field = jax.device_put(field, sharding)
    energy = engine(sharded_field, jnp.array([1.0], dtype=field.dtype))

    assert energy.dtype == field.dtype
 
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
