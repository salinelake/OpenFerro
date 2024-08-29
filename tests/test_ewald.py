from openferro.ewald import *
from openferro.lattice import BravaisLattice3D
from jax import jit, grad
from time import time as timer

def test_ewald():
    l1,l2,l3 = 3,2,2
    latt = BravaisLattice3D(l1, l2, l3)
    latt_vec = latt.latt_vec
    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (l1, l2, l3, 3))
    paras = {'a1': latt_vec[0][0], 'a2': latt_vec[1][1], 'a3': latt_vec[2][2], 'Z_star': 1.0, 'epsilon_inf': 1.0}
    ## check dipole-dipole interaction energy calculation
    E1 = dipole_dipole_ewald_slow(field,  paras)
    dipole_dipole_ewald_engine = jit(get_dipole_dipole_ewald(latt))
    
    t1 = timer()
    E2 = dipole_dipole_ewald_engine(field,  paras)
    jax.block_until_ready(E2)
    t2 = timer()
    print("Time for Ewald summation: initialization:", t2 - t1)

    t1 = timer()
    E2 = dipole_dipole_ewald_engine(field,  paras)
    jax.block_until_ready(E2)
    t2 = timer()
    print("Time for Ewald summation: second time:", t2 - t1)

    print('Energy from exact calculation: ', E1)
    print('Energy from approximate Ewald summation: '  , E2)
    assert abs(E1 - E2) <  (abs(E1) / 100)

# if __name__ == "__main__":
#     test_ewald()

# ## check dipole-dipole interaction force calculation
# grad_slow = grad( dipole_dipole_ewald_slow  )
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