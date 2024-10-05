from memory_profiler import memory_usage
from openferro.ewald import *
from openferro.lattice import BravaisLattice3D
from jax import jit, grad
from time import time as timer
import os 

def test_ewald():
    l1,l2,l3 = 100,100,100
    latt = BravaisLattice3D(l1, l2, l3)
    latt_vec = latt.latt_vec
    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (l1, l2, l3, 3))
    paras = {'a1': latt_vec[0][0], 'a2': latt_vec[1][1], 'a3': latt_vec[2][2], 'Z_star': 1.0, 'epsilon_inf': 1.0}

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
    print('Energy from approximate Ewald summation: '  , E2)

if __name__ == "__main__":
    # test_ewald()
    mem_usage = memory_usage(test_ewald, interval=0.01)
    # print('Memory usage (in chunks of 0.01 seconds): %s' % mem_usage)
    print('Maximum memory usage: %s MB' % max(mem_usage))