from time import time as timer

import jax
import jax.numpy as jnp
from jax import jit
from memory_profiler import memory_usage

from openferro.engine.ewald import build_dipole_dipole_ewald
from openferro.lattice import BravaisLattice3D


def test_ewald():
    l1, l2, l3 = 100, 100, 100
    latt = BravaisLattice3D(l1, l2, l3)
    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (l1, l2, l3, 3))
    parameters = jnp.asarray([1.0])

    dipole_dipole_ewald_engine, UkGG = build_dipole_dipole_ewald(latt)
    dipole_dipole_ewald_engine = jit(dipole_dipole_ewald_engine)
    t1 = timer()
    E2 = dipole_dipole_ewald_engine(field, UkGG, parameters)
    jax.block_until_ready(E2)
    t2 = timer()
    print("Time for Ewald summation: initialization:", t2 - t1)

    t1 = timer()
    E2 = dipole_dipole_ewald_engine(field, UkGG, parameters)
    jax.block_until_ready(E2)
    t2 = timer()
    print("Time for Ewald summation: second time:", t2 - t1)
    print('Energy from approximate Ewald summation: '  , E2)

if __name__ == "__main__":
    # test_ewald()
    mem_usage = memory_usage(test_ewald, interval=0.01)
    # print('Memory usage (in chunks of 0.01 seconds): %s' % mem_usage)
    print('Maximum memory usage: %s MB' % max(mem_usage))
