"""
utilities functions for Ewald summation
"""
# This file is part of OpenFerro.

import numpy as np
import jax
import jax.numpy as jnp

def get_ewald_tensor(latt):
    l1, l2, l3 = latt.size
    b1, b2, b3 = latt.reciprocal_latt_vec


# def generate_dipole_matrix(self):
#     pi = 4.0 * atan(1.0)
#     pi2 = pi * 2.0

#     aa = [0.0, 0.0, 0.0]
#     for i in range(3):
#         aa[i] = np.linalg.norm(self.lattice[i])
#     a0 = min(aa)

#     tol = 1.0e-12
#     eta = sqrt(-log(tol)) / a0
#     gcut = 2.0 * eta ** 2
#     gcut2 = gcut ** 2
#     eta4 = 1.0 / (4 * eta ** 2)

#     am = np.zeros((3))
#     for i in range(3):
#         for k in range(3):
#             am[i] += self.a[k, i] ** 2
#         am[i] = sqrt(am[i])

#     mg1 = int(gcut * am[0] / pi2) + 1
#     mg2 = int(gcut * am[1] / pi2) + 1
#     mg3 = int(gcut * am[2] / pi2) + 1
#     print('Gcut: ', gcut, ' mg1, mg2, mg3: ', mg1, mg2, mg3)

#     pos0 = self.ixa[0] * self.lattice[0, :] \
#             + self.iya[0] * self.lattice[1, :] \
#             + self.iza[0] * self.lattice[2, :]

#     for ia in range(self.nsites):
#         print('site: ', ia)
#         pos = np.zeros(3)
#         pos = self.ixa[ia] * self.lattice[0, :] \
#                 + self.iya[ia] * self.lattice[1, :] \
#                 + self.iza[ia] * self.lattice[2, :]
#         rx = pos[0]
#         ry = pos[1]
#         rz = pos[2]

#         origin = (rx == 0) and (ry == 0) and (rz == 0)

#         c = 2.0 * pi / self.celvol
#         residue = 2.0 * eta ** 3 / (3.0 * sqrt(pi))

#         dum = dd_sum_over_k(self.b, mg1, mg2, mg3, gcut2, eta4, rx - pos0[0], ry - pos0[1], rz - pos0[2])

#         # The 2.0 comes from how the sum on k was done.
#         self.dpij[ia, :] = dum[:] * c * 2.0
#         if origin:
#             self.dpij[ia, 0:3] -= residue
 