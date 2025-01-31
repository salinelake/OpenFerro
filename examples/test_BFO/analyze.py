import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['lines.linestyle'] = 'dashed'
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['legend.frameon'] = False

L=16
temp_list = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]).astype(int)
dipole_avg_list = []
afd_avg_list = []
spin_avg_list = []
dipole2polar = 5.868 / 3.98**3 * 16.0217646  # eA -> C/m^2
i, j, k = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing="ij")
sign_array = (-1) ** (i + j + k)

for temp in temp_list:
    dipole_history = np.loadtxt(  './output/field_{}K_avg.log'.format(temp), comments='#', delimiter=',')
    dipole_avg = dipole_history[:,1:4].mean(axis=0)
    afd_history = []
    spin_history = []
    for i in range(25,75):
        idx = int(i*1000)
        afd_traj = np.load('./output/AFD_{}K_dump_{}.npy'.format(temp, idx))
        afd_R_avg = afd_traj * sign_array[...,None]
        afd_R_avg = afd_R_avg.mean(axis=(0,1,2))
        afd_history.append(afd_R_avg)
        spin_traj = np.load('./output/spin_{}K_dump_{}.npy'.format(temp, idx))
        spin_R_avg = spin_traj * sign_array[...,None]
        spin_R_avg = spin_R_avg.mean(axis=(0,1,2))
        spin_history.append(spin_R_avg)
    dipole_avg_list.append(dipole_avg)
    afd_avg_list.append(np.array(afd_history).mean(axis=0))
    spin_avg_list.append(np.array(spin_history).mean(axis=0))
 
dipole_list = np.abs(np.array(dipole_avg_list).reshape(len(temp_list), 3)) * dipole2polar
afd_list = np.abs(np.array(afd_avg_list).reshape(len(temp_list), 3))
spin_list = np.abs(np.array(spin_avg_list).reshape(len(temp_list), 3))
fig, ax = plt.subplots(1,3, figsize=(10, 3))
ax[0].plot(temp_list, dipole_list[:,0], label=r'$P_{1}$')
ax[0].plot(temp_list, dipole_list[:,1], label=r'$P_{2}$')
ax[0].plot(temp_list, dipole_list[:,2], label=r'$P_{3}$') 
ax[0].plot(temp_list, (dipole_list**2).sum(-1)**0.5 , label=r'$|P|$')
print((dipole_list**2).sum(-1)**0.5)
ax[0].set_xlabel('T [K]')
ax[0].set_ylabel(r'Polarization [$\mathrm{C/m^2}$] ')
ax[0].legend()
ax[1].plot(temp_list, afd_list[:,0], label=r'$\omega_{R,x}$')
ax[1].plot(temp_list, afd_list[:,1], label=r'$\omega_{R,y}$')
ax[1].plot(temp_list, afd_list[:,2], label=r'$\omega_{R,z}$')
ax[1].set_xlabel('T [K]')
ax[1].set_ylabel(r'AFD distortion [rad]')
ax[1].legend()
ax[0].set_xlim(0,1600)
ax[1].set_xlim(0,1600)
ax[2].plot(temp_list, spin_list[:,0], label=r'$M_{R,x}$')
ax[2].plot(temp_list, spin_list[:,1], label=r'$M_{R,y}$')
ax[2].plot(temp_list, spin_list[:,2], label=r'$M_{R,z}$')
ax[2].set_xlabel('T [K]')
ax[2].set_ylabel(r'Anti-ferromagnetic moment [$\mu_B$]')
ax[2].legend()
plt.tight_layout()
plt.savefig('field_avg.png')