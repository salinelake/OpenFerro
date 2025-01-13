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

temp_list = np.array([ 140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320, 350, 400]).astype(int)
field_list = []
strain_list = []
dipole2polar = 9.956 / 3.9477**3 * 16.0217646  # eA -> C/m^2

for temp in temp_list:
    field_history = np.loadtxt(  './output/field_{}K_avg.log'.format(temp), comments='#', delimiter=',')
    thermo_history = np.loadtxt( './output/thermo_{}K.log'.format(temp), comments='#', delimiter=',')
    field_traj = field_history[:,1:7]
    strain_traj = thermo_history[...,1:7] * 1e2

    field_abs = np.abs(np.array(field_traj))
    strain_abs =np.abs( np.array(strain_traj))
    field_1_idx = np.argmax(field_abs, axis=-1)
    field_1 = [field_traj[i, field_1_idx[i]] for i in range(len(field_1_idx))]
    field_3_idx = np.argmin(field_abs, axis=-1)
    field_3 = [field_traj[i, field_3_idx[i]] for i in range(len(field_3_idx))]
    field_2 = [field_traj[i].sum() - field_1[i] - field_3[i] for i in range(len(field_1_idx))]
    field_1 = np.array(field_1)
    field_2 = np.array(field_2)
    field_3 = np.array(field_3)

    strain_1 = np.max(strain_abs[...,:3], axis=-1)
    strain_3 = np.min(strain_abs[...,:3], axis=-1)
    strain_2 = strain_abs[...,:3].sum(-1) - strain_1 - strain_3
    strain_4 =  np.max(strain_abs[...,3:], axis=-1)
    strain_6 =  np.min(strain_abs[...,3:], axis=-1)
    strain_5 =  strain_abs[...,3:].sum(-1) - strain_4 - strain_6


    field_list.append([field_1.mean(), field_2.mean(), field_3.mean()])
    strain_list.append([strain_1.mean(), strain_2.mean(), strain_3.mean(), strain_4.mean(), strain_5.mean(), strain_6.mean()])

field_list = np.abs(np.array(field_list).reshape(len(temp_list), 3)) * dipole2polar
strain_list = np.abs(np.array(strain_list).reshape(len(temp_list), 6))

fig, ax = plt.subplots(1,2, figsize=(8, 3))
ax[0].plot(temp_list, field_list[:,0], label='$P_{1}$' )
ax[0].plot(temp_list, field_list[:,1], label='$P_{2}$' )
ax[0].plot(temp_list, field_list[:,2], label='$P_{3}$' ) 
ax[0].set_xlabel('T [K]')
ax[0].set_ylabel(r'Polarization [$\mathrm{C/m^2}$] ')
ax[0].legend()
ax[1].plot(temp_list, strain_list[:,0], label=r'$\eta_1$')
ax[1].plot(temp_list, strain_list[:,1], label=r'$\eta_2$')
ax[1].plot(temp_list, strain_list[:,2], label=r'$\eta_3$')
ax[1].plot(temp_list, strain_list[:,3], label=r'$\eta_4$')
ax[1].plot(temp_list, strain_list[:,4], label=r'$\eta_5$')
ax[1].plot(temp_list, strain_list[:,5], label=r'$\eta_6$')
ax[1].set_xlabel('T [K]')
ax[1].set_ylabel(r'Strain [$10^{-2}$]')
ax[1].set_ylim(0,2.2) 
ax[1].legend()

plt.tight_layout()
plt.savefig('field_avg.png')