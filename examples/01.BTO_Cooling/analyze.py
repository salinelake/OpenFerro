import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
# Physical Review B line-art defaults (readable at two-column width).
mpl.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 14,
    "axes.linewidth": 1.5,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.0,
    "lines.linestyle": "dashed",
    "lines.marker": "o",
    "legend.fontsize": 10,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

temp_list =  [320, 310, 300, 290, 280, 270, 260, 250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150]
temp_list = np.flip(np.array(temp_list).astype(int))

field_list = []
strain_list = []
dipole2polar = 9.956 / 3.9477**3 * 16.0217646 * 100 # eA -> uC/cm^2

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

fig, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
ax[0].plot(temp_list, field_list[:,0], label='$P_{1}$' )
ax[0].plot(temp_list, field_list[:,1], label='$P_{2}$' )
ax[0].plot(temp_list, field_list[:,2], label='$P_{3}$' ) 
ax[0].set_ylabel(r'$P$ [$\mathrm{\mu C/cm^2}$] ')
ax[0].legend()
ax[0].set_ylim(-2, 32)
ax[1].plot(temp_list, strain_list[:,0], label=r'$\eta_1$')
ax[1].plot(temp_list, strain_list[:,1], label=r'$\eta_2$')
ax[1].plot(temp_list, strain_list[:,2], label=r'$\eta_3$')
ax[1].plot(temp_list, strain_list[:,3], label=r'$\eta_4$')
ax[1].plot(temp_list, strain_list[:,4], label=r'$\eta_5$')
ax[1].plot(temp_list, strain_list[:,5], label=r'$\eta_6$')
ax[1].set_xlabel('T [K]')
ax[1].set_ylabel(r'$\eta$ [$\%$]')
ax[1].set_ylim(-0.1, 2.2) 
ax[1].legend(ncols=2, loc='lower right')

plt.tight_layout()
plt.savefig('field_avg.png')