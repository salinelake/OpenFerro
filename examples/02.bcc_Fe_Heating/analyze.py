import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['lines.linestyle'] = 'dashed'
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['legend.frameon'] = False

Ms= 2.23
M_abs_list = []
temp_list = [10,  100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]
for temp in temp_list:
    file = np.loadtxt("output/spin_{:d}K_avg.log".format(temp), comments="#", delimiter=",", dtype=float)
    M_abs = np.linalg.norm(file[:, 1:4], axis=1) / Ms
    M_abs_list.append(M_abs.mean())

M_abs_list = np.array(M_abs_list)
fig, ax = plt.subplots(1, 1, figsize=(4,3))
ax.plot(temp_list, M_abs_list )
ax.axvline(x=1043, color='k', marker='o', markersize=0, linestyle='--', label=r"EXP $T_c$")
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("M / Ms")
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("M_avg.png", dpi=200)