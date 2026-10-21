import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

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

Ms= 2.23
M_abs_list = []
temp_list = [ 10, 200, 400, 600, 700, 800, 900, 1000, 1200]

for temp in temp_list:
    file = np.loadtxt("output/spin_{:d}K_avg.log".format(temp), comments="#", delimiter=",", dtype=float)
    M_abs = np.linalg.norm(file[:, 1:4], axis=1) / Ms
    M_abs_list.append(M_abs.mean())

M_abs_list = np.array(M_abs_list)
fig, ax = plt.subplots(1, 1, figsize=(4,2.1))
ax.plot(temp_list, M_abs_list )
ax.axvline(x=1043, color='k', marker='o', markersize=0, linestyle='--', label=r"EXP $T_c$")
ax.set_xlabel("T [K]")
ax.set_ylabel(r"$M$ / $M_s$")
# ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("M_avg.png")