"""Plot magnetization with one-standard-error bars from ten time blocks."""

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


def _block_standard_error(values, n_blocks=10):
    """Estimate mean uncertainty using contiguous blocks of the trajectory."""
    if len(values) < 2 * n_blocks:
        raise ValueError(f"Need at least {2 * n_blocks} samples for {n_blocks} blocks.")
    block_means = np.array([block.mean() for block in np.array_split(values, n_blocks)])
    return block_means.std(ddof=1) / np.sqrt(n_blocks)


Ms= 2.23
M_abs_list = []
M_error_list = []
temp_list = [ 10, 200, 400, 600, 700, 800, 900, 1000, 1200]

for temp in temp_list:
    file = np.loadtxt("output/spin_{:d}K_avg.log".format(temp), comments="#", delimiter=",", dtype=float)
    M_abs = np.linalg.norm(file[:, 1:4], axis=1) / Ms
    M_abs_list.append(M_abs.mean())
    M_error_list.append(_block_standard_error(M_abs))

M_abs_list = np.array(M_abs_list)
fig, ax = plt.subplots(1, 1, figsize=(4,2.1))
ax.errorbar(temp_list, M_abs_list, yerr=M_error_list, fmt='o--', capsize=2, elinewidth=1)
ax.axvline(x=1043, color='k', marker='o', markersize=0, linestyle='--', label=r"EXP $T_c$")
ax.set_xlabel("T [K]")
ax.set_ylabel(r"$M$ / $M_s$")
# ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("M_avg.png")
