import numpy as np
from matplotlib import pyplot as plt

Ms= 2.23
M_abs_list = []
temp_list = [10,  100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]
for temp in temp_list:
    file = np.loadtxt("output/spin_{:d}K_avg.log".format(temp), comments="#", delimiter=",", dtype=float)
    M_abs = np.linalg.norm(file[:, 1:4], axis=1) / Ms
    M_abs_list.append(M_abs.mean())

M_abs_list = np.array(M_abs_list)
fig, ax = plt.subplots(1, 1)
ax.plot(temp_list, M_abs_list, label="M")
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("M / Ms")
ax.legend()
plt.savefig("M_avg.png", dpi=200)