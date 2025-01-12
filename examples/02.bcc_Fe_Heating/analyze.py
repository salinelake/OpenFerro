import numpy as np
from matplotlib import pyplot as plt

Ms= 2.23
M_avg_list = []
temp_list = [10,  100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]
for temp in temp_list:
    file = np.loadtxt("spin_{}K_avg.log".format(temp), comments="#", delimiter=",", dtype=float)
    M_avg = file[:, 1:4].mean(0) / Ms
    M_avg_list.append(M_avg)

M_avg_list = np.array(M_avg_list)
fig, ax = plt.subplots(1, 1)
ax.plot(temp_list, M_avg_list[:, 0], label="Mx")
ax.plot(temp_list, M_avg_list[:, 1], label="My")
ax.plot(temp_list, M_avg_list[:, 2], label="Mz")
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("M / Ms")
ax.legend()
plt.savefig("M_avg.png", dpi=200)
