from matplotlib import pyplot as plt
import numpy as np

m=48
n=16
dmax=4.5
# dipole_array = np.load('output/relax_field_dump_200000.npy')
dipole_array = np.load('output/drive_field_dump_1500000.npy')
dipole_array = np.roll(dipole_array, n//2, axis=2)
 
# Plot slices of a 3D array as parallel planes in 3D space
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Add each slice as a 2D plane in 3D space
for y_idx in range(0, 256, 32):
    xz_plane = dipole_array[:, y_idx, :, 2]  # Extract the 2D slice
    X_plane, Z_plane = np.meshgrid(np.arange(xz_plane.shape[0]), np.arange(xz_plane.shape[1]))  # Meshgrid for the plane
    Y_plane = np.full_like(X_plane, y_idx)  # Z-coordinate
    xz_plane = xz_plane.T
    facecolor = (xz_plane / dmax + 1) /2
    ax.plot_surface(X_plane, Y_plane, Z_plane, facecolors=plt.cm.seismic(facecolor),
                    rstride=1, cstride=1, antialiased=False, shade=False )
# Set axis labels and view
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
## set ticks
ax.set_xticks(np.arange(0, 258, 64))
ax.set_yticks(np.arange(0, 258, 64))
ax.set_zticks(np.arange(0, 66, 32))
ax.set_box_aspect((256, 256, 64))
# ax.set_title("3D Slices as Parallel Planes")
plt.savefig('3d_visualization.png', dpi=300)
plt.close()
 