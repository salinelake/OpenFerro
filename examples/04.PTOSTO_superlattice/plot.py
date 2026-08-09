"""Visualize z-directed local dipoles on parallel planes."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import colors, pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Field dump written by npt.py.")
    parser.add_argument("--output", type=Path, default=Path("3d_visualization.png"))
    parser.add_argument("--sto-layers", type=int, default=16)
    parser.add_argument("--max-dipole", type=float, default=4.5)
    return parser.parse_args()


def main():
    args = parse_args()
    dipole = np.load(args.input)
    dipole = np.roll(dipole, args.sto_layers // 2, axis=2)
    nx, ny, nz, _ = dipole.shape

    normalization = colors.Normalize(vmin=-args.max_dipole, vmax=args.max_dipole)
    colormap = plt.get_cmap("seismic")
    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection="3d")
    for y_index in np.unique(np.linspace(0, ny - 1, min(8, ny), dtype=int)):
        xz_plane = dipole[:, y_index, :, 2].T
        x_plane, z_plane = np.meshgrid(np.arange(nx), np.arange(nz))
        y_plane = np.full_like(x_plane, y_index)
        axis.plot_surface(
            x_plane,
            y_plane,
            z_plane,
            facecolors=colormap(normalization(xz_plane)),
            rstride=1,
            cstride=1,
            antialiased=False,
            shade=False,
        )

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_xticks(np.linspace(0, nx - 1, min(5, nx), dtype=int))
    axis.set_yticks(np.linspace(0, ny - 1, min(5, ny), dtype=int))
    axis.set_zticks(np.linspace(0, nz - 1, min(5, nz), dtype=int))
    axis.set_box_aspect((nx, ny, nz))
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=colormap),
        ax=axis,
        shrink=0.7,
        pad=0.1,
    )
    colorbar.set_label(r"$p_z$ [e Angstrom]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
