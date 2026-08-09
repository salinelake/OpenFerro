"""Animate z-directed local dipoles from a PTO/STO trajectory."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import animation, colors, pyplot as plt


EXAMPLE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=EXAMPLE_DIR / "output")
    parser.add_argument(
        "--output",
        type=Path,
        default=EXAMPLE_DIR / "output" / "drive_field.webm",
    )
    parser.add_argument("--pto-layers", type=int, default=48)
    parser.add_argument("--sto-layers", type=int, default=16)
    parser.add_argument("--max-dipole", type=float, default=4.5)
    parser.add_argument("--dt-ps", type=float, default=0.002)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--surface-stride", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    field_files = sorted(
        args.input_dir.glob("drive_field_dump_*.npy"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if not field_files:
        raise FileNotFoundError(
            f"No drive_field_dump_*.npy files found in {args.input_dir}"
        )

    first_field = np.load(field_files[0], mmap_mode="r")
    nx, ny, nz, _ = first_field.shape
    middle_pto_layer = args.pto_layers // 2
    y_indices = np.unique(np.linspace(0, ny - 1, min(8, ny), dtype=int))
    x_plane, z_plane = np.meshgrid(np.arange(nx), np.arange(nz))

    normalization = colors.Normalize(
        vmin=-args.max_dipole, vmax=args.max_dipole
    )
    colormap = plt.get_cmap("seismic")
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    axis_3d = fig.add_subplot(121, projection="3d")
    axis_xy = fig.add_subplot(122)
    image = axis_xy.imshow(
        first_field[:, :, middle_pto_layer, 2].T,
        origin="lower",
        extent=(0, nx - 1, 0, ny - 1),
        cmap=colormap,
        norm=normalization,
        interpolation="nearest",
    )
    axis_xy.set_title("(b) Middle PTO layer, viewed along z")
    axis_xy.set_xlabel("x")
    axis_xy.set_ylabel("y")
    axis_xy.set_aspect("equal")
    time_label = fig.suptitle("")
    colorbar = fig.colorbar(
        image,
        ax=(axis_3d, axis_xy),
        shrink=0.8,
        pad=0.03,
    )
    colorbar.set_label(r"$p_z$ [e Angstrom]")

    def update(frame_index):
        field_file = field_files[frame_index]
        dipole_z = np.load(field_file, mmap_mode="r")[..., 2]
        rolled_dipole_z = np.roll(
            dipole_z, args.sto_layers // 2, axis=2
        )

        axis_3d.clear()
        for y_index in y_indices:
            y_plane = np.full_like(x_plane, y_index)
            axis_3d.plot_surface(
                x_plane,
                y_plane,
                z_plane,
                facecolors=colormap(
                    normalization(rolled_dipole_z[:, y_index, :].T)
                ),
                rstride=args.surface_stride,
                cstride=args.surface_stride,
                antialiased=False,
                shade=False,
            )

        step = int(field_file.stem.rsplit("_", 1)[1])
        time_label.set_text(f"t = {step * args.dt_ps:g} ps")
        axis_3d.set_title("(a) Parallel x-z sections")
        axis_3d.set_xlabel("x")
        axis_3d.set_ylabel("y")
        axis_3d.set_zlabel("z")
        axis_3d.set_xlim(0, nx - 1)
        axis_3d.set_ylim(0, ny - 1)
        axis_3d.set_zlim(0, nz - 1)
        axis_3d.set_xticks(
            np.linspace(0, nx - 1, min(5, nx), dtype=int)
        )
        axis_3d.set_yticks(
            np.linspace(0, ny - 1, min(5, ny), dtype=int)
        )
        axis_3d.set_zticks(
            np.linspace(0, nz - 1, min(5, nz), dtype=int)
        )
        axis_3d.set_box_aspect((nx, ny, nz))
        image.set_data(dipole_z[:, :, middle_pto_layer].T)

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(field_files),
        interval=1000 / args.fps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=args.fps,
        codec="libvpx-vp9",
        extra_args=["-crf", "18", "-b:v", "0", "-pix_fmt", "yuv420p"],
    )
    movie.save(args.output, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {len(field_files)} frames to {args.output}")


if __name__ == "__main__":
    main()
