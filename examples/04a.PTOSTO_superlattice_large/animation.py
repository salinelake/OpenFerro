"""Animate z-directed local dipoles from a PTO/STO cooling run."""

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import animation, colors, pyplot as plt
from matplotlib.animation import adjusted_figsize


EXAMPLE_DIR = Path(__file__).resolve().parent
FRAME_PATTERN = re.compile(
    r"^(?P<kind>field|sample)_(?P<temp>\d+(?:\.\d+)?)K_dump_(?P<step>\d+)$"
)
# Cubic cells. A strictly equal aspect makes the 24-layer PTO slab a
# hairline across 2500 sites, so a short axis may be stretched, at most
# by this factor, until it is about 15% of the long axis.
MAX_SHORT_AXIS_STRETCH = 10.0
THIN_AXIS_FRACTION = 0.15


def panel_box_aspect(n_horizontal, n_vertical):
    """Return the axes height/width for one cell per pixel."""
    physical = n_vertical / n_horizontal
    if physical < THIN_AXIS_FRACTION:
        stretch = min(MAX_SHORT_AXIS_STRETCH, THIN_AXIS_FRACTION / physical)
        return physical * stretch
    if physical > 1.0 / THIN_AXIS_FRACTION:
        stretch = min(MAX_SHORT_AXIS_STRETCH, physical * THIN_AXIS_FRACTION)
        return physical / stretch
    return physical


def cell_output_dir(lateral_size, pto_layers, sto_layers):
    """Return the directory written by npt.py for this cell."""
    return EXAMPLE_DIR / f"output_L{lateral_size}_PTO{pto_layers}_STO{sto_layers}"


def collect_frames(input_dir, stage):
    """Return (path, label-temperature, dump-step) in playback order.

    Cooling frames are the end-of-plateau dumps, hottest first. Sample frames
    are the final-temperature series, in time order.
    """
    kind = "field" if stage == "cooling" else "sample"
    frames = []
    for path in input_dir.glob("*.npy"):
        match = FRAME_PATTERN.match(path.stem)
        if match is None or match.group("kind") != kind:
            continue
        frames.append(
            (path, float(match.group("temp")), int(match.group("step")))
        )
    if stage == "cooling":
        frames.sort(key=lambda frame: (-frame[1], frame[2]))
    else:
        frames.sort(key=lambda frame: (frame[1], frame[2]))
    return frames


def frame_title(stage, temperature, step, dt_ps):
    if stage == "cooling":
        return f"T = {temperature:g} K"
    return f"T = {temperature:g} K, t = {step * dt_ps:g} ps"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lateral-size", type=int, default=2000)
    parser.add_argument("--pto-layers", type=int, default=12)
    parser.add_argument("--sto-layers", type=int, default=12)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument(
        "--stage",
        type=str,
        choices=["cooling", "sample"],
        default="cooling",
        help="cooling uses field_{T}K dumps; sample uses the final 1 ps series.",
    )
    parser.add_argument("--max-dipole", type=float, default=4.5)
    parser.add_argument("--dt-ps", type=float, default=0.002)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_dir is None:
        args.input_dir = cell_output_dir(
            args.lateral_size, args.pto_layers, args.sto_layers
        )
    output_file = args.input_dir / f"{args.stage}_field.webm"
    field_frames = collect_frames(args.input_dir, args.stage)
    if not field_frames:
        kind = "field" if args.stage == "cooling" else "sample"
        raise FileNotFoundError(
            f"No {kind}_*K_dump_*.npy files found in {args.input_dir}"
        )

    first_field = np.load(field_frames[0][0], mmap_mode="r")
    nx, ny, nz, _ = first_field.shape
    if args.pto_layers < 1 or args.pto_layers > nz:
        raise ValueError(
            f"--pto-layers must lie in 1..{nz}, got {args.pto_layers}."
        )
    middle_y = ny // 2
    middle_pto_layer = args.pto_layers // 2

    normalization = colors.Normalize(
        vmin=-args.max_dipole, vmax=args.max_dipole
    )
    colormap = plt.get_cmap("seismic")
    xz_aspect = panel_box_aspect(nx, nz)
    xy_aspect = panel_box_aspect(nx, ny)
    # FFMpegWriter sizes the raw frame with int(inches * dpi), while the
    # canvas rounds. A one-pixel mismatch shears every frame, so snap both
    # sides to an even pixel count before the first draw.
    axes_width = 6.4
    left, right = 0.85, 1.30
    bottom, gap, top = 0.55, 0.65, 0.72
    xz_height = axes_width * xz_aspect
    xy_height = axes_width * xy_aspect
    fig_width, fig_height = adjusted_figsize(
        left + axes_width + right,
        bottom + xy_height + gap + xz_height + top,
        args.dpi,
        2,
    )
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=args.dpi)

    def axes_rect(x_inch, y_inch, width_inch, height_inch):
        return [
            x_inch / fig_width,
            y_inch / fig_height,
            width_inch / fig_width,
            height_inch / fig_height,
        ]

    axis_xy = fig.add_axes(
        axes_rect(left, bottom, axes_width, xy_height)
    )
    axis_xz = fig.add_axes(
        axes_rect(left, bottom + xy_height + gap, axes_width, xz_height),
        sharex=axis_xy,
    )
    colorbar_axis = fig.add_axes(
        axes_rect(
            left + axes_width + 0.18,
            bottom,
            0.22,
            xy_height + gap + xz_height,
        )
    )
    # The side view is the full PTO+STO stack. PTO is z = 0 .. pto_layers-1.
    image_xz = axis_xz.imshow(
        first_field[:, middle_y, :, 2].T,
        origin="lower",
        extent=(0, nx - 1, 0, nz - 1),
        cmap=colormap,
        norm=normalization,
        interpolation="nearest",
        aspect="auto",
    )
    axis_xz.set_title(f"(a) Middle x-z section (y = {middle_y})")
    axis_xz.set_ylabel("z")
    image_xy = axis_xy.imshow(
        first_field[:, :, middle_pto_layer, 2].T,
        origin="lower",
        extent=(0, nx - 1, 0, ny - 1),
        cmap=colormap,
        norm=normalization,
        interpolation="nearest",
        aspect="auto",
    )
    axis_xy.set_title("(b) Middle PTO layer, viewed along z")
    axis_xy.set_xlabel("x")
    axis_xy.set_ylabel("y")
    time_label = fig.suptitle("")
    colorbar = fig.colorbar(image_xy, cax=colorbar_axis)
    colorbar.set_label(r"$p_z$ [e Angstrom]")

    def update(frame_index):
        field_file, temperature, step = field_frames[frame_index]
        dipole_z = np.load(field_file, mmap_mode="r")[..., 2]
        time_label.set_text(frame_title(args.stage, temperature, step, args.dt_ps))
        image_xz.set_data(dipole_z[:, middle_y, :].T)
        image_xy.set_data(dipole_z[:, :, middle_pto_layer].T)

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(field_frames),
        interval=1000 / args.fps,
    )
    writer = animation.FFMpegWriter(
        fps=args.fps,
        codec="libvpx-vp9",
        extra_args=["-crf", "18", "-b:v", "0", "-pix_fmt", "yuv420p"],
    )
    movie.save(output_file, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {len(field_frames)} frames to {output_file}")


if __name__ == "__main__":
    main()
