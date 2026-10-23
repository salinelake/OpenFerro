"""Plot nanoparticle domain structure from the last sampled frame.

The cutaway is a block average colored by the local soft-mode direction, and
the dipole histogram shows where those block averages sit in ``(u_x, u_y, u_z)``.
A separate figure draws the raw soft-mode displacement on a slice through a
meeting point of the four in-plane ``[110]`` domains.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy import ndimage

from geometry import build_spherical_geometry


EXAMPLE_DIR = Path(__file__).resolve().parent
MATERIAL_FILE = EXAMPLE_DIR / "BaTiO3.json"
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "T250D502L512_shell8_seed0"


def list_sample_dumps(output_dir):
    """Return numerically ordered sampling dipole dumps."""
    paths = list((Path(output_dir) / "traj").glob("sample_dipole_dump_*.npy"))
    paths.sort(key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if not paths:
        raise FileNotFoundError(
            f"No sample_dipole_dump_*.npy files in {Path(output_dir) / 'traj'}."
        )
    return paths


def radius_squared(box_size, center):
    """Squared distance from ``center`` on the lattice-index grid."""
    offsets = [np.arange(box_size, dtype=np.float32) - float(origin) for origin in center]
    return (
        offsets[0][:, None, None] ** 2
        + offsets[1][None, :, None] ** 2
        + offsets[2][None, None, :] ** 2
    )


def _unit(vec):
    """Return ``vec`` scaled to unit length."""
    vec = np.asarray(vec, dtype=float)
    return vec / np.linalg.norm(vec)


def plot_block_dipole_distribution(axis, dipoles):
    """Draw a 3D histogram of block-averaged dipoles.

    Hue follows ``1/2 + 1/2 u_hat``. The bin count sets both the color strength,
    from white up to that hue, and the opacity. Both rise linearly and reach
    full strength at half the peak count. Markers have a fixed size.
    """
    limit = max(float(np.percentile(np.abs(dipoles), 99.5)), 0.05)
    edges = np.linspace(-limit, limit, 49)
    counts, bin_edges = np.histogramdd(dipoles, bins=[edges, edges, edges])
    peak = float(counts.max())
    if peak <= 0.0:
        return
    centers = [0.5 * (edge[:-1] + edge[1:]) for edge in bin_edges]
    grid = np.meshgrid(*centers, indexing="ij")
    keep = counts >= max(peak * 0.01, 1.0)
    points = np.column_stack([component[keep] for component in grid])
    density = counts[keep] / peak
    # Linear in the count, with full strength at half the peak.
    strength = np.clip(density / 0.5, 0.0, 1.0)
    direction = points / np.clip(np.linalg.norm(points, axis=1, keepdims=True), 1.0e-12, None)
    direction_rgb = 0.5 + 0.5 * direction
    colors = np.empty((len(points), 4))
    colors[:, :3] = (1.0 - strength)[:, None] + strength[:, None] * direction_rgb
    colors[:, 3] = strength
    axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=11,
        depthshade=False,
        linewidths=0,
    )
    axis.set_facecolor("white")
    for spine in (axis.xaxis, axis.yaxis, axis.zaxis):
        spine.pane.fill = False
        spine.pane.set_edgecolor("0.75")
    step = 0.05 if limit <= 0.25 else 0.1
    tick_max = np.floor(limit / step) * step
    ticks = np.arange(-tick_max, tick_max + 0.5 * step, step)
    # Extend past the outer ticks so the labels at a shared corner do not collide.
    span = max(limit, tick_max) * 1.18
    axis.set_box_aspect((1, 1, 1))
    axis.set(
        xlabel=r"$u_x$ [Å]",
        ylabel=r"$u_y$ [Å]",
        zlabel=r"$u_z$ [Å]",
        xlim=(-span, span),
        ylim=(-span, span),
        zlim=(-span, span),
        xticks=ticks,
        yticks=ticks,
        zticks=ticks,
        title="block-averaged dipole",
    )
    axis.tick_params(labelsize=8, pad=2)
    axis.view_init(elev=20, azim=-70)


def plot_domain_cutaway(field, cell_mask, center, radius, output_path, title):
    """Draw a quarter cutaway colored by the local soft-mode direction.

    The removed wedge opens along +x and is bounded by the (110) and (1-10)
    planes.  Each exposed face is colored by ``1/2 + 1/2 u_hat``.  Panes,
    ticks, labels, and the grid are omitted.
    """
    center = np.asarray(center, dtype=float)
    box = cell_mask.shape[0]
    x_offset = np.arange(box, dtype=np.float32) - np.float32(center[0])
    y_offset = np.arange(box, dtype=np.float32) - np.float32(center[1])
    wedge = ((x_offset[:, None] + y_offset[None, :]) > 0.0) & (
        (x_offset[:, None] - y_offset[None, :]) > 0.0
    )
    kept = cell_mask & ~wedge[:, :, None]
    padded_kept = np.pad(kept, 1, mode="constant", constant_values=False)
    boundary = kept & (
        ~padded_kept[:-2, 1:-1, 1:-1]
        | ~padded_kept[2:, 1:-1, 1:-1]
        | ~padded_kept[1:-1, :-2, 1:-1]
        | ~padded_kept[1:-1, 2:, 1:-1]
        | ~padded_kept[1:-1, 1:-1, :-2]
        | ~padded_kept[1:-1, 1:-1, 2:]
    )
    del padded_kept
    idx_plot = np.argwhere(boundary)
    u_plot = np.asarray(field[boundary], dtype=np.float32)
    del boundary
    u_hat = u_plot / np.clip(np.linalg.norm(u_plot, axis=1, keepdims=True), 1.0e-12, None)
    colors = np.ones((len(u_plot), 4), dtype=np.float32)
    colors[:, :3] = 0.5 + 0.5 * u_hat

    face_dirs = (
        ((-1, 0, 0), ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))),
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0, 1, 0), ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))),
        ((0, 0, -1), ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))),
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    )
    verts = []
    face_colors = []
    for (di, dj, dk), corners in face_dirs:
        neigh = idx_plot + np.array((di, dj, dk))
        in_box = np.all((neigh >= 0) & (neigh < box), axis=1)
        exposed = np.ones(len(idx_plot), dtype=bool)
        exposed[in_box] = ~kept[neigh[in_box, 0], neigh[in_box, 1], neigh[in_box, 2]]
        if not np.any(exposed):
            continue
        quads = idx_plot[exposed, None, :].astype(np.float32) + np.asarray(corners, dtype=np.float32)
        verts.append(quads)
        face_colors.append(colors[exposed])
    verts = np.concatenate(verts, axis=0)
    face_colors = np.concatenate(face_colors, axis=0)
    del idx_plot, colors, kept

    figure = plt.figure(figsize=(7.2, 6.0))
    axis = figure.add_subplot(111, projection="3d")
    axis.add_collection3d(
        Poly3DCollection(verts, facecolors=face_colors, edgecolors="none", zorder=1)
    )
    del verts, face_colors

    eps = 0.15
    r_arc = float(radius) + 0.45
    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 256)
    n_110 = _unit((1.0, 1.0, 0.0))
    n_1m10 = _unit((1.0, -1.0, 0.0))
    zhat = np.array((0.0, 0.0, 1.0))
    arc_110 = center + eps * n_110 + r_arc * (
        np.cos(theta)[:, None] * n_1m10 + np.sin(theta)[:, None] * zhat
    )
    arc_1m10 = center + eps * n_1m10 + r_arc * (
        np.cos(theta)[:, None] * n_110 + np.sin(theta)[:, None] * zhat
    )
    z_edge = np.linspace(center[2] - r_arc, center[2] + r_arc, 64)
    corner = center + eps * _unit((1.0, 0.0, 0.0))
    axis.computed_zorder = False
    contour_kw = dict(color="k", linewidth=1, solid_capstyle="round", zorder=20)
    axis.plot(arc_110[:, 0], arc_110[:, 1], arc_110[:, 2], **contour_kw)
    axis.plot(arc_1m10[:, 0], arc_1m10[:, 1], arc_1m10[:, 2], **contour_kw)
    axis.plot(np.full_like(z_edge, corner[0]), np.full_like(z_edge, corner[1]), z_edge, **contour_kw)
    pad = float(radius) + 1.5
    axis.set(
        xlim=(center[0] - pad, center[0] + pad),
        ylim=(center[1] - pad, center[1] + pad),
        zlim=(center[2] - pad, center[2] + pad),
        title=title,
    )
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=22, azim=0)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def block_average_field(field, mask, block):
    """Replace each masked cell by the mean dipole of its cubic block.

    Blocks are non-overlapping cubes of ``block`` cells on a side, aligned to
    the origin. Only cells selected by ``mask`` enter the mean, and the mean
    is written back onto those cells. A trailing slab thinner than ``block``
    is left unchanged; for this nanoparticle that slab lies outside the
    sphere. Vacuum cells stay zero.
    """
    box = field.shape[0]
    repeats = box // block
    if repeats == 0:
        raise ValueError(f"Lattice size {box} is smaller than block size {block}.")
    covered = repeats * block
    field = np.asarray(field)
    mask = np.asarray(mask)
    shaped = field[:covered, :covered, :covered].reshape(
        repeats, block, repeats, block, repeats, block, 3
    )
    selected = mask[:covered, :covered, :covered].reshape(
        repeats, block, repeats, block, repeats, block
    )
    count = selected.sum(axis=(1, 3, 5))
    total = (shaped * selected[..., None]).sum(axis=(1, 3, 5))
    mean = np.divide(
        total,
        count[..., None],
        out=np.zeros_like(total),
        where=count[..., None] > 0,
    )
    expanded = np.repeat(np.repeat(np.repeat(mean, block, axis=0), block, axis=1), block, axis=2)
    averaged = np.array(field, dtype=np.float32, copy=True)
    averaged[:covered, :covered, :covered] = expanded
    return np.where(mask[..., None], averaged, 0).astype(np.float32)


def plot_block_averaged_cutaway(output_dir, block=3):
    """Cut away the last frame after averaging each dipole over a local block."""
    output_dir = Path(output_dir)
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    material = json.loads(MATERIAL_FILE.read_text(encoding="utf-8"))
    geometry = build_spherical_geometry(
        int(config["size"]),
        float(config["radius"]),
        float(material["lattice"]["a1"]),
        center=config["center"],
    )
    path = list_sample_dumps(output_dir)[-1]
    time_ps = int(path.stem.rsplit("_", 1)[1]) * float(config["dt"])
    field = np.asarray(np.load(path, mmap_mode="r"))
    cell_mask = np.asarray(geometry.cell_mask)
    averaged = block_average_field(field, cell_mask, block)
    del field
    diameter = 2.0 * float(config["radius"])
    diameter_nm = diameter * float(material["lattice"]["a1"]) / 10.0
    shell = float(config["surface_shell_thickness_cells"])
    core = cell_mask & (
        radius_squared(int(config["size"]), geometry.center) < (geometry.radius_cells - shell) ** 2
    )
    offset = block // 2
    block_dipoles = averaged[offset::block, offset::block, offset::block][
        core[offset::block, offset::block, offset::block]
    ]
    output_path = output_dir / f"domain_cutaway_block{block}.png"
    plot_domain_cutaway(
        averaged,
        cell_mask,
        geometry.center,
        geometry.radius_cells,
        output_path,
        title=(
            f"3D cutaway  {block}x{block}x{block} block average  "
            f"D={diameter:.0f} ({diameter_nm:.1f} nm)  {time_ps:.0f} ps"
        ),
    )
    distribution_path = output_dir / f"domain_dipole_xy_block{block}.png"
    distribution_figure = plt.figure(figsize=(6.2, 5.4))
    distribution_axis = distribution_figure.add_subplot(111, projection="3d")
    plot_block_dipole_distribution(distribution_axis, block_dipoles)
    distribution_figure.savefig(distribution_path, dpi=300, bbox_inches="tight")
    plt.close(distribution_figure)
    print(f"Wrote {output_path}")
    print(f"Wrote {distribution_path}")
    return output_path


# The four orthorhombic variants that lie in the xy plane.  Listed in the
# order [110], [1-10], [-110], [-1-10].
INPLANE_110 = np.array(
    [
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
INPLANE_110 /= np.linalg.norm(INPLANE_110, axis=1, keepdims=True)
# Quadrant order is NE, NW, SW, SE.  A counterclockwise vortex places
# [-110], [-1-10], [1-10], [110] in those quadrants; clockwise is the reverse.
_CCW_QUADRANT_LABELS = np.array([2, 3, 1, 0])
_CW_QUADRANT_LABELS = np.array([1, 0, 2, 3])


def coarse_dipole_means(field, mask, block):
    """Return mean dipoles on non-overlapping cubes of ``block`` cells.

    ``field`` may be a memory map.  Only cells selected by ``mask`` enter the
    mean.  The result is the coarse grid used to locate a domain junction;
    plotted displacements stay on the original cells.
    """
    box = int(field.shape[0])
    repeats = box // block
    if repeats == 0:
        raise ValueError(f"Lattice size {box} is smaller than block size {block}.")
    total = np.zeros((repeats, repeats, repeats, 3), dtype=np.float64)
    count = np.zeros((repeats, repeats, repeats), dtype=np.int32)
    for index in range(repeats):
        start = index * block
        slab = np.asarray(field[start : start + block])
        selected = np.asarray(mask[start : start + block])
        shaped = slab.reshape(1, block, repeats, block, repeats, block, 3)
        chosen = selected.reshape(1, block, repeats, block, repeats, block)
        count[index] = chosen.sum(axis=(1, 3, 5))[0]
        total[index] = (shaped * chosen[..., None]).sum(axis=(1, 3, 5))[0]
    mean = np.divide(
        total,
        count[..., None],
        out=np.zeros_like(total),
        where=count[..., None] > 0,
    )
    return mean.astype(np.float32), count


def _label_inplane_110(dipoles, valid):
    """Assign each valid dipole to the nearest in-plane ``[110]`` variant."""
    labels = np.full(dipoles.shape[:-1], -1, dtype=np.int8)
    if not np.any(valid):
        return labels
    direction = dipoles[valid] @ INPLANE_110.T
    labels[valid] = direction.argmax(axis=1).astype(np.int8)
    return labels


def _ring_winding_and_quadrants(dipoles, labels, sites, radius):
    """Return the xy winding and how many quadrants match a vortex."""
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    limit = dipoles.shape[0] - 1
    x = np.clip(np.rint(sites[:, 0:1] + radius * np.cos(theta)).astype(int), 0, limit)
    y = np.clip(np.rint(sites[:, 1:2] + radius * np.sin(theta)).astype(int), 0, limit)
    z = sites[:, 2]
    sampled = dipoles[x, y, z[:, None]]
    angle = np.arctan2(sampled[..., 1], sampled[..., 0])
    step = np.diff(angle, axis=1, append=angle[:, :1])
    step = (step + np.pi) % (2.0 * np.pi) - np.pi
    winding = step.sum(axis=1) / (2.0 * np.pi)

    ring_labels = labels[x, y, z[:, None]]
    sector = np.floor(theta / (0.5 * np.pi)).astype(int)
    match_ccw = np.zeros(len(sites))
    match_cw = np.zeros(len(sites))
    for quadrant in range(4):
        counts = np.stack(
            [(ring_labels[:, sector == quadrant] == label).sum(axis=1) for label in range(4)],
            axis=1,
        )
        majority = counts.argmax(axis=1)
        match_ccw += majority == _CCW_QUADRANT_LABELS[quadrant]
        match_cw += majority == _CW_QUADRANT_LABELS[quadrant]
    return winding, np.maximum(match_ccw, match_cw), np.all(ring_labels >= 0, axis=1)


def _refine_intersection_cell(field, cell, search=16):
    """Move ``cell`` onto the local meeting point of the four variants.

    A five-cell moving average is used only to decide which variant a cell
    belongs to.  The returned index is a single raw cell.
    """
    x0, y0, z0 = (int(value) for value in cell)
    pad = 4
    x_start = max(0, x0 - search - pad)
    y_start = max(0, y0 - search - pad)
    x_stop = min(field.shape[0], x0 + search + pad + 1)
    y_stop = min(field.shape[1], y0 + search + pad + 1)
    slab = np.asarray(field[x_start:x_stop, y_start:y_stop, z0])
    smoothed = ndimage.uniform_filter(slab, size=(5, 5, 1), mode="nearest")
    labels = (smoothed @ INPLANE_110.T).argmax(axis=-1)
    distance = np.stack(
        [ndimage.distance_transform_edt(labels != label) for label in range(4)]
    )
    score = distance.max(axis=0) + 0.25 * distance.sum(axis=0)
    x_lo = x0 - search - x_start
    y_lo = y0 - search - y_start
    x_hi = x_lo + 2 * search + 1
    y_hi = y_lo + 2 * search + 1
    window = score[x_lo:x_hi, y_lo:y_hi]
    meeting = np.argwhere(window <= np.min(window) + 0.5)
    offset_x, offset_y = np.rint(meeting.mean(axis=0)).astype(int)
    return (x0 - search + int(offset_x), y0 - search + int(offset_y), z0)


def find_four_domain_intersection(field, core_mask, block=4):
    """Locate a raw cell where the four in-plane ``[110]`` domains meet.

    Domain labels are assigned on a coarse grid so thermal noise does not
    create false junctions.  The returned cell is refined onto that junction
    and is the center of the un-averaged displacement plot.
    """
    dipoles, counts = coarse_dipole_means(field, core_mask, block)
    occupied = counts > (block**3 // 2)
    magnitude = np.linalg.norm(dipoles, axis=-1)
    valid = occupied & (magnitude > 1.0e-8)
    labels = _label_inplane_110(dipoles, valid)
    neighborhood = 2
    nearby = [
        ndimage.maximum_filter((labels == label).astype(np.uint8), size=2 * neighborhood + 1) > 0
        for label in range(4)
    ]
    candidates = nearby[0] & nearby[1] & nearby[2] & nearby[3] & valid
    sites = np.argwhere(candidates)
    if len(sites) == 0:
        raise RuntimeError("No junction of the four in-plane [110] domains was found.")

    ring_radius = 8
    winding = np.empty(len(sites))
    quadrants = np.empty(len(sites))
    ring_inside = np.empty(len(sites), dtype=bool)
    for start in range(0, len(sites), 25000):
        stop = start + 25000
        winding[start:stop], quadrants[start:stop], ring_inside[start:stop] = (
            _ring_winding_and_quadrants(dipoles, labels, sites[start:stop], ring_radius)
        )
    vortex = ring_inside & (quadrants == 4) & (np.abs(np.abs(winding) - 1.0) < 0.15)
    if not np.any(vortex):
        raise RuntimeError("Four-domain junctions were found, but none had vortex winding.")

    mask = np.zeros(labels.shape, dtype=np.uint8)
    chosen = sites[vortex]
    mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = 1
    component, _ = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=int))
    sizes = np.bincount(component.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    distance = np.stack(
        [ndimage.distance_transform_edt(labels != label) for label in range(4)]
    )
    score = distance.max(axis=0) + 0.25 * distance.sum(axis=0)
    score = np.where(component == largest, score + 0.05 * magnitude, np.inf)
    block_index = np.unravel_index(np.argmin(score), score.shape)
    in_largest = component[chosen[:, 0], chosen[:, 1], chosen[:, 2]] == largest
    winding_value = float(np.sign(np.median(winding[vortex][in_largest])))
    cell = tuple(int(index) * block + block // 2 for index in block_index)
    cell = _refine_intersection_cell(field, cell)
    return {
        "cell": cell,
        "winding": winding_value,
        "block_index": tuple(int(index) for index in block_index),
        "n_vortex_blocks": int(sizes[largest]),
    }


def plot_soft_mode_intersection(field, center, winding, output_path, half_width=30, arrow_stride=2):
    """Draw the raw soft-mode displacement around a four-domain junction.

    Each pixel is one cell.  Arrows are that cell's ``(u_x, u_y)``, drawn on
    every ``arrow_stride`` cell so neighboring arrows remain visible.  Color
    follows ``1/2 + 1/2 u_hat``, mixed toward white so the arrows stay visible.
    """
    x_center, y_center, z_center = (int(value) for value in center)
    x_start = max(0, x_center - half_width)
    x_stop = min(field.shape[0], x_center + half_width + 1)
    y_start = max(0, y_center - half_width)
    y_stop = min(field.shape[1], y_center + half_width + 1)
    slab = np.asarray(field[x_start:x_stop, y_start:y_stop, z_center])
    magnitude = np.linalg.norm(slab, axis=-1)
    direction = slab / np.clip(magnitude[..., None], 1.0e-12, None)
    colors = np.clip(0.5 + 0.5 * direction, 0.0, 1.0)
    colors = 0.45 + 0.55 * colors

    figure, axis = plt.subplots(figsize=(6.2, 5.6))
    axis.imshow(
        np.transpose(colors, (1, 0, 2)),
        origin="lower",
        interpolation="nearest",
        extent=(x_start - 0.5, x_stop - 0.5, y_start - 0.5, y_stop - 0.5),
    )
    x_coords = np.arange(x_start, x_stop)
    y_coords = np.arange(y_start, y_stop)
    sampled_x = x_coords[::arrow_stride]
    sampled_y = y_coords[::arrow_stride]
    sampled = slab[::arrow_stride, ::arrow_stride]
    grid_x, grid_y = np.meshgrid(sampled_x, sampled_y, indexing="ij")
    axis.quiver(
        grid_x,
        grid_y,
        sampled[..., 0],
        sampled[..., 1],
        angles="xy",
        scale_units="xy",
        scale=0.05,
        width=0.0028,
        color="black",
        pivot="middle",
    )
    sense = "counterclockwise" if winding > 0.0 else "clockwise"
    axis.set(
        xlabel="x",
        ylabel="y",
        aspect="equal",
        title=(
            f"soft mode at a [110] four-domain intersection\n"
            f"z = {z_center}, {sense} winding {winding:+.0f}"
        ),
    )
    axis.xaxis.label.set_size(18)
    axis.yaxis.label.set_size(18)
    axis.tick_params(axis="both", which="major", labelsize=16, length=7, width=1.4)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_four_domain_intersection(output_dir, block=4):
    """Find and plot a vortex-like junction in the last sampled frame."""
    output_dir = Path(output_dir)
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    material = json.loads(MATERIAL_FILE.read_text(encoding="utf-8"))
    geometry = build_spherical_geometry(
        int(config["size"]),
        float(config["radius"]),
        float(material["lattice"]["a1"]),
        center=config["center"],
    )
    shell = float(config["surface_shell_thickness_cells"])
    core = np.asarray(geometry.cell_mask) & (
        radius_squared(int(config["size"]), geometry.center)
        < (geometry.radius_cells - shell) ** 2
    )
    path = list_sample_dumps(output_dir)[-1]
    field = np.load(path, mmap_mode="r")
    found = find_four_domain_intersection(field, core, block=block)
    output_path = output_dir / "domain_vortex_intersection.png"
    plot_soft_mode_intersection(field, found["cell"], found["winding"], output_path)
    x_center, y_center, z_center = found["cell"]
    print(
        "Four in-plane [110] domains meet at cell "
        f"({x_center}, {y_center}, {z_center}) "
        f"with winding {found['winding']:+.2f} "
        f"({found['n_vortex_blocks']} coarse blocks)."
    )
    print(f"Wrote {output_path}")
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    plot_block_averaged_cutaway(args.output_dir)
    plot_four_domain_intersection(args.output_dir)
