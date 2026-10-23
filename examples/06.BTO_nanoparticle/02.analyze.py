"""Plot dipole snapshots and calculate polarization and susceptibility."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from openferro.units import Constants

from geometry import build_spherical_geometry, radial_surface_shell_mask


MATERIAL_FILE = Path(__file__).with_name("BaTiO3.json")
BULK_FIELD_LOG = (
    Path(__file__).resolve().parents[1]
    / "01.BTO_Cooling"
    / "output"
    / "field_270K_avg.log"
)
POLARIZATION_TO_SI = 16.02176634  # e / Angstrom**2 to C / m**2


def load_bulk_soft_mode(path=BULK_FIELD_LOG):
    """Return the mean bulk soft-mode magnitude |u| in Angstrom."""
    data = np.loadtxt(path, comments="#", delimiter=",", ndmin=2)
    return float(np.linalg.norm(data[:, 1:4], axis=1).mean())


def list_dipole_dumps(output_dir, stage):
    """Return numerically ordered dump paths from ``<output_dir>/traj``."""
    paths = list((Path(output_dir) / "traj").glob(f"{stage}_dipole_dump_*.npy"))
    paths.sort(key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    return paths


def load_dipole_trajectory(output_dir, stage):
    """Load numerically ordered ``Field_Reporter`` dumps for one stage."""
    paths = list_dipole_dumps(output_dir, stage)
    if not paths:
        raise FileNotFoundError(
            f"No {stage} dipole dumps found under {Path(output_dir) / 'traj'}. "
            "Run the simulation with --dump-field."
        )
    return np.stack([np.load(path, allow_pickle=False) for path in paths])


def calculate_radial_soft_mode_profile(dump_paths, geometry, bin_width_cells=1.0):
    """Average ``|u|`` over snapshots and one-cell radial bins."""
    dump_paths = list(dump_paths)
    if not dump_paths:
        raise ValueError("At least one dipole dump is required.")
    if not np.isfinite(bin_width_cells) or bin_width_cells <= 0.0:
        raise ValueError("bin_width_cells must be finite and positive.")

    active = tuple(np.asarray(geometry.active_indices).T)
    radii = np.linalg.norm(geometry.relative_coordinates, axis=1)
    n_bins = max(1, int(np.ceil(geometry.radius_cells / bin_width_cells)))
    bin_indices = np.minimum(
        np.floor(radii / bin_width_cells).astype(int),
        n_bins - 1,
    )
    counts = np.bincount(bin_indices, minlength=n_bins)
    radius_sums = np.bincount(bin_indices, weights=radii, minlength=n_bins)
    magnitude_sums = np.zeros(n_bins, dtype=float)
    expected_shape = geometry.cell_mask.shape + (3,)

    for path in dump_paths:
        field = np.load(path, mmap_mode="r", allow_pickle=False)
        if field.shape != expected_shape:
            raise ValueError(
                f"Dipole dump {path} has shape {field.shape}, "
                f"expected {expected_shape}."
            )
        magnitudes = np.linalg.norm(np.asarray(field[active]), axis=1)
        magnitude_sums += np.bincount(
            bin_indices,
            weights=magnitudes,
            minlength=n_bins,
        )

    populated = counts > 0
    return {
        "n_snapshots": len(dump_paths),
        "bin_width_cells": float(bin_width_cells),
        "bins": [
            {
                "mean_radius_cells": float(radius_sums[index] / counts[index]),
                "n_cells": int(counts[index]),
                "mean_abs_u_A": float(
                    magnitude_sums[index] / (counts[index] * len(dump_paths))
                ),
            }
            for index in np.flatnonzero(populated)
        ],
    }


def _thermo_columns(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            labels = [label.strip() for label in line.lstrip("#").split(",")]
            if labels and labels[0] == "Step":
                return labels
    raise ValueError(f"Could not find column headers in {path}.")


def load_particle_volumes(output_dir, stage="sample"):
    """Load step indices and particle volumes from a thermo trajectory."""
    path = Path(output_dir) / f"{stage}_thermo.log"
    if not path.is_file():
        raise FileNotFoundError(f"Missing thermo log: {path}")
    columns = _thermo_columns(path)
    if "Volume" not in columns:
        raise ValueError(f"Could not find the Volume column in {path}.")
    data = np.loadtxt(path, comments="#", delimiter=",", ndmin=2)
    return data[:, 0], data[:, columns.index("Volume")]


def load_field_averages(output_dir, stage="sample"):
    """Load step indices and box-averaged soft-mode vectors."""
    path = Path(output_dir) / f"{stage}_dipole_avg.log"
    if not path.is_file():
        raise FileNotFoundError(f"Missing field-average log: {path}")
    data = np.loadtxt(path, comments="#", delimiter=",", ndmin=2)
    return data[:, 0], data[:, 1:4]


def sampling_was_run(output_dir, config):
    """Return True if sampling averages and thermo logs are available."""
    if int(config.get("sampling_steps", 0)) <= 0:
        return False
    output_dir = Path(output_dir)
    return (
        (output_dir / "sample_dipole_avg.log").is_file()
        and (output_dir / "sample_thermo.log").is_file()
    )


def calculate_polarization_and_chi(
    box_means, volumes, z_star, temperature, box_size
):
    """Return sampling polarization in C/m2 and dimensionless soft-mode chi.

    ``box_means`` are ``Field_Reporter`` averages over all ``L**3`` stored
    sites.  The total soft-mode sum is therefore ``box_means * L**3``.
    """
    box_means = np.asarray(box_means)
    volumes = np.asarray(volumes)
    if box_means.ndim != 2 or box_means.shape[-1] != 3:
        raise ValueError("box_means must have shape (n_samples, 3).")
    if volumes.shape != (box_means.shape[0],):
        raise ValueError("One particle volume is required per dipole sample.")
    if box_means.shape[0] < 2:
        raise ValueError("At least two sampling configurations are required for chi.")

    n_sites = int(box_size) ** 3
    moments = float(z_star) * box_means * n_sites
    polarization = moments / volumes[:, None]
    centered_polarization = polarization - polarization.mean(axis=0)
    centered_moment = moments - moments.mean(axis=0)
    chi = (
        centered_polarization.T
        @ centered_moment
        / box_means.shape[0]
        / (Constants.epsilon0 * Constants.kb * float(temperature))
    )

    return polarization * POLARIZATION_TO_SI, chi


def calculate_polarization_autocorrelation(polarization, max_lag=None):
    """Return normalized Cartesian autocorrelation of polarization.

    Parameters
    ----------
    polarization : ndarray, shape (n_samples, 3)
        Polarization trajectory (any consistent units).
    max_lag : int, optional
        Maximum lag index. Defaults to ``n_samples // 2``.

    Returns
    -------
    lags : ndarray, shape (n_lags,)
        Integer lag indices ``0, 1, ..., max_lag``.
    acf : ndarray, shape (n_lags, 3)
        Biased, mean-subtracted, variance-normalized autocorrelation for
        ``P_x``, ``P_y``, and ``P_z``.
    """
    polarization = np.asarray(polarization)
    if polarization.ndim != 2 or polarization.shape[-1] != 3:
        raise ValueError("polarization must have shape (n_samples, 3).")
    n_samples = polarization.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required for autocorrelation.")

    if max_lag is None:
        max_lag = n_samples // 2
    max_lag = int(max_lag)
    if max_lag < 0 or max_lag >= n_samples:
        raise ValueError("max_lag must satisfy 0 <= max_lag < n_samples.")

    centered = polarization - polarization.mean(axis=0)
    lags = np.arange(max_lag + 1)
    acf = np.empty((max_lag + 1, 3), dtype=float)
    for component in range(3):
        series = centered[:, component]
        variance = float(np.dot(series, series) / n_samples)
        if variance <= 0.0:
            acf[:, component] = 0.0
            continue
        full = np.correlate(series, series, mode="full")
        acf[:, component] = full[n_samples - 1 : n_samples + max_lag] / (
            n_samples * variance
        )
    return lags, acf


def plot_polarization_and_acf(times_ps, polarization, acf_times_ps, acf, output_path):
    """Plot polarization trajectory and its autocorrelation to one figure."""
    polarization_uC_cm2 = np.asarray(polarization) * 100.0  # C/m^2 -> uC/cm^2
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)

    axes[0].plot(times_ps, polarization_uC_cm2[:, 0], label=r"$P_x$")
    axes[0].plot(times_ps, polarization_uC_cm2[:, 1], label=r"$P_y$")
    axes[0].plot(times_ps, polarization_uC_cm2[:, 2], label=r"$P_z$")
    axes[0].set(xlabel="Time (ps)", ylabel=r"Polarization ($\mathrm{\mu C/cm^2}$)")
    axes[0].legend(frameon=False)

    axes[1].plot(acf_times_ps, acf[:, 0], label=r"$C_{xx}$")
    axes[1].plot(acf_times_ps, acf[:, 1], label=r"$C_{yy}$")
    axes[1].plot(acf_times_ps, acf[:, 2], label=r"$C_{zz}$")
    axes[1].axhline(0.0, color="k", linewidth=0.8, linestyle=":")
    axes[1].set(xlabel=r"Lag $\tau$ (ps)", ylabel=r"$C(\tau)$")
    axes[1].legend(frameon=False)

    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_snapshots(
    equilibration, cell_mask, output_path, bulk_u, z_center=None
):
    """Plot minimized and final-equilibration central xy soft-mode sections."""
    # relax dump at step 0 is written at the start of equilibration, i.e. the
    # minimized configuration before Langevin dynamics.
    panels = [
        ("Minimized", equilibration[0]),
        ("Equilibration", equilibration[-1]),
    ]

    if z_center is None:
        z_center = cell_mask.shape[2] // 2
    z_center = int(z_center)
    if z_center < 0 or z_center >= cell_mask.shape[2]:
        raise ValueError("z_center is outside the field.")
    plane_mask = cell_mask[:, :, z_center]
    active_x, active_y = np.nonzero(plane_mask)

    slices = []
    for _, field in panels:
        plane = field[:, :, z_center]
        u_z = np.where(plane_mask, plane[..., 2], np.nan)
        slices.append((u_z, plane[active_x, active_y, 0], plane[active_x, active_y, 1]))

    finite_uz = np.concatenate(
        [u_z[np.isfinite(u_z)] for u_z, _, _ in slices]
    )
    vmax = float(np.max(np.abs(finite_uz))) if finite_uz.size else 1.0
    vmax = max(vmax, float(bulk_u), 1e-12)
    in_plane = np.concatenate(
        [np.hypot(u_x, u_y) for _, u_x, u_y in slices]
    )
    arrow_scale = float(in_plane.max()) if in_plane.size and in_plane.max() > 0.0 else 1.0

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    image = None
    for axis, (title, _), (u_z, u_x, u_y) in zip(axes, panels, slices):
        image = axis.imshow(
            u_z.T,
            origin="lower",
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.quiver(
            active_x,
            active_y,
            u_x / arrow_scale,
            u_y / arrow_scale,
            color="black",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
        )
        axis.set(
            title=f"{title}: central xy (z = {z_center})",
            xlabel="x",
            ylabel="y",
            aspect="equal",
        )
    colorbar = figure.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    colorbar.set_label(r"$u_z$ [Angstrom]")
    for value in (-bulk_u, bulk_u):
        colorbar.ax.axhline(value, color="k", linestyle="--", linewidth=1.0)
    colorbar.ax.text(
        1.15,
        bulk_u,
        rf"$|u|_{{\mathrm{{bulk}}}}={bulk_u:.3f}$",
        transform=colorbar.ax.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=8,
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args(argv)

    with (args.output_dir / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    with MATERIAL_FILE.open(encoding="utf-8") as stream:
        material = json.load(stream)
    lattice_constant = float(material["lattice"]["a1"])
    z_star = float(material["parameters"]["born"]["Z_star"])

    
    if sampling_was_run(args.output_dir, config):
        avg_steps, box_means = load_field_averages(args.output_dir, stage="sample")
        thermo_steps, volumes = load_particle_volumes(
            args.output_dir, stage="sample"
        )
        volume_by_step = dict(zip(thermo_steps.astype(int), volumes))
        try:
            matched_volumes = np.array(
                [volume_by_step[int(step)] for step in avg_steps]
            )
        except KeyError as exc:
            raise ValueError(
                "Could not match sample_dipole_avg.log steps to "
                "sample_thermo.log volumes."
            ) from exc
        polarization, chi = calculate_polarization_and_chi(
            box_means,
            matched_volumes,
            z_star,
            config["temperature"],
            config["size"],
        )
        lags, acf = calculate_polarization_autocorrelation(polarization)
        times_ps = avg_steps * float(config["dt"])
        if len(avg_steps) >= 2:
            sample_dt_ps = float(avg_steps[1] - avg_steps[0]) * float(config["dt"])
        else:
            sample_dt_ps = float(config["dt"])
        acf_times_ps = lags * sample_dt_ps
        polarization_plot = args.output_dir / "polarization.png"
        plot_polarization_and_acf(
            times_ps, polarization, acf_times_ps, acf, polarization_plot
        )
    else:
        polarization = None
        chi = None
        polarization_plot = None

    geometry = build_spherical_geometry(
        config["size"],
        config["radius"],
        lattice_constant,
        center=config.get("center"),
    )
    profile_paths = list_dipole_dumps(args.output_dir, "sample")
    if not profile_paths:
        profile_paths = list_dipole_dumps(args.output_dir, "relax")
    radial_profile_path = None
    if profile_paths:
        radial_profile = calculate_radial_soft_mode_profile(
            profile_paths,
            geometry,
        )
        shell_thickness = float(config.get("surface_shell_thickness_cells", 0.0))
        k2_shift = float(config.get("surface_k2_shift", 0.0))
        radial_profile["shell_thickness_cells"] = shell_thickness
        radial_profile["surface_k2_shift_eV_per_A2"] = k2_shift
        if shell_thickness > 0.0:
            shell = radial_surface_shell_mask(geometry, shell_thickness)
            radial_profile["n_shell_cells"] = int(shell.sum())
            radial_profile["shell_fraction"] = float(
                shell.sum() / geometry.n_particle
            )
            radial_profile["shell_inner_radius_cells"] = float(
                geometry.radius_cells - shell_thickness
            )
        else:
            radial_profile["n_shell_cells"] = 0
            radial_profile["shell_fraction"] = 0.0
            radial_profile["shell_inner_radius_cells"] = None
        radial_profile_path = args.output_dir / "radial_soft_mode_profile.json"
        with radial_profile_path.open("w", encoding="utf-8") as stream:
            json.dump(radial_profile, stream, indent=2)
            stream.write("\n")

    bulk_u = load_bulk_soft_mode()
    plot_path = args.output_dir / "dipole_snapshots.png"
    z_center = int(np.floor(float(geometry.center[2]) + 0.5))
    try:
        equilibration = load_dipole_trajectory(args.output_dir, "relax")
        plot_snapshots(
            equilibration,
            geometry.cell_mask,
            plot_path,
            bulk_u,
            z_center=z_center,
        )
    except:
        pass


    mean_polarization = None if polarization is None else polarization.mean(axis=0)
    results = {
        "temperature_K": float(config["temperature"]),
        "radius_cells": float(config["radius"]),
        "diameter_cells": 2.0 * float(config["radius"]),
        "size": int(config["size"]),
        "n_particle": int(geometry.n_particle),
        "n_samples": None if polarization is None else int(polarization.shape[0]),
        "mean_polarization_C_per_m2": (
            None if mean_polarization is None else mean_polarization.tolist()
        ),
        "chi": None if chi is None else np.asarray(chi).tolist(),
    }
    results_path = args.output_dir / "polarization_chi.json"
    with results_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")

    if polarization is None:
        print("Sampling stage was skipped; polarization/chi not computed.")
    else:
        print("Mean sampling polarization (C/m^2):")
        print(mean_polarization)
        print("Ionic soft-mode chi:")
        print(chi)
        print(f"Saved {polarization_plot}")
    if radial_profile_path is None:
        print("No dipole dumps found; radial profile not computed.")
    else:
        print(f"Saved {radial_profile_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
