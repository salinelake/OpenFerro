"""Average magnetization, susceptibility, local mode, and exchange-fit check.

Reads ``spin_{T}K_avg.log`` files under ``output_frozen/L{size}`` and
``output_coupled/L{size}``. Each spin log stores the spatial average of the
unit-length spin field, ``m(t) = <S>_sites``. The reported magnetization is
the time average of ``|m|``, and the susceptibility follows the fluctuation
formula

    chi = N * (<|m|^2> - <|m|>^2) / (k_B * T)

with ``N = size**3`` and units of eV^{-1}. ``parameter.png`` compares
``evaluate_exchange`` with the published ``exchange_vs_normalized_u_PRM_2018``
values. ``magnetic.png`` shows magnetization and susceptibility.
``distortion.png`` shows the local-mode amplitude. Coupled runs also write
``local_mode_{T}K_avg.log``; frozen local-mode logs are not written, so that
panel uses the fixed [110] amplitude from ``run_frozen.py``.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from openferro.units import Constants

from model import MEV_TO_EV, SMO_CONFIG, evaluate_exchange, load_smo_parameters

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

HERE = Path(__file__).resolve().parent
CASES = (("frozen", HERE / "output_frozen"), ("coupled", HERE / "output_coupled"))
SPIN_AVG_PATTERN = re.compile(r"spin_([0-9]+(?:\.[0-9]+)?)K_avg\.log$")
LOCAL_MODE_COMPONENTS = (
    ("abs_ux", r"$|u_x|$", "C2"),
    ("abs_uy", r"$|u_y|$", "C3"),
    # ("abs_uz", r"$|u_z|$", "C4"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Cubic lattice size L to analyze (reads output_*/L{L}).",
    )
    return parser.parse_args(argv)


def load_vector_average(path):
    """Return the spatial-average vector trajectory from a field-average log."""
    data = np.loadtxt(path, comments="#", delimiter=",", ndmin=2)
    if data.shape[1] < 4:
        raise ValueError(f"Expected Step, vx, vy, vz columns in {path}.")
    return data[:, 1:4]


def load_magnetization(path):
    """Return the per-sample magnetization magnitude from a spin average log."""
    return np.linalg.norm(load_vector_average(path), axis=1)


def load_abs_local_mode(path):
    """Return time-averaged |<u_x>|, |<u_y>|, |<u_z>| from a local-mode log."""
    return np.mean(np.abs(load_vector_average(path)), axis=0)


def frozen_local_mode_abs(parameters):
    """Return the fixed [110] local-mode amplitude used in the frozen run."""
    u_eq = float(parameters["effective_hamiltonian_u_eq_angstrom"])
    return np.abs(np.asarray((u_eq / np.sqrt(2.0), u_eq / np.sqrt(2.0), 0.0)))


def load_exchange_reference():
    """Return tabulated u/u_eq, J1_ip, and J1_op from the 2018 exchange curve."""
    with SMO_CONFIG.open(encoding="utf-8") as handle:
        curve = json.load(handle)["exchange_vs_normalized_u_PRM_2018"]
    return (
        np.asarray(curve["u_over_u_eq"], dtype=float),
        np.asarray(curve["J1_ip_meV"], dtype=float),
        np.asarray(curve["J1_op_meV"], dtype=float),
    )


def fitted_exchange_meV(parameters, u_over_u_eq):
    """Return evaluate_exchange in meV at one or more normalized amplitudes."""
    ratios = np.atleast_1d(np.asarray(u_over_u_eq, dtype=float))
    values = np.asarray(
        [evaluate_exchange(parameters, float(ratio)) for ratio in ratios]
    )
    return values[:, 0] / MEV_TO_EV, values[:, 1] / MEV_TO_EV


def summarize_temperature(path, temperature, n_sites):
    """Return mean |m| and chi for one temperature."""
    magnetization = load_magnetization(path)
    if magnetization.size < 2:
        raise ValueError(f"Need at least two samples in {path}.")
    mean_m = float(np.mean(magnetization))
    mean_m2 = float(np.mean(magnetization**2))
    chi = n_sites * (mean_m2 - mean_m**2) / (Constants.kb * float(temperature))
    return mean_m, chi, magnetization.size


def collect_case(output_root, size):
    """Return temperature-sorted magnetization data for one cubic cell size."""
    records = []
    size_dir = Path(output_root) / f"L{size}"
    if not size_dir.is_dir():
        return records
    n_sites = int(size) ** 3
    for path in sorted(size_dir.glob("spin_*K_avg.log")):
        match = SPIN_AVG_PATTERN.search(path.name)
        if match is None:
            continue
        temperature = float(match.group(1))
        mean_m, chi, n_samples = summarize_temperature(path, temperature, n_sites)
        record = {
            "size_dir": size_dir.name,
            "temperature": temperature,
            "magnetization": mean_m,
            "chi": chi,
            "n_sites": n_sites,
            "n_samples": n_samples,
        }
        mode_path = size_dir / f"local_mode_{match.group(1)}K_avg.log"
        if mode_path.exists():
            abs_u = load_abs_local_mode(mode_path)
            record["abs_ux"] = float(abs_u[0])
            record["abs_uy"] = float(abs_u[1])
            record["abs_uz"] = float(abs_u[2])
        records.append(record)
    records.sort(key=lambda item: item["temperature"])
    return records


def plot_exchange_fit(ax, parameters):
    """Compare evaluate_exchange with the published J1(u/u_eq) points."""
    u_over_u_eq, j_ip_ref, j_op_ref = load_exchange_reference()
    j_ip_fit, j_op_fit = fitted_exchange_meV(parameters, u_over_u_eq)
    ratio_grid = np.linspace(0.0, 1.0, 101)
    j_ip_curve, j_op_curve = fitted_exchange_meV(parameters, ratio_grid)

    ax.plot(ratio_grid, j_ip_curve, color="C0", linestyle="solid", marker="None", label=r"$J_1^{ip}$")
    ax.plot(ratio_grid, j_op_curve, color="C1", linestyle="solid", marker="None", label=r"$J_1^{op}$")
    ax.plot(u_over_u_eq, j_ip_ref, color="C0", linestyle="None", marker="o", )  # reference data
    ax.plot(u_over_u_eq, j_op_ref, color="C1", linestyle="None", marker="s", )  # reference data
    ax.set_xlabel(r"$u/u_{\mathrm{eq}}$")
    ax.set_ylabel(r"$J$ [meV]")
    ax.legend()

    print("\nExchange fit versus exchange_vs_normalized_u_PRM_2018")
    print("u/u_eq,  J_ip fit,  J_ip ref,  J_op fit,  J_op ref")
    for ratio, ip_fit, ip_ref, op_fit, op_ref in zip(
        u_over_u_eq, j_ip_fit, j_ip_ref, j_op_fit, j_op_ref
    ):
        print(
            f"{ratio:6.1f},  {ip_fit:8.4f},  {ip_ref:8.4f},  "
            f"{op_fit:8.4f},  {op_ref:8.4f}"
        )


def main(argv=None):
    args = parse_args(argv)
    if args.size < 1:
        raise ValueError("--size must be >= 1.")

    results = {}
    parameter_path = HERE / "parameter.png"
    magnetic_path = HERE / "magnetic.png"
    distortion_path = HERE / "distortion.png"
    parameters = load_smo_parameters()
    frozen_u = frozen_local_mode_abs(parameters)
    for case, output_root in CASES:
        records = collect_case(output_root, args.size)
        results[case] = records
        if not records:
            print(f"No spin average logs found under {output_root}")
            continue
        print(f"\n{case} ({records[0]['size_dir']}, N={records[0]['n_sites']})")
        print("T [K],  |M|,  chi [eV^-1],  |ux|,  |uy|,  |uz|,  n_samples")
        for item in records:
            if "abs_ux" in item:
                abs_u = (item["abs_ux"], item["abs_uy"], item["abs_uz"])
            else:
                abs_u = frozen_u
            print(
                f"{item['temperature']:6.1f},  {item['magnetization']:.6f},  "
                f"{item['chi']:.4e},  {abs_u[0]:.6f},  {abs_u[1]:.6f},  "
                f"{abs_u[2]:.6f},  {item['n_samples']}"
            )

    parameter_fig, parameter_ax = plt.subplots(figsize=(3, 2.5))
    plot_exchange_fit(parameter_ax, parameters)
    parameter_fig.tight_layout()
    parameter_fig.savefig(parameter_path)
    print(f"Wrote {parameter_path}")

    magnetic_fig, magnetic_axes = plt.subplots(2, 1, figsize=(4, 3.5), sharex=True)
    distortion_fig, distortion_ax = plt.subplots(figsize=(3.5, 2.5))
    styles = {
        "frozen": {"color": "C0", "linestyle": "dashed"},
        "coupled": {"color": "C1", "linestyle": "solid"},
    }
    for case, records in results.items():
        if not records:
            continue
        temperatures = [item["temperature"] for item in records]
        magnetization = [item["magnetization"] for item in records]
        chi = [item["chi"] for item in records]
        label = f"{case}"
        magnetic_axes[0].plot(temperatures, magnetization, label=label, **styles[case])
        magnetic_axes[1].plot(temperatures, chi, label=label, **styles[case])
        mode_records = [item for item in records if "abs_ux" in item]
        if not mode_records:
            continue
        mode_temperatures = [item["temperature"] for item in mode_records]
        for key, component_label, color in LOCAL_MODE_COMPONENTS:
            distortion_ax.plot(
                mode_temperatures,
                [item[key] for item in mode_records],
                label=f"{case}",
                color=color,
                linestyle=styles[case]["linestyle"],
            )

    for (_, component_label, color), value in zip(LOCAL_MODE_COMPONENTS, frozen_u):
        distortion_ax.axhline(
            value,
            color=color,
            linestyle=styles["frozen"]["linestyle"],
            marker="None",
            label=f"frozen {component_label}",
        )

    magnetic_axes[0].set_ylabel(r"$M\ /\ M_s$")
    magnetic_axes[0].set_ylim(-0.1, 1.05)
    magnetic_axes[1].set_xlabel("T [K]")
    magnetic_axes[1].set_ylabel(r"$\chi$ [arb. units]")
    magnetic_axes[1].set_yscale("log")
    magnetic_axes[1].set_ylim(25, 2e5)
    for ax in magnetic_axes:
        ax.axvline(100.0, color='k', marker='o', markersize=0, linestyle='--')
        ax.legend()
    magnetic_fig.tight_layout()
    magnetic_fig.savefig(magnetic_path)
    print(f"Wrote {magnetic_path}")

    distortion_ax.set_xlabel("T [K]")
    distortion_ax.set_ylabel(r"$|u|$ [$\mathrm{\AA}$]")
    distortion_ax.axvline(100.0, color='k', marker='o', markersize=0, linestyle='--')
    distortion_ax.legend()
    distortion_fig.tight_layout()
    distortion_fig.savefig(distortion_path)
    print(f"Wrote {distortion_path}")


if __name__ == "__main__":
    main()

