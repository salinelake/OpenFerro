"""Benchmark single-node GPU scaling for BaTiO3 NPT dynamics at 270 K."""

import argparse
import csv
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import openferro as of
from openferro.parallelism import DeviceMesh
from openferro.simulation import SimulationNPTLangevin
from openferro.units import Constants


TEMPERATURE_K = 270.0
PRESSURE_BAR = -4.8e4
TIME_STEP_PS = 0.002
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "01.BTO_Cooling" / "BaTiO3.json"


def build_system(config, size, device_mesh):
    """Build the BaTiO3 system used by ``01.BTO_Cooling``."""
    lattice_parameters = config["lattice"]
    lattice = of.SimpleCubic3D(
        size,
        size,
        size,
        jnp.asarray((lattice_parameters["a1"], 0.0, 0.0)),
        jnp.asarray((0.0, lattice_parameters["a2"], 0.0)),
        jnp.asarray((0.0, 0.0, lattice_parameters["a3"])),
    )
    system = of.System(lattice)
    mass = 200 * Constants.amu

    dipole = system.add_field("dipole", ftype="Rn", dim=3, value=0.0, mass=mass)
    local_strain = system.add_field(
        "lstrain", ftype="LocalStrain3D", value=0.0, mass=mass
    )
    global_strain = system.add_global_strain(
        value=jnp.asarray((0.01, 0.01, 0.01, 0.0, 0.0, 0.0)),
        mass=mass * size**3,
    )
    system.move_fields_to_multi_devs(device_mesh)

    parameters = config["parameters"]
    onsite = parameters["onsite"]
    short = parameters["short_range"]
    elastic = parameters["elastic"]
    elastic_dipole = parameters["elastic_dipole"]
    born = parameters["born"]

    system.add_dipole_onsite_interaction(
        "self_onsite",
        field_ID="dipole",
        K2=onsite["k2"],
        alpha=onsite["alpha"],
        gamma=onsite["gamma"],
    )
    system.add_dipole_interaction_1st_shell(
        "short_range_1", field_ID="dipole", j1=short["j1"], j2=short["j2"]
    )
    system.add_dipole_interaction_2nd_shell(
        "short_range_2",
        field_ID="dipole",
        j3=short["j3"],
        j4=short["j4"],
        j5=short["j5"],
    )
    system.add_dipole_interaction_3rd_shell(
        "short_range_3", field_ID="dipole", j6=short["j6"], j7=short["j7"]
    )
    system.add_dipole_dipole_interaction(
        "dipole_ewald",
        field_ID="dipole",
        prefactor=born["Z_star"] ** 2 / born["epsilon_inf"],
    )
    system.add_homo_elastic_interaction(
        "homo_elastic", field_ID="gstrain", **elastic
    )
    system.add_homo_strain_dipole_interaction(
        "homo_strain_dipole",
        field_1_ID="gstrain",
        field_2_ID="dipole",
        **elastic_dipole,
    )
    system.add_inhomo_elastic_interaction(
        "inhomo_elastic", field_ID="lstrain", **elastic
    )
    system.add_inhomo_strain_dipole_interaction(
        "inhomo_strain_dipole",
        field_1_ID="lstrain",
        field_2_ID="dipole",
        **elastic_dipole,
    )
    return system, dipole, local_strain, global_strain


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device_mesh = DeviceMesh()
    system, dipole, local_strain, global_strain = build_system(
        config, args.size, device_mesh
    )

    dipole.set_integrator(
        "isothermal", dt=TIME_STEP_PS, temp=TEMPERATURE_K, tau=0.1
    )
    local_strain.set_integrator(
        "isothermal", dt=TIME_STEP_PS, temp=TEMPERATURE_K, tau=1.0
    )
    global_strain.set_integrator(
        "isothermal", dt=TIME_STEP_PS, temp=TEMPERATURE_K, tau=1.0
    )
    simulation = SimulationNPTLangevin(system, pressure=PRESSURE_BAR, seed=args.seed)
    simulation.init_velocity(mode="gaussian", temp=TEMPERATURE_K)

    start = perf_counter()
    simulation.run(1)
    jax.block_until_ready([field.get_values() for field in system.get_all_fields()])
    compile_seconds = perf_counter() - start

    simulation.run(args.warmup_steps)
    jax.block_until_ready([field.get_values() for field in system.get_all_fields()])

    repeat_seconds = []
    for _ in range(args.repeats):
        start = perf_counter()
        simulation.run(args.steps)
        jax.block_until_ready(
            [field.get_values() for field in system.get_all_fields()]
        )
        repeat_seconds.append(perf_counter() - start)

    peak_memory = np.asarray(
        [
            (device.memory_stats() or {}).get("peak_bytes_in_use", np.nan) / 2**30
            for device in jax.devices()
        ]
    )
    elapsed_seconds = float(np.median(repeat_seconds))
    sites = args.size**3
    local_mode_rms = jnp.sqrt(jnp.mean(jnp.sum(dipole.get_values() ** 2, axis=-1)))
    volume_per_cell = system.calc_volume() / sites
    local_mode_rms, volume_per_cell = jax.device_get(
        (local_mode_rms, volume_per_cell)
    )

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "jax_version": jax.__version__,
        "gpu_model": jax.devices()[0].device_kind,
        "gpu_count": jax.device_count(),
        "mesh_rows": int(device_mesh.mesh.shape["x"]),
        "mesh_cols": int(device_mesh.mesh.shape["y"]),
        "lattice_size": args.size,
        "sites": sites,
        "temperature_k": TEMPERATURE_K,
        "pressure_bar": PRESSURE_BAR,
        "time_step_ps": TIME_STEP_PS,
        "compile_seconds": compile_seconds,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.steps,
        "repeats": args.repeats,
        "median_seconds": elapsed_seconds,
        "milliseconds_per_step": 1000 * elapsed_seconds / args.steps,
        "steps_per_second": args.steps / elapsed_seconds,
        "peak_memory_per_gpu_gib": float(np.max(peak_memory)),
        "aggregate_peak_memory_gib": float(np.sum(peak_memory)),
        "final_local_mode_rms_angstrom": float(local_mode_rms),
        "final_volume_per_cell_angstrom3": float(volume_per_cell),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=row, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"L={args.size}, {row['gpu_count']} GPU: "
        f"{row['milliseconds_per_step']:.3f} ms/step, "
        f"{row['peak_memory_per_gpu_gib']:.2f} GiB peak/GPU"
    )


if __name__ == "__main__":
    main()
