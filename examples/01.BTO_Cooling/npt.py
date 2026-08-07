"""NPT cooling example for the BaTiO3 effective Hamiltonian."""

import argparse
import json
import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import openferro as of
from openferro.simulation import MDMinimize, SimulationNPTLangevin
from openferro.units import Constants


EXAMPLE_DIR = Path(__file__).resolve().parent


def _load_model_record(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def build_system(config, size, pressure_volume=None):
    """Build the BaTiO3 system from a documented model record.

    Parameters
    ----------
    config : mapping
        BaTiO3 model parameters, units, provenance, and conventions.
    size : int
        Cubic supercell extent.
    pressure_volume : {"determinant", "linearized_small_strain"}, optional
        Override the config's pressure-volume convention for comparison runs.
    """
    if config["model"]["kind"] != "ferroelectric_effective_hamiltonian":
        raise ValueError("The BTO example requires a ferroelectric model record.")
    lattice_parameters = config["lattice"]
    if any(
        lattice_parameters[angle] != 0 for angle in ("alpha", "beta", "gamma")
    ):
        raise ValueError("The BTO example requires an axis-aligned orthogonal cell.")
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

    # The field stores the soft-mode displacement in Angstrom, not its dipole.
    dipole = system.add_field(
        ID="dipole", ftype="Rn", dim=3, value=0.0, mass=mass
    )
    local_strain = system.add_field(
        ID="lstrain", ftype="LocalStrain3D", value=0.0, mass=mass
    )
    if pressure_volume is None:
        pressure_volume = config["conventions"]["pressure_volume"]
    global_strain = system.add_global_strain(
        value=jnp.asarray((0.01, 0.01, 0.01, 0.0, 0.0, 0.0)),
        mass=mass * size**3,
        pressure_volume=pressure_volume,
    )

    parameters = config["parameters"]
    onsite = parameters["onsite"]
    short = parameters["short_range"]
    elastic = parameters["elastic"]
    elastic_dipole = parameters["elastic_dipole"]
    system.add_dipole_onsite_interaction(
        "self_onsite",
        field_ID="dipole",
        K2=onsite["k2"],
        alpha=onsite["alpha"],
        gamma=onsite["gamma"],
    )
    system.add_dipole_interaction_1st_shell(
        "short_range_1",
        field_ID="dipole",
        j1=short["j1"],
        j2=short["j2"],
    )
    system.add_dipole_interaction_2nd_shell(
        "short_range_2",
        field_ID="dipole",
        j3=short["j3"],
        j4=short["j4"],
        j5=short["j5"],
    )
    system.add_dipole_interaction_3rd_shell(
        "short_range_3",
        field_ID="dipole",
        j6=short["j6"],
        j7=short["j7"],
    )
    born = parameters["born"]
    system.add_dipole_dipole_interaction(
        "dipole_ewald",
        field_ID="dipole",
        prefactor=born["Z_star"] ** 2 / born["epsilon_inf"],
    )
    system.add_homo_elastic_interaction(
        "homo_elastic",
        field_ID="gstrain",
        B11=elastic["B11"],
        B12=elastic["B12"],
        B44=elastic["B44"],
    )
    system.add_homo_strain_dipole_interaction(
        "homo_strain_dipole",
        field_1_ID="gstrain",
        field_2_ID="dipole",
        B1xx=elastic_dipole["B1xx"],
        B1yy=elastic_dipole["B1yy"],
        B4yz=elastic_dipole["B4yz"],
    )
    system.add_inhomo_elastic_interaction(
        "inhomo_elastic",
        field_ID="lstrain",
        B11=elastic["B11"],
        B12=elastic["B12"],
        B44=elastic["B44"],
    )
    system.add_inhomo_strain_dipole_interaction(
        "inhomo_strain_dipole",
        field_1_ID="lstrain",
        field_2_ID="dipole",
        B1xx=elastic_dipole["B1xx"],
        B1yy=elastic_dipole["B1yy"],
        B4yz=elastic_dipole["B4yz"],
    )
    return system, dipole, local_strain, global_strain


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXAMPLE_DIR / "BaTiO3.json",
        help="Documented ferroelectric JSON model record.",
    )
    parser.add_argument("--size", type=int, default=None, help="Cubic supercell size.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "output",
        help="Directory for logs and field reports.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Simulation RNG seed.")
    parser.add_argument(
        "--pressure-volume",
        choices=("determinant", "linearized_small_strain"),
        default=None,
        help="Override the config's pressure-volume convention.",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Run a two-site-per-axis CPU smoke calculation.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    size = args.size if args.size is not None else (2 if args.tiny else 20)
    if size < 2:
        raise ValueError("size must be at least 2 for the configured interactions.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=args.output_dir / "simulation.log",
        force=True,
    )
    config = _load_model_record(args.config)
    system, dipole, local_strain, global_strain = build_system(
        config, size, pressure_volume=args.pressure_volume
    )
    pressure_bar = -4.8e4

    logging.info("Structure relaxation")
    dipole.set_integrator("optimization", dt=0.0001)
    global_strain.set_integrator("optimization", dt=0.0001)
    local_strain.set_integrator("optimization", dt=0.0001)
    minimizer = MDMinimize(system, max_iter=1 if args.tiny else 1000, tol=1e-5)
    minimizer.add_thermo_reporter(
        file=str(args.output_dir / "optimization.log"),
        log_interval=1 if args.tiny else 10,
        global_strain=True,
        volume=True,
        potential_energy=True,
        kinetic_energy=False,
        temperature=False,
    )
    minimizer.run(variable_cell=True, pressure=pressure_bar)

    dt = 0.002
    temperatures = np.asarray([300] if args.tiny else [
        400, 350, 320, 310, 300, 290, 280, 270, 260, 250, 240,
        230, 220, 210, 200, 190, 180, 170, 160, 150, 140,
    ])
    relax_steps = 1 if args.tiny else int(10 / dt)
    sampling_steps = 1 if args.tiny else int(100 / dt)
    log_interval = 1 if args.tiny else 100

    simulation = SimulationNPTLangevin(system, pressure=pressure_bar, seed=args.seed)
    simulation.init_velocity(mode="gaussian", temp=float(temperatures[0]))
    for temperature in temperatures:
        temperature = int(temperature)
        dipole.set_integrator("isothermal", dt=dt, temp=temperature, tau=0.1)
        global_strain.set_integrator("isothermal", dt=dt, temp=temperature, tau=1)
        local_strain.set_integrator("isothermal", dt=dt, temp=temperature, tau=1)

        logging.info("T=%s K, NPT equilibration", temperature)
        simulation.remove_all_reporters()
        simulation.run(relax_steps)

        logging.info("T=%s K, NPT sampling", temperature)
        simulation.add_thermo_reporter(
            file=str(args.output_dir / f"thermo_{temperature}K.log"),
            log_interval=log_interval,
            global_strain=True,
            excess_stress=True,
            volume=True,
            potential_energy=True,
            kinetic_energy=True,
            temperature=True,
        )
        simulation.add_field_reporter(
            file_prefix=str(args.output_dir / f"field_{temperature}K"),
            field_ID="dipole",
            log_interval=log_interval,
            field_average=True,
            dump_field=not args.tiny,
        )
        simulation.run(sampling_steps)


if __name__ == "__main__":
    main()
