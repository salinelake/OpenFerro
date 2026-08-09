"""Simulate electric-field-driven domain motion in a PTO/STO superlattice."""

import argparse
import json
import logging
from pathlib import Path

import jax.numpy as jnp

import openferro as of
from openferro.engine.ferroelectric_superlatt import (
    get_homo_strain_dipole_interaction_on_AmBnLattice,
    get_self_energy_onsite_on_AmBnLattice,
    get_short_range_1stnn_on_AmBnLattice,
    get_short_range_2ednn_on_AmBnLattice,
    get_short_range_3rdnn_on_AmBnLattice,
)
from openferro.simulation import SimulationNPTLangevin
from openferro.units import Constants


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PTO_CONFIG = REPOSITORY_ROOT / "model_configs" / "PbTiO3_SCAN.json"
DEFAULT_STO_CONFIG = REPOSITORY_ROOT / "model_configs" / "SrTiO3_WC.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def build_system(
    pto_config,
    sto_config,
    lateral_size,
    pto_layers,
    sto_layers,
):
    """Build the PTO/STO effective-Hamiltonian system."""
    layer_count = pto_layers + sto_layers
    lattice_constant = pto_config["lattice"]["a1"]
    lattice = of.SimpleCubic3D(
        lateral_size,
        lateral_size,
        layer_count,
        lattice_constant * jnp.asarray((1.0, 0.0, 0.0)),
        lattice_constant * jnp.asarray((0.0, 1.0, 0.0)),
        lattice_constant * jnp.asarray((0.0, 0.0, 1.0)),
    )
    system = of.System(lattice)
    site_count = lateral_size**2 * layer_count
    mass = 200 * Constants.amu

    # This field stores the physical local dipole in e Angstrom, not the
    # soft-mode displacement used by the material parameter records.
    dipole = system.add_field(
        "dipole",
        ftype="Rn",
        dim=3,
        value=jnp.asarray((0.0, 0.0, -3.0)),
        mass=mass,
    )
    dipole.set_values(
        dipole.get_values().at[:, :, pto_layers - 1 :, :].set(0.0)
    )
    global_strain = system.add_global_strain(
        value=jnp.asarray((-0.012, -0.012, 0.051, 0.0, 0.0, 0.0)),
        mass=mass * site_count,
    )

    pto = pto_config["parameters"]
    sto = sto_config["parameters"]
    z_pto = pto["born"]["Z_star"]
    z_sto = sto["born"]["Z_star"]

    system.add_self_interaction(
        "self_onsite",
        field_ID="dipole",
        energy_engine=get_self_energy_onsite_on_AmBnLattice(
            lattice, pto_layers, sto_layers
        ),
        parameters=[
            pto["onsite"]["k2"] / z_pto**2,
            pto["onsite"]["alpha"] / z_pto**4,
            pto["onsite"]["gamma"] / z_pto**4,
            sto["onsite"]["k2"] / z_sto**2,
            sto["onsite"]["alpha"] / z_sto**4,
            sto["onsite"]["gamma"] / z_sto**4,
        ],
    )
    system.add_self_interaction(
        "short_range_1",
        field_ID="dipole",
        energy_engine=get_short_range_1stnn_on_AmBnLattice(
            lattice, pto_layers, sto_layers
        ),
        parameters=[
            pto["short_range"]["j1"] / z_pto**2,
            pto["short_range"]["j2"] / z_pto**2,
            sto["short_range"]["j1"] / z_sto**2,
            sto["short_range"]["j2"] / z_sto**2,
        ],
    )
    system.add_self_interaction(
        "short_range_2",
        field_ID="dipole",
        energy_engine=get_short_range_2ednn_on_AmBnLattice(
            lattice, pto_layers, sto_layers
        ),
        parameters=[
            pto["short_range"]["j3"] / z_pto**2,
            pto["short_range"]["j4"] / z_pto**2,
            pto["short_range"]["j5"] / z_pto**2,
            sto["short_range"]["j3"] / z_sto**2,
            sto["short_range"]["j4"] / z_sto**2,
            sto["short_range"]["j5"] / z_sto**2,
        ],
    )
    system.add_self_interaction(
        "short_range_3",
        field_ID="dipole",
        energy_engine=get_short_range_3rdnn_on_AmBnLattice(
            lattice, pto_layers, sto_layers
        ),
        parameters=[
            pto["short_range"]["j6"] / z_pto**2,
            pto["short_range"]["j7"] / z_pto**2,
            sto["short_range"]["j6"] / z_sto**2,
            sto["short_range"]["j7"] / z_sto**2,
        ],
    )
    system.add_homo_elastic_interaction(
        "homo_elastic", field_ID="gstrain", **pto["elastic"]
    )
    # The legacy model deliberately omits STO homogeneous strain coupling.
    system.add_mutual_interaction(
        "homo_strain_dipole",
        field_1_ID="gstrain",
        field_2_ID="dipole",
        energy_engine=get_homo_strain_dipole_interaction_on_AmBnLattice(
            lattice, pto_layers, sto_layers
        ),
        parameters=[
            pto["elastic_dipole"]["B1xx"] / z_pto**2,
            pto["elastic_dipole"]["B1yy"] / z_pto**2,
            pto["elastic_dipole"]["B4yz"] / z_pto**2,
            0.0,
            0.0,
            0.0,
        ],
    )
    system.add_dipole_dipole_interaction(
        "dipole_ewald",
        field_ID="dipole",
        prefactor=1.0 / pto["born"]["epsilon_inf"],
    )
    electric_field = system.add_dipole_efield_interaction(
        "dipole_efield", field_ID="dipole", E=(0.0, 0.0, 0.0)
    )
    return system, dipole, global_strain, electric_field


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pto-config", type=Path, default=DEFAULT_PTO_CONFIG)
    parser.add_argument("--sto-config", type=Path, default=DEFAULT_STO_CONFIG)
    parser.add_argument("--lateral-size", type=int, default=256)
    parser.add_argument("--pto-layers", type=int, default=48)
    parser.add_argument("--sto-layers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--dt-ps", type=float, default=0.002)
    parser.add_argument("--relax-time-ps", type=float, default=500.0)
    parser.add_argument("--drive-time-ps", type=float, default=4000.0)
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument("--dump-interval", type=int, default=50000)
    parser.add_argument("--electric-field", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=args.output_dir / "simulation.log",
        force=True,
    )

    pto_config = json.loads(args.pto_config.read_text(encoding="utf-8"))
    sto_config = json.loads(args.sto_config.read_text(encoding="utf-8"))
    relax_steps = int(args.relax_time_ps / args.dt_ps)
    drive_steps = int(args.drive_time_ps / args.dt_ps)

    system, dipole, global_strain, electric_field = build_system(
        pto_config,
        sto_config,
        args.lateral_size,
        args.pto_layers,
        args.sto_layers,
    )
    logging.info(
        "PTO/STO cell: %dx%dx%d on one device",
        args.lateral_size,
        args.lateral_size,
        args.pto_layers + args.sto_layers,
    )

    dt = args.dt_ps * Constants.ps
    dipole.set_integrator(
        "isothermal", dt=dt, temp=args.temperature, tau=0.1
    )
    global_strain.set_integrator(
        "isothermal",
        dt=dt,
        temp=args.temperature,
        tau=1.0,
        freeze_x=True,
        freeze_y=True,
        freeze_z=False,
    )
    simulation = SimulationNPTLangevin(system, pressure=0.0, seed=args.seed)
    simulation.init_velocity(mode="gaussian", temp=args.temperature)

    logging.info("Relaxing for %d steps at zero electric field", relax_steps)
    simulation.add_thermo_reporter(
        file=str(args.output_dir / "relax.log"),
        log_interval=args.log_interval,
        global_strain=True,
        volume=True,
        potential_energy=True,
        kinetic_energy=True,
        temperature=True,
    )
    simulation.add_field_reporter(
        file_prefix=str(args.output_dir / "relax_field"),
        field_ID="dipole",
        log_interval=args.dump_interval,
        field_average=False,
        dump_field=True,
    )
    simulation.run(relax_steps)

    applied_field = args.electric_field * Constants.V_Angstrom
    electric_field.set_parameters(jnp.asarray((0.0, 0.0, applied_field)))
    simulation.remove_all_reporters()
    logging.info(
        "Driving for %d steps at E_z=%g V/Angstrom",
        drive_steps,
        args.electric_field,
    )
    simulation.add_thermo_reporter(
        file=str(args.output_dir / "drive.log"),
        log_interval=args.log_interval,
        global_strain=True,
        volume=True,
        potential_energy=True,
        kinetic_energy=True,
        temperature=True,
    )
    simulation.add_field_reporter(
        file_prefix=str(args.output_dir / "drive_field"),
        field_ID="dipole",
        log_interval=args.dump_interval,
        field_average=False,
        dump_field=True,
    )
    simulation.run(drive_steps)
    print(f"Output written to {args.output_dir}")


if __name__ == "__main__":
    main()
