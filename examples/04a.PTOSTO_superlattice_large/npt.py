"""Cool a PTO/STO superlattice from 300 K to 10 K at zero electric field."""

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
from openferro.parallelism import DeviceMesh
from openferro.units import Constants


DEFAULT_PTO_CONFIG = Path("PbTiO3_SCAN.json")
DEFAULT_STO_CONFIG = Path("SrTiO3_WC.json")


def build_system(
    pto_config,
    sto_config,
    lateral_size,
    pto_layers,
    sto_layers,
    device_mesh=None,
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
    if device_mesh is not None:
        system.move_fields_to_multi_devs(device_mesh)

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
        "homo_elastic", field_ID="gstrain", **pto["elastic"], n_cells=lateral_size**2 * pto_layers
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
    return system, dipole, global_strain


def output_dir_for_cell(lateral_size, pto_layers, sto_layers):
    """Return an output directory named by the simulation cell."""
    return Path(f"output_L{lateral_size}_PTO{pto_layers}_STO{sto_layers}")


def cooling_temperatures(t_start, t_end, t_step):
    """Return the plateau temperatures from t_start down to t_end, inclusive."""
    if t_step <= 0:
        raise ValueError("--t-step must be positive.")
    if t_start <= t_end:
        raise ValueError("--t-start must be above --t-end.")
    temperatures = []
    temperature = t_start
    while temperature > t_end:
        temperatures.append(temperature)
        temperature -= t_step
    temperatures.append(t_end)
    return temperatures


def format_temperature(temperature):
    rounded = round(temperature)
    if abs(temperature - rounded) < 1e-6:
        return str(int(rounded))
    return f"{temperature:.1f}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pto-config", type=Path, default=DEFAULT_PTO_CONFIG)
    parser.add_argument("--sto-config", type=Path, default=DEFAULT_STO_CONFIG)
    parser.add_argument("--lateral-size", type=int, default=1200)
    parser.add_argument("--pto-layers", type=int, default=12)
    parser.add_argument("--sto-layers", type=int, default=12)
    parser.add_argument("--t-start", type=float, default=300.0)
    parser.add_argument("--t-end", type=float, default=10.0)
    parser.add_argument("--t-step", type=float, default=10.0)
    parser.add_argument("--dt-ps", type=float, default=0.002)
    parser.add_argument("--hold-time-ps", type=float, default=50.0)
    parser.add_argument("--sample-time-ps", type=float, default=10.0)
    parser.add_argument("--sample-interval-ps", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to output_L{lateral}_PTO{pto}_STO{sto}.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir is None:
        args.output_dir = output_dir_for_cell(
            args.lateral_size, args.pto_layers, args.sto_layers
        )
    temperatures = cooling_temperatures(args.t_start, args.t_end, args.t_step)
    hold_steps = int(round(args.hold_time_ps / args.dt_ps))
    sample_steps = int(round(args.sample_time_ps / args.dt_ps))
    sample_interval = int(round(args.sample_interval_ps / args.dt_ps))
    if hold_steps < 1:
        raise ValueError("--hold-time-ps must cover at least one time step.")
    if sample_interval < 1 or sample_steps < sample_interval:
        raise ValueError("The final sampling window must cover at least one dump interval.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=args.output_dir / "simulation.log",
        force=True,
    )

    pto_config = json.loads(args.pto_config.read_text(encoding="utf-8"))
    sto_config = json.loads(args.sto_config.read_text(encoding="utf-8"))

    system, dipole, global_strain = build_system(
        pto_config,
        sto_config,
        args.lateral_size,
        args.pto_layers,
        args.sto_layers,
        device_mesh=DeviceMesh(),
    )
    logging.info(
        "PTO/STO cell: %dx%dx%d distributed across visible devices",
        args.lateral_size,
        args.lateral_size,
        args.pto_layers + args.sto_layers,
    )
    logging.info(
        "Cooling %s K -> %s K by %s K, %s ps (%d steps) at each plateau, E = 0",
        format_temperature(temperatures[0]),
        format_temperature(temperatures[-1]),
        format_temperature(args.t_step),
        args.hold_time_ps,
        hold_steps,
    )

    dt = args.dt_ps * Constants.ps
    simulation = SimulationNPTLangevin(system, pressure=0.0, seed=args.seed)
    simulation.init_velocity(mode="gaussian", temp=temperatures[0])
    for temperature in temperatures:
        label = format_temperature(temperature)
        dipole.set_integrator(
            "isothermal", dt=dt, temp=temperature, tau=0.1
        )
        global_strain.set_integrator(
            "isothermal",
            dt=dt,
            temp=temperature,
            tau=1.0,
            freeze_x=True,
            freeze_y=True,
            freeze_z=False,
        )
        logging.info("T=%s K, holding for %d steps at E = 0", label, hold_steps)
        simulation.remove_all_reporters()
        simulation.add_thermo_reporter(
            file=str(args.output_dir / f"thermo_{label}K.log"),
            log_interval=args.log_interval,
            global_strain=True,
            volume=True,
            potential_energy=True,
            kinetic_energy=True,
            temperature=True,
        )
        simulation.run(hold_steps)
        # One more step records the configuration at the end of this plateau.
        simulation.remove_all_reporters()
        simulation.add_field_reporter(
            file_prefix=str(args.output_dir / f"field_{label}K"),
            field_ID="dipole",
            log_interval=1,
            field_average=False,
            dump_field=True,
        )
        simulation.run(1)
    final_label = format_temperature(temperatures[-1])
    logging.info(
        "T=%s K, sampling %s ps and dumping every %s ps",
        final_label,
        args.sample_time_ps,
        args.sample_interval_ps,
    )
    simulation.remove_all_reporters()
    simulation.add_thermo_reporter(
        file=str(args.output_dir / f"sample_{final_label}K.log"),
        log_interval=sample_interval,
        global_strain=True,
        volume=True,
        potential_energy=True,
        kinetic_energy=True,
        temperature=True,
    )
    simulation.add_field_reporter(
        file_prefix=str(args.output_dir / f"sample_{final_label}K"),
        field_ID="dipole",
        log_interval=sample_interval,
        field_average=False,
        dump_field=True,
    )
    # The reporter writes when its counter is a multiple of the interval,
    # including the first and last step, so this spans 0 to sample-time.
    simulation.run(sample_steps + 1)
    print(f"Output written to {args.output_dir}")


if __name__ == "__main__":
    main()
