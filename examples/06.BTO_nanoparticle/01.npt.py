"""NPT simulation of a BaTiO3 nanoparticle in a dielectric continuum."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from openferro.simulation import SimulationNPTLangevin
from model import build_system

"""Parse command-line arguments for the continuum-medium simulation."""
parser = argparse.ArgumentParser(description=__doc__)
## system parameters
parser.add_argument("--material-config", type=Path, default=Path("BaTiO3.json"))
parser.add_argument("--size", type=int, default=41, help="Padded cubic box size.")
parser.add_argument("--radius", type=float, default=15.5, help="Particle and dielectric-boundary radius in unit cells.")
parser.add_argument("--center", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Particle center in lattice-index coordinates.")
parser.add_argument("--temperature", type=float, default=250.0, help="Temperature in K.")
parser.add_argument("--pressure", type=float, default=-4.8e4, help="External pressure in bar. The default negative value compensates the intrinsic LDA error.")
parser.add_argument("--soft-mode-mass", type=float, default=200.0, help="Mass in amu.")
parser.add_argument("--initial-dipole", choices=("random", "z"), default="random")
## Interaction switches
parser.add_argument("--electric-field", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Field conjugate to the BaTiO3 soft-mode dipole, in V/Angstrom= 10^6 kV/cm")
parser.add_argument("--surface-mode", choices=("conducting", "spherical_vacuum"), default="spherical_vacuum")
parser.add_argument("--surface-shell-thickness-cells", type=float, default=8.0, help="Thickness of the hardened radial surface shell in reference-lattice cell units. Use 0 with --surface-k2-shift 0 to disable.")
parser.add_argument("--surface-k2-shift", type=float, default=5.502, help="Positive quadratic onsite shift in the surface shell, in eV/Angstrom^2. Use 0 with --surface-shell-thickness-cells 0 to disable.")
## simulation parameters
parser.add_argument("--dt", type=float, default=0.002, help="Langevin timestep in ps.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--perturbation", type=float, default=0.005, help="Standard deviation of initial local modes in Angstrom.")
parser.add_argument("--cooldown_init", type=float, default=320.0, help="Temperature in K.")
parser.add_argument("--cooldown_stages", type=int, default=8, help="Number of cooldown stages.")
parser.add_argument("--cooldown-steps", type=int, default=12500)
parser.add_argument("--cooldown-log-interval", type=int, default=100)
parser.add_argument("--sampling-steps", type=int, default=100000)
parser.add_argument("--sample-log-interval", type=int, default=100)
parser.add_argument("--sample-dump-interval", type=int, default=10000, help="Disable field dumping with a value of 0.")
parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory.")
args = parser.parse_args()
args.distributed = False

"""Create output directory and log file."""
args.output_dir.mkdir(parents=True, exist_ok=True)
traj_dir = args.output_dir / "traj"
traj_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, filename=args.output_dir / "simulation.log", force=True)
with args.material_config.open(encoding="utf-8") as stream:
    material = json.load(stream)

"""Run NPT equilibration and sampling."""
### Build system
system, dipole, gstrain, geometry = build_system(material, args)
args.n_particle = geometry.n_particle
with (args.output_dir / "config.json").open("w", encoding="utf-8") as stream:
    json.dump(vars(args), stream, indent=2, default=str)
    stream.write("\n")

logging.info("Active cells: %d", geometry.n_particle)
logging.info("Inactive face padding: %s", geometry.face_padding_by_name)

### Equilibration
temp_hierarchy = np.linspace(args.cooldown_init, args.temperature, args.cooldown_stages)
for temperature in temp_hierarchy:
    logging.info("T=%s K, NPT equilibration", temperature)
    dipole.set_integrator("isothermal", args.dt, temp=temperature, tau=0.1)
    gstrain.set_integrator("isothermal", args.dt, temp=temperature, tau=1.0)
    simulation = SimulationNPTLangevin(system, pressure=args.pressure,seed=args.seed)
    if temperature == args.cooldown_init:
        simulation.init_velocity(mode="gaussian", temp=temperature)
    simulation.add_thermo_reporter(
        file=str(args.output_dir / f"cooldown_T{temperature}K_thermo.log"),
        log_interval=args.cooldown_log_interval,
        global_strain=True,
        excess_stress=True,
        volume=True,
        potential_energy=True,
        kinetic_energy=True,
        temperature=True,
    )
    simulation.add_field_reporter(
        file_prefix=str(args.output_dir / f"cooldown_T{temperature}K_dipole"),
        field_ID="dipole",
        log_interval=args.cooldown_log_interval,
        field_average=True,
        dump_field=False,
    )
    simulation.run(args.cooldown_steps)
    simulation.remove_all_reporters()

### Sampling
logging.info("T=%s K, NPT sampling", args.temperature)

simulation.add_thermo_reporter(
    file=str(args.output_dir / "sample_thermo.log"),
    log_interval=args.sample_log_interval,
    global_strain=True,
    excess_stress=True,
    volume=True,
    potential_energy=True,
    kinetic_energy=True,
    temperature=True,
)
simulation.add_field_reporter(
    file_prefix=str(args.output_dir / "sample_dipole"),
    field_ID="dipole",
    log_interval=args.sample_log_interval,
    field_average=True,
    dump_field=False,
)
if args.sample_dump_interval > 0:
    simulation.add_field_reporter(
        file_prefix=str(traj_dir / "sample_dipole"),
        field_ID="dipole",
        log_interval=args.sample_dump_interval,
        field_average=False,
        dump_field=True,
    )
simulation.run(args.sampling_steps)
