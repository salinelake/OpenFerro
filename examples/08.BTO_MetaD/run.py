"""Explore BaTiO3 polarization phases in one cubic-symmetry chamber."""

import argparse
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp

import openferro as of
from openferro.simulation import (
    MDMinimize,
    MetadynamicsNPT,
    SimulationNPTLangevin,
)
from openferro.units import Constants

from helper import build_system, polarization_difference_xy, polarization_difference_yz, polarization_z


E_PER_ANGSTROM2_TO_C_PER_M2 = 16.02176634

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, default="BaTiO3.json")
    parser.add_argument("--settings", type=Path, default="settings.json")
    parser.add_argument("--output-dir", type=Path, default="output")
    parser.add_argument("--temperature", type=float, default=None, help="override the temperature in the settings file")
    parser.add_argument("--local-strain-field", action="store_true", help="Use local strain field.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    model = json.loads(args.model_file.read_text(encoding="utf-8"))
    size = settings["lattice_size"]
    if args.temperature is not None:
        settings["temperature_K"] = args.temperature
    temperature = settings["temperature_K"]
    pressure = settings["pressure_bar"]
    dynamics = settings["dynamics"]
    metadynamics = settings["metadynamics"]
    seed = settings["seed"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=args.output_dir / "simulation.log",
        force=True,
    )
    logging.info("JAX backend: %s", jax.default_backend())
    logging.info("BaTiO3 size: %d^3; temperature: %.3f K", size, temperature)
    ## copy the model and settings files to the output directory
    json.dump(model, (args.output_dir / "model.json").open(mode="w", encoding="utf-8"), indent=2)
    json.dump(settings, (args.output_dir / "settings.json").open(mode="w", encoding="utf-8"), indent=2)

    ## Step 0: Build the system
    if args.local_strain_field:
        system, dipole, local_strain, global_strain = build_system(
            model, settings, size, local_strain_field=True
        )
    else:
        system, dipole, global_strain = build_system(
            model, settings, size, local_strain_field=False
        )

    ## Step 1: Metadynamics
    dt = metadynamics["dt_ps"] * Constants.ps
    dipole_tau = metadynamics["dipole_tau_ps"] * Constants.ps
    strain_tau = metadynamics["strain_tau_ps"] * Constants.ps
    dipole.set_integrator("isothermal", dt=dt, temp=temperature, tau=dipole_tau)
    if args.local_strain_field:
        local_strain.set_integrator("isothermal", dt=dt, temp=temperature, tau=strain_tau)
    global_strain.set_integrator("isothermal", dt=dt, temp=temperature, tau=strain_tau)
    log_interval = metadynamics["log_interval"]
    born_charge = model["parameters"]["born"]["Z_star"]   # e/A^2
    reference_volume = float(jax.device_get(system.lattice.ref_volume))
    cv_parameters = [
        born_charge * E_PER_ANGSTROM2_TO_C_PER_M2,
        reference_volume,
    ]
    ## Define the collective variables
    CV1 = {
                "id": "q1_Px_minus_Py",
                "field_ids": ("dipole", "gstrain"),
                "engine": polarization_difference_xy,
                "parameters": cv_parameters,
            }
    CV2 = {
                "id": "q2_Py_minus_Pz",
                "field_ids": ("dipole", "gstrain"),
                "engine": polarization_difference_yz,
                "parameters": cv_parameters,
            }
    CV3 = {
                "id": "q3_Pz",
                "field_ids": ("dipole", "gstrain"),
                "engine": polarization_z,
                "parameters": cv_parameters,
            }

    ## Define the metadynamics simulation
    wall_epsilon = metadynamics["epsilon_C_per_m2"]
    simulation = MetadynamicsNPT(
        system, [CV1, CV2, CV3],
        pace=metadynamics["pace"],
        sigma=metadynamics["sigma_C_per_m2"],
        height=metadynamics["height_eV"],
        grid_min=metadynamics["grid_min_C_per_m2"],
        grid_max=metadynamics["grid_max_C_per_m2"],
        pressure=pressure,
        grid_bin=metadynamics["grid_bin"],
        lower_walls={
            "at": [-wall_epsilon, -wall_epsilon, -wall_epsilon],
            "kappa": metadynamics["lower_wall_kappa_eV"],
            "exp": [2.0, 2.0, 2.0],
            "eps": metadynamics["lower_wall_scale_C_per_m2"],
        },
        upper_walls={
            "at": metadynamics["upper_wall_at_C_per_m2"],
            "kappa": metadynamics["upper_wall_kappa_eV"],
            "exp": [2.0, 2.0, 2.0],
            "eps": metadynamics["upper_wall_scale_C_per_m2"],
        },
        hills_file=args.output_dir / "HILLS",
        seed=seed + 1,
    )
    simulation.init_velocity(mode="gaussian", temp=temperature)

    ## Check if the initial collective variables are within the grid
    initial_q = jax.device_get(simulation.calc_collective_variables())
    grid_min = jax.device_get(simulation.grid_min)
    grid_max = jax.device_get(simulation.grid_max)
    if jnp.any(initial_q <= grid_min) or jnp.any(initial_q >= grid_max):
        raise ValueError(
            f"Initial q={initial_q.tolist()} lies outside the metadynamics grid."
        )

    simulation.add_thermo_reporter(
        file=str(args.output_dir / "metadynamics_thermo.log"),
        log_interval=log_interval,
        global_strain=True,
        volume=True,
        potential_energy=True,
        kinetic_energy=True,
        temperature=True,
    )
    simulation.add_field_reporter(
        file_prefix=str(args.output_dir / "metadynamics_dipole"),
        field_ID="dipole",
        log_interval=log_interval,
        field_average=True,
        dump_field=False,
    )
    simulation.run(metadynamics["steps"])

if __name__ == "__main__":
    main()
