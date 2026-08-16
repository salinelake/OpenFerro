"""Run fixed-height metadynamics for an analytically solvable toy lattice."""

import argparse
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp

import openferro as of
from openferro.simulation import MetadynamicsNVT


DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def toy_energy(dipole, parameters):
    """Four-well collective energy plus decoupled harmonic fluctuations."""
    coefficient, minimum, transverse = parameters
    sx = jnp.sum(dipole[..., 0])
    sy = jnp.sum(dipole[..., 1])
    mean_x = jnp.mean(dipole[..., 0])
    mean_y = jnp.mean(dipole[..., 1])
    collective = coefficient * (
        (sx**2 - minimum**2) ** 2 + (sy**2 - minimum**2) ** 2
    )
    orthogonal = 0.5 * transverse * jnp.sum(
        (dipole[..., 0] - mean_x) ** 2
        + (dipole[..., 1] - mean_y) ** 2
        + dipole[..., 2] ** 2
    )
    return collective + orthogonal


def total_dipole_x(dipole, parameters):
    """Return the total x dipole without adding a Hamiltonian term."""
    del parameters
    return jnp.sum(dipole[..., 0])


def total_dipole_y(dipole, parameters):
    """Return the total y dipole without adding a Hamiltonian term."""
    del parameters
    return jnp.sum(dipole[..., 1])


def load_config(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def build_simulation(config, output_dir, seed):
    """Build the toy system and its metadynamics simulation."""
    lattice_config = config["lattice"]
    size = lattice_config["size"]
    lattice_constant = lattice_config["lattice_constant_angstrom"]
    lattice = of.SimpleCubic3D(
        *size,
        lattice_constant * jnp.asarray((1.0, 0.0, 0.0)),
        lattice_constant * jnp.asarray((0.0, 1.0, 0.0)),
        lattice_constant * jnp.asarray((0.0, 0.0, 1.0)),
    )
    system = of.System(lattice)
    model = config["model"]
    nsites = int(jnp.prod(jnp.asarray(size)))
    initial_site_dipole = model["S0_P"] / nsites
    dipole = system.add_field(
        "dipole",
        ftype="R3",
        value=(initial_site_dipole, initial_site_dipole, 0.0),
        mass=config["field"]["mass"],
    )
    system.add_self_interaction(
        "toy_hamiltonian",
        "dipole",
        toy_energy,
        parameters=[
            model["A_eV_per_P4"],
            model["S0_P"],
            model["K_perp_eV_per_P2"],
        ],
    )

    dynamics = config["dynamics"]
    dipole.set_integrator(
        "isothermal",
        dt=dynamics["dt"],
        temp=dynamics["temperature_K"],
        tau=dynamics["tau"],
    )
    metadynamics = config["metadynamics"]
    simulation = MetadynamicsNVT(
        system,
        [
            {
                "id": "total_dipole_x",
                "field_ids": "dipole",
                "engine": total_dipole_x,
            },
            {
                "id": "total_dipole_y",
                "field_ids": "dipole",
                "engine": total_dipole_y,
            },
        ],
        pace=metadynamics["pace"],
        sigma=metadynamics["sigma"],
        height=metadynamics["height_eV"],
        grid_min=metadynamics["grid_min"],
        grid_max=metadynamics["grid_max"],
        grid_bin=metadynamics.get("grid_bin"),
        upper_walls=metadynamics["upper_walls"],
        lower_walls=metadynamics["lower_walls"],
        hills_file=output_dir / "HILLS",
        seed=seed,
    )
    simulation.init_velocity(mode="gaussian", temp=dynamics["temperature_K"])
    return simulation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--steps", type=int, help="Override the configured step count.")
    parser.add_argument("--seed", type=int, help="Override the configured RNG seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the short execution smoke mode; this does not validate FES accuracy.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    seed = config["seed"] if args.seed is None else args.seed
    if args.steps is not None:
        steps = args.steps
    elif args.quick:
        steps = config["dynamics"]["quick_steps"]
    else:
        steps = config["dynamics"]["steps"]
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=args.output_dir / "simulation.log",
        force=True,
    )
    logging.info("JAX backend: %s", jax.default_backend())
    logging.info("Running %d metadynamics steps with seed %d", steps, seed)
    simulation = build_simulation(config, args.output_dir, seed)
    simulation.run(steps)
    final_cv = jax.device_get(simulation.calc_collective_variables()).tolist()
    summary = {
        "steps": steps,
        "simulation_time_ps": steps * config["dynamics"]["dt"],
        "seed": seed,
        "quick": args.quick,
        "hill_count": int(simulation.get_hill_centers().shape[0]),
        "hill_interval_ps": (
            config["metadynamics"]["pace"] * config["dynamics"]["dt"]
        ),
        "final_collective_variables": final_cv,
        "physical_potential_energy_eV": float(
            jax.device_get(simulation.system.calc_total_potential_energy())
        ),
        "bias_energy_eV": float(jax.device_get(simulation.calc_total_bias())),
    }
    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
