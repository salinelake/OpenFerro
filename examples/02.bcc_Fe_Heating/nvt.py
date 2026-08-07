"""NVT heating example for the bcc Fe classical Heisenberg model."""

import argparse
import json
import logging
from pathlib import Path

import jax.numpy as jnp

import openferro as of
from openferro.simulation import SimulationNVTLangevin
from openferro.units import Constants


EXAMPLE_DIR = Path(__file__).resolve().parent


def _load_model_record(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _exchange_couplings(config):
    conventions = config["conventions"]
    if conventions["engine_pair_counting"] != "unique":
        raise ValueError("The bcc Fe example requires unique engine bond counting.")
    source_pair_counting = conventions["source_pair_counting"]
    if source_pair_counting not in {"ordered", "unique"}:
        raise ValueError("Unsupported source pair-counting convention.")
    unit_to_ev = {"mRy": Constants.mRy, "J": Constants.Joule}[
        config["units"]["exchange_source"]
    ]
    parameters = config["parameters"]
    pair_factor = 2.0 if source_pair_counting == "ordered" else 1.0
    return (
        jnp.asarray(parameters["exchange_source_values"])
        * unit_to_ev
        * pair_factor
        / parameters["moment_mu_B"] ** 2
    )


def build_system(config, size):
    """Build the bcc Fe spin system from a documented model record."""
    if config["model"]["kind"] != "classical_heisenberg":
        raise ValueError("The bcc Fe example requires a Heisenberg model record.")
    if config["lattice"]["type"] != "body_centered_cubic":
        raise ValueError(
            "The bcc Fe example requires a body-centered-cubic lattice."
        )
    lattice_constant = config["lattice"]["lattice_constant_angstrom"]
    lattice = of.BodyCenteredCubic3D(
        size,
        size,
        size,
        0.5 * lattice_constant * jnp.asarray((-1.0, 1.0, 1.0)),
        0.5 * lattice_constant * jnp.asarray((1.0, -1.0, 1.0)),
        0.5 * lattice_constant * jnp.asarray((1.0, 1.0, -1.0)),
    )
    system = of.System(lattice)
    spin = system.add_field(
        ID="spin", ftype="SO3", value=jnp.asarray((0.0, 0.0, 1.0))
    )
    spin.set_magnitude(config["parameters"]["moment_mu_B"])

    interaction_methods = (
        system.add_isotropic_exchange_interaction_1st_shell,
        system.add_isotropic_exchange_interaction_2nd_shell,
        system.add_isotropic_exchange_interaction_3rd_shell,
        system.add_isotropic_exchange_interaction_4th_shell,
    )
    couplings = _exchange_couplings(config)
    if len(couplings) != config["lattice"]["shells"] or len(couplings) > 4:
        raise ValueError("The bcc Fe record has an incompatible exchange shell count.")
    for shell, (method, coupling) in enumerate(
        zip(interaction_methods, couplings), start=1
    ):
        method(
            ID=f"exchange_{shell}_shell",
            field_ID="spin",
            coupling=coupling,
        )
    return system, spin


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXAMPLE_DIR / "bcc_Fe.json",
        help="Documented magnetic JSON model record.",
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
    system, spin = build_system(config, size)

    dt = 0.0002
    temperatures = [10] if args.tiny else [
        10, 200, 400, 600, 700, 800, 900, 1000, 1200
        ]
    equilibration_steps = 1 if args.tiny else 5000
    sampling_steps = 1 if args.tiny else 10000
    log_interval = 1 if args.tiny else 100

    simulation = SimulationNVTLangevin(system, seed=args.seed)
    for temperature in temperatures:
        spin.set_integrator("isothermal", dt=dt, temp=temperature, alpha=0.5)

        logging.info("T=%s K, NVT equilibration", temperature)
        simulation.remove_all_reporters()
        simulation.run(equilibration_steps)

        simulation.add_thermo_reporter(
            file=str(args.output_dir / f"thermo_{temperature}K.log"),
            log_interval=log_interval,
            global_strain=False,
            volume=True,
            potential_energy=True,
            kinetic_energy=True,
            temperature=True,
        )
        simulation.add_field_reporter(
            file_prefix=str(args.output_dir / f"spin_{temperature}K"),
            field_ID="spin",
            log_interval=log_interval,
            field_average=True,
            dump_field=False,
        )
        logging.info("T=%s K, NVT sampling", temperature)
        simulation.run(sampling_steps)


if __name__ == "__main__":
    main()
