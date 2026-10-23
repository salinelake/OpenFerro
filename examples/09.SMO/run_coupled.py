"""Run the coupled spin--local-mode SrMnO3 heating calculation.

Raw averages and snapshots of the evolving fields are written for later
postprocessing; no order parameters are evaluated during the simulation.
"""

import argparse
from pathlib import Path

import jax.numpy as jnp

from openferro.simulation import SimulationNVTLangevin

from model import build_smo_coupled_system, load_smo_parameters


DEFAULT_TEMPERATURES = (
    20.0,
    40.0,
    60.0,
    80.0,
    90.0,
    95.0,
    100.0,
    105.0,
    110.0,
    120.0,
    140.0,
    160.0,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("output_coupled"))
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES
    )
    parser.add_argument("--equilibration-steps", type=int, default=5_000)
    parser.add_argument("--sampling-steps", type=int, default=10_000)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--polar-dt-ps", type=float, default=2.0e-4)
    parser.add_argument("--local-strain-dt-ps", type=float, default=2.0e-4)
    parser.add_argument("--langevin-tau-ps", type=float, default=0.1)
    parser.add_argument("--spin-dt-ps", type=float, default=2.0e-4)
    parser.add_argument("--spin-alpha", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    parameters = load_smo_parameters()
    system, fields = build_smo_coupled_system(
        parameters,
        size=args.size,
        enable_jit=True,
        include_ewald=True, 
    )

    u_eq = parameters["effective_hamiltonian_u_eq_angstrom"]
    local_mode = jnp.asarray((u_eq / jnp.sqrt(2.0), u_eq / jnp.sqrt(2.0), 0.0))
    fields["local_mode"].set_values(
        jnp.broadcast_to(local_mode, (args.size, args.size, args.size, 3))
    )
    fields["local_strain"].set_values(
        jnp.zeros((args.size, args.size, args.size, 3))
    )
    fields["spin"].set_values(
        jnp.broadcast_to(
            jnp.asarray((0.0, 0.0, 1.0)),
            (args.size, args.size, args.size, 3),
        )
    )

    output_dir = args.output_dir / f"L{args.size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation = SimulationNVTLangevin(system, seed=args.seed)
    simulation.init_velocity(mode="gaussian", temp=float(args.temperatures[0]))

    for temperature in args.temperatures:
        temperature = float(temperature)
        fields["local_mode"].set_integrator(
            "isothermal",
            dt=args.polar_dt_ps,
            temp=temperature,
            tau=args.langevin_tau_ps,
        )
        fields["local_strain"].set_integrator(
            "isothermal",
            dt=args.local_strain_dt_ps,
            temp=temperature,
            tau=args.langevin_tau_ps,
        )
        fields["spin"].set_integrator(
            "isothermal",
            dt=args.spin_dt_ps,
            temp=temperature,
            alpha=args.spin_alpha,
        )

        simulation.remove_all_reporters()
        simulation.run(args.equilibration_steps)

        tag = f"{temperature:g}K"
        simulation.add_thermo_reporter(
            file=str(output_dir / f"thermo_{tag}.log"),
            log_interval=args.report_interval,
            global_strain=True,
        )
        for field_id in ("spin", "local_mode"):
            simulation.add_field_reporter(
                file_prefix=str(output_dir / f"{field_id}_{tag}"),
                field_ID=field_id,
                log_interval=args.report_interval,
                field_average=True,
                dump_field=True,
            )
        simulation.run(args.sampling_steps)


if __name__ == "__main__":
    main()
