"""
Lightweight baseline runner for dipole-dipole Ewald memory and timing.

Run from the repository root, for example:

    JAX_PLATFORMS=cpu python tests/profiling/ewald_baseline.py --sizes 3x2x2
"""

import argparse
import json

import jax.numpy as jnp

from openferro.engine.ewald import benchmark_dipole_dipole_ewald
from openferro.lattice import BravaisLattice3D


def _parse_size(size):
    parts = size.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("sizes must be formatted as L1xL2xL3")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must contain integers") from exc


def main():
    parser = argparse.ArgumentParser(
        description="Measure current Ewald energy and autodiff-force baseline."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=_parse_size,
        default=[(3, 2, 2)],
        help="One or more lattice sizes formatted as L1xL2xL3.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of post-compile timing repeats.",
    )
    parser.add_argument(
        "--include-force",
        action="store_true",
        help="Also time the current autodiff force path.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Field and coefficient dtype.",
    )
    args = parser.parse_args()

    dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    results = []
    for size in args.sizes:
        latt = BravaisLattice3D(*size)
        results.append(
            benchmark_dipole_dipole_ewald(
                latt,
                repeat=args.repeat,
                dtype=dtype,
                include_force=args.include_force,
            )
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
