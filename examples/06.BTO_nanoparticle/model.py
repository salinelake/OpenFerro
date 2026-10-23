import logging
import numpy as np
import jax
import jax.numpy as jnp

import openferro as of
from openferro.parallelism import DeviceMesh
from openferro.units import Constants

from custom_engines import (
    surface_term_energy,
    make_surface_k2_hardening_potential,
)
from geometry import build_spherical_geometry


def build_system(material, args):
    """Build and initialize the padded nanoparticle in its dielectric medium."""
    lattice_constant = float(material["lattice"]["a1"])
    geometry = build_spherical_geometry(
        args.size,
        args.radius,
        lattice_constant,
        center=args.center,
        required_inactive_layers=1,
    )
    lattice = of.SimpleCubic3D(
        args.size,
        args.size,
        args.size,
        jnp.asarray((lattice_constant, 0.0, 0.0)),
        jnp.asarray((0.0, lattice_constant, 0.0)),
        jnp.asarray((0.0, 0.0, lattice_constant)),
    )
    system = of.System(lattice)
    mass = args.soft_mode_mass * Constants.amu

    dipole = system.add_field(
        ID="dipole",
        ftype="MaskedRn",
        dim=3,
        value=0.0,
        mass=mass,
        active_mask=geometry.cell_mask,
    )
    values = np.zeros((args.size, args.size, args.size, 3))
    if args.initial_dipole == "random":
        rng = np.random.default_rng(args.seed)
        values[geometry.cell_mask] = rng.normal(
            scale=args.perturbation,
            size=(geometry.n_particle, 3),
        )
        gstrain_value = jnp.asarray((0.01, 0.01, 0.01, 0.0, 0.0, 0.0))
    else:
        initial_values = np.zeros((geometry.n_particle, 3))
        initial_values[:, 2] = 0.1
        values[geometry.cell_mask] = initial_values
        gstrain_value = jnp.asarray((0.01, 0.01, 0.02, 0.0, 0.0, 0.0))
    dipole.set_values(values)

    gstrain = system.add_global_strain(
        value=gstrain_value,
        mass=mass * geometry.n_particle,
        reference_volume=geometry.reference_volume,
    )

    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(gpu_devices) > 1:
        mesh = DeviceMesh(devices=gpu_devices)
        mesh_rows, mesh_columns = mesh.mesh.devices.shape
        if args.size % mesh_rows == 0 and args.size % mesh_columns == 0:
            system.move_fields_to_multi_devs(mesh)
            args.distributed = True
            logging.info("Using %d x %d device mesh", mesh_rows, mesh_columns)
        else:
            logging.warning(
                "Keeping fields on one device because box size %d is not "
                "divisible by the %dx%d device mesh. Use a compatible box "
                "size and an explicit --center for even boxes.",
                args.size,
                mesh_rows,
                mesh_columns,
            )

    parameters = material["parameters"]
    onsite = parameters["onsite"]
    short_range = parameters["short_range"]
    elastic = parameters["elastic"]
    coupling = parameters["elastic_dipole"]
    born = parameters["born"]

    system.add_dipole_onsite_interaction(
        "self_onsite",
        "dipole",
        onsite["k2"],
        onsite["alpha"],
        onsite["gamma"],
    )
    system.add_dipole_interaction_1st_shell(
        "short_range_1",
        "dipole",
        short_range["j1"],
        short_range["j2"],
    )
    system.add_dipole_interaction_2nd_shell(
        "short_range_2",
        "dipole",
        short_range["j3"],
        short_range["j4"],
        short_range["j5"],
    )
    system.add_dipole_interaction_3rd_shell(
        "short_range_3",
        "dipole",
        short_range["j6"],
        short_range["j7"],
    )
    dipole_prefactor = born["Z_star"] ** 2 / born["epsilon_inf"]
    system.add_dipole_dipole_interaction(
        "dipole_ewald",
        "dipole",
        prefactor=dipole_prefactor,
    )
    system.add_homo_elastic_interaction(
        "homo_elastic",
        "gstrain",
        elastic["B11"],
        elastic["B12"],
        elastic["B44"],
        n_cells=geometry.n_particle,
    )
    system.add_homo_strain_dipole_interaction(
        "homo_strain_dipole",
        "gstrain",
        "dipole",
        coupling["B1xx"],
        coupling["B1yy"],
        coupling["B4yz"],
    )

    # The field stores u in Angstrom, so the electric enthalpy is -Z_star * sum_i(u_i . E). E is the directly conjugate field here.
    electric_field = jnp.asarray(args.electric_field) * Constants.V_Angstrom
    if np.any(np.asarray(args.electric_field) != 0.0):
        system.add_dipole_efield_interaction(
            "dipole_efield",
            "dipole",
            E=born["Z_star"] * electric_field,
        )
    
    if args.surface_mode == "spherical_vacuum":
        system.add_self_interaction(
            "spherical_surface",
            "dipole",
            surface_term_energy,
            parameters=jnp.asarray((dipole_prefactor, lattice.ref_volume)),
        )

    if args.surface_k2_shift > 0.0:
        system.add_self_interaction(
            "surface_k2_hardening",
            "dipole",
            make_surface_k2_hardening_potential(
                geometry,
                args.surface_shell_thickness_cells,
                args.surface_k2_shift,
            ),
        )

    return system, dipole, gstrain, geometry
