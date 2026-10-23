import openferro as of
import jax.numpy as jnp
from openferro.units import Constants
from openferro.engine.elastic import deformed_volume


def instantaneous_polarization(dipole, global_strain, parameters):
    """Return the instantaneous Cartesian polarization in C/m^2."""
    dipole_moment = parameters[0] * jnp.sum(dipole, axis=(0, 1, 2))
    volume = deformed_volume(global_strain, parameters[1])
    return dipole_moment / volume


def polarization_difference_xy(dipole, global_strain, parameters):
    """Return q1 = Px - Py in C/m^2."""
    polarization = instantaneous_polarization(dipole, global_strain, parameters)
    return polarization[0] - polarization[1]


def polarization_difference_yz(dipole, global_strain, parameters):
    """Return q2 = Py - Pz in C/m^2."""
    polarization = instantaneous_polarization(dipole, global_strain, parameters)
    return polarization[1] - polarization[2]


def polarization_z(dipole, global_strain, parameters):
    """Return q3 = Pz in C/m^2."""
    polarization = instantaneous_polarization(dipole, global_strain, parameters)
    return polarization[2]


def build_system(model, settings, size, local_strain_field=False):
    """Build the BaTiO3 effective-Hamiltonian system."""
    lattice_parameters = model["lattice"]
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
    dipole = system.add_field(
        "dipole",
        ftype="Rn",
        dim=3,
        value=jnp.asarray(settings["initial_local_mode_angstrom"]),
        mass=mass,
    )
    if local_strain_field:
        local_strain = system.add_field(
            "lstrain", ftype="LocalStrain3D", value=0.0, mass=mass
        )
    global_strain = system.add_global_strain(
        value=jnp.asarray(settings["initial_global_strain"]),
        mass=mass * size**3,
    )

    parameters = model["parameters"]
    onsite = parameters["onsite"]
    short = parameters["short_range"]
    elastic = parameters["elastic"]
    elastic_dipole = parameters["elastic_dipole"]
    born = parameters["born"]
    system.add_dipole_onsite_interaction(
        "self_onsite",
        field_ID="dipole",
        K2=onsite["k2"],
        alpha=onsite["alpha"],
        gamma=onsite["gamma"],
    )
    system.add_dipole_interaction_1st_shell(
        "short_range_1", field_ID="dipole", j1=short["j1"], j2=short["j2"]
    )
    system.add_dipole_interaction_2nd_shell(
        "short_range_2",
        field_ID="dipole",
        j3=short["j3"],
        j4=short["j4"],
        j5=short["j5"],
    )
    system.add_dipole_interaction_3rd_shell(
        "short_range_3", field_ID="dipole", j6=short["j6"], j7=short["j7"]
    )
    system.add_dipole_dipole_interaction(
        "dipole_ewald",
        field_ID="dipole",
        prefactor=born["Z_star"] ** 2 / born["epsilon_inf"],
    )
    system.add_homo_elastic_interaction(
        "homo_elastic", field_ID="gstrain", **elastic
    )
    system.add_homo_strain_dipole_interaction(
        "homo_strain_dipole",
        field_1_ID="gstrain",
        field_2_ID="dipole",
        **elastic_dipole,
    )
    if local_strain_field:
        system.add_inhomo_elastic_interaction(
            "inhomo_elastic", field_ID="lstrain", **elastic
        )
        system.add_inhomo_strain_dipole_interaction(
            "inhomo_strain_dipole",
            field_1_ID="lstrain",
            field_2_ID="dipole",
            **elastic_dipole,
        )
        return system, dipole, local_strain, global_strain
    else:
        return system, dipole, global_strain
