"""Example-local SrMnO3 Hamiltonians and OpenFerro system builders.

The ferroelectric and frozen-structure magnetic builders reproduce separate
published limits.  The coupled builder is intentionally labeled as a new,
first-principles-informed model extension.
"""

import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import openferro as of
from openferro.units import Constants
from custom_engine import (
    anisotropic_first_neighbor_spin_energy,
    isotropic_second_neighbor_spin_energy,
    reference_matched_anisotropic_second_neighbor_spin_energy,
    reference_matched_first_neighbor_spin_local_energy,
    smo_onsite_energy,
)

MEV_TO_EV = 1.0e-3
SMO_CONFIG = Path(__file__).with_name("reference_data.json")

def _polyval_ascending_numpy(value, coefficients):
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result

def load_smo_parameters():
    """Build the 5% strain model parameters from the published source data."""
    with SMO_CONFIG.open(encoding="utf-8") as handle:
        source = json.load(handle)

    table = source["table_I_PRM_2018"]
    snapshot = source["five_percent_snapshot_PRM_2018"]
    curve = source["exchange_vs_normalized_u_PRM_2018"]
    cubic_J2 = source["cubic_second_neighbor_PRM_2018"]
    relaxed_exchange = source["relaxed_five_percent_exchange_PRM_2018"]

    s_nodes = np.square(np.asarray(curve["u_over_u_eq"], dtype=float))
    J1_ip_coefficients = np.polynomial.polynomial.polyfit(
        s_nodes, np.asarray(curve["J1_ip_meV"], dtype=float), deg=3
    )
    J1_op_coefficients = np.polynomial.polynomial.polyfit(
        s_nodes, np.asarray(curve["J1_op_meV"], dtype=float), deg=3
    )
    J1_ip_anchored = J1_ip_coefficients.copy()
    J1_op_anchored = J1_op_coefficients.copy()
    relaxed_J1_ip = 0.5 * (
        relaxed_exchange["J1_ip_x_meV"] + relaxed_exchange["J1_ip_y_meV"]
    )
    J1_ip_anchored[1] += relaxed_J1_ip - _polyval_ascending_numpy(
        1.0, J1_ip_coefficients
    )
    J1_op_anchored[1] += relaxed_exchange[
        "J1_op_meV"
    ] - _polyval_ascending_numpy(1.0, J1_op_coefficients)

    onsite = table["onsite"]
    elastic = table["elastic"]
    strain_mode = table["strain_mode"]
    return {
        "lattice_constant_angstrom": table["lattice_constant"]["value"],
        "fixed_strain": np.asarray(
            (
                snapshot["strain"],
                snapshot["strain"],
                snapshot["effective_hamiltonian_eta3"],
                0.0,
                0.0,
                0.0,
            )
        ),
        "onsite_coefficients_eV": np.asarray(
            (
                snapshot["kappa2"],
                onsite["alpha"],
                onsite["gamma"],
                onsite["k1"],
                onsite["k2"],
                onsite["k3"],
                onsite["k4"],
            )
        ),
        "short_range_eV_per_angstrom2": np.asarray(
            tuple(snapshot[f"j{index}"] for index in range(1, 8))
        ),
        "elastic_eV": np.asarray(
            (elastic["B11"], elastic["B12"], elastic["B44"])
        ),
        "strain_mode_eV_per_angstrom2": np.asarray(
            (strain_mode["B1xx"], strain_mode["B1yy"], strain_mode["B4yz"])
        ),
        "mode_effective_charge_e": table["born_charges"]["mode_effective_charge"],
        "epsilon_xx": snapshot["epsilon_xx"],
        "soft_mode_mass_amu": table["effective_mass"]["value"],
        # Numerical mass for the acoustic-displacement field, approximately one
        # SrMnO3 formula mass. It is not a fitted material parameter.
        "local_strain_mass_amu": 190.56,
        "effective_hamiltonian_u_eq_angstrom": snapshot[
            "effective_hamiltonian_u_eq"
        ],
        "exchange_curve_u_eq_angstrom": snapshot["dft_u_eq"],
        "J1_ip_coefficients_meV": J1_ip_coefficients,
        "J1_op_coefficients_meV": J1_op_coefficients,
        "J1_ip_anchored_coefficients_meV": J1_ip_anchored,
        "J1_op_anchored_coefficients_meV": J1_op_anchored,
        "J2_meV": 0.5 * (cubic_J2["J2_ip_meV"] + cubic_J2["J2_op_meV"]),
        # Direction order: [110], [1 -1 0], and the common xz/yz branch.
        "J2_relaxed_fe_meV": np.asarray(
            (
                relaxed_exchange["J2_ip_branch_1_meV"],
                relaxed_exchange["J2_ip_branch_2_meV"],
                relaxed_exchange["J2_op_meV"],
            )
        ),
    }


def evaluate_exchange(parameters, u_over_u_eq, allow_extrapolation=False):
    """Evaluate the frozen homogeneous first-neighbor exchange fit in eV."""
    ratio = float(u_over_u_eq)
    if not np.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        if not allow_extrapolation:
            raise ValueError(
                f"u/u_eq={ratio!r} is outside the fitted [0, 1] interval. "
                "Pass allow_extrapolation=True only for an explicitly labeled sensitivity test."
            )
    J_ip = _polyval_ascending_numpy(
        ratio**2, parameters["J1_ip_coefficients_meV"]
    ) * MEV_TO_EV
    J_op = _polyval_ascending_numpy(
        ratio**2, parameters["J1_op_coefficients_meV"]
    ) * MEV_TO_EV
    return float(J_ip), float(J_op)

def _add_ferroelectric_fields_and_interactions(
    system, parameters, enable_jit, include_ewald
):
    local_mode_mass = parameters["soft_mode_mass_amu"] * Constants.amu
    acoustic_mass = parameters["local_strain_mass_amu"] * Constants.amu
    local_mode = system.add_field(
        ID="local_mode", ftype="R3", value=jnp.zeros(3), mass=local_mode_mass
    )
    local_strain = system.add_field(
        ID="local_strain", ftype="LocalStrain3D", value=jnp.zeros(3), mass=acoustic_mass
    )
    global_strain = system.add_global_strain(
        value=jnp.asarray(parameters["fixed_strain"]),
        mass=acoustic_mass * system.lattice.nsites,
    )

    system.add_self_interaction(
        "onsite",
        field_ID="local_mode",
        energy_engine=smo_onsite_energy,
        parameters=jnp.asarray(parameters["onsite_coefficients_eV"]),
        enable_jit=enable_jit,
    )

    j1, j2, j3, j4, j5, j6, j7 = parameters["short_range_eV_per_angstrom2"]
    system.add_dipole_interaction_1st_shell(
        "short_range_1",
        field_ID="local_mode",
        j1=j1,
        j2=j2,
        enable_jit=enable_jit,
    )
    system.add_dipole_interaction_2nd_shell(
        "short_range_2",
        field_ID="local_mode",
        j3=j3,
        j4=j4,
        j5=j5,
        enable_jit=enable_jit,
    )
    system.add_dipole_interaction_3rd_shell(
        "short_range_3",
        field_ID="local_mode",
        j6=j6,
        j7=j7,
        enable_jit=enable_jit,
    )
    if include_ewald:
        prefactor = (
            parameters["mode_effective_charge_e"] ** 2 / parameters["epsilon_xx"]
        )
        system.add_dipole_dipole_interaction(
            "dipole_ewald",
            field_ID="local_mode",
            prefactor=prefactor,
            enable_jit=enable_jit,
        )

    B11, B12, B44 = parameters["elastic_eV"]
    system.add_homo_elastic_interaction(
        "homo_elastic",
        field_ID="gstrain",
        B11=B11,
        B12=B12,
        B44=B44,
        enable_jit=enable_jit,
    )
    system.add_inhomo_elastic_interaction(
        "inhomo_elastic",
        field_ID="local_strain",
        B11=B11,
        B12=B12,
        B44=B44,
        enable_jit=enable_jit,
    )
    B1xx, B1yy, B4yz = parameters["strain_mode_eV_per_angstrom2"]
    system.add_homo_strain_dipole_interaction(
        "homo_strain_mode",
        field_1_ID="gstrain",
        field_2_ID="local_mode",
        B1xx=B1xx,
        B1yy=B1yy,
        B4yz=B4yz,
        enable_jit=enable_jit,
    )
    system.add_inhomo_strain_dipole_interaction(
        "inhomo_strain_mode",
        field_1_ID="local_strain",
        field_2_ID="local_mode",
        B1xx=B1xx,
        B1yy=B1yy,
        B4yz=B4yz,
        enable_jit=enable_jit,
    )
    return {
        "local_mode": local_mode,
        "local_strain": local_strain,
        "global_strain": global_strain,
    }



def build_smo_spin_system(
    parameters,
    size=2,
    u_over_u_eq=0.0,
    enable_jit=True,
    allow_extrapolation=False,
):
    """Build the separate frozen-structure magnetic limit (Stage C)."""
    J_ip, J_op = evaluate_exchange(
        parameters,
        u_over_u_eq=u_over_u_eq,
        allow_extrapolation=allow_extrapolation,
    )
    J2_eV = parameters["J2_meV"] * MEV_TO_EV
    a0 = parameters["lattice_constant_angstrom"]
    lattice = of.SimpleCubic3D(size, size, size, jnp.asarray((a0, 0.0, 0.0)), jnp.asarray((0.0, a0, 0.0)), jnp.asarray((0.0, 0.0, a0)))
    system = of.System(lattice)
    spin = system.add_field(
        ID="spin", ftype="SO3", value=jnp.asarray((0.0, 0.0, 1.0))
    )
    system.add_self_interaction(
        "spin_first_neighbor",
        field_ID="spin",
        energy_engine=anisotropic_first_neighbor_spin_energy,
        parameters=jnp.asarray((J_ip, J_op)),
        enable_jit=enable_jit,
    )
    system.add_self_interaction(
        "spin_second_neighbor",
        field_ID="spin",
        energy_engine=isotropic_second_neighbor_spin_energy,
        parameters=jnp.asarray((J2_eV,)),
        enable_jit=enable_jit,
    )
    return system, {"spin": spin, "u_over_u_eq": float(u_over_u_eq)}


def build_smo_coupled_system(
    parameters,
    size=2,
    enable_jit=True,
    include_ewald=True,
):
    """Build the coupled first-principles-informed extension (Stage D).

    This is a new unified Hamiltonian, not a reproduction of either SrMnO3
    reference.
    """
    a0 = parameters["lattice_constant_angstrom"]
    lattice = of.SimpleCubic3D(size, size, size, jnp.asarray((a0, 0.0, 0.0)), jnp.asarray((0.0, a0, 0.0)), jnp.asarray((0.0, 0.0, a0)))
    system = of.System(lattice)
    fields = _add_ferroelectric_fields_and_interactions(
        system, parameters, enable_jit=enable_jit, include_ewald=include_ewald
    )
    ## add spin field and spin-local interaction
    spin = system.add_field(
        ID="spin", ftype="SO3", value=jnp.asarray((0.0, 0.0, 1.0))
    )
    fields["spin"] = spin
    u_eq = parameters["effective_hamiltonian_u_eq_angstrom"]
    first_neighbor_parameters = np.concatenate(
        (
            (u_eq,),
            parameters["J1_ip_anchored_coefficients_meV"] * MEV_TO_EV,
            parameters["J1_op_anchored_coefficients_meV"] * MEV_TO_EV,
        )
    )
    system.add_mutual_interaction(
        "reference_matched_spin_local_first_neighbor",
        field_1_ID="local_mode",
        field_2_ID="spin",
        energy_engine=reference_matched_first_neighbor_spin_local_energy,
        parameters=jnp.asarray(first_neighbor_parameters),
        enable_jit=enable_jit,
    )
    system.add_self_interaction(
        "reference_matched_spin_second_neighbor",
        field_ID="spin",
        energy_engine=reference_matched_anisotropic_second_neighbor_spin_energy,
        parameters=jnp.asarray(parameters["J2_relaxed_fe_meV"] * MEV_TO_EV),
        enable_jit=enable_jit,
    )
    return system, fields
