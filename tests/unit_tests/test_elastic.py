import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openferro.engine.elastic import (
    deformed_volume,
    deformed_volume_change,
    homo_elastic_energy,
    inhomo_elastic_energy,
    linearized_volume,
    linearized_volume_change,
    pV_energy,
    pV_energy_linearized,
)
from openferro.lattice import BravaisLattice3D
from openferro.reporter import Thermo_Reporter
from openferro.system import System
from scientific_helpers import (
    assert_eager_jit_parity,
    assert_float32_float64_parity,
    assert_force_matches_finite_difference,
)


pytestmark = pytest.mark.scientific


def test_homogeneous_elastic_analytic_strains():
    parameters = jnp.array([12.0, 3.0, 8.0, 5.0])

    uniaxial = jnp.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    hydrostatic = jnp.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
    engineering_shear = jnp.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0])

    np.testing.assert_allclose(
        homo_elastic_energy(uniaxial, parameters),
        5.0 * 0.5 * 12.0 * 0.2**2,
    )
    np.testing.assert_allclose(
        homo_elastic_energy(hydrostatic, parameters),
        5.0 * (1.5 * 12.0 + 3.0 * 3.0) * 0.1**2,
    )
    np.testing.assert_allclose(
        homo_elastic_energy(engineering_shear, parameters),
        5.0 * 0.5 * 8.0 * 0.3**2,
    )


def test_deformed_volume_and_pressure_force():
    strain = jnp.array([0.03, -0.01, 0.02, 0.4, -0.2, 0.1])
    reference_volume = 125.0
    pressure = 0.07
    parameters = jnp.array([pressure, reference_volume])
    strain_tensor = np.array(
        [
            [0.03, 0.05, -0.1],
            [0.05, -0.01, 0.2],
            [-0.1, 0.2, 0.02],
        ]
    )
    expected_volume = reference_volume * np.linalg.det(np.eye(3) + strain_tensor)

    np.testing.assert_allclose(
        deformed_volume(strain, reference_volume), expected_volume
    )
    np.testing.assert_allclose(
        deformed_volume_change(strain, reference_volume),
        expected_volume - reference_volume,
    )
    np.testing.assert_allclose(
        pV_energy(strain, parameters),
        pressure * (expected_volume - reference_volume),
    )
    assert_force_matches_finite_difference(
        lambda value: pV_energy(value, parameters), strain
    )

    diagonal = jnp.array([0.03, -0.01, 0.02, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        -jax.grad(pV_energy, argnums=0)(diagonal, parameters),
        -pressure
        * reference_volume
        * np.array([0.99 * 1.02, 1.03 * 1.02, 1.03 * 0.99, 0.0, 0.0, 0.0]),
    )


def test_linearized_volume_is_the_first_order_compatibility_reference():
    strain = jnp.array([0.03, -0.01, 0.02, 0.4, -0.2, 0.1])
    reference_volume = 125.0
    parameters = jnp.array([0.07, reference_volume])

    np.testing.assert_allclose(linearized_volume(strain, reference_volume), 130.0)
    np.testing.assert_allclose(
        linearized_volume_change(strain, reference_volume), 5.0
    )
    np.testing.assert_allclose(pV_energy_linearized(strain, parameters), 0.35)

    direction = jnp.array([0.3, -0.1, 0.2, 0.4, -0.2, 0.1])
    coarse = abs(
        deformed_volume(0.02 * direction, 1.0)
        - linearized_volume(0.02 * direction, 1.0)
    )
    fine = abs(
        deformed_volume(0.01 * direction, 1.0)
        - linearized_volume(0.01 * direction, 1.0)
    )
    assert coarse / fine > 3.8
    assert_force_matches_finite_difference(
        lambda value: pV_energy_linearized(value, parameters), strain
    )


def test_volume_helpers_reject_non_voigt_shape():
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        deformed_volume(jnp.zeros(3), 1.0)
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        linearized_volume(jnp.zeros(3), 1.0)


def test_local_elastic_coefficients_and_translation_invariance():
    shape = (4, 4, 4, 3)
    longitudinal = np.zeros(shape)
    longitudinal[1, :, :, 0] = 1.0
    transverse = np.zeros(shape)
    transverse[1, :, :, 1] = 1.0
    parameters = jnp.array([12.0, 3.0, 8.0])

    # A unit-displacement plane varying only along x isolates a longitudinal
    # B11 mode or a transverse B44 mode at each of the 4x4 plane sites.
    np.testing.assert_allclose(
        inhomo_elastic_energy(jnp.asarray(longitudinal), parameters), 16 * 12.0
    )
    np.testing.assert_allclose(
        inhomo_elastic_energy(jnp.asarray(transverse), parameters), 16 * 8.0
    )
    np.testing.assert_allclose(
        inhomo_elastic_energy(jnp.ones(shape), parameters), 0.0
    )

    nonsymmetric = jnp.asarray(longitudinal + 0.37 * transverse)
    assert_force_matches_finite_difference(
        lambda value: inhomo_elastic_energy(value, parameters), nonsymmetric
    )


def test_elastic_jit_and_dtype_parity():
    strain = jnp.array([0.03, -0.01, 0.02, 0.04, -0.02, 0.01])
    parameters = jnp.array([12.0, 3.0, 8.0, 5.0])

    assert_eager_jit_parity(homo_elastic_energy, strain, parameters)
    assert_float32_float64_parity(homo_elastic_energy, strain, parameters)
    pressure_parameters = jnp.array([0.07, 125.0])
    assert_eager_jit_parity(pV_energy, strain, pressure_parameters)
    assert_float32_float64_parity(pV_energy, strain, pressure_parameters)
    assert_eager_jit_parity(pV_energy_linearized, strain, pressure_parameters)
    assert_float32_float64_parity(
        pV_energy_linearized, strain, pressure_parameters
    )


def test_reporter_uses_the_selected_shared_volume(tmp_path):
    lattice = BravaisLattice3D(2, 2, 2, a1=jnp.array([2.0, 0.0, 0.0]))
    system = System(lattice)
    system.add_global_strain(
        value=jnp.array([0.03, -0.01, 0.02, 0.5, 0.4, 0.3])
    )
    output = tmp_path / "thermo.log"
    reporter = Thermo_Reporter(file=output, log_interval=1, volume=True)

    reporter.initialize(system)
    reporter.step(system)

    data_line = [
        line for line in output.read_text().splitlines() if not line.startswith("#")
    ][0]
    reported_volume = float(data_line.split(", ")[1])
    expected = float(
        deformed_volume(
            system.get_field_by_ID("gstrain").get_values(), lattice.ref_volume
        )
    )
    np.testing.assert_allclose(reported_volume, expected)

    linear_system = System(lattice)
    linear_system.add_global_strain(
        value=jnp.array([0.03, -0.01, 0.02, 0.5, 0.4, 0.3]),
        pressure_volume="linearized_small_strain",
    )
    np.testing.assert_allclose(
        linear_system.calc_volume(),
        linearized_volume(
            linear_system.get_field_by_ID("gstrain").get_values(),
            lattice.ref_volume,
        ),
    )


def test_fixed_cell_reporter_uses_reference_volume(tmp_path):
    lattice = BravaisLattice3D(
        2,
        3,
        4,
        a1=jnp.array([2.0, 0.0, 0.0]),
        a2=jnp.array([0.0, 3.0, 0.0]),
        a3=jnp.array([0.0, 0.0, 4.0]),
    )
    system = System(lattice)
    output = tmp_path / "fixed_cell_thermo.log"
    reporter = Thermo_Reporter(file=output, log_interval=1, volume=True)

    reporter.initialize(system)
    reporter.step(system)

    data_line = [
        line for line in output.read_text().splitlines() if not line.startswith("#")
    ][0]
    reported_volume = float(data_line.split(", ")[1])
    np.testing.assert_allclose(reported_volume, lattice.ref_volume)
