import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openferro.engine.elastic import deformed_volume, linearized_volume
from openferro.engine.ewald import build_dipole_dipole_ewald
from openferro.integrator.md import LangevinIntegrator
from openferro.lattice import SimpleCubic3D
from openferro.parallelism import DeviceMesh
from openferro.simulation import SimulationNPTLangevin, SimulationNVTLangevin
from openferro.system import System


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not any(device.platform == "gpu" for device in jax.devices()),
        reason="requires a JAX GPU backend",
    ),
]


ROOT = Path(__file__).resolve().parents[2]


def _load_bto_builder():
    path = ROOT / "examples/01.BTO_Cooling/npt.py"
    spec = importlib.util.spec_from_file_location("openferro_bto_npt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_system


def _run_bto_volume_trajectory(mode, *, size, warmup_steps, sample_steps):
    config_path = ROOT / "examples/01.BTO_Cooling/BaTiO3.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    build_system = _load_bto_builder()
    system, dipole, local_strain, global_strain = build_system(
        config, size, pressure_volume=mode
    )
    simulation = SimulationNPTLangevin(system, pressure=-4.8e4, seed=101)
    dipole.set_integrator("isothermal", dt=0.002, temp=300.0, tau=0.1)
    global_strain.set_integrator("isothermal", dt=0.002, temp=300.0, tau=1.0)
    local_strain.set_integrator("isothermal", dt=0.002, temp=300.0, tau=1.0)
    simulation.init_velocity(mode="gaussian", temp=300.0)

    determinant_volumes = []
    linearized_volumes = []
    strains = []
    local_mode_rms = []
    reference_volume = system.lattice.ref_volume
    for step in range(warmup_steps + sample_steps):
        simulation.run(1)
        if step < warmup_steps:
            continue
        strain = global_strain.get_values()
        determinant_volumes.append(
            float(deformed_volume(strain, reference_volume))
        )
        linearized_volumes.append(
            float(linearized_volume(strain, reference_volume))
        )
        strains.append(np.asarray(strain))
        local_mode_rms.append(float(jnp.sqrt(jnp.mean(dipole.get_values() ** 2))))

    return {
        "determinant_volumes": np.asarray(determinant_volumes),
        "linearized_volumes": np.asarray(linearized_volumes),
        "strains": np.asarray(strains),
        "local_mode_rms": np.asarray(local_mode_rms),
    }


def compare_bto_volume_conventions(*, size=2, warmup_steps=20, sample_steps=80):
    """Run paired BTO NPT trajectories and return comparison metrics."""
    determinant = _run_bto_volume_trajectory(
        "determinant",
        size=size,
        warmup_steps=warmup_steps,
        sample_steps=sample_steps,
    )
    linearized = _run_bto_volume_trajectory(
        "linearized_small_strain",
        size=size,
        warmup_steps=warmup_steps,
        sample_steps=sample_steps,
    )

    det_mean = np.mean(determinant["determinant_volumes"])
    linear_trajectory_det_mean = np.mean(linearized["determinant_volumes"])
    formula_relative_difference = np.mean(
        np.abs(
            determinant["determinant_volumes"]
            - determinant["linearized_volumes"]
        )
        / determinant["determinant_volumes"]
    )
    volume_relative_difference = abs(det_mean - linear_trajectory_det_mean) / det_mean
    mean_strain_max_abs_difference = np.max(
        np.abs(
            np.mean(determinant["strains"], axis=0)
            - np.mean(linearized["strains"], axis=0)
        )
    )
    det_rms = np.mean(determinant["local_mode_rms"])
    linear_rms = np.mean(linearized["local_mode_rms"])
    local_mode_rms_relative_difference = abs(det_rms - linear_rms) / det_rms
    return {
        "determinant_mean_volume": float(det_mean),
        "linearized_mode_mean_determinant_volume": float(linear_trajectory_det_mean),
        "formula_relative_difference": float(formula_relative_difference),
        "volume_relative_difference": float(volume_relative_difference),
        "mean_strain_max_abs_difference": float(mean_strain_max_abs_difference),
        "local_mode_rms_relative_difference": float(
            local_mode_rms_relative_difference
        ),
    }


def test_gpu_ewald_energy_and_autodiff_force_are_finite_float32():
    lattice = SimpleCubic3D(
        2,
        2,
        2,
        a1=jnp.asarray((3.9, 0.0, 0.0), dtype=jnp.float32),
        a2=jnp.asarray((0.0, 3.9, 0.0), dtype=jnp.float32),
        a3=jnp.asarray((0.0, 0.0, 3.9), dtype=jnp.float32),
    )
    field = jnp.arange(24, dtype=jnp.float32).reshape((2, 2, 2, 3)) / 100
    parameters = jnp.asarray((1.7,), dtype=jnp.float32)
    engine, UkGG = build_dipole_dipole_ewald(lattice, dtype=jnp.float32)
    energy_and_force = jax.jit(jax.value_and_grad(engine, argnums=0))

    energy, gradient = energy_and_force(field, UkGG, parameters)
    energy.block_until_ready()

    assert energy.dtype == jnp.float32
    assert gradient.dtype == jnp.float32
    assert all(device.platform == "gpu" for device in gradient.devices())
    assert bool(jnp.isfinite(energy))
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_gpu_stochastic_sib_exchange_step_preserves_spin_norm():
    system = System(SimpleCubic3D(2, 2, 2))
    spin = system.add_field(
        "spin", ftype="SO3", value=jnp.asarray((0.3, -0.4, 0.5))
    )
    spin.set_magnitude(1.5)
    system.add_isotropic_exchange_interaction_1st_shell(
        "exchange", "spin", coupling=0.02
    )
    spin.set_integrator("isothermal", dt=0.0002, temp=700.0, alpha=1.0)

    SimulationNVTLangevin(system, seed=23).run(1)
    values = spin.get_values()
    values.block_until_ready()

    assert all(device.platform == "gpu" for device in values.devices())
    assert bool(jnp.all(jnp.isfinite(values)))
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(values), axis=-1), 1.5, rtol=2e-6, atol=2e-6
    )


def test_gpu_langevin_noise_and_velocity_preserve_field_sharding():
    device_count = jax.device_count()
    system = System(SimpleCubic3D(2, 2 * device_count, 2))
    field = system.add_field("x", ftype="Rn", dim=3, value=0.0, mass=1.0)
    field.set_velocity(jnp.zeros_like(field.get_values()))
    field.set_force(jnp.ones_like(field.get_values()))
    system.move_fields_to_multi_devs(DeviceMesh())
    integrator = LangevinIntegrator(dt=0.01, temp=300.0, tau=0.5)
    key = jax.random.PRNGKey(31)

    reference = jax.random.normal(
        key, field.get_values().shape, dtype=field.get_values().dtype
    )
    noise = integrator.get_noise(key, field)
    np.testing.assert_array_equal(np.asarray(noise), np.asarray(reference))
    assert noise.sharding == field._sharding

    integrator.step(key, field)
    field.get_velocity().block_until_ready()
    assert field.get_velocity().sharding == field._sharding


@pytest.mark.stochastic
def test_gpu_bto_npt_determinant_is_consistent_with_linearized_reference():
    metrics = compare_bto_volume_conventions()

    assert metrics["formula_relative_difference"] < 1.0e-3
    assert metrics["volume_relative_difference"] < 1.0e-3
    assert metrics["mean_strain_max_abs_difference"] < 1.0e-3
    assert metrics["local_mode_rms_relative_difference"] < 1.0e-2
