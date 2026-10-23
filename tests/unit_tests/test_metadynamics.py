import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openferro.lattice import SimpleCubic3D
from openferro.parallelism import DeviceMesh
from openferro.simulation import (
    MetadynamicsNPT,
    MetadynamicsNVT,
    SimulationNPTLangevin,
)
from openferro.system import System
from openferro.units import Constants


def _component_sum(component):
    def engine(field, parameters):
        del parameters
        return jnp.sum(field[..., component])

    return engine


def _vector_simulation(
    dimension=1,
    *,
    value=(0.0, 0.0, 0.0),
    sigma=1.0,
    upper_walls=None,
    lower_walls=None,
    hills_file=None,
    grid_min=-2.0,
    grid_max=2.0,
    grid_bin=16,
):
    system = System(SimpleCubic3D(1, 1, 1))
    field = system.add_field("dipole", ftype="R3", value=value)
    definitions = [
        {
            "id": f"s{component}",
            "field_ids": "dipole",
            "engine": _component_sum(component),
        }
        for component in range(dimension)
    ]
    simulation = MetadynamicsNVT(
        system,
        definitions,
        pace=2,
        sigma=sigma,
        height=0.25,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_bin=grid_bin,
        upper_walls=upper_walls,
        lower_walls=lower_walls,
        hills_file=hills_file,
    )
    return simulation, field


@pytest.mark.parametrize(
    "definitions, error",
    [
        ([], ValueError),
        (
            [
                {"id": str(index), "field_ids": "dipole", "engine": _component_sum(0)}
                for index in range(4)
            ],
            ValueError,
        ),
        ([{"id": "bad", "field_ids": "missing", "engine": _component_sum(0)}], ValueError),
        ([{"id": "bad", "field_ids": "dipole", "engine": 3}], TypeError),
    ],
)
def test_collective_variable_definition_validation(definitions, error):
    system = System(SimpleCubic3D(1, 1, 1))
    system.add_field("dipole", ftype="R3")
    with pytest.raises(error):
        MetadynamicsNVT(
            system,
            definitions,
            pace=1,
            sigma=1.0,
            height=1.0,
            grid_min=-2.0,
            grid_max=2.0,
        )


def test_collective_variable_rejects_duplicates_parameters_and_vector_output():
    system = System(SimpleCubic3D(1, 1, 1))
    system.add_field("dipole", ftype="R3")
    duplicate = [
        {"id": "same", "field_ids": "dipole", "engine": _component_sum(0)},
        {"id": "same", "field_ids": "dipole", "engine": _component_sum(1)},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        MetadynamicsNVT(
            system,
            duplicate,
            pace=1,
            sigma=1.0,
            height=1.0,
            grid_min=-2.0,
            grid_max=2.0,
        )
    with pytest.raises(ValueError, match="repeat"):
        MetadynamicsNVT(
            system,
            [
                {
                    "id": "repeated_field",
                    "field_ids": ("dipole", "dipole"),
                    "engine": lambda first, second, parameters: jnp.sum(first + second),
                }
            ],
            pace=1,
            sigma=1.0,
            height=1.0,
            grid_min=-2.0,
            grid_max=2.0,
        )
    with pytest.raises(TypeError, match="parameters"):
        MetadynamicsNVT(
            system,
            [
                {
                    "id": "bad_parameters",
                    "field_ids": "dipole",
                    "engine": _component_sum(0),
                    "parameters": {"not": "numeric"},
                }
            ],
            pace=1,
            sigma=1.0,
            height=1.0,
            grid_min=-2.0,
            grid_max=2.0,
        )

    def vector_output(field, parameters):
        del parameters
        return jnp.sum(field, axis=(0, 1, 2))

    with pytest.raises(ValueError, match="one real scalar"):
        MetadynamicsNVT(
            system,
            [{"id": "vector", "field_ids": "dipole", "engine": vector_output}],
            pace=1,
            sigma=1.0,
            height=1.0,
            grid_min=-2.0,
            grid_max=2.0,
        )


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("pace", 0),
        ("pace", True),
        ("sigma", 0.0),
        ("sigma", [1.0, 2.0]),
        ("height", 0.0),
        ("height", np.inf),
    ],
)
def test_metadynamics_parameter_validation(keyword, value):
    system = System(SimpleCubic3D(1, 1, 1))
    system.add_field("dipole", ftype="R3")
    arguments = {"pace": 1, "sigma": 1.0, "height": 1.0}
    arguments[keyword] = value
    with pytest.raises((TypeError, ValueError)):
        MetadynamicsNVT(
            system,
            [{"id": "sx", "field_ids": "dipole", "engine": _component_sum(0)}],
            **arguments,
            grid_min=-2.0,
            grid_max=2.0,
        )


@pytest.mark.parametrize(
    "wall",
    [
        {"kappa": 1.0},
        {"at": 0.0},
        {"at": 0.0, "kappa": -1.0},
        {"at": 0.0, "kappa": 1.0, "eps": 0.0},
        {"at": 0.0, "kappa": 1.0, "exp": 1.5},
        {"at": 0.0, "kappa": 1.0, "unknown": 2.0},
    ],
)
def test_wall_validation(wall):
    with pytest.raises((TypeError, ValueError)):
        _vector_simulation(upper_walls=wall)


def test_wall_vector_shape_validation():
    with pytest.raises(ValueError, match="shape"):
        _vector_simulation(
            2,
            upper_walls={"at": (0.0, 0.0, 0.0), "kappa": 1.0},
        )


def test_active_walls_must_begin_inside_grid():
    with pytest.raises(ValueError, match="upper wall"):
        _vector_simulation(
            grid_min=-1.0,
            grid_max=1.0,
            upper_walls={"at": 1.1, "kappa": 1.0},
        )
    with pytest.raises(ValueError, match="lower wall"):
        _vector_simulation(
            grid_min=-1.0,
            grid_max=1.0,
            lower_walls={"at": -1.1, "kappa": 1.0},
        )


@pytest.mark.parametrize(
    "arguments, error",
    [
        ({"grid_min": (0.0, 0.0)}, ValueError),
        ({"grid_min": 1.0, "grid_max": 1.0}, ValueError),
        ({"grid_bin": 1}, ValueError),
        ({"grid_bin": 2.5}, TypeError),
        ({"grid_bin": True}, TypeError),
    ],
)
def test_grid_parameter_validation(arguments, error):
    defaults = {"grid_min": -1.0, "grid_max": 1.0, "grid_bin": 10}
    defaults.update(arguments)
    with pytest.raises(error):
        _vector_simulation(**defaults)


def test_default_grid_spacing_is_at_most_one_fifth_sigma():
    simulation, _ = _vector_simulation(
        2,
        sigma=(0.4, 0.8),
        grid_min=(-1.0, -2.0),
        grid_max=(1.0, 2.0),
        grid_bin=None,
    )

    np.testing.assert_array_equal(simulation.grid_bin, (25, 25))
    np.testing.assert_array_less(
        np.asarray(simulation.grid_spacing),
        0.2 * np.asarray(simulation.sigma) + 1e-7,
    )
    assert simulation._bias_grid.shape == (26, 26)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_gaussian_energy_and_derivative_match_analytic_result(dimension):
    value = np.asarray((0.3, -0.2, 0.1))
    sigma = np.asarray((0.4, 0.7, 1.1))[:dimension]
    simulation, _ = _vector_simulation(
        dimension,
        value=value,
        sigma=sigma,
        grid_min=-1.0,
        grid_max=1.0,
        grid_bin=20,
    )
    centers = jnp.asarray(
        [[0.0, 0.1, -0.2], [0.5, -0.4, 0.3]], dtype=jnp.float32
    )[:, :dimension]
    for center in centers:
        simulation._bias_grid = simulation._add_grid_hill_engine(
            simulation._bias_grid,
            simulation._grid_points,
            center,
            simulation.sigma,
            simulation.height,
        )
    values = jnp.asarray(value[:dimension], dtype=jnp.float32)

    energy, derivative = simulation._metadynamics_bias_and_derivative(values)
    displacement = np.asarray(values)[None, :] - np.asarray(centers)
    hills = simulation.height * np.exp(
        -0.5 * np.sum((displacement / sigma) ** 2, axis=1)
    )
    expected_derivative = -np.sum(
        hills[:, None] * displacement / sigma**2, axis=0
    )

    np.testing.assert_allclose(energy, np.sum(hills), rtol=1e-6)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=0.03, atol=2e-3)


def test_gaussian_derivative_is_zero_at_single_hill_center():
    simulation, _ = _vector_simulation(1)
    center = jnp.asarray((0.25,))
    simulation._bias_grid = simulation._add_grid_hill_engine(
        simulation._bias_grid,
        simulation._grid_points,
        center,
        simulation.sigma,
        simulation.height,
    )

    energy, derivative = simulation._metadynamics_bias_and_derivative(
        center
    )

    np.testing.assert_allclose(energy, simulation.height)
    np.testing.assert_array_equal(derivative, (0.0,))


def test_combined_bias_derivative_matches_jax_grad_and_wall_formulas():
    simulation, _ = _vector_simulation(
        2,
        sigma=(0.4, 0.7),
        upper_walls={
            "at": (0.2, 2.0),
            "kappa": (1.5, 0.0),
            "offset": (0.1, 0.0),
        },
        lower_walls={
            "at": (-2.0, 0.5),
            "kappa": (0.0, 2.0),
            "offset": (0.0, 0.1),
        },
    )
    for center in jnp.asarray(((0.0, 0.0), (0.5, -0.5))):
        simulation._bias_grid = simulation._add_grid_hill_engine(
            simulation._bias_grid,
            simulation._grid_points,
            center,
            simulation.sigma,
            simulation.height,
        )
    values = jnp.asarray((0.4, 0.2))
    metadynamics, walls, derivative = simulation._bias_components_and_derivative(
        values
    )
    autodiff = jax.grad(
        lambda point: sum(simulation._bias_components_and_derivative(point)[:2])
    )(values)

    np.testing.assert_allclose(walls, 1.5 * 0.3**2 + 2.0 * 0.4**2)
    np.testing.assert_allclose(derivative, autodiff, rtol=1e-6)
    assert float(metadynamics) > 0.0


def test_walls_are_inactive_on_allowed_side_and_at_boundary():
    simulation, _ = _vector_simulation(
        upper_walls={"at": 0.5, "kappa": 2.0},
        lower_walls={"at": -0.5, "kappa": 3.0},
    )

    for value in (-0.5, 0.0, 0.5):
        energy, derivative = simulation._bias_components_and_derivative(
            jnp.asarray((value,))
        )[1:]
        np.testing.assert_array_equal(energy, 0.0)
        np.testing.assert_array_equal(derivative, (0.0,))


def test_cv_is_not_hamiltonian_and_chain_rule_force_has_correct_sign():
    def harmonic(field, parameters):
        return 0.5 * parameters[0] * jnp.sum(field**2)

    system = System(SimpleCubic3D(1, 1, 1))
    field = system.add_field("dipole", ftype="R3", value=(0.2, -0.1, 0.0))
    system.add_self_interaction("harmonic", "dipole", harmonic, parameters=[3.0])
    physical_energy = system.calc_total_potential_energy()
    interaction_ids = set(system.interaction_dict)
    simulation = MetadynamicsNVT(
        system,
        [{"id": "sx", "field_ids": "dipole", "engine": _component_sum(0)}],
        pace=2,
        sigma=0.5,
        height=0.1,
        upper_walls={"at": 0.0, "kappa": 2.0},
        hills_file=None,
        grid_min=-2.0,
        grid_max=2.0,
    )

    simulation._update_force()

    expected = -3.0 * np.asarray(field.get_values())
    expected[..., 0] -= 4.0 * 0.2
    np.testing.assert_allclose(field.get_force(), expected)
    np.testing.assert_allclose(system.calc_total_potential_energy(), physical_energy)
    assert set(system.interaction_dict) == interaction_ids == {"harmonic"}


def test_shared_two_and_three_field_cv_forces_accumulate():
    def cv_abc(a, b, c, parameters):
        del parameters
        return jnp.sum(a * b + c)

    def cv_ab(a, b, parameters):
        del parameters
        return jnp.sum(a + b)

    system = System(SimpleCubic3D(1, 1, 1))
    a = system.add_field("a", value=0.2)
    b = system.add_field("b", value=0.3)
    c = system.add_field("c", value=0.4)
    simulation = MetadynamicsNVT(
        system,
        [
            {"id": "abc", "field_ids": ("a", "b", "c"), "engine": cv_abc},
            {"id": "ab", "field_ids": ("a", "b"), "engine": cv_ab},
        ],
        pace=2,
        sigma=(1.0, 1.0),
        height=0.1,
        upper_walls={"at": (-1.0, -1.0), "kappa": (0.5, 0.25)},
        hills_file=None,
        grid_min=-2.0,
        grid_max=2.0,
    )

    simulation._update_force()

    coefficient_abc = 0.46 + 1.0
    coefficient_ab = 0.5 * (0.5 + 1.0)
    np.testing.assert_allclose(a.get_force(), -(coefficient_abc * 0.3 + coefficient_ab))
    np.testing.assert_allclose(b.get_force(), -(coefficient_abc * 0.2 + coefficient_ab))
    np.testing.assert_allclose(c.get_force(), -coefficient_abc)


class _NoOpLangevin:
    def step(self, key, field):
        del key, field


class _CountingNoOpLangevin:
    def __init__(self):
        self.steps = 0

    def step(self, key, field):
        del key, field
        self.steps += 1


def test_deposition_schedule_repeated_runs_and_hills_output(tmp_path):
    hills_file = tmp_path / "HILLS"
    simulation, field = _vector_simulation(value=(0.25, 0.0, 0.0), hills_file=hills_file)
    field.integrator = _NoOpLangevin()

    simulation.run(1)
    assert simulation.get_hill_centers().shape == (0, 1)
    simulation._update_force()
    simulation._update_force()
    assert simulation.get_hill_centers().shape == (0, 1)
    simulation.run(1)
    simulation.run(2)

    np.testing.assert_allclose(simulation.get_hill_centers(), ((0.25,), (0.25,)))
    assert simulation._bias_grid.shape == (17,)
    np.testing.assert_allclose(simulation.calc_metadynamics_bias(), 0.5, rtol=1e-6)
    lines = hills_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "#! FIELDS step s0 sigma_s0 height"
    assert lines[1] == "#! SET pace 2"
    assert [int(line.split()[0]) for line in lines[2:]] == [2, 4]


def test_hills_file_is_not_overwritten(tmp_path):
    hills_file = tmp_path / "HILLS"
    hills_file.write_text("existing\n", encoding="utf-8")
    simulation, field = _vector_simulation(hills_file=hills_file)
    field.integrator = _NoOpLangevin()

    with pytest.raises(FileExistsError):
        simulation.run(0)
    assert hills_file.read_text(encoding="utf-8") == "existing\n"


@pytest.mark.parametrize("simulation_class", [MetadynamicsNVT, MetadynamicsNPT])
@pytest.mark.parametrize("distributed_field", ["dipole", "spectator"])
def test_metadynamics_rejects_distributed_fields(
    simulation_class, distributed_field, tmp_path
):
    devices = jax.devices()[:4]
    if len(devices) < 2:
        pytest.skip("Requires at least two devices.")
    system = System(SimpleCubic3D(2, 2 * len(devices), 1))
    system.add_field("dipole", ftype="R3")
    system.add_field("spectator", ftype="R3")
    if simulation_class is MetadynamicsNPT:
        system.add_global_strain()
    mesh = DeviceMesh(devices=devices, num_rows=1, num_cols=len(devices))
    system.get_field_by_ID(distributed_field).to_multi_devs(mesh)
    hills_file = tmp_path / "HILLS"

    with pytest.raises(NotImplementedError, match="distributed fields"):
        simulation_class(
            system,
            [{"id": "sx", "field_ids": "dipole", "engine": _component_sum(0)}],
            pace=1,
            sigma=0.5,
            height=0.1,
            grid_min=-2.0,
            grid_max=2.0,
            hills_file=hills_file,
        )

    assert not hills_file.exists()


@pytest.mark.parametrize("simulation_class", [MetadynamicsNVT, MetadynamicsNPT])
def test_metadynamics_rejects_distribution_after_construction(
    simulation_class, tmp_path
):
    devices = jax.devices()[:4]
    if len(devices) < 2:
        pytest.skip("Requires at least two devices.")
    system = System(SimpleCubic3D(2, 2 * len(devices), 1))
    field = system.add_field("dipole", ftype="R3")
    if simulation_class is MetadynamicsNPT:
        system.add_global_strain()
    hills_file = tmp_path / "HILLS"
    simulation = simulation_class(
        system,
        [{"id": "sx", "field_ids": "dipole", "engine": _component_sum(0)}],
        pace=1,
        sigma=0.5,
        height=0.1,
        grid_min=-2.0,
        grid_max=2.0,
        hills_file=hills_file,
    )
    for active_field in simulation.all_fields:
        active_field.integrator = _NoOpLangevin()
    field.to_multi_devs(
        DeviceMesh(devices=devices, num_rows=1, num_cols=len(devices))
    )

    with pytest.raises(NotImplementedError, match="distributed fields"):
        simulation.run(1)

    assert simulation.get_hill_centers().shape == (0, 1)
    assert not hills_file.exists()


@pytest.mark.parametrize("simulation_class", [MetadynamicsNVT, MetadynamicsNPT])
def test_bias_force_preserves_single_device_mesh_sharding(simulation_class):
    system = System(SimpleCubic3D(1, 1, 1))
    field = system.add_field("dipole", ftype="R3", value=(0.2, 0.0, 0.0))
    mesh = DeviceMesh(devices=jax.devices()[:1], num_rows=1, num_cols=1)
    system.move_fields_to_multi_devs(mesh)
    if simulation_class is MetadynamicsNPT:
        system.add_global_strain()
    simulation = simulation_class(
        system,
        [{"id": "sx", "field_ids": "dipole", "engine": _component_sum(0)}],
        pace=2,
        sigma=1.0,
        height=0.1,
        upper_walls={"at": 0.0, "kappa": 1.0},
        hills_file=None,
        grid_min=-2.0,
        grid_max=2.0,
    )

    simulation._update_force()

    assert field.get_force().sharding == field._sharding


def test_npt_metadynamics_sets_pressure_and_integrates_global_strain():
    system = System(SimpleCubic3D(1, 1, 1))
    dipole = system.add_field("dipole", ftype="R3", value=(0.1, 0.0, 0.0))
    global_strain = system.add_global_strain()
    reference_volume = np.asarray(
        system.get_interaction_by_ID("pV").get_parameters()
    )[1]
    simulation = MetadynamicsNPT(
        system,
        [{"id": "sx", "field_ids": "dipole", "engine": _component_sum(0)}],
        pace=1,
        sigma=0.5,
        height=0.1,
        pressure=2500.0,
        hills_file=None,
        grid_min=-2.0,
        grid_max=2.0,
        grid_bin=16,
    )
    dipole.integrator = _CountingNoOpLangevin()
    global_strain.integrator = _CountingNoOpLangevin()

    simulation.run(2)

    pressure_parameters = np.asarray(
        system.get_interaction_by_ID("pV").get_parameters()
    )
    np.testing.assert_allclose(pressure_parameters[0], 2500.0 * Constants.bar)
    np.testing.assert_allclose(pressure_parameters[1], reference_volume)
    assert simulation.pressure == 2500.0
    assert isinstance(simulation, SimulationNPTLangevin)
    assert global_strain in simulation.non_SO3_fields
    assert global_strain in simulation.all_fields
    assert simulation.nfields == 2
    assert dipole.integrator.steps == 2
    assert global_strain.integrator.steps == 2
    assert simulation.get_hill_centers().shape == (2, 1)


def test_npt_metadynamics_cv_force_propagates_to_global_strain():
    def strain_xx(global_strain, parameters):
        del parameters
        return global_strain[0]

    system = System(SimpleCubic3D(1, 1, 1))
    global_strain = system.add_global_strain(value=(0.2, 0.0, 0.0, 0.0, 0.0, 0.0))
    simulation = MetadynamicsNPT(
        system,
        [{"id": "strain_xx", "field_ids": "gstrain", "engine": strain_xx}],
        pace=1,
        sigma=0.5,
        height=0.1,
        pressure=0.0,
        upper_walls={"at": 0.0, "kappa": 2.0},
        hills_file=None,
        grid_min=-1.0,
        grid_max=1.0,
        grid_bin=16,
    )

    simulation._update_force()

    np.testing.assert_allclose(
        global_strain.get_force(),
        (-0.8, 0.0, 0.0, 0.0, 0.0, 0.0),
        atol=1e-7,
    )
