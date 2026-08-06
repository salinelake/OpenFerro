import numpy as np
import jax.numpy as jnp

from openferro.lattice import SimpleCubic3D
from openferro.system import System


def _trilinear_energy(field1, field2, field3, parameters):
    return parameters[0] * jnp.sum(field1 * field2 * field3)


def test_triple_interaction_energy_lookup_and_force():
    system = System(SimpleCubic3D(1, 1, 1))
    field1 = system.add_field("a", value=1.0, mass=None)
    system.add_field("b", value=2.0, mass=None)
    system.add_field("c", value=3.0, mass=None)

    interaction = system.add_triple_interaction(
        "abc",
        "a",
        "b",
        "c",
        _trilinear_energy,
        parameters=[2.0],
        enable_jit=False,
    )

    assert system.get_interaction_by_ID("abc") is interaction
    np.testing.assert_allclose(system.calc_energy_by_ID("abc"), 12.0)

    force1, force2, force3 = system.calc_force_by_ID("abc")
    np.testing.assert_allclose(force1, -12.0)
    np.testing.assert_allclose(force2, -6.0)
    np.testing.assert_allclose(force3, -4.0)

    original = field1.get_values()
    index = (0, 0, 0, 0)
    epsilon = 1e-2
    try:
        field1.set_values(original.at[index].add(epsilon))
        energy_plus = float(system.calc_energy_by_ID("abc"))
        field1.set_values(original.at[index].add(-epsilon))
        energy_minus = float(system.calc_energy_by_ID("abc"))
    finally:
        field1.set_values(original)

    finite_difference_force = -(energy_plus - energy_minus) / (2 * epsilon)
    np.testing.assert_allclose(
        force1[index], finite_difference_force, rtol=1e-4, atol=1e-4
    )


def test_triple_interaction_parameter_update():
    system = System(SimpleCubic3D(1, 1, 1))
    for field_id in ("a", "b", "c"):
        system.add_field(field_id, value=1.0, mass=None)
    interaction = system.add_triple_interaction(
        "abc",
        "a",
        "b",
        "c",
        _trilinear_energy,
        parameters=[1.0],
        enable_jit=False,
    )

    interaction.set_parameters([3.0])

    np.testing.assert_allclose(interaction.get_parameters(), jnp.array([3.0]))
    np.testing.assert_allclose(system.calc_energy_by_ID("abc"), 3.0)
