import numpy as np
import jax.numpy as jnp

from openferro.lattice import SimpleCubic3D
from openferro.interaction import self_interaction
from openferro.system import System


def _trilinear_energy(field1, field2, field3, parameters):
    return parameters[0] * jnp.sum(field1 * field2 * field3)


def _weighted_quadratic_energy(field, weights, parameters):
    return parameters[0] * jnp.sum(weights * field**2)


def _bilinear_energy(field1, field2, parameters):
    return parameters[0] * jnp.sum(field1 * field2)


def test_self_interaction_passes_engine_data_to_energy_and_force():
    system = System(SimpleCubic3D(1, 1, 1))
    field = system.add_field("a", ftype="Rn", dim=1, value=2.0, mass=None)
    interaction = self_interaction("a")
    interaction.set_engine_data(jnp.full(field.get_values().shape, 3.0))
    interaction.set_energy_engine(_weighted_quadratic_energy)
    interaction.create_force_engine()
    interaction.set_parameters([2.0])

    np.testing.assert_allclose(interaction.calc_energy(field), 24.0)
    np.testing.assert_allclose(interaction.calc_force(field), -24.0)
    np.testing.assert_allclose(
        interaction._accumulate_force(field, jnp.ones_like(field.get_values())),
        -23.0,
    )
    assert interaction._force_accumulator is None


def test_compiled_force_accumulators_match_term_forces():
    system = System(SimpleCubic3D(1, 1, 1))
    field1 = system.add_field("a", value=2.0, mass=None)
    field2 = system.add_field("b", value=3.0, mass=None)
    field3 = system.add_field("c", value=4.0, mass=None)

    self_term = system.add_self_interaction(
        "aa", "a", lambda field, p: p[0] * jnp.sum(field**2), [0.5]
    )
    mutual_term = system.add_mutual_interaction(
        "ab", "a", "b", _bilinear_energy, [2.0]
    )
    triple_term = system.add_triple_interaction(
        "abc", "a", "b", "c", _trilinear_energy, [1.5]
    )

    current1 = jnp.full_like(field1.get_values(), 10.0)
    current2 = jnp.full_like(field2.get_values(), 20.0)
    current3 = jnp.full_like(field3.get_values(), 30.0)

    np.testing.assert_allclose(
        self_term._accumulate_force(field1, current1),
        current1 + self_term.calc_force(field1),
    )
    force1, force2 = mutual_term._accumulate_force(
        field1, field2, current1, current2
    )
    expected1, expected2 = mutual_term.calc_force(field1, field2)
    np.testing.assert_allclose(force1, current1 + expected1)
    np.testing.assert_allclose(force2, current2 + expected2)

    force1, force2, force3 = triple_term._accumulate_force(
        field1, field2, field3, current1, current2, current3
    )
    expected1, expected2, expected3 = triple_term.calc_force(
        field1, field2, field3
    )
    np.testing.assert_allclose(force1, current1 + expected1)
    np.testing.assert_allclose(force2, current2 + expected2)
    np.testing.assert_allclose(force3, current3 + expected3)


def test_triple_interaction_energy_lookup_and_force():
    system = System(SimpleCubic3D(1, 1, 1))
    field1 = system.add_field("a", value=1.0, mass=None)
    field2 = system.add_field("b", value=2.0, mass=None)
    field3 = system.add_field("c", value=3.0, mass=None)

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

    system.update_force()
    np.testing.assert_allclose(field1.get_force(), force1)
    np.testing.assert_allclose(field2.get_force(), force2)
    np.testing.assert_allclose(field3.get_force(), force3)

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
