Logic of OpenFerro
==================

An OpenFerro program follows a small set of explicit objects. Lattices define
geometry, fields store state, interaction wrappers assemble the Hamiltonian,
integrators update individual fields, and simulations coordinate stepping and
reporting. The :doc:`quickstart` gives a complete minimal program.

Define a lattice
----------------

A three-dimensional Bravais lattice is defined by its number of cells and
three primitive vectors. For example:

.. code-block:: python

   import jax.numpy as jnp
   import openferro as of


   lattice = of.BodyCenteredCubic3D(
       10,
       10,
       10,
       a1=jnp.array((-0.5, 0.5, 0.5)),
       a2=jnp.array((0.5, -0.5, 0.5)),
       a3=jnp.array((0.5, 0.5, -0.5)),
   )

Lattice classes also provide neighbor-shell rollers for translationally
invariant short-range interactions. Constructor availability does not imply a
validated end-to-end workflow; :doc:`feature_status` is authoritative for
geometry support.

Create a system and fields
--------------------------

The :class:`~openferro.system.System` owns one lattice and dictionaries of
fields and interactions:

.. code-block:: python

   system = of.System(lattice)
   displacement = system.add_field(
       "displacement", ftype="Rn", dim=3, value=0.0, mass=1.0
   )

Local field arrays have shape ``(l1, l2, l3, field_dimension)``. Supported
public field kinds include scalar, ``R3``, ``Rn``, ``SO3``, and
``LocalStrain3D``. ``system.add_global_strain`` creates the six-component
engineering-Voigt global strain field.

Assemble the Hamiltonian
------------------------

Built-in ``System.add_*_interaction`` methods register common ferroelectric,
elastic, Ewald, and magnetic terms. The general API accepts pure scalar energy
functions through three wrappers:

* ``add_self_interaction`` for one field;
* ``add_mutual_interaction`` for two fields;
* ``add_triple_interaction`` for three fields.

For example:

.. code-block:: python

   def onsite_energy(values, parameters):
       coefficient = parameters[0]
       return coefficient * jnp.sum(values**2)


   system.add_self_interaction(
       "onsite",
       field_ID="displacement",
       energy_engine=onsite_energy,
       parameters=[1.0],
   )

The wrapper JIT-compiles the energy engine and derives force as its negative
gradient. Energy engines receive field values first and parameters last, must
not perform I/O or mutate state, and return one scalar.

Select dynamics
---------------

Each field owns its integrator. Real-valued fields accept ``optimization``,
``adiabatic``, or ``isothermal`` aliases for gradient descent, Leapfrog, and
LFMiddle Langevin integration. ``SO3`` fields map the same aliases to damped,
conservative, and stochastic SIB methods.

.. code-block:: python

   displacement.set_integrator("adiabatic", dt=0.001)
   simulation = of.SimulationNVE(system, seed=42)
   simulation.init_velocity(mode="zero")
   simulation.run(100)

``MDMinimize``, ``SimulationNVE``, ``SimulationNVTLangevin``, and
``SimulationNPTLangevin`` provide the promoted orchestration paths. A
simulation may attach thermo and field reporters before ``run``. Restart uses
explicit field values, velocities, and the simulation random key; a general
checkpoint-file format remains experimental.

Parallel execution
------------------

Ordinary unsharded arrays use the default JAX device. Experimental
multi-device execution constructs a :class:`~openferro.parallelism.DeviceMesh`
and calls ``system.move_fields_to_multi_devs(mesh)`` after fields are created.
The maintained ``05.BTO_GPU_Parallel`` example records the current one-node
benchmark workflow and its limitations.
