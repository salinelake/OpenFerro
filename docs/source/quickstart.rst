Quickstart
==========

OpenFerro models follow one explicit construction sequence: create a lattice,
create a :class:`~openferro.system.System`, add fields, register Hamiltonian
terms, select field integrators, and run a simulation.

Minimal custom interaction
--------------------------

The example below defines a three-component field with a harmonic onsite
energy. Only the scalar energy engine is supplied; OpenFerro derives the force
with JAX automatic differentiation.

.. code-block:: python

   import jax.numpy as jnp

   import openferro as of


   lattice = of.SimpleCubic3D(2, 2, 2)
   system = of.System(lattice)
   field = system.add_field("u", ftype="R3", value=(0.0, 0.0, 0.1))


   def harmonic_energy(u, parameters):
       k = parameters[0]
       return 0.5 * k * jnp.sum(u**2)


   system.add_self_interaction(
       "harmonic",
       field_ID="u",
       energy_engine=harmonic_energy,
       parameters=[2.0],
   )
   system.update_force()

   print(system.calc_total_potential_energy())
   print(field.get_force()[0, 0, 0])

An energy engine receives field values first and parameters last, performs no
I/O or mutation, and returns one scalar energy. Mutual and triple interaction
engines follow the same convention with additional field-value arguments.

To evolve this field with conservative Leapfrog dynamics:

.. code-block:: python

   field.set_integrator("adiabatic", dt=0.01)
   simulation = of.SimulationNVE(system)
   simulation.init_velocity()
   simulation.run(10)

See the tutorial_ for a longer model-building walkthrough and the maintained
examples_ for complete ferroelectric and magnetic workflows.

Running OpenFerro on CPU
------------------------

Set ``JAX_PLATFORMS`` before Python starts when a reproducible CPU-only run is
required:

.. code-block:: bash

   export JAX_PLATFORMS=cpu

JAX controls CPU thread use according to its backend configuration.

Running OpenFerro on one GPU
----------------------------

With a supported JAX accelerator build and one visible NVIDIA GPU, ordinary
OpenFerro arrays and JIT-compiled calculations use that GPU automatically. See
:doc:`installation` for the supported versions and installation sequence.

Experimental multi-device execution
-----------------------------------

Multi-device and multi-host behavior remains experimental. Read
:doc:`feature_status` and validate the physical result independently. To shard
fields over all devices visible to one JAX process, move them after creating all
fields and before constructing the simulation:

.. code-block:: python

   from openferro.parallelism import DeviceMesh


   gpu_mesh = DeviceMesh()
   system.move_fields_to_multi_devs(gpu_mesh)

On one process the default mesh is a ``1 x N`` slab over visible devices. An
explicit device list and mesh dimensions may be passed when required. The
single-node BTO benchmark_ is the maintained complete example; its measurements
show that sharding helps only when the workload is large enough to amortize
communication.

.. _tutorial: https://github.com/salinelake/OpenFerro/blob/main/tutorials/quickstart.ipynb
.. _examples: https://github.com/salinelake/OpenFerro/tree/main/examples
.. _benchmark: https://github.com/salinelake/OpenFerro/tree/main/examples/05.BTO_GPU_Parallel
