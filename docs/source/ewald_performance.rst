Ewald Performance
=================

The dipole-dipole Ewald term is often the dominant GPU-memory consumer for
large supercells. OpenFerro stores the reciprocal-space kernel in Voigt form
with shape ``(l1, l2, l3, 6)`` and evaluates the energy through a 3D FFT of the
field with shape ``(l1, l2, l3, 3)``.

Memory Model
------------

Use ``openferro.engine.ewald.estimate_dipole_dipole_ewald_memory`` to estimate
the arrays explicitly tracked by the OpenFerro Ewald path. The estimate includes
the real-space field, the stored ``UkGG`` kernel, the complex FFT field, and the
complex kernel-applied FFT field. It does not include backend FFT workspaces,
compiler temporaries, or autodiff residuals, which depend on JAX, jaxlib, and
the GPU backend.

For ``float32`` data and ``N = l1 * l2 * l3`` sites, the tracked arrays are:

* field: ``3 * N`` floats
* ``UkGG``: ``6 * N`` floats
* FFT field: ``3 * N`` complex values
* kernel-applied FFT field: ``3 * N`` complex values

The current force calculation is still autodiff-derived from the scalar energy,
so force updates can require additional full-field intermediates during the
reverse pass.

Sharding
--------

``build_dipole_dipole_ewald`` accepts an optional ``sharding`` argument and
returns both the energy engine and ``UkGG``. Pass the kernel as an explicit
engine argument so JAX can use a global array spanning multiple processes;
such an array cannot be captured as a JIT closure constant. When a field is
sharded with ``DeviceMesh.partition_sharding()``, use the same sharding for
the kernel:

.. code-block:: python

   import jax

   from openferro.engine.ewald import build_dipole_dipole_ewald

   sharding = gpu_mesh.partition_sharding()
   energy_engine, UkGG = build_dipole_dipole_ewald(
       lattice, dtype=field.get_values().dtype, sharding=sharding
   )
   energy = jax.jit(energy_engine)(field.get_values(), UkGG, parameters)

Partitioning distributes the ``6 * N`` kernel values across the mesh. Using
replicated sharding instead stores the complete kernel on every GPU, so its
per-GPU memory cost does not decrease as devices are added.

For very large cells, remember that distributed FFTs can still allocate backend
workspace and communication buffers beyond the arrays listed above.

Baseline Timing
---------------

A small baseline runner is available for comparing energy timing, memory
estimates, and the current autodiff force path:

.. code-block:: bash

   JAX_PLATFORMS=cpu python tests/profiling/ewald_baseline.py --sizes 3x2x2
   JAX_PLATFORMS=cpu python tests/profiling/ewald_baseline.py --sizes 3x2x2 --include-force

On Perlmutter login nodes, keep this to small CPU-only checks. Use an
interactive GPU allocation before running large GPU baselines.
