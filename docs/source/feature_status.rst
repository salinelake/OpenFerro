Feature Status
==============

OpenFerro is currently a research alpha. A callable API is not necessarily a
scientifically validated API. This page is the authoritative status statement
for the 0.2 series. Exact promoted equations and units are in
:doc:`scientific_conventions`.

Status definitions
------------------

``Stable``
   The listed behavior has focused analytical or invariant tests and is
   intended to remain compatible within the 0.2 series, within the stated
   geometry and field-count limits.

``Experimental``
   The API is available for research and development, but its physical
   convention, numerical method, or distributed behavior still needs the
   named reference validation. Results require independent validation.

``Not implemented``
   The symbol may exist for API planning, but calling it raises
   ``NotImplementedError`` or no end-to-end workflow is supported.

Supported environment
---------------------

.. list-table:: Package compatibility contract
   :header-rows: 1
   :widths: 25 30 45

   * - Component
     - Supported range
     - Validation evidence
   * - Python
     - 3.13 and 3.14
     - Python 3.14.6 in the Perlmutter ``of_dev`` conda environment.
   * - JAX
     - >=0.10,<0.12
     - JAX 0.11.0 CPU reference and JIT suite.
   * - NumPy
     - >=2.0,<3
     - NumPy 2.5.1 reference calculations.
   * - Single GPU
     - Promoted paths only
     - JAX 0.11.0 on one NVIDIA A100 ``CudaDevice`` allocated on Perlmutter
       passed float32 Ewald JIT/autodiff, stochastic SIB, sharded Langevin
       state, and the BTO determinant-volume NPT smoke test. This is not a
       performance guarantee.
   * - Multi-device and multi-host
     - Experimental
     - No correctness or scaling guarantee; see the experimental benchmark
       example below.

Core API
--------

.. list-table:: Core feature status
   :header-rows: 1
   :widths: 29 18 53

   * - Feature
     - Status
     - Scope or limitation
   * - Clean wheel and source distribution
     - Stable
     - The packaging extra mirrors the no-isolation build requirements. Engine
       and integrator subpackages are included and imported from outside the
       checkout in clean wheel and source-distribution environments.
   * - ``scalar``, ``R3``, and ``Rn`` fields
     - Stable
     - Construction, broadcasting, integer/Boolean-to-floating promotion,
       shape and finite-value validation, sharding-preserving assignment,
       positive masses, finite velocities, local updates, and inertial state
       are tested on 3-D lattices.
   * - ``MaskedRn`` field constraints
     - Stable core invariant
     - A required immutable Boolean site mask projects values, velocities, and
       assembled forces to exact zero outside the active region. Active
       temperature and partition-sharding commits are tested. Inherited mean
       and variance, standard reporters, stored array size, and positive
       inactive masses retain ordinary ``Rn`` behavior.
   * - ``MaskedLocalStrain`` constraints
     - Stable for the nanoparticle workflow
     - A dedicated three-component acoustic field combines an immutable node
       mask with a required finite linear basis that removes translation and
       affine modes. Projected state, active temperature, uniform active mass,
       and partition-sharding commits are tested.
   * - ``SO3`` field state
     - Stable
     - Assignment, local mutation, and magnitude changes preserve finite,
       nonzero configured magnitudes. Stable dynamics is limited to one SO(3)
       field and the SIB aliases listed below.
   * - ``LocalStrain3D`` and ``GlobalStrain`` state
     - Stable
     - Validated real-valued state, Voigt mapping, elastic energies, pressure,
       reported volume, masks, and the tested NPT path share one selected
       volume convention. Global strain may use a positive thermodynamic
       reference volume distinct from the lattice/Ewald reference volume;
       excess stress is a reference-volume-normalized nominal measure.
   * - Self, mutual, and triple interaction wrappers
     - Stable
     - Registration, lookup, energy, autodiff force sign, parameter updates,
       and finite-difference forces are covered.
   * - Single-process random streams
     - Stable
     - A simulation-owned key is split per field and step. Restoring values,
       velocities, and key exactly continues Langevin state. A general
       checkpoint file format remains experimental.
   * - Maintained model records
     - Stable example data
     - Shipped JSON records carry provenance, units, conventions, and tested
       reference observables. They are not a universal package schema;
       arbitrary models continue to use the ordinary construction APIs.
   * - Explicit custom force engines
     - Not implemented
     - Only autodiff force creation has a public API.
   * - Reciprocal lattice vectors
     - Stable
     - Duality is tested for orthogonal, rotated, skew, left-handed, BCC, and
       hexagonal primitive vectors and is independent of supercell size.
   * - Two-dimensional system workflows
     - Experimental
     - Standalone 2-D geometry exists, but Hamiltonians and simulations are not
       validated end to end.
   * - FCC, ``Hexagonal2D``, and ring-polymer workflows
     - Not implemented
     - Public constructors fail explicitly or lack required engines.

Hamiltonians
------------

.. list-table:: Hamiltonian feature status
   :header-rows: 1
   :widths: 31 18 51

   * - Feature
     - Status
     - Scope or limitation
   * - Custom energy plus autodiff force
     - Stable
     - The wrapper contract is analytically and finite-difference tested on
       self, mutual, and triple energies.
   * - Vector external magnetic field
     - Stable
     - The parameter has shape ``(3,)``; all components and force signs are
       tested.
   * - Dipole Ewald energy and autodiff force
     - Stable
     - Limited to positive, axis-aligned orthogonal primitive vectors and a
       three-component field. General cells and an explicit analytic force are
       not implemented.
   * - Spherical Ewald surface correction
     - Experimental application helper
     - The nanoparticle example provides only the fixed named conducting and
       spherical-vacuum conventions and a standalone same-parity nonperiodic
       energy-and-force validation plot. It is not a dielectric-interface or
       atomic-surface model.
   * - Ferroelectric onsite and short-range terms
     - Stable
     - BTO source matrices, independent energies, finite-difference forces,
       cubic/inversion symmetry, JIT, and dtype parity are covered.
   * - Elastic, strain-dipole, and pressure terms
     - Stable
     - Engineering Voigt strain and ``B44 / 8`` local shear are supported.
       Pressure uses ``V0 * det(I + eta)`` by default; the prior linearized
       small-strain volume remains an explicit compatibility mode.
   * - Isotropic magnetic exchange
     - Stable
     - The default is unique undirected displacement bonds. Simple-cubic and
       BCC shell geometry, source conversions, aliasing multiplicity, forces,
       JIT, and dtype parity are covered. ``ordered`` is legacy compatibility.
   * - Cubic anisotropy
     - Experimental
     - Available, but a source-locked force and invariant matrix is incomplete.
   * - Dzyaloshinskii-Moriya energy
     - Not implemented
     - It raises until a bond and orientation convention is implemented and
       tested.
   * - Multiferroic and superlattice engines
     - Experimental
     - Source TODOs retain unresolved prefactors, coordination divisions, and
       diagonal-term factors.

Dynamics and simulations
------------------------

.. list-table:: Dynamics feature status
   :header-rows: 1
   :widths: 31 18 51

   * - Feature
     - Status
     - Scope or limitation
   * - Gradient descent
     - Stable
     - Exact one-step behavior, masks, invalid state, and current-state force
       convergence in minimization orchestration are tested.
   * - Leapfrog MD
     - Stable
     - Stored velocities are at half time steps. Exact updates, harmonic
       second-order convergence, and bounded energy error are tested.
   * - LFMiddle Langevin MD
     - Stable
     - The half-step ``B-A-O-A`` state, fixed-seed updates, canonical scales,
       dtype-preserving noise, and manual restart equivalence are tested.
   * - Conservative, damped, and stochastic SIB
     - Stable
     - Limited to exactly one SO(3) field. Predictor/corrector midpoint,
       precession, damping, exchange invariants, stochastic norm/equilibrium,
       and bounded nonconvergence reporting are tested.
   * - Multiple coupled SO(3) fields
     - Not implemented
     - Simulation loops fail before stepping; a simultaneous implicit solve is
       required to avoid field-order dependence.
   * - Non-SIB ``ConservativeLLIntegrator``, ``LLIntegrator``, and
       ``LLLangevinIntegrator``
     - Experimental
     - These directly constructible legacy classes are not included in the
       promoted validation matrix.
   * - NVE, NVT, and NPT orchestration
     - Stable
     - Stable for promoted field/integrator combinations. NPT uses only the
       documented determinant volume by default and can select the linearized
       compatibility mode; restart is manual state plus key restoration rather
       than a checkpoint file.

Reference examples
------------------

Examples 01 through 03 and 06 have explicit entry points, reproducible seeds,
and automated reduced CPU execution tests. Examples 01 through 03 expose
``--tiny``; Example 06 supplies reduced values through its ordinary arguments.
Examples 01 through 03 retain working-directory-relative defaults; Example 06
resolves repository defaults. Example 04 resolves its default paths from the
repository and script locations and has an automated, explicitly reduced
4x4x4 CPU smoke run. These checks validate construction and short execution;
they do not regress full phase-transition curves or establish convergence of a
production run. Example 05 is a metadata-rich performance benchmark rather
than a scientific reference trajectory.

.. list-table:: Maintained example status
   :header-rows: 1
   :widths: 31 20 49

   * - Example
     - Status
     - Validated scope
   * - ``01.BTO_Cooling``
     - Stable entry point
     - BTO config, Ewald, local/global strain, minimization, NPT Langevin,
       determinant-volume pressure, and output reporting execute in tiny mode.
   * - ``02.bcc_Fe_Heating``
     - Stable entry point
     - Four-shell ordered-source to unique-engine conversion and stochastic
       SIB heating execute in tiny mode.
   * - ``03.sc_Ising_Heating``
     - Stable entry point
     - The historical path now identifies its continuous SO(3) dot-product
       model as Heisenberg; strict unique-link conversion and stochastic SIB
       execute in tiny mode.
   * - ``04.PTOSTO_superlattice``
     - Runnable experimental example
     - Single-device construction, Ewald, NPT Langevin, reporting, and an
       explicitly reduced trajectory execute. The superlattice engines remain
       experimental.
   * - ``05.BTO_GPU_Parallel``
     - Experimental performance example
     - Reproducible one-node A100 measurements cover 1, 2, 3, and 4 devices for
       three BTO cell sizes. They do not promote multi-device correctness.
   * - ``06.BTO_Nanoparticle``
     - Runnable Phase-B workflow; production validation pending
     - Masked soft modes, affine-constrained acoustic nodes, fully integrated
       finite-cell free-surface mechanics, Born-charge mappings, and reduced
       minimization/NPT execution are tested. The closure preserves cubic
       elastic constants but not finite-wave-vector ZVR acoustic dispersion.
       A physical dielectric interface, surface chemistry, and production box
       convergence remain outside the promoted scope.

A feature moves from experimental to stable only with a declared equation and
parameter convention, independent energy or one-step coverage, force or
invariant validation, and relevant JIT and dtype checks.
