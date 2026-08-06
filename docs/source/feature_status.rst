Feature Status
==============

OpenFerro is currently a research alpha. A callable API is not necessarily a
scientifically validated API. This page is the authoritative status statement
for the 0.1 series.

Status definitions
------------------

``Stable``
   The listed behavior has focused analytical or invariant tests and is
   intended to remain compatible within the 0.1 series.

``Experimental``
   The API is available for research and development, but its physical
   convention, numerical method, or distributed behavior still needs the
   reference validation named below. Results require independent validation.

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
     - Milestone A validation
   * - Python
     - 3.13 and 3.14
     - Python 3.14.6
   * - JAX
     - >=0.10,<0.12
     - JAX 0.11.0, CPU backend
   * - NumPy
     - >=2.0,<3
     - NumPy 2.5.1
   * - GPU and multi-host
     - Experimental
     - Not validated by Milestone A

Core API
--------

.. list-table:: Core feature status
   :header-rows: 1
   :widths: 28 18 54

   * - Feature
     - Status
     - Scope or limitation
   * - Clean wheel and source distribution
     - Stable
     - Engine and integrator subpackages are included and are installed and
       imported from outside the checkout in the packaging test.
   * - ``scalar``, ``R3``, and ``Rn`` fields
     - Stable
     - Construction, broadcasting, mass defaults, and local immutable updates
       are tested on three-dimensional lattices.
   * - ``SO3`` field state
     - Stable
     - Assignment, local mutation, and magnitude changes preserve finite,
       nonzero configured magnitudes. LLG dynamics remain experimental.
   * - ``LocalStrain3D`` and ``GlobalStrain`` state
     - Stable
     - State construction is tested. Elastic energies, pressure/volume, and
       NPT dynamics remain experimental.
   * - Self, mutual, and triple interaction wrappers
     - Stable
     - Registration, lookup, scalar energy, autodiff force sign, parameter
       update, and a finite-difference triple-force check are covered.
   * - Single-process random streams
     - Stable
     - A simulation-owned key is split per field and Langevin step. Saved key
       state restores the stream; repeated runs continue it. Full state
       checkpoints and process/device-count invariance remain experimental.
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
     - Public constructors fail explicitly or lack the required engines.

Hamiltonians and dynamics
-------------------------

.. list-table:: Scientific feature status
   :header-rows: 1
   :widths: 31 18 51

   * - Feature
     - Status
     - Reason
   * - Custom energy plus autodiff force
     - Stable
     - The interaction wrapper contract is analytically and
       finite-difference tested on a trilinear energy.
   * - Vector external magnetic field
     - Stable
     - A three-component parameter convention and force are tested.
   * - Dipole Ewald energy
     - Experimental
     - Limited to positive, axis-aligned orthogonal primitive vectors. Rotated
       and skew cells now fail explicitly; broader reference coverage and an
       analytic force remain pending.
   * - Ferroelectric onsite and short-range terms
     - Experimental
     - Available to existing models, pending the full per-engine analytical
       and finite-difference reference suite.
   * - Elastic, strain-dipole, and pressure terms
     - Experimental
     - The B44 prefactor and finite-strain volume/pressure convention are not
       resolved. These are the COR-09 and COR-10 quarantine.
   * - Isotropic magnetic exchange
     - Experimental
     - The factor-of-two and unique-bond convention need hand-counted lattice
       tests. This is the COR-11 quarantine.
   * - Cubic anisotropy
     - Experimental
     - Available, but reference and force-validation coverage is incomplete.
   * - Dzyaloshinskii-Moriya energy
     - Not implemented
     - It raises immediately until a bond and orientation convention is
       implemented and tested.
   * - Multiferroic and superlattice engines
     - Experimental
     - Source TODOs identify unresolved prefactors, coordination divisions,
       and diagonal-term factors. These terms must not be treated as validated.
   * - Gradient descent, MD/Langevin, and LLG/SIB integrators
     - Experimental
     - Discrete-state conventions, conservation/equilibrium behavior, and the
       deterministic SIB midpoint update await reference validation (COR-12).
   * - Fixed-cell minimization orchestration
     - Stable
     - Frozen global strain is excluded from integrator validation and
       convergence, and nonconvergence is reported explicitly.
   * - NVE, NVT, and NPT simulation results
     - Experimental
     - Their integrators are experimental; NPT also depends on the quarantined
       pressure/volume convention.

Reference examples
------------------

The field-construction calls and overall public workflow used by these examples
remain compatibility targets. Their scientific results are still experimental:

.. list-table:: Maintained example status
   :header-rows: 1
   :widths: 31 20 49

   * - Example
     - Status
     - Experimental dependencies
   * - ``01.BTO_Cooling``
     - Demonstration
     - Ewald, elastic/strain coupling, pressure/volume, minimization
       integrator, and NPT Langevin dynamics.
   * - ``02.bcc_Fe_Heating``
     - Demonstration
     - Exchange bond counting and stochastic LLG/SIB dynamics.
   * - ``03.sc_Ising_Heating``
     - Demonstration
     - Exchange bond counting and stochastic LLG/SIB dynamics.

A feature moves from experimental to stable only with a declared equation and
parameter convention, analytical/reference energy coverage, force validation,
and the relevant JIT, dtype, and invariant tests.
