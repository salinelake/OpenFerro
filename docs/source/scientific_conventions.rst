Scientific Conventions
======================

This page defines the conventions promoted by Milestone B. The status and
scope of other features remain in :doc:`feature_status`. In every engine below,
field values precede the parameter array, the energy is a scalar, and force is
the negative energy gradient.

BaTiO3 effective Hamiltonian
----------------------------

The onsite, short-range, elastic, and strain-mode conventions follow Zhong,
Vanderbilt, and Rabe, *Phys. Rev. B* **52**, 6301 (1995),
DOI ``10.1103/PhysRevB.52.6301``. Their local soft mode is represented by
``dipole`` in the maintained example, but its stored value is a displacement
``u`` in Angstrom. Multiplication by ``Z_star`` is required to obtain a dipole.

Onsite term
^^^^^^^^^^^

The onsite engine implements Eq. (3):

.. math::

   E_{\mathrm{self}} = \sum_i \left[
   \kappa_2 |\mathbf{u}_i|^2 + \alpha |\mathbf{u}_i|^4
   + \gamma(u_{ix}^2u_{iy}^2 + u_{iy}^2u_{iz}^2
   + u_{iz}^2u_{ix}^2)\right].

The parameter order is ``(k2, alpha, gamma)`` with units
``(eV/Angstrom^2, eV/Angstrom^4, eV/Angstrom^4)``.

Short-range term
^^^^^^^^^^^^^^^^

The source Eqs. (9)-(10) are implemented as

.. math::

   E_{\mathrm{short}} = \sum_{s}\sum_i
   \sum_{\mathbf{d}\in D_s}
   \mathbf{u}_i^T J_s(\mathbf{d})\mathbf{u}_{i-\mathbf{d}},

where each ``D_s`` is one half of a cubic neighbor shell. Thus each
undirected displacement bond is represented once. For unit direction
``n = d / |d|``, the independently tested matrices are:

* first shell ``(j1, j2)``: diagonal entry
  ``j1 + (j2 - j1) * abs(n_a)``;
* second shell ``(j3, j4, j5)``: diagonal entry
  ``j4 + sqrt(2) * (j3 - j4) * abs(n_a)`` and off-diagonal entry
  ``2 * j5 * n_a * n_b``;
* third shell ``(j6, j7)``: diagonal entry ``j6`` and off-diagonal entry
  ``3 * j7 * n_a * n_b``.

All seven values have units ``eV/Angstrom^2``. The tests enumerate these
matrices without calling the production rollers or energy engines.

Strain, elasticity, and pressure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Global and local strain use engineering Voigt order

.. math::

   \boldsymbol{\eta} =
   (\eta_{xx},\eta_{yy},\eta_{zz},2\eta_{yz},2\eta_{xz},2\eta_{xy}).

The homogeneous elastic energy from Eq. (12) is

.. math::

   E_{\mathrm{elas}} = N\left[
   \frac{B_{11}}{2}\sum_{a=1}^3\eta_a^2
   + B_{12}(\eta_1\eta_2+\eta_2\eta_3+\eta_3\eta_1)
   + \frac{B_{44}}{2}\sum_{a=4}^6\eta_a^2\right].

The parameter order is ``(B11, B12, B44, N)``. ``N`` is the number of
primitive cells. ``System.add_homo_elastic_interaction(..., n_cells=N)`` can
select this scale explicitly; omitting it preserves ``lattice.nsites``. For
the local acoustic-displacement form of Eq. (13), the
production coefficients are ``g11 = B11 / 4``, ``g12 = B12 / 8``, and
``g44 = B44 / 8``. A uniform local displacement has zero elastic energy.

The homogeneous and inhomogeneous strain-mode interactions implement Eq. (14)
with parameter order ``(B1xx, B1yy, B4yz)``. For homogeneous strain, the
matrix in ``0.5 * sum_i u_i^T B(eta) u_i`` is

.. math::

   B(\eta) = \begin{pmatrix}
   B_{1xx}\eta_1+B_{1yy}(\eta_2+\eta_3) & B_{4yz}\eta_6 & B_{4yz}\eta_5 \\
   B_{4yz}\eta_6 & B_{1xx}\eta_2+B_{1yy}(\eta_1+\eta_3) & B_{4yz}\eta_4 \\
   B_{4yz}\eta_5 & B_{4yz}\eta_4 & B_{1xx}\eta_3+B_{1yy}(\eta_1+\eta_2)
   \end{pmatrix}.

OpenFerro maps the engineering Voigt components to a symmetric strain tensor,
uses ``F = I + eta`` as the deformation gradient, and computes the default
variable-cell volume from its determinant:

.. math::

   \eta = \begin{pmatrix}
   \eta_1 & \eta_6/2 & \eta_5/2 \\
   \eta_6/2 & \eta_2 & \eta_4/2 \\
   \eta_5/2 & \eta_4/2 & \eta_3
   \end{pmatrix}, \qquad
   V(\eta)=V_0\det(I+\eta), \qquad
   E_p=p[V(\eta)-V_0].

``SimulationNPTLangevin`` converts pressure from bar to ``eV/Angstrom^3``.
The pressure engine, ``System.calc_volume()``, and thermo reporter share the
selected volume convention; a fixed-cell reporter returns ``V0``. The
``linearized_small_strain`` compatibility mode uses
``V = V0 * (1 + eta1 + eta2 + eta3)``, the first-order Taylor expansion of the
determinant. Select it with
``add_global_strain(..., pressure_volume="linearized_small_strain")`` when
reproducing a pre-determinant workflow.

``System.add_global_strain(reference_volume=V0)`` may associate homogeneous
strain with a positive thermodynamic reference volume distinct from the
lattice reference volume. The selected value is used consistently by the
pressure interaction, ``System.calc_volume()``, and excess-stress
normalization without modifying ``lattice.ref_volume``. Excess stress is
therefore a reference-volume-normalized generalized or nominal stress, not a
finite-strain Cauchy stress. This distinction lets a padded nanoparticle use
``N_particle * a0**3`` for homogeneous mechanics while retaining
``(L * a0)**3`` for the fixed Ewald geometry.

Masked Euclidean fields
^^^^^^^^^^^^^^^^^^^^^^^

``MaskedFieldRn`` stores a full ``(*lattice.size, dim)`` array and a required,
immutable lattice-shaped Boolean mask ``m``. Public state assignments and the
built-in Euclidean integrator commit paths enforce

.. math::

   \mathbf{u}_i \leftarrow
   \begin{cases}
   \mathbf{u}_i, & m_i=1,\\
   \mathbf{0}, & m_i=0.
   \end{cases}

The same projection applies to velocity and assembled force.
``MaskedFieldRn.active_dof`` is ``N_active * dim``.

``MaskedLocalStrain`` is a distinct three-component subclass for constrained
acoustic displacement. It requires a full-lattice ``constraint_basis`` with
rank ``r`` and additionally applies the active-site orthogonal projection

.. math::

   P(\mathbf{u}) =
   \mathbf{u} - X(X^T X)^{-1}X^T\mathbf{u}

independently to each component. Only the basis and its small inverse Gram
matrix are stored. Basis columns must be finite and linearly independent on
the active sites. ``MaskedLocalStrain`` requires uniform mass on its active
sites so this Euclidean projection is also the mass-metric projection.

Mass remains finite and positive at inactive sites. Kinetic energy therefore
needs no special mask. ``MaskedLocalStrain`` temperature uses
``active_dof = 3 * (N_active - r)``:

.. math::

   T=\frac{2K}
   {3 k_B (N_{\mathrm{active}}-r)}.

The inherited ``mean`` and ``var`` and the standard field reporter still
reduce over every stored padded site. Exact zero padding allows recovery of an
active mean as ``mean_reported * N_box / N_active``; active variance requires a
full field sample. The mask constrains degrees of freedom and is not a general
material-topology deletion rule for arbitrary energy engines.
``System.calc_force_by_ID`` remains a raw interaction diagnostic and may show
nonzero exterior components; ``System.update_force`` stores the projected
assembled force used by integration and minimization.

Dipole Ewald term
^^^^^^^^^^^^^^^^^

The reciprocal-space implementation follows Sec. 5.3 of Wang et al.,
*Computational Materials Science* **162**, 314-321 (2019),
DOI ``10.1016/j.commatsci.2019.03.006``. The maintained BTO mapping supplies a
one-element parameter array containing
``Z_star**2 / epsilon_inf``. The field must have shape ``(l1, l2, l3, 3)``.

Milestone B supports only finite, positive, axis-aligned orthogonal primitive
vectors. Rotated, left-handed, skew, and general triclinic cells fail
explicitly. Energy is tested against a direct real-space dipole sum for two
cell sizes and several fields; force uses JAX autodiff and is checked against
float64 finite differences. A separate analytic Ewald force is not provided.

The reciprocal zero vector is omitted, corresponding to the conducting or
tin-foil macroscopic boundary convention. Under the same homogeneous
``1 / epsilon_inf`` screening convention, the optional application-local
spherical boundary term is

.. math::

   \Delta H_{\mathrm{sphere}} =
   \frac{Z_*^2}{\epsilon_\infty}
   \frac{|\sum_i \mathbf{u}_i|^2}{6\epsilon_0 V_{\mathrm{box}}}.

This is the shape-dependent Ewald zero-mode boundary energy, not a physical
surface free energy proportional to particle area. It does not solve a
spatially varying BaTiO3-vacuum dielectric interface. The nanoparticle example
keeps it off by default and provides a standalone same-parity nonperiodic
energy-and-force comparison in ``surface_validation.py``.

Magnetic exchange
-----------------

The public convention is

.. math::

   E_{\mathrm{ex}} = -\sum_s\sum_i
   \sum_{\mathbf{d}\in D_s} J_s\,
   \mathbf{m}_i\mathbin{\cdot}\mathbf{m}_{i-\mathbf{d}},

where ``D_s`` contains half of shell ``s``. The default
``bond_counting="unique"`` therefore counts every periodic displacement bond
once. Positive ``J`` is ferromagnetic. Parameter arrays have shape ``(1,)``;
the field and coupling units must be mutually consistent.

.. list-table:: Validated three-dimensional shells
   :header-rows: 1
   :widths: 30 20 25 25

   * - Lattice
     - Shells
     - Coordination
     - Half-shell rollers
   * - Simple cubic
     - 1-3
     - 6, 12, 8
     - 3, 6, 4
   * - Body-centered cubic
     - 1-4
     - 8, 6, 12, 24
     - 4, 3, 6, 12

In undersized periodic cells, different displacement bonds may map to the same
site. OpenFerro retains their displacement multiplicity, treating the periodic
system as a multigraph. This preserves shell coordination but does not mean
that every rolled array index names a distinct neighbor site.

bcc Fe conversion
^^^^^^^^^^^^^^^^^

Tao et al., *Phys. Rev. Lett.* **95**, 087207 (2005),
DOI ``10.1103/PhysRevLett.95.087207``, write an unnumbered Hamiltonian directly
before Eq. (1) as an ordered sum over ``r != r'``. Their spins have unit length,
the moment is absorbed into ``J``, and the four values are reported in meV.
Their Eq. (1) effective field contains the corresponding factor of two.

The maintained config stores the same values in mRy and a moment ``M_s`` in
``mu_B``. Because OpenFerro stores ``m = M_s S`` and counts unique bonds,

.. math::

   J_s^{\mathrm{engine}} =
   \frac{2 J_s^{\mathrm{source}} C_{\mathrm{mRy\ to\ eV}}}{M_s^2}.

The `official VAMPIRE Curie-temperature tutorial
<https://vampire.york.ac.uk/tutorials/curie-temperature-simulation/>`_ gives the
simple-cubic example's ``6.72e-21 J/link`` value. It specifies energy per unique
link, so the conversion has no factor of two. Evans et al., *J. Phys.: Condens.
Matter* **26**, 103202 (2014), DOI ``10.1088/0953-8984/26/10/103202``, provide
the peer-reviewed atomistic-spin model convention cited by the config.
``bond_counting="ordered"`` remains an explicit compatibility mode: it
multiplies the unique engine energy by two.

Molecular dynamics
------------------

``GradientDescentIntegrator`` stores only the on-step coordinate and applies
``x <- x + dt * force / mass``. Mass, time step, temperature, relaxation time,
velocity, and tolerance inputs reject non-finite or invalid values.

``LeapFrogIntegrator`` stores ``(x_n, v_(n-1/2))``. One step applies a full
force kick followed by a full drift and stores ``(x_(n+1), v_(n+1/2))``.
Kinetic reporters therefore observe the stored half-step velocity.

``LangevinIntegrator`` is the LFMiddle scheme described by Zhang et al.,
*J. Phys. Chem. A* **123**, 6056-6079 (2019),
DOI ``10.1021/acs.jpca.9b02771``. With the same half-step state, one call uses
the operator order ``B-A-O-A``: full kick, half drift, exact
Ornstein-Uhlenbeck thermostat, and half drift. The last kick of one
velocity-Verlet-middle step is merged into the first kick of the next call.

For reproducible continuation, save and restore field values, stored
velocities, and ``Simulation.get_random_key()``. Restoring coordinates alone
is not a complete restart. ``SimulationNPTLangevin`` uses this same LFMiddle
state for local modes and global strain.

Spin dynamics
-------------

The ``SO3`` aliases ``adiabatic``, ``optimization``, and ``isothermal`` select
the conservative, damped, and stochastic SIB integrators. They follow the SIB
method of Mentink et al., *J. Phys.: Condens. Matter* **22**, 176001 (2010),
DOI ``10.1088/0953-8984/22/17/176001``, especially its predictor and corrector
in Eq. (18).

For each stage, OpenFerro solves the fixed-point relation

.. math::

   Y = M - \Delta t\,\gamma\,
   \frac{M+Y}{2}\mathbin{\times}B_{\mathrm{stage}}.

The predictor uses the field at ``M``. The corrector evaluates the Hamiltonian
force at the actual, generally non-unit predictor midpoint ``(M+Y_pred)/2``.
For damped dynamics, the damping cross product also uses that midpoint. The
public ``SO3`` state is normalized only after the corrector. Stochastic SIB
uses the same Gaussian realization in both stages and the source's cut-off
Gaussian bound.

``last_iterations`` and ``last_converged`` expose bounded fixed-point status;
nonconvergence logs a warning and returns the bounded iterate. Exactly one
``SO3`` field is supported by simulation loops. Multiple coupled ``SO3``
fields fail before stepping because their correct solution requires one
simultaneous implicit stage rather than field-order updates.

Maintained model records
------------------------

The JSON files under ``model_configs`` and the maintained examples are
inspectable parameter records rather than inputs to a closed package schema.
They declare source, DOI, units, conventions, conversion formulas, and a small
reference observable. ``schema_version`` identifies the current record layout;
it does not limit the model kinds or parameter groups that OpenFerro can use.

OpenFerro has no model-specific configuration loader. The examples read JSON
with the standard library and map their own named values to the public lattice,
field, and interaction APIs. Record tests recompute the maintained reference
observables, but arbitrary user models are not required to adopt this layout.
See ``model_configs/README.md`` for the data boundary and extension guidance.

Compatibility and reversal
--------------------------

Milestone B changes the default magnetic engine from the old implicit
factor-two behavior to unique bonds. To run a pre-Milestone-B parameter set
unchanged, pass ``bond_counting="ordered"`` to the System exchange helper.
Do not also double its coupling.

The SIB midpoint correction intentionally changes trajectories for
state-dependent fields; it has no compatibility switch. The previous behavior
can be isolated to ``openferro/integrator/llg.py``, while random-stream and
field validation changes are separate files. Model-record metadata is isolated
to the JSON files, the examples that interpret them, and record-specific tests;
it does not add a production configuration layer. The pressure convention is
centralized in ``openferro/engine/elastic.py`` and
``openferro/reporter.py``.

Validation evidence
-------------------

The independent scientific checks are in:

* ``tests/unit_tests/test_lattice.py`` and ``test_magnetic.py``;
* ``tests/unit_tests/test_elastic.py`` and ``test_ferroelectric.py``;
* ``tests/unit_tests/test_ewald.py``;
* ``tests/unit_tests/test_integrator_md.py`` and ``test_integrator_llg.py``;
* ``tests/unit_tests/test_model_records.py`` and ``test_examples.py``.
* ``tests/unit_tests/test_masked_field.py``, ``test_particle_mechanics.py``,
  ``test_nanoparticle_phase_a.py``, and ``test_nanoparticle_phase_b.py``.

Reference tests enable float64 only inside pytest. Production precision still
follows the user's JAX configuration. Float32 parity is checked with explicit
tolerances appropriate to each engine.
