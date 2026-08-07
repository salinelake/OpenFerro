# Milestone B Scientific Validation Plan

- Prepared: 2026-08-06
- Starting branch: `multinode`
- Starting revision: `abf2f2f`
- Environment: Della `openferro` conda environment
- Status: implementation and validation complete; determinant-volume and
  model-record-boundary amendments completed 2026-08-07

## Goal

Milestone B builds the scientific validation harness needed to promote selected
OpenFerro Hamiltonians and integrators from experimental to stable. Promotion
requires an explicit equation, units and parameter convention, independent
reference energy, force validation, relevant invariants, JIT parity, and dtype
coverage.

The primary compatibility targets are the maintained workflows:

- `examples/01.BTO_Cooling`
- `examples/02.bcc_Fe_Heating`
- `examples/03.sc_Ising_Heating`

Scientific behavior is not changed merely to remove a TODO. Every coefficient,
counting factor, or algorithmic change must first be resolved against a primary
source and an independent test.

## Scope

### Included

- Scientific test infrastructure for analytical energies, independent neighbor
  sums, float64 finite differences, invariants, JIT, and dtype parity.
- Simple-cubic and BCC neighbor-shell displacement and bond-count conventions.
- Magnetic isotropic exchange conventions used by the two magnetic examples.
- BaTiO3 onsite, short-range, elastic, strain-dipole, pressure, and supported
  orthogonal-cell Ewald terms.
- Gradient-descent, inertial MD, Langevin, NPT, and single-spin-field LLG/SIB
  algorithms used by the maintained examples.
- Persistent random-stream continuation and split-run restart equivalence.
- Strict, versioned, unit-aware model configuration loading with provenance and
  one reference observable per maintained model.
- Tiny deterministic CPU smoke modes for the maintained examples.
- Feature-status and scientific-convention documentation for validated paths.

### Deferred

- Resolving multiferroic and ferroelectric-superlattice coefficient TODOs unless
  complete primary derivations become available during this milestone.
- Implementing Dzyaloshinskii-Moriya interactions.
- General-cell Ewald support or an explicit analytic Ewald force.
- Performance optimization, benchmark/scaling work, and multi-device tuning.
- Multi-host initialization, process-safe reporting, and distributed checkpoint
  formats.
- A simultaneous implicit solver for multiple coupled SO(3) fields. Such use
  must remain experimental or fail explicitly if order independence is not
  established.

## Primary Source Lock

The convention table must cite exact equations and record how source symbols map
to code parameters. Initial sources are:

- BaTiO3 effective Hamiltonian: Zhong, Vanderbilt, and Rabe, Physical Review B
  52, 6301 (1995), DOI 10.1103/PhysRevB.52.6301.
- bcc Fe exchange parameters: Physical Review Letters 95, 087207 (2005),
  DOI 10.1103/PhysRevLett.95.087207.
- Semi-implicit spin integration: Mentink et al., Journal of Physics:
  Condensed Matter 22, 176001 (2010),
  DOI 10.1088/0953-8984/22/17/176001.
- Middle-scheme Langevin integration: Zhang et al., Journal of Physical
  Chemistry A 123, 6056-6079 (2019),
  DOI 10.1021/acs.jpca.9b02771.

Each validated term must record:

- source DOI and equation number;
- field definition and physical units;
- parameter order and units;
- energy and force sign convention;
- neighbor direction and bond-count convention;
- strain and Voigt mapping where relevant;
- expected symmetry or invariant;
- supported dtype and geometry;
- independently computed reference values and tolerances.

## Decision Gates

### Strain and pressure

Determine whether the effective-Hamiltonian strain is explicitly infinitesimal
and whether pressure couples through the source's linearized volume. Do not
replace it with a determinant by assumption. Whichever convention is selected
must be shared by `pV_energy`, volume reporting, documentation, and pressure
finite-difference tests.

The original B3 resolution selected the linearized expression. On 2026-08-07,
the user explicitly superseded that choice and requested the standard
determinant volume plus a direct BTO NPT comparison. The implementation and
evidence for that amendment are recorded in the final section of this file;
the text above is retained as the original decision gate.

Derive the local-strain shear normalization before resolving `B44 / 8` versus
`B44 / 4`.

### Magnetic exchange

Choose and document one public convention, preferably
`E = -sum_(unique undirected bonds) J_ij m_i dot m_j`. Resolve whether source
couplings already contain moment normalization and remove example-side factors
only after hand-counted energies identify the required conversion.

Small periodic cells that alias distinct displacements onto the same site must
either be rejected for that shell or have explicitly documented multiplicity.

### MD state

Declare whether stored velocities are on-step or half-step. Validate the
implemented operator ordering against the chosen convention before retaining the
`LeapFrogIntegrator` name. Compatibility aliases are required if the accurate
name or implementation changes.

### LLG/SIB state

Resolve the deterministic second-stage damping-field update against the SIB
paper. Single-SO(3)-field dynamics is the promotion target. Multiple coupled
spin fields remain experimental unless field-order independence is proven.

### Example 03 identity

`03.sc_Ising_Heating` currently uses continuous SO(3) vectors and isotropic
dot-product exchange, which is Heisenberg-like rather than an Ising model.
Retain the directory path for compatibility, but correct the scientific
description unless an actual discrete Ising implementation is introduced.

## Work Packages

### B1 - Independent scientific test harness

- Add x64 reference-test setup without changing production default precision.
- Add central finite-difference force helpers with componentwise error reports.
- Add independent NumPy neighbor enumeration that does not call production
  rollers or production energy functions.
- Add eager-versus-JIT and float32-versus-float64 comparison helpers.
- Use fixed, nonsymmetric tiny fields so sign and index mistakes cannot cancel.
- Add pytest markers for scientific, stochastic, GPU, and slow tests.

### B2 - Geometry and exchange

- Verify every simple-cubic and BCC shell displacement and coordination count.
- Test non-aliasing cells and explicitly characterize undersized cells.
- Hand-count uniform, antiferromagnetic, and nonsymmetric spin configurations.
- Resolve the exchange factor of two and parameter sign convention.
- Update magnetic example conversions and documentation together with the code.

### B3 - Strain, elasticity, pressure, and reporting

- Write the exact Voigt-to-strain mapping used by the effective Hamiltonian.
- Derive homogeneous and local elastic coefficients, including shear factors.
- Implement one pure volume/volume-change helper for energy and reporting.
- Test hydrostatic, uniaxial, and shear strain analytically.
- Compare pressure forces against float64 central finite differences.
- Ensure reporter volume and the pressure Hamiltonian use the same convention.

### B4 - BaTiO3 Hamiltonian and Ewald

- Validate onsite and first-, second-, and third-shell short-range energies.
- Validate homogeneous and inhomogeneous elastic and strain-dipole energies.
- Test translational, inversion, permutation, and cubic symmetry as applicable.
- Expand orthogonal-cell Ewald comparison across zero, uniform, and
  nonsymmetric fields and several small sizes.
- Check Ewald autodiff forces against finite differences in float64.
- Keep rotated/skew cells explicitly unsupported in Milestone B.

### B5 - MD, Langevin, and NPT

- Test exact one-step updates for every promoted deterministic operator.
- Test zero-force motion and harmonic-oscillator period/convergence.
- Bound NVE energy drift for the declared velocity convention.
- Test Langevin zero-temperature reduction and finite-temperature equipartition.
- Test strain masks and a solvable harmonic NPT strain system.
- Test invalid time step, mass, temperature, friction, and tolerance inputs.
- Compare uninterrupted runs with manually restored values, velocities, and
  random key state.

### B6 - LLG/SIB

- Test constant-field precession and analytical damping direction.
- Test zero-field stability and long-run norm preservation.
- Test SIB convergence, bounded nonconvergence reporting, and the midpoint stage.
- Test a fixed-seed stochastic single-spin equilibrium observable.
- Define and test the behavior for more than one SO(3) field.

### B7 - Model configurations and examples

- Add standard-library dataclasses and strict JSON loading; do not add a schema
  framework dependency.
- Require schema version, material/model identity, units, citation/DOI,
  parameter conventions, and conversion formulas.
- Reject unknown and missing fields with path-specific messages.
- Convert validated sections explicitly into JAX engine parameter arrays.
- Migrate maintained example parameters and existing `model_configs` files.
- Add one source-backed reference observable and tolerance per parameter set.
- Add explicit imports, `main()`, and tiny CPU options while preserving each
  example's default scientific workflow and own-directory execution.

This was the original B7 plan. Its closed production loader was removed on
2026-08-07 after review found that two hard-coded model families were too narrow
for OpenFerro's general lattice-model scope. The enriched records, conversions,
reference observables, and example improvements were retained. The final
boundary and validation are recorded in the amendment at the end of this file.

### B8 - Status, documentation, and acceptance

- Add a scientific conventions page and link each promoted feature to evidence.
- Promote only terms that pass the full validation matrix.
- Keep unresolved multiferroic/superlattice rows experimental.
- Run the complete CPU suite, clean artifact test, strict Sphinx build, and
  example smoke tests in the `openferro` conda environment.
- Use `gputest` for the final one-GPU JIT/dtype smoke test; do not run heavy
  simulations on the login node.

## Validation Matrix

Every stable Hamiltonian must have:

- primary equation and declared units;
- independent analytical or enumerated reference energy;
- float64 central finite-difference force check;
- physical symmetry/invariant test;
- eager/JIT parity;
- float32/float64 coverage;
- invalid parameter/shape behavior.

Every stable integrator must have:

- an independently written one-step reference;
- expected convergence order or exact invariant;
- conservation, damping, or equilibrium behavior appropriate to its ensemble;
- deterministic fixed-seed behavior for stochastic methods;
- uninterrupted versus split-run continuation;
- explicit invalid-configuration behavior.

## Reversible Commit Boundaries

Implementation should remain separable into these commits:

1. scientific reference harness;
2. lattice and exchange convention;
3. strain and pressure convention;
4. BaTiO3 and Ewald reference coverage;
5. MD/Langevin/NPT validation and corrections;
6. LLG/SIB validation and corrections;
7. model schema and maintained examples;
8. feature-status and Milestone B completion record.

Shared files must be changed in focused hunks. Each commit must pass its focused
tests and the complete CPU unit suite so a later revocation can use a normal
commit revert or a small documented hunk reversal.

## Exit Gate

Milestone B is complete only when:

- every feature labeled stable passes the full Hamiltonian or integrator matrix;
- no stable engine retains an unresolved coefficient, counting, or unit TODO;
- pressure energy and reported volume share one documented convention;
- magnetic example coupling conversions match the documented unique-bond rule;
- stochastic continuation and manual restart reproduce uninterrupted state;
- each maintained model has a strict validated config and reference observable;
- maintained examples pass tiny deterministic CPU smoke tests;
- the full CPU, packaging, and warnings-as-errors documentation checks pass;
- an allocated one-GPU smoke check passes;
- all unvalidated terms remain explicitly experimental or not implemented.

## Execution Record

- Implementation completed: 2026-08-06
- Validation completed: 2026-08-06; determinant amendment revalidated
  2026-08-07
- Final environment: Python 3.14.6, JAX 0.11.0, and NumPy 2.5.1 in the
  Della `openferro` conda environment
- CPU host: `della9`
- GPU hosts: `della-l05g4` and `della-l07g4`, allocated through `gputest`
- Result: every Milestone B exit gate passed within the scope and limitations
  recorded below

The plan above was retained as written so its intended scope can be compared
with the implementation. This execution record identifies the exact decisions,
files, behavior changes, validation evidence, and selective rollback boundaries.

## Source and Convention Decisions

### BaTiO3 effective Hamiltonian

- The onsite term follows Zhong, Vanderbilt, and Rabe, Physical Review B 52,
  6301 (1995), Eq. (3), with parameter order `(k2, alpha, gamma)` and local
  soft-mode displacement in Angstrom.
- First-, second-, and third-shell short-range matrices follow Eqs. (9) and
  (10). Independent tests construct the source matrices directly rather than
  calling production rollers or production energy engines.
- Homogeneous and inhomogeneous elasticity follow Eqs. (12) and (13).
  Engineering Voigt order is `(exx, eyy, ezz, 2eyz, 2exz, 2exy)`, and the
  local shear coefficient is `g44 = B44 / 8`.
- Homogeneous and inhomogeneous strain-mode coupling follow Eq. (14).
- Pressure now uses `V = V0 * det(I + epsilon)` by default, where engineering
  Voigt shear components are halved to construct the symmetric strain tensor.
  The Hamiltonian stores `p * (V - V0)`; the omitted constant cannot affect
  forces. The first-order expression `V0 * (1 + exx + eyy + ezz)` remains the
  explicit `linearized_small_strain` compatibility mode.
- The current Ewald engine remains limited to positive, axis-aligned,
  orthogonal primitive vectors. General rotated or skew cells and an explicit
  analytical Ewald force remain outside Milestone B.

### Magnetic exchange

- The public engine convention is
  `E = -sum_(unique undirected displacement bonds) J_ij m_i dot m_j`.
  Lattice shell rollers enumerate a half-shell and each listed displacement
  retains its multiplicity even when a small periodic cell maps two
  displacements to the same site.
- The pre-Milestone-B factor-of-two behavior remains available only when a
  caller explicitly passes `bond_counting="ordered"`.
- The bcc Fe source Hamiltonian sums ordered pairs of unit spins and absorbs
  the magnetic moment into its published `J`. Its config therefore applies
  `J_engine = 2 * J_source * unit_to_eV / moment_mu_B**2` before using the
  unique-bond engine.
- The official VAMPIRE Curie-temperature tutorial supplies the simple-cubic
  value `6.72e-21 J/link`; the Evans et al. DOI in the config supplies the
  peer-reviewed atomistic-spin convention. The config converts the tutorial's
  unique-link coupling with
  `J_engine = J_source * unit_to_eV / moment_mu_B**2`.
- The historical `03.sc_Ising_Heating` path is retained, but its continuous
  fixed-magnitude SO(3) field and dot-product exchange are now described as a
  classical Heisenberg model rather than a discrete Ising model.

### Dynamics

- Leapfrog and LFMiddle store velocities at half time steps. A leapfrog state
  `(x_n, v_(n-1/2))` receives a full kick and full drift. LFMiddle applies the
  half-step-state form `B-A-O-A`; its final kick is merged into the next
  step's first kick.
- Gaussian velocity initialization is already stationary for the stored
  half-step momentum and does not receive an implicit half kick.
- SIB predictor and corrector stages now evaluate interaction forces at the
  raw arithmetic midpoint required by Mentink et al., Journal of Physics:
  Condensed Matter 22, 176001 (2010), Eq. (18). The midpoint is not normalized
  before force evaluation; the final SO(3) assignment restores the configured
  magnitude invariant.
- Conservative, damped, and stochastic SIB promotion is limited to one SO(3)
  field. Simulation loops reject multiple coupled SO(3) fields before stepping
  until a simultaneous implicit solver exists.
- Fixed-point stages expose their iteration count and convergence result and
  issue one bounded warning when `max_iter` is exhausted.

## Detailed Implementation Changes

### B1 - Scientific reference harness

- `pyproject.toml` registers `scientific`, `stochastic`, `gpu`, and `slow`
  pytest markers.
- `tests/unit_tests/conftest.py` enables x64 only for reference tests, without
  changing OpenFerro's production precision default.
- `tests/unit_tests/scientific_helpers.py` adds a central-difference negative
  gradient check with component diagnostics, an independent NumPy bond sum,
  eager/JIT parity checks, and float32/float64 comparison helpers.
- Reference fields are small, fixed, and nonsymmetric so sign, axis, and
  pair-count errors cannot cancel through symmetry.

### B2 - Geometry and exchange

- `openferro/engine/magnetic.py` makes unique bond counting the default and
  validates the optional legacy `ordered` mode.
- The four isotropic-exchange helpers in `openferro/system.py` expose the same
  explicit `bond_counting` argument and document the energy sign.
- `tests/unit_tests/test_lattice.py` locks exact simple-cubic and BCC shell
  displacements, distances, and coordination counts.
- `tests/unit_tests/test_magnetic.py` compares every promoted shell with an
  independent sum, covers nonsymmetric and uniform fields, finite-difference
  forces, eager/JIT and dtype parity, bcc source conversion, invalid modes,
  legacy compatibility, and aliased small-cell multiplicity.
- The bcc Fe and simple-cubic configs and examples were converted with the
  declared source-unit and pair-count formulas rather than retaining hidden
  example-side factors.

### B3 - Strain, elasticity, pressure, and reporting

- `openferro/engine/elastic.py` adds pure `deformed_volume()` and
  `deformed_volume_change()` helpers for `V0 * det(I + epsilon)`.
  `pV_energy()` uses them by default. `linearized_volume()`, its change helper,
  and `pV_energy_linearized()` retain the first-order compatibility path. The
  resolved `B44 / 8` coefficient still cites the source equation.
- `openferro/system.py` selects the determinant mode by default, stores the
  selected mode transactionally with global-strain creation, and exposes
  `calc_volume()` as the shared energy/reporting convention boundary.
- `openferro/reporter.py` calls `System.calc_volume()` for variable-cell output
  and reports the lattice reference volume when no global strain exists.
- `openferro/field.py` documents the exact `GlobalStrain` engineering-Voigt
  order, determinant default, and linearized compatibility mode.
- `tests/unit_tests/test_elastic.py` covers hydrostatic, uniaxial, and shear
  cases, determinant and linearized pressure forces, Taylor convergence, local
  elastic translation invariance, JIT/dtype parity, reporter agreement, and
  fixed-cell reporter volume.

### B4 - BaTiO3 and Ewald

- `tests/unit_tests/test_ferroelectric.py` validates onsite, all three
  short-range shells, finite-difference forces, eager/JIT and dtype parity,
  cubic and inversion symmetry, and homogeneous/inhomogeneous strain-mode
  coupling against independent source expressions.
- `openferro/engine/ewald.py` now rejects a field or parameter array whose
  exact shape is incompatible with the precomputed engine.
- `openferro/system.py` builds Ewald setup arrays with the target field dtype
  and sharding instead of relying on ambient defaults.
- `tests/unit_tests/test_ewald.py` compares zero, uniform, and nonsymmetric
  fields on multiple small cells with an independent direct dipole sum; it
  also covers scaling, cubic symmetry, float64 finite-difference force, JIT,
  float32 tolerance, reciprocal component form, and invalid shapes.

### B5 - MD, Langevin, NPT, and restart state

- `openferro/integrator/base.py` rejects non-finite or non-positive time steps.
- `openferro/integrator/md.py` declares the half-step velocity state, validates
  temperature and relaxation time, and creates thermostat noise in the stored
  velocity dtype.
- `openferro/field.py` requires finite strictly positive mass, finite velocity,
  and finite non-negative initialization temperature, while restoring configured
  sharding after mass and velocity updates.
- `tests/unit_tests/test_integrator_md.py` covers independent one-step updates,
  zero-force motion, harmonic convergence and energy bounds, Langevin
  zero-temperature reduction and canonical scales, masks, invalid inputs,
  fixed-seed behavior, sharding/dtype preservation, and manual split-run
  continuation of values, velocities, and random key.

### B6 - LLG and SIB

- `openferro/integrator/llg.py` validates physical and solver scalars, shares
  one bounded fixed-point stage solver, records convergence telemetry, uses
  dtype-preserving noise, restores the stochastic truncation from the source
  algorithm, and corrects predictor/corrector midpoint force and damping
  evaluation.
- `FieldSO3._set_values_for_force_evaluation()` provides the narrowly scoped
  unnormalized midpoint state needed inside SIB and rejects non-finite or
  incorrectly shaped stage values.
- `openferro/simulation.py` rejects more than one SO(3) field in minimization,
  NVE, and NVT loops before any field is advanced.
- `tests/unit_tests/test_integrator_llg.py` covers constant-field precession,
  damping direction, zero-field stability, midpoint evaluation, convergence
  and bounded nonconvergence, exchange energy/norm behavior, fixed-seed
  stochastic behavior, equilibrium scale, dtype, invalid inputs, and the
  multiple-SO(3)-field rejection.

### B7 - Model records and maintained examples

- OpenFerro core remains model-agnostic and has no package-level model-record
  loader, closed dataclass hierarchy, or hard-coded model-kind dispatcher.
- All four JSON files under `model_configs/` and the BTO example config
  carry record version, identity, DOI, units, conventions, model-specific
  parameter groups, and a recomputed reference observable. The two PbTiO3
  citation texts were corrected to Paul et al. for DOI
  `10.1103/PhysRevB.95.054111`. All five ferroelectric records select
  determinant pressure volume.
- Documented records were added for bcc Fe and the simple-cubic Heisenberg
  example. `model_configs/README.md` defines them as open-ended example data and
  documents provenance, engine conversions, reference checks, extension, and
  rollback without claiming a universal schema.
- The three maintained scripts now have explicit imports, `build_system()`,
  `parse_args()`, `main()`, script-relative defaults, configurable output
  directories and seeds, and a bounded `--tiny` mode. Each reads JSON through
  the standard library and owns only its required geometry and conversion
  assumptions. Normal production schedules remain unchanged.
- `tests/unit_tests/test_model_records.py` checks provenance and finite data for
  every maintained record, then performs record-specific production-engine,
  reference-observable, dtype, determinant-volume, and exchange-conversion
  checks. `test_examples.py` launches every script from its own directory and
  verifies tiny-mode output files.

### B8 - Status and documentation

- New `docs/source/scientific_conventions.rst` records equations, field and
  parameter units, pair counting, strain mapping, force signs, supported
  geometry/dtype, integrator state, restart requirements, evidence files, and
  compatibility limits.
- `docs/source/feature_status.rst` promotes only the validated subset and keeps
  cubic anisotropy, legacy non-SIB spin integrators, multiferroic,
  superlattice, multi-device, and multi-host paths experimental. DMI and
  multiple coupled SO(3) dynamics remain not implemented.
- API and theory pages link the convention record and correct the
  simple-cubic model identity. Each example README documents record identity,
  default execution, tiny execution, and output location.
- `tests/unit_tests/test_gpu_smoke.py` contains bounded allocated-GPU exit
  checks for float32 Ewald JIT/autodiff, stochastic SIB device placement and
  norm preservation, and paired determinant/linearized BTO NPT trajectories.

## User-Visible Behavior Changes

| Call or condition | Previous behavior | Milestone B behavior | Rollback group |
| --- | --- | --- | --- |
| Isotropic exchange helper with no counting argument | Multiplied the half-shell sum by two. | Counts each undirected displacement bond once. Pass `bond_counting="ordered"` to reproduce the old factor immediately. | B2 |
| bcc Fe example coupling setup | Contained implicit moment/unit/count factors in the script. | Loads source mRy values and applies the declared ordered-source to unique-engine conversion. | B2/B7 |
| Simple-cubic example identity | Called a continuous SO(3) dot-product model Ising. | Retains the path but labels and configures the model as classical Heisenberg. | B2/B7 |
| Pressure or reported variable-cell volume | Used separate inline trace expressions; the first B3 implementation unified them on the linearized expression. | Both use `V0 * det(I + epsilon)` by default through one selected convention. `linearized_small_strain` remains available for immediate comparison or rollback. | B3 |
| Volume reporting without `gstrain` | Relied on a three-item fallback that happened to work only through broad exception handling. | Reports the lattice reference volume explicitly. | B3 |
| Ewald field or parameter with a wrong shape | Failed later or broadcast ambiguously. | Raises a shape-specific `ValueError` before FFT evaluation. | B4 |
| Zero, negative, NaN, or infinite mass | Zero was accepted and could divide by zero during integration. | Mass must be finite and strictly positive. | B5 |
| Non-finite velocity or invalid thermostat scalar | Could enter a trajectory and fail downstream. | Fails at assignment or integrator construction. | B5 |
| Leapfrog/Langevin velocity interpretation | Not declared consistently. | Public docstrings and tests define stored values as half-step velocities. | B5 |
| Damped or stochastic SIB corrector | Used a normalized midpoint and stale damping state. | Uses the raw predictor midpoint and recomputes damping at that midpoint. This intentionally changes affected trajectories. | B6 |
| SIB fixed-point exhaustion | Emitted repeated long warnings without machine-readable state. | Records `last_iterations` and `last_converged` per stage and emits one bounded warning. | B6 |
| Simulation with multiple SO(3) fields | Advanced fields sequentially with order-dependent implicit dynamics. | Raises `NotImplementedError` before stepping. | B6 |
| Maintained example import or quick validation | Executed top-level setup with working-directory-dependent paths and full schedules. | Uses `main()`, script-relative configs, explicit CLI options, and isolated `--tiny` schedules. | B7 |
| Model-record consumption | Was unpacked ad hoc; the first B7 implementation replaced this with a closed two-family production loader. | Examples use ordinary JSON mappings and keep model-specific extraction local. Maintained records retain provenance and recomputed test observables without restricting arbitrary OpenFerro models. | B7 |

## Regression Tests Added or Expanded

| File | Specific coverage |
| --- | --- |
| `tests/unit_tests/scientific_helpers.py` | Independent bond sum, finite-difference force diagnostics, eager/JIT parity, and dtype parity. |
| `tests/unit_tests/test_lattice.py` | Exact simple-cubic/BCC shell shifts, distances, coordination, and small-cell displacement multiplicity. |
| `tests/unit_tests/test_magnetic.py` | Unique/ordered exchange, independent energies, all four bcc shells, source conversion, forces, JIT, dtypes, and invalid mode. |
| `tests/unit_tests/test_elastic.py` | Voigt mapping, elastic shear, independent determinant volume/cofactor force, linearized Taylor reference, JIT/dtypes, reporter agreement, and fixed-cell volume. |
| `tests/unit_tests/test_ferroelectric.py` | BTO onsite/short-range source expressions, forces, symmetries, strain-mode terms, JIT, and dtypes. |
| `tests/unit_tests/test_ewald.py` | Direct-sum references, multiple fields/sizes, force, symmetry/scaling, reciprocal form, JIT/dtypes, shapes, and memory metadata. |
| `tests/unit_tests/test_integrator_md.py` | Exact MD/LFMiddle steps, convergence/energy/equilibrium behavior, masks, invalid inputs, RNG continuation, dtype, and sharding. |
| `tests/unit_tests/test_integrator_llg.py` | SIB predictor/corrector, precession, damping, norm/energy, stochastic scale, convergence state, invalid inputs, and field-count limit. |
| `tests/unit_tests/test_model_records.py` | Every maintained record's provenance and finite values, production conversion arrays, reference observables, determinant convention, exchange units/counting, and example-local conversions. |
| `tests/unit_tests/test_examples.py` | Own-directory subprocess execution and output checks for all three `--tiny` workflows. |
| `tests/unit_tests/test_gpu_smoke.py` | Allocated-GPU float32 Ewald JIT/autodiff, stochastic SIB placement/norm, and paired BTO NPT volume-convention metrics. |

## Selective Rollback Groups

No files were staged and no commits were created during Milestone B. The groups
below are logical publication and rollback boundaries. The worktree also
contains the user's pre-existing tracked `AGENTS.md` edit and unrelated
untracked submission/profiling artifacts; those are not part of Milestone B
and must not be swept into a commit or removed with `git clean`.

| Group | Purpose | Files and anchors |
| --- | --- | --- |
| B1 reference harness | Shared scientific-test primitives and markers. | Marker hunk in `pyproject.toml`; new `tests/unit_tests/conftest.py` and `scientific_helpers.py`. |
| B2 exchange convention | Unique-bond API, shell/source evidence, and magnetic conversion. | Exchange engine in `openferro/engine/magnetic.py`; four exchange helper hunks in `openferro/system.py`; exchange additions in `test_lattice.py` and `test_magnetic.py`; magnetic config/example conversion hunks. |
| B3 strain and pressure | Engineering Voigt, `B44 / 8`, determinant default, named linearized compatibility, and reporter alignment. | Volume and pressure engines in `openferro/engine/elastic.py`; mode and `calc_volume()` hunks in `openferro/system.py`; volume hunk in `openferro/reporter.py`; `GlobalStrain` docstring; BTO config/example volume hunks; `test_elastic.py`, NPT test hunk, and BTO GPU comparison. |
| B4 BTO and Ewald | Independent BTO/Ewald validation and shape/dtype safety. | Ewald shape hunk in `openferro/engine/ewald.py`; Ewald construction hunk in `openferro/system.py`; expanded `test_ewald.py`; new `test_ferroelectric.py`. |
| B5 MD and restart | Half-step state contract, scalar/state validation, dtype/sharding, and MD/LFMiddle evidence. | `openferro/integrator/base.py` and `md.py`; mass/velocity/init hunks in `openferro/field.py`; new `test_integrator_md.py`. |
| B6 SIB | Correct midpoint algorithm, convergence telemetry, one-spin-field boundary, and evidence. | `openferro/integrator/llg.py`; SIB stage setter hunk in `openferro/field.py`; SO(3)-count hunks in `openferro/simulation.py`; new `test_integrator_llg.py`; SIB half of `test_gpu_smoke.py`. |
| B7 records and examples | Open-ended documented records, reference observables, local conversion, and executable entry points. | Five `model_configs/*.json`, record README, three example scripts/READMEs/configs, `test_model_records.py`, and `test_examples.py`. No production loader or package export belongs to this group. |
| B8 status and record | Public scientific claims, source/convention guide, GPU evidence, and this completion record. | New `docs/source/scientific_conventions.rst`; `feature_status.rst`, `api.rst`, `index.rst`, two theory pages; Ewald half of `test_gpu_smoke.py`; this file. |

### Shared-file dependencies and reversal order

- `openferro/system.py` contains independent B2 exchange and B4 Ewald hunks.
  Reverse selected hunks, not the entire file.
- `openferro/field.py` contains B3 documentation, B5 inertial-state
  validation, and B6 SIB midpoint support. The SIB helper can be removed only
  with the corrected SIB implementation that calls it.
- Magnetic example records and scripts combine B2 convention decisions with
  B7 local extraction. To restore legacy exchange, set
  the call to `bond_counting="ordered"` and change the declared conversion and
  stored reference together; do not alter only the coupling factor.
- B3 pressure energy and reporter behavior share the mode stored by `System`.
  Change a config or `add_global_strain()` call to
  `linearized_small_strain` for selective behavioral rollback. Removing the
  determinant API itself requires reverting the B3 engine, system, reporter,
  config, example, test, and documentation hunks together.
- B6 intentionally changes SIB trajectories. There is no runtime compatibility
  flag for the source-correct algorithm; revoke the B6 code/tests/status hunks
  as one group if exact legacy trajectories are required.
- B7 JSON files and tests are group-owned. Example-local extraction can be
  changed independently because no package configuration API depends on it.
- B8 status claims must be downgraded whenever a corresponding B2-B7 behavior
  or validation file is revoked.

For a tracked shared file, inspect `git diff -- <path>` and reverse only the
selected hunk, for example with `git restore -p -- <path>`. New group-owned
files can be removed only when their complete group is revoked. Never use
`git clean` in this worktree. After any selective reversal, run the group's
focused tests, all unit tests, the packaging test when B7 or exported modules
change, and the strict Sphinx build when behavior or status claims change.

## Explicitly Deferred or Unchanged

- Multiferroic and ferroelectric-superlattice coefficient TODOs remain
  unresolved and experimental; their production formulas were not edited.
- Cubic magnetic anisotropy remains experimental, and DMI remains explicitly
  not implemented.
- Directly constructed non-SIB conservative, damped, and stochastic LL classes
  remain experimental even though their input/noise safety improved.
- General-cell Ewald, an explicit analytic Ewald force, multiple coupled SO(3)
  integration, checkpoint files, multi-device correctness, multi-host
  initialization/reporting, and scaling claims remain outside this milestone.
- No production-size phase-transition trajectory or benchmark was run. Default
  reference-example schedules were preserved. In addition to bounded tiny entry
  points, the determinant amendment ran a short paired `4x4x4` NPT comparison;
  it is not evidence for a converged phase-transition curve.
- No existing generated output, submission script, profiling script, image, or
  other unrelated untracked artifact was modified or deleted.

## Validation Record

The original baseline commands below ran after the first implementation in the
Della `openferro` conda environment. CPU commands ran on `della9`; the original
two bounded GPU smoke tests ran on the allocated GPU node. Amendment-specific
commands and results are recorded in the final section.

- `python -m pip install -e .`: built and installed OpenFerro 0.1.0 in editable
  mode with the declared NumPy/JAX dependencies already satisfied.
- `python -m compileall -q openferro tests/unit_tests examples/01.BTO_Cooling examples/02.bcc_Fe_Heating examples/03.sc_Ising_Heating`: succeeded.
- `python -m pytest tests/unit_tests -q`: 115 passed and 2 GPU-only tests
  skipped on the CPU login node in 58.47 s.
- `python -m pytest tests/unit_tests/test_model_config.py -q`: 14 passed in
  2.83 s in the final targeted rerun after the citation/provenance corrections.
- `python -m pytest tests/packaging -q`: 1 passed in 8.36 s after installing
  the test-environment prerequisites `build` and `wheel`; clean wheel and sdist
  installation/import outside the checkout succeeded.
- `python -m sphinx -W --keep-going -b html docs/source /tmp/openferro-docs-milestone-b`:
  succeeded for all 15 documentation sources under Sphinx 9.1.0.
- `python -c "import openferro as of; print(of.System, of.load_model_config)"`:
  imported and printed both public symbols from the editable installation.
- `gputest` allocated Slurm job `12094733` on `della-l05g4`. Inside a freshly
  activated `openferro` environment, `jax.devices()` returned
  `[CudaDevice(id=0)]`; `python -m pytest tests/unit_tests/test_gpu_smoke.py -q`
  reported 2 passed in 4.89 s. The allocation was then released.
- `git diff --check`: succeeded after the completion record and final citation
  correction.

No heavy CPU or GPU simulation was run on a login node.

## Exit Gate Result

| Gate | Result | Evidence |
| --- | --- | --- |
| Stable Hamiltonians have source, independent energy, force, invariant, JIT, dtype, and invalid-input coverage. | Passed | B2-B4 tests and `scientific_conventions.rst`. |
| Stable integrators have independent step/invariant, stochastic, continuation, and invalid-input coverage. | Passed | B5-B6 tests and documented half-step/SIB state. |
| Pressure energy and reported volume share one convention. | Passed | Shared B3 helpers and reporter tests. |
| Magnetic example conversions match unique-bond counting. | Passed | Documented records, reference observables, local conversion tests, and B2 tests. |
| Maintained records and tiny entry points validate. | Passed | All shipped records plus three own-directory subprocess tests. |
| CPU, artifact, import, and strict documentation checks pass. | Passed | Validation record above. |
| One allocated-GPU promoted-path smoke passes. | Passed | Slurm job `12094733`, one CUDA device, 2 GPU tests passed. |
| Unvalidated work remains quarantined. | Passed | Feature matrix and deferred-scope list. |

## Determinant-Volume Amendment - 2026-08-07

### Requested behavior and exact convention

The user requested that variable-cell simulations use the standard volume
rather than being restricted to its linearized approximation, followed by a
direct BTO NPT consistency check. The new default maps engineering Voigt strain
`(exx, eyy, ezz, 2eyz, 2exz, 2exy)` to

```text
epsilon = [[exx,   exy,   exz],
           [exy,   eyy,   eyz],
           [exz,   eyz,   ezz]]
F = I + epsilon
V = V0 * det(F)
E_pressure = pressure * (V - V0)
```

The subtraction of `V0` removes only a constant from the Hamiltonian. The
linearized formula `V0 * (1 + exx + eyy + ezz)` is its first-order Taylor
reference and remains selectable as `linearized_small_strain`.

### Changes made

- `openferro/engine/elastic.py` adds `deformed_volume()` and
  `deformed_volume_change()`, makes `pV_energy()` determinant-based, and moves
  the previous behavior to the named `pV_energy_linearized()` engine. Both
  paths are pure JAX functions and retain negative-gradient force generation.
- `openferro/system.py` adds the two explicit mode maps, stores the selected
  pressure-volume mode, validates it before mutating the system, rolls it back
  if pressure registration fails, and adds `System.calc_volume()`. Both
  `add_global_strain()` and direct `add_pressure()` default to determinant mode.
- `openferro/reporter.py` now obtains volume from `System.calc_volume()`, so
  pressure energy and reported volume cannot silently choose different
  formulas. `openferro/field.py` documents the same Voigt and volume contract.
- The BTO example record and all four maintained ferroelectric records in
  `model_configs/` declare `determinant`; the BTO example passes the selected
  value to the system API, which also accepts `linearized_small_strain`.
- `examples/01.BTO_Cooling/npt.py` propagates the config convention and adds
  `--pressure-volume` for controlled comparisons. Its README records the
  command-level rollback and the measured GPU result.
- `docs/source/scientific_conventions.rst`, `theory-dynamics.rst`, and
  `feature_status.rst` now state the determinant equation, engineering-shear
  mapping, shared reporting behavior, and linearized compatibility path.
- `tests/unit_tests/test_elastic.py` checks an independent NumPy determinant,
  diagonal cofactor forces, float64 finite differences, second-order
  determinant-versus-linear convergence, shape failures, JIT/dtype parity,
  reporter agreement, and both modes. `test_integrator_md.py` checks the
  determinant NPT stationary solution. `test_system.py` checks transactional
  rejection, and `test_model_records.py` checks the maintained record default.
- `tests/unit_tests/test_gpu_smoke.py` builds the maintained BTO Hamiltonian and
  advances paired NPT trajectories from the same seed and initial state, with
  pressure-volume mode as the only controlled difference.

### BTO NPT consistency evidence

Slurm job `12103623`, allocated with `gputest` on `della-l07g4`, exposed one
NVIDIA A100 as `[CudaDevice(id=0)]` under JAX 0.11.0.

The automated `2x2x2` regression used 20 warmup and 80 sampled steps at 300 K,
`dt = 0.002 ps`, and `pressure = -4.8e4 bar`. It measured:

| Metric | Result |
| --- | ---: |
| Determinant-mode mean physical volume | `510.739093 Angstrom^3` |
| Linearized-mode mean physical volume, evaluated by determinant | `510.375838 Angstrom^3` |
| Mean formula difference along the determinant trajectory | `0.0375%` |
| Mean physical-volume difference between trajectories | `0.0711%` |
| Maximum absolute difference between mean strain components | `3.27044e-4` |
| Local-mode RMS difference | `0.2916%` |

The same helper was then run on a `4x4x4` cell for 200 warmup and 800 sampled
steps. It measured determinant and linearized-mode physical volumes of
`4090.295165` and `4087.146605 Angstrom^3`, respectively: a `0.0770%`
difference. The formula difference was `0.0467%`, the maximum difference in
mean strain components was `7.49820e-4`, and local-mode RMS differed by
`0.5300%`.

These paired checks show that the determinant implementation is consistent
with the previous linearized result for the tested short BTO trajectories. They
do not establish a converged equation of state, transition temperature,
thermodynamic-limit result, or production cooling curve.

### Amendment validation

- `python -m compileall -q openferro tests/unit_tests examples/01.BTO_Cooling`:
  succeeded on the login node.
- Focused pressure, MD/NPT, system, and config suite: 45 passed in 12.22 s.
- `python -m pytest tests/unit_tests -q`: 119 passed and 3 GPU-only tests
  skipped on `della9` in 56.15 s.
- `python -m pytest tests/packaging -q`: 1 passed in 8.48 s.
- Strict Sphinx build of all 15 sources with `-W --keep-going`: succeeded.
- `python -m pytest tests/unit_tests/test_gpu_smoke.py -q -s`: 3 passed on the
  allocated A100 in 24.94 s.
- The separate 1,000-step `4x4x4` paired comparison completed on the same
  allocation with the metrics above.
- No simulation work was run on the login node, and the allocation was released
  immediately after the comparison.

### Selective revocation

The compatibility path avoids an all-or-nothing revert:

1. For one BTO run, pass
   `--pressure-volume linearized_small_strain`. No file edit is required.
2. For one model, set `conventions.pressure_volume` to
   `linearized_small_strain` in that JSON record. The BTO example forwards it,
   and pressure energy plus reporting switch together.
3. To restore linearized behavior as the project default while retaining the
   determinant option, change the defaults in `System.add_global_strain()` and
   `System.add_pressure()`, update the five maintained ferroelectric config
   values, and adjust only the default-status documentation and tests.
4. To remove this amendment entirely, revert the determinant helper/engine
   hunks in `openferro/engine/elastic.py`; the mode state and `calc_volume()`
   hunks in `openferro/system.py`; the reporter and field-doc hunks; the
   five JSON values; the BTO CLI/README hunks; the determinant-specific test
   hunks; and this documentation amendment as one B3 group. Do not revert all
   of these shared files wholesale because they also contain independent
   Milestone B work.

## Model-Record Boundary Amendment - 2026-08-07

### Reason and architectural decision

Review found that the first B7 implementation put two specific material-model
families into OpenFerro core. `openferro/model_config.py` contained separate
ferroelectric and magnetic dataclass trees, exact parameter names, a two-value
model-kind dispatcher, fixed unit choices, fixed geometries, and two hard-coded
reference-observable formulas. That boundary did not match OpenFerro's goal of
supporting arbitrary lattice models.

The remedy removes the production configuration layer rather than replacing it
with a premature generalized loader. OpenFerro continues to accept lattice,
field, interaction, and simulation construction directly. Model serialization
and parameter naming remain application concerns until a genuinely extensible
format or registration design is justified.

### Preserved work

- No JSON parameter record was removed or edited by this remedy. The enriched
  identity, citation/DOI, units, conventions, source parameters, conversion
  declarations, and reference observables remain intact.
- The BTO, bcc Fe, and simple-cubic Heisenberg examples retain explicit entry
  points, script-relative paths, seeds, output directories, and tiny modes.
- The ferroelectric engine mappings, magnetic unit and pair-count conversions,
  determinant pressure-volume selection, and reference energies are still
  numerically checked.
- `schema_version` remains record-layout metadata for the shipped data. It is
  not presented as a package-wide list of permitted model kinds or parameters.

### Removed or relocated work

- `openferro/model_config.py` was deleted. `openferro/__init__.py` no longer
  exports `load_model_config`, the two model dataclasses, or `ModelConfigError`,
  and `docs/source/api.rst` no longer advertises that module.
- The three maintained examples now use `json.load()` and ordinary mappings.
  BTO performs its parameter mapping beside `build_system()`. Each magnetic
  example has a private `_exchange_couplings()` helper that applies only the
  source-unit and pair-count assumptions documented by its record.
- Geometry, model-kind, and shell-count checks now live beside the examples
  that require them. They are no longer restrictions on unrelated OpenFerro
  applications.
- Loader-specific synthetic rejection tests were removed. They were replaced
  by `tests/unit_tests/test_model_records.py`, which checks provenance and finite
  data generically, then uses record-specific test code to reproduce maintained
  reference observables and conversion arrays.
- `tests/unit_tests/test_gpu_smoke.py` now reads the BTO record directly before
  running the paired NPT comparison.
- Documentation now calls the JSON files model records or example data and
  explicitly states that OpenFerro has no universal configuration schema.
- The bcc Fe README and tiny-output expectation were aligned with the existing
  10 K tiny schedule. The schedule itself was not changed by this remedy.

### Reversal boundaries

The retained record data and removed production layer are independent:

1. The JSON enrichment can be kept or reverted independently; no core import
   depends on those files.
2. Example-local JSON extraction is isolated to the three maintained scripts
   and the BTO GPU helper. It can be changed without editing OpenFerro core.
3. Record validation is test-only and isolated to `test_model_records.py`; it
   does not constrain user models at runtime.
4. Restoring the deleted two-family loader would require reintroducing
   `openferro/model_config.py`, its `__init__` export, API docs, typed call sites,
   and loader-specific tests as one group. It should not be restored partially.
5. A future general serialization feature should be a separate design with an
   extensible payload or registry, rather than another branch added to the
   removed dispatcher.

### Validation

- Direct record plus own-directory example suite: 21 passed in 17.36 s after
  the production module was removed.
- `python -m pytest tests/unit_tests -q`: 120 passed and 3 GPU-only tests skipped
  on `della9` in 55.13 s.
- `python -m pytest tests/packaging -q`: 1 passed in 8.21 s; the built package
  imports without `openferro.model_config`.
- Strict Sphinx build of all 15 sources with `-W --keep-going`: succeeded.
- Slurm job `12105034`, allocated through `gputest` on `della-l07g5`, exposed
  one `[CudaDevice(id=0)]`; all 3 GPU smoke tests passed in 24.78 s.
- The paired BTO GPU metrics were unchanged after the mapping migration:
  `0.0711%` mean physical-volume difference, `3.27044e-4` maximum mean-strain
  difference, and `0.2916%` local-mode-RMS difference.
- Import-level verification confirmed `openferro.System` remains available,
  `load_model_config` is absent, and `find_spec("openferro.model_config")`
  returns `None`.
- Compilation and whitespace checks succeeded. No heavy simulation ran on the
  login node, and the GPU allocation was released after validation.
