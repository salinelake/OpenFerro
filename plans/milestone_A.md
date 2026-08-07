# Milestone A Completion Record

- Implementation completed: 2026-07-21
- Detailed change log expanded: 2026-08-06
- Pre-publish review completed: 2026-08-06
- Branch: `multinode`
- Base revision: `214f72e`
- Status: implementation and validation complete for the Milestone A scope
  defined in `plans/review.md`; publication awaits commit-scope confirmation

## Scope and decisions

Milestone A restores basic API correctness, release artifact integrity, and
honest scientific status reporting. It does not validate or change unresolved
Hamiltonian prefactors, pressure/volume conventions, exchange bond counting,
or integration algorithms. Those paths are marked experimental pending the
Milestone B reference suite.

The maintained examples `01.BTO_Cooling`, `02.bcc_Fe_Heating`, and
`03.sc_Ising_Heating` were not edited. Their field-construction and simulation
call patterns remain compatibility targets and are covered by focused tests.
Scientific outputs from those examples remain experimental because they use
one or more quarantined engines or integrators.

## Pre-publish review corrections

- The review found that the reciprocal-lattice source hunk was absent even
  though its regression tests and documentation were present. This produced
  seven failures in `tests/unit_tests/test_lattice.py`. The primitive-cell,
  signed-volume implementation was restored in `openferro/lattice.py`, after
  which the complete unit suite returned to 40 passing tests.
- `git diff --check HEAD` found one trailing space on a blank line in
  `docs/source/installation.rst`. The space was removed without changing the
  rendered documentation.
- No additional correctness finding remained in the Milestone A code,
  packaging, test, or documentation boundary after the checks below.

## Detailed implementation changes

### Packaging and supported environment

- `pyproject.toml` now declares the `setuptools.build_meta` backend and requires
  `setuptools>=77` plus `wheel` to build the project through PEP 517.
- PEP 621 metadata now supplies the package name and version, README, MIT
  license expression, author, classifiers, and documentation/repository/issue
  URLs in one place. `setup.py` is reduced to a compatibility `setup()` shim.
- Runtime dependencies are limited to `numpy>=2,<3` and `jax>=0.10,<0.12`.
  `pytest` was removed from runtime installation and placed in the `test`
  extra; build/release and Sphinx tools are in separate `package` and `docs`
  extras.
- Setuptools package discovery uses `include = ["openferro*"]`. This replaces
  the old `packages=['openferro']` declaration that omitted
  `openferro.engine` and `openferro.integrator` from wheels.
- Pytest discovery and the `packaging` marker are centralized in
  `pyproject.toml`; there is no separate project-wide formatter or linter
  configuration in this milestone.
- `tests/packaging/test_artifacts.py` copies a clean source tree while excluding
  bytecode, egg-info, build directories, notebook checkpoints, and the
  untracked `archived` directory. It builds both formats, checks that every
  clean-source Python module is present in both archives, installs each
  artifact into a fresh environment, imports every installed OpenFerro module
  from outside the checkout, and executes a default scalar-field smoke test.
- `.readthedocs.yaml` moves the documentation build from Python 3.10 to 3.13.
  `docs/requirements.txt` bounds Sphinx to `>=8,<10` and the Read the Docs
  theme to `>=3,<4`.
- `docs/source/installation.rst` now gives distinct CPU and NVIDIA GPU install
  flows, states the Python/JAX/NumPy contract, records the Perlmutter
  environment, and documents unit, artifact, and import validation commands.

### System and field state

- `System.add_field()` now makes its default `ftype='scalar'` usable. It maps
  `scalar`/`FieldScalar`, `R3`/`FieldR3`, `Rn`, `SO3`, and `LocalStrain3D` to
  the corresponding field classes.
- New scalar fields have shape `(l1, l2, l3, 1)` and zero values; `R3`, `SO3`,
  and local-strain fields use three components. `Rn` requires a positive
  integer `dim`; booleans, omitted dimensions, zero, and negative values are
  rejected.
- Initial values are converted with `jnp.asarray` and broadcast to the complete
  field shape. Field construction, value validation, and mass setup complete
  before the field is inserted into `System._fields_dict`, so failed setup no
  longer leaves a partial registration.
- `_DEFAULT_FIELD_MASS` distinguishes an omitted mass from explicit
  `mass=None`. Omitted non-spin fields retain the historical mass `1.0`;
  omitted SO(3) fields are massless; explicitly passing `None` leaves any
  supported field massless.
- `System.add_global_strain()` validates the reserved `gstrain` field ID,
  reserved `pV` interaction ID, and exact Voigt shape `(6,)` before mutation.
  If pressure registration fails after the field is inserted, both new entries
  are removed. Existing fields or pressure interactions are never overwritten.
- `FieldRn.set_local_value()` validates a lattice-dimension tuple of integer
  indices, accepts in-range negative indices, validates a `(field_dim,)` value
  (or a scalar for a scalar field), performs immutable `.at[loc].set`, and
  restores configured sharding on the result.
- `FieldSO3.__init__()` now starts every site at the positive z-axis with unit
  magnitude instead of exposing zero vectors. `set_values()` and
  `set_local_value()` normalize finite nonzero inputs to the configured site
  magnitude.
- `FieldSO3.set_magnitude()` accepts a finite positive scalar or a lattice-
  shaped magnitude array, rejects zero/negative/non-finite values, reshapes
  orientations to the new magnitude, and keeps magnitude sharding aligned with
  field values. `perturb()` and `normalize()` reject invalid source vectors
  before division.
- `Field.init_velocity()` accepts either a legacy `seed` or an explicit JAX
  `key`. Gaussian initialization now rejects missing temperature or mass, and
  unknown modes raise `ValueError` instead of silently leaving velocity
  unchanged. Velocity arrays are returned to the field sharding.

### Interactions, geometry, and engines

- `triple_interaction` now inherits `interaction_base`, gaining parameter and
  energy-engine storage. Its existing autodiff engine can therefore produce
  three negative energy gradients as forces.
- `System.get_interaction_by_ID()`, `calc_energy_by_ID()`, and
  `calc_force_by_ID()` now recognize the triple-interaction registry and report
  all registered interaction IDs in lookup errors.
- `BravaisLattice3D.reciprocal_latt_vec` now computes each reciprocal vector
  from JAX cross products divided by the signed primitive-cell volume
  `dot(a1, cross(a2, a3))`. Supercell replication counts no longer scale the
  reciprocal basis, and left-handed primitive bases retain their orientation.
- `external_field_energy()` treats `parameters` as exactly one vector with
  shape `(3,)` and evaluates `-sum(field * B_ext)`. The old implementation used
  only `parameters[0]`, which applied one component to every spin component.
- `Dzyaloshinskii_Moriya_energy()` now raises `NotImplementedError` with the
  missing bond/orientation convention instead of returning `None`. It is
  omitted from `openferro.engine.magnetic.__all__` while the implemented
  exchange, anisotropy, and external-field functions remain exported.
- `_dipole_dipole_ewald_setup()` now requires a three-dimensional lattice,
  positive extents and finite `3x3` vectors. Positive axis-aligned
  diagonal cells remain supported; rotated, skew, or negative-axis cells raise
  `NotImplementedError`, and a zero diagonal raises `ValueError`.
- Ewald memory and benchmark helpers convert lattice extents, site counts,
  shapes, and byte counts to Python `int`. Their structural output can now be
  passed directly to `json.dumps` instead of containing JAX scalar objects.

### Simulation control and random state

- `Simulation` now owns `_random_key`. Constructors accept `seed=42` or a
  saved uint32 JAX `PRNGKey`; `reset_random_key()` explicitly restarts a
  stream, and `get_random_key()` exposes immutable state for external restart
  or checkpoint storage.
- `_next_random_keys(count)` splits the current key, retains one new key as
  simulation state, and returns distinct subkeys. `Simulation.init_velocity()`
  gives each field its own subkey, eliminating identical Gaussian velocities
  for equal-shaped fields.
- `SimulationNVE`, `SimulationNVTLangevin`, and `SimulationNPTLangevin` accept
  the same optional `seed` and `key` constructor arguments without changing
  existing calls that pass only a system (and pressure for NPT).
- `SimulationNVTLangevin.run(..., seed=None)` now draws new subkeys for every
  field on every step and advances state across repeated calls. Passing an
  integer `seed` to `run()` is an explicit reset; omitting it continues the
  previous stream.
- `MDMinimize.run()` builds one `active_fields` list and uses it for integrator
  validation and convergence. A fixed-cell run excludes `GlobalStrain` from
  both checks while still allowing force reporting on the frozen strain.
- `MDMinimize` now exposes `converged`, `iterations`, and
  `max_force_by_field`. These values reset at each run, update after every
  iteration, and accompany a warning if `max_iter` is exhausted.
- Only the random key is exposed for restart. Milestone A does not introduce a
  general checkpoint format or promise a trajectory invariant to process or
  device count.

### Documentation and scientific quarantine

- `docs/source/feature_status.rst` is the authoritative 0.1-series matrix. It
  defines `Stable`, `Experimental`, and `Not implemented`, records the tested
  environment, and separates state/API invariants from unvalidated scientific
  dynamics.
- Pressure/volume conventions (COR-09), unresolved Hamiltonian coefficients
  (COR-10), magnetic exchange counting (COR-11), and MD/LLG algorithms
  (COR-12) are explicitly experimental. No numerical formula in those areas
  was changed for Milestone A.
- The matrix identifies `01.BTO_Cooling`, `02.bcc_Fe_Heating`, and
  `03.sc_Ising_Heating` as demonstrations and names their experimental
  dependencies instead of presenting their outputs as validation results.
- `README.md` now calls OpenFerro a research alpha, limits multi-device claims
  to experimental sharding, describes available integrators as experimental,
  and relabels the old GPU plot as a historical illustration rather than a
  reproducible speedup claim.
- `docs/source/api.rst` adds scientific-status warnings and autodoc entries for
  the multiferroic and ferroelectric-superlattice engine modules.
- `docs/source/faq.rst` no longer says long constant-folding warnings are always
  harmless. It directs users to separate compilation from steady execution and
  inspect compile profiles and array placement.
- Documentation-only notation changes in `openferro/integrator/llg.py` and
  `openferro/utilities.py` replace RST-sensitive `*` and `|...|` forms with
  unambiguous shape and norm text. No integrator equations or executable
  statements were changed by that cleanup.

## User-visible behavior changes

| Call or condition | Previous behavior | Milestone A behavior | Rollback group |
| --- | --- | --- | --- |
| `system.add_field("x")` | Raised unknown-field-type despite the default signature. | Creates a zero scalar field with shape `(l1, l2, l3, 1)` and mass 1. | A4 |
| Invalid `Rn` dimension or unbroadcastable value | Could fail after partial setup or through an unclear array error. | Raises a specific `ValueError` before system registration. | A4 |
| Direct `FieldSO3(...)` construction | Initial values were zero vectors. | Initial values point along +z with unit magnitude. | A4 |
| Assigning an SO(3) vector with the wrong norm | Stored the supplied norm and violated `_magnitude`. | Normalizes to the configured site magnitude; zero/non-finite vectors fail. | A4 |
| `field.set_local_value(...)` on a JAX array | Attempted illegal in-place mutation. | Uses immutable indexed update with index/value validation. | A4 |
| Repeated `SimulationNVTLangevin.run()` calls | Recreated seed 42 and replayed the same noise. | Continue the simulation-owned stream; `run(seed=N)` deliberately resets it. | A8 |
| Gaussian velocity initialization for equal-shaped fields | Every field received the same default key. | Each field receives an independent split key. | A8 |
| External magnetic field `[Bx, By, Bz]` | Used only `Bx` and broadcast it over components. | Uses all three components and requires shape `(3,)`. | A6 |
| Rotated or skew cell passed to Ewald | Silently read diagonal entries and could return incorrect energy. | Fails immediately as unsupported. | A5 |
| Dzyaloshinskii-Moriya engine call | Returned `None`. | Raises an explicit `NotImplementedError` and is not wildcard-exported. | A6 |
| Fixed-cell minimization with nonzero strain force | Could fail convergence forever on a degree of freedom it never updated. | Excludes global strain from active-field validation and convergence. | A7 |
| Duplicate or failed global-strain setup | Could overwrite `gstrain` or leave `gstrain` without a valid `pV`. | Rejects duplicates and rolls back partial field/interaction state. | A7 |
| Reciprocal basis for an `l1*l2*l3` supercell | Was divided by the replicated reference volume. | Depends only on the signed primitive-cell volume. | A5 |
| Ewald memory estimate passed to `json.dumps` | Contained JAX scalar metadata and failed serialization. | Contains Python integers and serializes directly. | A5 |

## Regression tests added

| File | Specific coverage |
| --- | --- |
| `tests/packaging/test_artifacts.py` | Clean wheel/sdist construction, complete module membership, fresh-environment installation, all-module imports, and default-field smoke execution. |
| `tests/unit_tests/test_field.py` | Field aliases/defaults, failed-construction non-registration, valid SO(3) defaults, heterogeneous magnitude preservation, zero rejection, perturbation, local immutable update, and one-device sharding. |
| `tests/unit_tests/test_interaction.py` | Triple registration/lookup, scalar energy, three force signs, central finite-difference agreement, and parameter replacement. |
| `tests/unit_tests/test_lattice.py` | Reciprocal duality for orthogonal/rotated/skew/left-handed/BCC/hexagonal bases, supercell independence, Ewald geometry rejection, and JSON metadata. |
| `tests/unit_tests/test_magnetic.py` | Three-component external-field energy and force, ambiguous parameter-shape rejection, and explicit DMI quarantine. |
| `tests/unit_tests/test_rng.py` | Same-seed reproducibility, different-seed divergence, per-field independence, repeated-run continuity, saved-key continuation, and explicit seed reset. |
| `tests/unit_tests/test_simulation.py` | Fixed-cell convergence with frozen strain force and observable nonconvergence state/warning. |
| `tests/unit_tests/test_system.py` | Duplicate/invalid/pre-existing/partially-failed global-strain transactions and exact field calls used by the three reference examples. |

## Explicitly unchanged or deferred

- No pressure-volume equation, strain convention, elastic/multiferroic
  prefactor, or magnetic exchange coordination factor was changed.
- No MD, Langevin, LLG, or SIB numerical update formula was changed or promoted
  to scientifically validated status.
- No maintained reference-example script, material JSON file, reporter format,
  or generated simulation output was changed by Milestone A.
- No GPU, multi-GPU, multi-host, process-index RNG, distributed checkpoint, or
  device-count-invariant trajectory behavior was implemented.
- The unrelated current `.gitignore` modification, tracked deletions under
  `examples/test_BFO/`, and untracked example/development artifacts are outside
  this Milestone A record and should not be staged with it without an explicit
  scope decision.

## Acceptance checklist

| Audit item | Resolution | Regression evidence |
| --- | --- | --- |
| PKG-01 | Setuptools now discovers `openferro*`, including engine and integrator subpackages. | Clean wheel and sdist contents are inspected, installed, and imported outside the checkout. |
| PKG-02 | Added PEP 517/518 and PEP 621 metadata in `pyproject.toml`; retained a thin `setup.py` compatibility shim. | Both artifact formats build through `python -m build`. |
| PKG-04 | Declared Python `>=3.13,<3.15`, JAX `>=0.10,<0.12`, and NumPy `>=2,<3`; aligned Read the Docs and installation documentation. | Validated on Python 3.14.6, JAX 0.11.0, and NumPy 2.5.1. |
| PKG-06 | Artifact tests copy a clean source subset and exclude stale build, cache, archived, and egg-info content. | Wheel and sdist are each installed without dependencies into fresh environments and smoke-tested from an outside working directory. |
| COR-01 | `triple_interaction` inherits the interaction base, and system lookup/error paths include triple interactions. | Energy, all three force signs, parameter updates, lookup, and a finite-difference force check pass. |
| COR-02 | `System.add_field()` supports the documented `scalar`, `R3`, `Rn`, `SO3`, and `LocalStrain3D` forms, including legacy aliases, broadcasting, validation, and mutation-after-validation. | Default and reference-example construction patterns plus failed-construction rollback are tested. |
| COR-03 | Reciprocal vectors use the signed primitive-cell volume instead of supercell volume. | Duality is tested for orthogonal, rotated, skew, left-handed, BCC, and hexagonal bases and multiple supercell sizes. |
| COR-04 | External magnetic fields use and validate one three-component vector. | Directional energy, autodiff force, and invalid-shape behavior are tested. |
| COR-05 | SO(3) fields start valid, normalize assignments to finite positive configured magnitudes, reject zero vectors, keep magnitude sharding, and default to no mass through `System.add_field`. | Direct construction, assignment, local mutation, perturbation, heterogeneous magnitude, invalid input, and one-device sharding are tested. |
| COR-06 | Local field assignment uses JAX `.at[loc].set`, validates indices and value shape, and restores sharding. | Valid, malformed, out-of-range, and sharded assignments are tested. |
| COR-07 | Simulations own a persistent JAX key, split independent keys per field and Langevin step, continue across repeated runs, and accept a seed or saved key. | Same-seed identity, different-seed divergence, field independence, repeated-run continuity, explicit reset, and saved-key restart identity are tested. |
| COR-08 | Ewald setup rejects non-3D, non-finite, degenerate, negative-axis, rotated, and skew geometry instead of silently using diagonal entries. | Rotated and skew cells fail explicitly; supported axis-aligned cells retain the existing Ewald tests. |
| COR-14 | Fixed-cell minimization builds one active-field list, excludes frozen global strain from validation and convergence, exposes convergence state, and warns on exhaustion. | Frozen-strain convergence and explicit nonconvergence reporting are tested. |
| COR-15 | Global-strain creation validates `gstrain`, `pV`, and shape before mutation and rolls back if pressure registration fails. | Duplicate, invalid-shape, pre-existing-pressure, and injected partial-registration failures preserve prior state. |
| COR-18 | Dzyaloshinskii-Moriya energy raises `NotImplementedError` and is excluded from magnetic wildcard exports. | Explicit failure and export status are tested. |
| COR-20 | Ewald sizes, site counts, shapes, and byte counts are Python integers. | The complete memory estimate round-trips through JSON. |
| COR-09 through COR-12 | Pressure/volume, unresolved Hamiltonian coefficients, exchange counting, and bundled integrators are labeled experimental rather than changed without references. | `docs/source/feature_status.rst`, API warnings, README wording, and example status entries define the quarantine. |
| Feature matrix | Added one authoritative research-alpha matrix with stable, experimental, and not-implemented definitions and a tested environment table. | A Sphinx warnings-as-errors build succeeds. |

## Reference-example compatibility

`01.BTO_Cooling` continues to use explicit `Rn`, `LocalStrain3D`, and global
strain construction, followed by NPT construction, Gaussian velocity
initialization, and repeated `run()` calls. The first stochastic sequence still
defaults to seed 42. Later calls now continue the stream instead of replaying
it.

`02.bcc_Fe_Heating` and `03.sc_Ising_Heating` continue to construct `SO3`
fields, set their magnitudes, create `SimulationNVTLangevin` with only a
system argument, and call `run()` repeatedly. Spin values are normalized to
the configured magnitude, and repeated calls receive fresh keys.

No tracked or untracked file below those three example directories was changed
as part of this milestone.

## Rollback groups

The implementation is organized into the independent groups below even if the
groups are published in one commit. To revoke one behavior, reverse only that
group's files or shared-file hunks and its corresponding tests and status
claims; a whole-commit revert is not required.

| Group | Purpose | Files and anchors |
| --- | --- | --- |
| A1 build artifacts | PKG-01/02/06 and ancillary dependency metadata | New `pyproject.toml`; `setup.py`; new `tests/packaging/test_artifacts.py`. |
| A2 version contract | PKG-04 and install environment alignment | `.readthedocs.yaml`; `docs/requirements.txt`; supported-version and install sections in `docs/source/installation.rst`; environment table in `docs/source/feature_status.rst`. |
| A3 triple interactions | COR-01 | `triple_interaction` in `openferro/interaction.py`; triple lookup/error hunks in `openferro/system.py`; new `tests/unit_tests/test_interaction.py`. |
| A4 field construction and invariants | COR-02/05/06 | `_DEFAULT_FIELD_MASS` and `System.add_field` in `openferro/system.py`; local assignment and SO(3) methods in `openferro/field.py`; new `tests/unit_tests/test_field.py`; reference-construction test in `tests/unit_tests/test_system.py`. |
| A5 lattice and Ewald safety | COR-03/08/20 | `reciprocal_latt_vec` in `openferro/lattice.py`; setup and Python metadata conversions in `openferro/engine/ewald.py`; new `tests/unit_tests/test_lattice.py`. |
| A6 magnetic safety | COR-04/18 | `__all__`, DMI failure, and external-field validation in `openferro/engine/magnetic.py`; new `tests/unit_tests/test_magnetic.py`. |
| A7 minimization and strain transaction | COR-14/15 | `MDMinimize` state/run hunks in `openferro/simulation.py`; `add_global_strain` in `openferro/system.py`; new `tests/unit_tests/test_simulation.py`; transaction tests in `tests/unit_tests/test_system.py`. |
| A8 persistent RNG | COR-07 | `Field.init_velocity` and the SO(3) no-op signature in `openferro/field.py`; random-key methods and NVE/NVT/NPT signatures/run logic in `openferro/simulation.py`; new `tests/unit_tests/test_rng.py`; random-stream row in the feature matrix. |
| A9 scientific quarantine | COR-09/10/11/12 and honest public claims | New `docs/source/feature_status.rst`; warnings/entries in `docs/source/api.rst` and `docs/source/index.rst`; research-alpha, integrator, and benchmark wording in `README.md`; installation warnings. |
| A10 strict docs cleanup | Sphinx `-W` compatibility | Documentation-only notation edits in `openferro/integrator/llg.py` and `openferro/utilities.py`; FAQ underline and constant-folding guidance in `docs/source/faq.rst`; missing engine autodoc entries in `docs/source/api.rst`. |

### Shared-file dependencies

- `openferro/system.py` contains A3, A4, and A7. Reverse selected hunks, not the
  whole file, when removing only one group.
- `openferro/field.py` contains A4 and A8. The simulation RNG code requires the
  `key` parameter added to `Field.init_velocity` and its SO(3) override.
- `openferro/simulation.py` contains A7 and A8. Minimization state reporting is
  independent of the random-key methods and NVE/NVT/NPT signature changes.
- `tests/unit_tests/test_system.py` contains both A4 compatibility and A7
  transaction tests.
- The A1 artifact smoke test calls `System.add_field("x")`. If A4 is revoked,
  either revoke that smoke assertion or make its field type explicit.
- Status claims in A2/A8/A9 must be updated whenever the corresponding code or
  tests are revoked.

For tracked files, inspect `git diff -- <path>` and use interactive hunk
restoration such as `git restore -p -- <path>`. Do not restore an entire shared
file to revoke one group. New files are group-owned as listed above; remove a
new file only when its whole group is being revoked. Do not use `git clean`,
because the worktree contains unrelated untracked user files.

After a selective reversal, run that group's focused tests and then the full
unit suite. Reversing A1 or A4 also requires rerunning the artifact test.

## Validation record

The complete lightweight validation set was rerun during the pre-publish review
on 2026-08-06 after restoring the reciprocal-lattice implementation and fixing
documentation whitespace.

Validation environment: Python 3.14.6, JAX 0.11.0, NumPy 2.5.1, CPU backend.

- `JAX_PLATFORMS=cpu python -m pytest tests/unit_tests -q`: 40 passed in 13.75 s.
- `JAX_PLATFORMS=cpu python -m pytest tests/packaging/test_artifacts.py -q`:
  1 passed in 14.32 s; both clean artifacts were installed and imported outside
  the checkout.
- `python -m sphinx -W -E -b html docs/source /tmp/openferro-docs-build-werror`:
  succeeded with all 14 sources rebuilt.
- `python -m compileall -q openferro tests`: succeeded.
- `python -c "import openferro as of; print(of.System)"`: printed
  `<class 'openferro.system.System'>`.
- `git diff --check`: succeeded.

No heavy simulation was run on a login node. No GPU, multi-GPU, or
multi-host allocation was used, so those paths remain explicitly experimental.
