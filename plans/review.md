# OpenFerro codebase review

- Reviewed: 2026-07-21
- Branch/commit: `multinode` at `214f72e`
- Environment used for lightweight checks: Della login node, Python 3.14.6, JAX 0.11.0, one visible GPU

## Executive assessment

OpenFerro has a clear and useful architecture, but the current branch should be treated as a research/development snapshot rather than a release-ready large-scale simulator. There are confirmed correctness defects in public paths, an ordinary wheel omits most of the package, scientific prefactors remain explicitly unresolved, stochastic state is not continuous across runs, and multi-host execution lacks the initialization, I/O, restart, and end-to-end tests needed for production use.

The present large-scale bottleneck is workload-dependent:

| Workload | Current limiting path | Why | First remedy |
| --- | --- | --- | --- |
| Ferroelectric models with dipole Ewald | **Steady state: global 3-D FFT communication/redistribution and associated memory traffic/workspaces. Startup: Ewald-kernel construction plus JIT/autodiff compilation.** | Every force evaluation differentiates an FFT-based scalar energy over the full lattice. The spatial array is sharded only on axes 0 and 1, while the FFT spans axes 0, 1, and 2. Strong-scaling data in a local profiling script reaches only 2.68x on four A100s for a 250³ case (67% parallel efficiency), although the benchmark itself needs repair before it is authoritative. | Establish a synchronized interaction-level benchmark, implement an explicit analytic Ewald force, control and verify kernel sharding, cache/precompute the kernel, then profile alternative FFT mesh layouts. |
| Short-range MD without Ewald | Repeated full-array interaction kernels, `jnp.roll` communication, separate JIT dispatches, and Python step loops | Each interaction is differentiated and accumulated separately; no whole-Hamiltonian or multi-step fusion is used. | Compile a pure total-energy/force function over a state PyTree, reduce intermediates, and execute reporting-sized chunks with `lax.scan`. |
| Spin/SIB dynamics | Repeated global force evaluations plus a device-to-host convergence synchronization on every fixed-point iteration | SIB calls the force updater twice per field per step and tests a JAX scalar in a Python `if`. Multiple spin fields are advanced sequentially. | Put the fixed-point solve in `lax.while_loop`, update coupled spin fields together, and avoid redundant force calculations. |

For float32 Ewald alone, the arrays counted by `estimate_dipole_dipole_ewald_memory` require at least 84 bytes/site: field (12), six-component kernel (24), field FFT (24), and kernel-applied FFT (24). That is about 1.31 GiB at 256³ before forces, velocities, strain fields, autodiff residuals, FFT workspaces, compilation temporaries, and other interactions. Consequently, memory capacity and memory/collective bandwidth are at least as important as arithmetic throughput.

The most urgent order of work is:

1. Fix the clean-wheel failure and the confirmed public-API correctness defects.
2. Add small analytical and finite-difference tests for every claimed Hamiltonian/integrator; quarantine terms whose prefactors are still unresolved.
3. Make stochastic state, checkpoint/restart, and process-safe reporting reliable.
4. Create a trustworthy synchronized performance baseline before optimizing Ewald and distributed execution.
5. Complete multi-host initialization/topology handling and run real 2/4-GPU and two-node correctness tests.
6. Reconcile and complete the documentation only after the supported feature matrix is explicit.

Priority labels used below:

- **P0**: release/scientific blocker; fix before relying on results or publishing a package.
- **P1**: required for dependable large-scale production runs.
- **P2**: maintainability, usability, or completeness improvement.
- **Confirmed** means reproduced or directly determined from an executable path. **Risk** means the code needs a reference test or profiler measurement before deciding the exact fix.

## Audit evidence and limitations

Lightweight validation performed:

- `python -m pytest tests/unit_tests -q`: **9 passed** in 17.17 s. These are four Ewald tests and five mostly one-device mesh/sharding tests.
- `python -m compileall -q openferro tests`: passed.
- Targeted CPU probes reproduced the broken triple-interaction constructor, immutable `set_local_value`, reciprocal-vector scaling error, magnetic external-field indexing error, SO(3) magnitude inconsistency, correlated field velocity initialization, Ewald JSON serialization failure, and clean-wheel package omission.
- A wheel built from a clean Git archive contained root `openferro/*.py` only; `openferro/engine/*` and `openferro/integrator/*` were absent.
- An earlier in-tree build could accidentally include stale ignored files from `build/lib`, demonstrating why release validation must use a clean source tree.

Not performed:

- No heavy simulation was run on the login node.
- No real 2/4-GPU or multi-node job was allocated, so communication costs, numerical parity, distributed FFT layout, and multi-process I/O remain unverified.
- Sphinx is not installed in the active environment, so documentation was inspected but not built. The current Read the Docs configuration is independently incompatible with `python_requires`.
- Untracked user files and generated outputs were inspected only when useful as audit evidence and were not modified.

## 1. Correctness and scientific-integrity findings

### COR-01 — Triple interactions are unusable (**P0, Confirmed**)

`openferro/interaction.py:225` defines `triple_interaction` without inheriting `interaction_base`, but its constructor calls `super().__init__(parameters)`. Construction raises `TypeError`, and the class therefore also lacks `set_parameters` and `set_energy_engine`. In addition, `System.get_interaction_by_ID()` checks only the self and mutual registries even though energy/force code tries to retrieve triple interactions through it.

**Solution:** inherit from `interaction_base`, include the triple registry in lookup/error messages, and add constructor/energy/force/parameter-update tests. Consider replacing three parallel registries with one typed interaction registry so lookup behavior cannot diverge again.

### COR-02 — The default `System.add_field()` call is invalid (**P0, Confirmed**)

The signature defaults to `ftype='scalar'`, but the implementation accepts only `Rn`, `SO3`, and `LocalStrain3D`. Calling `system.add_field("x")` raises “Unknown field type.” The documented `FieldScalar` and `FieldR3` shortcuts also cannot be selected through this method.

**Solution:** either make `Rn` the explicit default with a required `dim`, or support stable names such as `scalar`, `R3`, `Rn`, `SO3`, and `LocalStrain3D` through a class registry. Validate `dim` and initial-value shape before mutating the system.

### COR-03 — 3-D reciprocal lattice vectors are scaled by the supercell volume (**P0, Confirmed**)

`BravaisLattice3D.reciprocal_latt_vec` divides by `ref_volume`, which includes `l1*l2*l3`, instead of the primitive-cell signed volume. For a 2×2×2 identity lattice it returns diagonal values `pi/4` rather than `2*pi`.

**Solution:** compute `b_i = 2*pi * cross(a_j, a_k) / dot(a1, cross(a2, a3))`, preserving orientation as appropriate. Test `a_i dot b_j = 2*pi*delta_ij` for orthogonal, rotated, skew, BCC, and hexagonal primitive vectors and for multiple supercell sizes.

### COR-04 — Vector magnetic fields are reduced to their first component (**P0, Confirmed**)

`openferro/engine/magnetic.py:68` assigns `B_ext = parameters[0]`. Passing `[Bx, By, Bz]` therefore applies scalar `Bx` to all spin components; a unit spin with `[1,2,3]` gives `-3` instead of `-6`.

**Solution:** define and validate one parameter convention, preferably `parameters.shape == (3,)`, and use the whole vector. Add direction-specific energy and force tests.

### COR-05 — SO(3) magnitude invariants are not enforced (**P0, Confirmed**)

Adding an SO(3) field with initial value `[0,0,2]` stores norm 2 while `_magnitude` remains 1. `Field.set_values()` can subsequently violate the invariant, and `normalize()` divides by zero for zero vectors. The default `mass=1.0` also allocates an unnecessary site-sized mass for an otherwise massless spin field.

**Solution:** give `FieldSO3` a validated override of `set_values`; either normalize to the configured magnitude or reject mismatched/zero input. Make spin mass default to `None`, keep magnitude sharded with values, and test initialization, mutation, perturbation, zero input, heterogeneous magnitudes, and integrator norm preservation.

### COR-06 — Local field assignment uses illegal JAX mutation (**P0, Confirmed**)

`FieldRn.set_local_value()` performs `self._values[loc] = value`, which fails because JAX arrays are immutable.

**Solution:** use `self._values = self._values.at[loc].set(value)`, validate index/value shape, and verify sharding is preserved.

### COR-07 — Random streams are correlated and replayed (**P0, Confirmed**)

`Simulation.init_velocity()` calls every field with the same default seed, producing identical Gaussian velocities for equal-shaped fields. `SimulationNVTLangevin.run()` recreates `PRNGKey(seed=42)` on every call; examples that call `run()` repeatedly replay the same noise sequence each time.

**Solution:** make the simulation own a persistent JAX key, split once per field and step, and update it after every use. Accept an initial key/seed at construction or explicit reset method. Fold in process index only according to a documented global-reproducibility policy. Include RNG state in checkpoints. Test same-seed reproducibility, different-field independence, repeated-run continuity, and restart equivalence.

### COR-08 — Ewald silently assumes axis-aligned orthogonal vectors (**P0, Confirmed limitation**)

`_dipole_dipole_ewald_setup()` reads only `latt_vec[0,0]`, `[1,1]`, and `[2,2]`; the prior orthogonality rejection is commented out. Rotated/skew lattices, BCC primitive vectors, and some hexagonal inputs therefore produce incorrect values or division by zero while appearing supported.

**Solution:** immediately validate and reject unsupported geometry with a precise error, then separately implement a general reciprocal-metric formulation using the primitive-cell volume and reciprocal basis. Add rotation-invariance and skew-cell reference tests before advertising general Bravais support.

### COR-09 — Pressure and reported volume use an undocumented linearized formula (**P0 for NPT science, Confirmed**)

`pV` and `Thermo_Reporter` use `(1 + eta1 + eta2 + eta3) * Vref`, while a source TODO proposes a product. Neither is the full determinant for general strain, and shear is ignored. The energy term and reported volume must use the same stated strain convention.

**Solution:** derive `F(eta)` from the package's Voigt/shear convention and compute `V = det(F)*Vref`, or explicitly name and bound a small-strain approximation. Verify pressure derivatives against finite differences and analytical hydrostatic/uniaxial/shear cases.

### COR-10 — Several Hamiltonian coefficients are explicitly unresolved (**P0, Confirmed**)

Release-critical TODOs remain in scientific kernels: `B44/8` versus `/4` in elastic energy, a factor 8 and other prefactors in multiferroic terms, coordination-number divisions, and multiple possible factors 0.5 for diagonal neighbor terms. A model-conversion script also questions spin units. These uncertainties can change physical results by order-one factors.

**Solution:** mark these engines experimental and prevent them from being presented as validated until each term has a written derivation, source equation/DOI, unit conversion, neighbor-count convention, and analytical/reference test. Store the convention and provenance next to each parameter schema rather than only in comments.

### COR-11 — Exchange bond-count convention is ambiguous (**P0 for magnetic models, Risk**)

The lattice rollers enumerate half a neighbor shell (each bond is intended to appear once), while `get_isotropic_exchange_energy_engine` multiplies the supplied coupling by two. Documentation states `-J sum_<ij>` without explaining this extra factor. Tiny periodic lattices can also alias the same neighbor through multiple rollers.

**Solution:** choose a single convention—unique undirected bonds is clearest—document it, and hand-count energies on 1-D-like, 2×2×2, and nondegenerate cells. Add guards or documented behavior for cells smaller than a neighbor displacement.

### COR-12 — MD and LLG algorithms lack reference validation (**P0/P1, Risk**)

The “LeapFrog” update is a kick followed by drift; correctness depends on velocities being interpreted as half-step values, but Gaussian initialization does not establish/document that convention. The Langevin implementation's stated splitting and actual operation order need an equilibrium reference test. In deterministic `LLSIBIntegrator`, the damping-field update for step 2 uses the original `M` even though the force was evaluated at the midpoint; the stochastic variant uses the midpoint. This inconsistency needs resolution against the cited algorithm.

**Solution:** write down the exact discrete state convention, compare one step to a reference implementation, and test convergence order, time reversibility where applicable, NVE energy drift, Langevin equipartition, and LLG norm/precession/damping. Fix the midpoint inconsistency only after the reference establishes the intended formula.

### COR-13 — Multiple SO(3) fields are advanced sequentially and order-dependently (**P1, Risk**)

Each spin field's SIB step invokes a global force update and mutates that field before the next spin field is advanced. Coupled spin sublattices can therefore see different time levels depending on dictionary insertion order.

**Solution:** represent all coupled spin fields in one integrator state and solve the implicit stage simultaneously, or explicitly document a splitting scheme and its order. Test field-order invariance or the expected splitting error.

### COR-14 — Fixed-cell minimization can check a frozen strain force forever (**P1, Confirmed by control flow**)

When a global strain exists and `variable_cell=False`, the step skips its update, but convergence still loops over all fields, including the frozen global strain. Nonzero frozen-cell stress can prevent convergence even when every active degree of freedom is converged.

**Solution:** construct the active-field list once, use it for integrator validation and convergence, report the final maximum force by field, and explicitly report non-convergence at `max_iter` rather than returning silently.

### COR-15 — Adding global strain is not transactional (**P1, Confirmed by control flow**)

`add_global_strain()` overwrites an existing `gstrain` field and then calls `add_pressure()`. If `pV` already exists, pressure insertion fails after the field has already changed, leaving an inconsistent system.

**Solution:** validate both reserved IDs before any mutation; reject duplicates or provide an explicit replace/update method. Add rollback/duplicate tests.

### COR-16 — Input validation is too weak for simulation state (**P1, Confirmed**)

Examples include nonpositive/float lattice sizes, degenerate vectors, arbitrary field shapes, zero mass even though integrators divide by mass, unknown velocity modes that silently do nothing, missing/negative temperature, invalid `dt/tau/alpha/tol/max_iter`, zero reporter intervals, and incompatible mesh divisibility. Several checks use Python `assert`, which disappears under `python -O` and can force device synchronization for JAX scalars.

**Solution:** validate Python-side configuration at object boundaries with `ValueError`/`TypeError`, use explicit expected shapes/dtypes/units in messages, require strictly positive masses for inertial integrators, and add negative tests for every public constructor/setter.

### COR-17 — Public incomplete features overstate supported dimensionality (**P1**)

`FaceCenteredCubic3D`, `Hexagonal2D`, `RingPolymerSystem`, and `OverdampedLangevinIntegrator` raise `NotImplementedError`; `Hexagonal3D` has no neighbor-shell rollers; nonperiodic boundaries are not implemented. Although 2-D lattice classes exist, `System.add_field()` hardcodes three spatial sizes and many engines hardcode axes `(0,1,2)`. Documentation names a nonexistent `Square2D` and says the same field strategy applies to 2-D.

**Solution:** publish a tested feature-status table. Remove incomplete symbols from the stable top-level API or label them experimental. Either make field/system/engine shapes dimension-generic and test 2-D end to end, or explicitly restrict `System` to 3-D.

### COR-18 — The Dzyaloshinskii–Moriya engine returns `None` (**P1**)

`Dzyaloshinskii_Moriya_energy` contains only `pass`, yet is imported through wildcard engine imports and appears like a callable feature.

**Solution:** implement it with a declared bond/orientation convention and tests, or replace it with an immediate `NotImplementedError` and remove it from public exports.

### COR-19 — Explicit force-engine support is promised but has no API (**P1**)

The interaction module says force engines may be explicitly supplied, but exposes only autodiff `create_force_engine`; users must mutate `force_engine` directly.

**Solution:** add validated `set_force_engine()` methods with clear self/mutual/triple signatures, sign convention, JIT option, and explicit-vs-autodiff parity tests.

### COR-20 — Ewald profiling output is not JSON serializable (**P2, Confirmed**)

Lattice sizes are JAX arrays, so `estimate_dipole_dipole_ewald_memory()` returns JAX scalar values in `shape`, `nsites`, and byte counts. `tests/profiling/ewald_baseline.py` fails at `json.dumps()`.

**Solution:** keep structural metadata such as lattice sizes and byte counts as Python `int`, reserve JAX arrays for numerical kernels, and test benchmark JSON serialization.

## 2. Large-scale performance findings

### PERF-01 — Global FFT communication is the dominant Ewald scaling ceiling (**P1, high-confidence design finding**)

`jnp.fft.fftn(field, axes=(0,1,2))` operates across the full domain while `PartitionSpec('x','y')` partitions the first two dimensions. Distributed FFTs require one or more redistributions/transposes; strong scaling worsens as local work shrinks and collective latency/bandwidth dominate. The local `benchmark.py` records 75.15, 44.90, and 28.06 seconds per 500 steps on 1/2/4 A100s for 250³: speedups 1.67x and 2.68x, efficiencies 84% and 67%. Treat these figures as provisional because the file lacks environment metadata, repeated trials, synchronization proof, and error bars.

**Solution:** profile with JAX/XProf or Nsight on allocated nodes; report FFT/transpose time, per-interaction force time, bytes/site, peak memory, compile time, and steady-state time. Compare 1-D and 2-D mesh layouts and weak scaling. Do not claim the exact limiting collective until traces confirm it.

### PERF-02 — Ewald kernel construction is an expensive uncached startup stage (**P1**)

`get_UkGG()` uses three Python loops over reciprocal replicas and runs full-grid JIT kernels for every offset. With the typical `(5,5,5)` bounds, the half-open loops execute about 1,000 full-grid accumulation passes. The resulting large array is captured in the returned energy closure, which can increase constant handling/compilation cost. Every call rebuilds it.

**Solution:** make kernel setup an explicit, timed object; cache by lattice vectors/shape/dtype/Ewald tolerance/sharding; optionally persist a versioned kernel file; replace host loops with a memory-bounded compiled/chunked reduction; and pass the kernel as an array argument rather than an opaque closure constant where that improves compilation. Benchmark setup separately from first compile and steady state.

### PERF-03 — Ewald forces use generic reverse-mode autodiff instead of the analytic linear operator (**P1**)

The energy is quadratic in the field, so the force can be evaluated directly by applying the reciprocal kernel and an inverse/adjoint FFT plus the real-space self term. Differentiating the scalar energy can retain extra complex intermediates and adds compile/memory overhead.

**Solution:** implement an explicit force engine, derive normalization/sign carefully for JAX FFT conventions, and compare it against `jax.grad`, finite differences, and brute-force small-cell references for float32/float64. Measure peak memory and time before making it the default.

### PERF-04 — Kernel/input sharding is not propagated by the public System API (**P1**)

`System.add_dipole_dipole_interaction()` calls `get_dipole_dipole_ewald(field.lattice)` without the field dtype or `field._sharding`. `move_fields_to_multi_devs()` moves only field-owned arrays, not interaction state or already-created kernels. The actual placement of the captured kernel is therefore implicit and unasserted.

**Solution:** make array placement part of interaction construction/state, propagate dtype and sharding, provide a system-wide `to_devices()` that covers fields and interactions, and inspect `addressable_shards`/compiled shardings in tests to prevent accidental replication.

### PERF-05 — Force evaluation is fragmented across interactions (**P1**)

Every interaction owns its own JIT/autodiff function; `System.update_force()` zeros arrays, dispatches each term, and performs another full-array accumulation. This repeats reads, temporaries, neighbor rolls/collectives, and dispatch overhead.

**Solution:** expose a pure `energy_terms(parameters, state)` plus `total_energy`, differentiate/fuse the total where memory permits, and return one force PyTree. Retain term-level functions for reporting/debugging. Benchmark fused, partially fused, and current forms because one giant compile may itself be costly.

### PERF-06 — The Python time-step loop prevents temporal fusion (**P1**)

Simulation loops dispatch every field/integrator operation for every step from Python. This becomes material as per-device work decreases, and it prevents buffer donation and multi-step compiler optimization.

**Solution:** separate immutable configuration from a JAX state PyTree and compile a pure step. Run chunks with `lax.scan`; return only report/checkpoint samples at chunk boundaries. Keep file I/O outside JIT and choose chunk size based on reporting and failure recovery.

### PERF-07 — Neighbor rolls can become repeated collectives (**P1**)

Short-range engines use many `jnp.roll` calls across sharded axes. These can require boundary exchange or redistribution for each term. A third-neighbor superlattice engine still uses a `jnp.dot` pattern that the standard ferroelectric engine notes was replaced because reshape/XLA behavior was unsuitable for sharded arrays.

**Solution:** inspect lowered HLO/communication traces, reuse rolled neighbor views across compatible terms, consider explicit halo exchange/domain decomposition for local Hamiltonians, and port the sharding-safe componentwise contraction to every related engine. Add numerical parity tests for sharded neighbor shells.

### PERF-08 — SIB convergence checks synchronize every iteration (**P1**)

The Python condition `if normalized_diff_avg < tol` reads a device scalar each fixed-point iteration. Each SO(3) field also triggers two full-system force evaluations per time step.

**Solution:** use `lax.while_loop` or a fixed bounded loop with a returned convergence flag, aggregate warnings outside JIT, and solve coupled fields together. Report iteration distributions and nonconvergence rates in profiling.

### PERF-09 — Reporting adds synchronous compute and I/O to the critical path (**P1**)

Converting arrays/scalars to strings forces host transfers. Potential-energy reporting recomputes all interactions, including an extra FFT. Full-field `jnp.save` materializes large arrays synchronously.

**Solution:** reuse energies produced by the step where possible; reduce scalars on device; transfer only process-owned/checkpoint data; buffer or asynchronously write through a bounded queue; and report I/O time separately. Ensure queue backpressure and errors are visible.

### PERF-10 — Existing benchmark scripts cannot support performance claims (**P1**)

Tracked profiling files are scripts, not regression tests. The local 256³ script times `jax.block_until_ready(simulation.run(...))`, but `run()` returns `None`, so that call does not synchronize the simulation's final array. Static results omit commit, JAX/jaxlib/CUDA/NCCL versions, clocks, mesh, precision, compilation policy, repetitions, peak memory, and uncertainty.

**Solution:** create a tracked benchmark CLI that blocks on a real final array, separates setup/compile/warmup/steady-state/I/O, emits JSON with environment metadata, and provides strong- and weak-scaling plots from raw data. Establish a baseline before optimizing and use tolerant regression thresholds on dedicated hardware.

## 3. Multi-device, multi-host, and HPC reliability

### HPC-01 — Multi-host initialization is absent (**P1**)

There is no library helper, entry-point documentation, or example that calls `jax.distributed.initialize()` before devices or computations are accessed. Official JAX documentation requires distributed initialization before JAX computations for multi-host use.

**Solution:** add an explicit launcher/bootstrap layer that initializes JAX first, validates coordinator/process IDs/counts, logs the global topology once, and shuts down cleanly. Provide Slurm templates for Perlmutter and Della. See the official [JAX distributed initialization documentation](https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html) and [multi-process guide](https://docs.jax.dev/en/latest/multi_process.html).

### HPC-02 — Mesh creation is single-host and not topology-aware (**P1**)

`DeviceMesh` describes itself as single-host, converts `jax.devices()` to an array, and reshapes it into a 2-D mesh. It has no process/topology assertions, array-dimension divisibility checks, or mesh-axis policy. Simple reshaping is not generally reliable for multi-slice/multi-host topology.

**Solution:** distinguish single-host and distributed mesh construction; use JAX mesh utilities appropriate to the hardware topology (including hybrid meshes where needed); validate global/local devices and array divisibility; and expose named layout policies rather than one fixed `('x','y')` choice. The official [JAX multi-process guide](https://docs.jax.dev/en/latest/multi_process.html) discusses topology-aware hybrid mesh construction.

### HPC-03 — Global arrays are first built on each process and only later resharded (**P1**)

Fields and Ewald grids are created as complete `jnp` arrays before `device_put`. At large size this can duplicate host/device memory on every process and does not define how process-local data becomes a global array.

**Solution:** construct distributed arrays directly from process-local shards (`make_array_from_process_local_data` or equivalent), avoid materializing global initial conditions on every host, and document which state is replicated versus partitioned. Test peak host/device memory per process.

### HPC-04 — Reporters are not multi-process safe (**P1**)

Every process executes the same Python reporter code and opens the same path. This risks duplicate headers, interleaved/corrupt logs, duplicate dumps, and invalid host materialization of non-addressable global arrays.

**Solution:** process-zero gate scalar output after explicit reductions; use per-process shard filenames or a coordinated global checkpoint format for arrays; add atomic write/rename and schema/version metadata; and test concurrent two-process reporting.

### HPC-05 — There is no checkpoint/restart facility (**P1**)

Long Slurm jobs cannot reliably survive wall-time limits or failures. Required state includes field values, masses/velocities/forces as appropriate, SO(3) magnitudes, global strain, integrator parameters/state, RNG key, step/reporter counters, mesh metadata, interaction/config version, and provenance.

**Solution:** define a versioned checkpoint schema with atomic completion markers, configurable cadence, process-local/sharded writes, and restart compatibility checks. Test uninterrupted versus split-run equivalence and recovery from an incomplete checkpoint.

### HPC-06 — Distributed reproducibility is unspecified (**P1**)

There is no policy for whether the same global trajectory should be invariant to device/process count, how process index enters keys, or how random arrays are generated with global shardings.

**Solution:** document the reproducibility contract, derive keys from `(base_key, global_step, field_id, stream_id[, process/shard])`, generate global random arrays with the intended sharding, and test at two device counts.

### HPC-07 — No end-to-end distributed validation exists (**P1**)

Current parallel tests use available devices and generally exercise one-device placement. They do not run a real force, FFT, integrator, reporter, or restart across multiple devices/processes.

**Solution:** add layers: simulated multi-device CPU tests in CI if possible, 2/4-GPU scheduled tests on one node, and a two-node eight-GPU Slurm smoke/regression job. Assert energy/force parity, finite outputs, sharding, no duplicate I/O, and restart equivalence.

## 4. Tests missing from `/tests`

The current suite is far too narrow for a scientific dynamics package: nine unit tests cover mainly one Ewald implementation and mesh-shape helpers. The profiling scripts have no assertions and one currently fails JSON serialization. There is no CI configuration, coverage target, shared fixtures, test markers, or supported-version matrix.

The following are the key missing test groups, in recommended implementation order:

| Priority | Test module/group | Minimum cases and acceptance criteria |
| --- | --- | --- |
| P0 | `test_interaction.py` | Construct self/mutual/triple interactions; set/update parameters; explicit and autodiff force engines; registry lookup; duplicate IDs; missing engines/fields; force sign and output shapes. |
| P0 | `test_lattice.py` | Reciprocal duality for orthogonal/rotated/skew/BCC/hexagonal vectors; positive integer sizes and nondegenerate vectors; exact neighbor-shell counts/displacements; small-periodic-cell alias behavior; unsupported PBC/features fail clearly. |
| P0 | `test_field.py` | Every field type and `System.add_field` spelling/default; shape/dtype validation; immutable local update; mass validation; SO(3) normalization/zero handling/magnitude preservation; single- and multi-device sharding preservation. |
| P0 | `test_engine_forces.py` | For **every** advertised engine, compare `-grad(E)` with central finite differences in float64 on tiny non-symmetric fields. Check translational, inversion, rotation, permutation, and lattice symmetry as physically applicable. |
| P0 | `test_engine_reference.py` | Hand-computed onsite/exchange/elastic/electric/magnetic energies; bond counts; pressure derivative; published reference configurations for ferro-, magneto-, multiferroic, and superlattice terms. Do not enable unresolved engines until they pass. |
| P0 | `test_ewald.py` expansion | Brute-force/reference agreement over several sizes/configurations; zero/uniform/random fields; force finite differences; analytic-vs-autodiff force; dtype/JIT parity; supported geometry validation; sharded parity; memory-estimate Python types/JSON. A 1% single-case tolerance is not enough. |
| P0 | `test_rng.py` | Same seed reproducibility, independent streams by field, repeated `run()` continuity, different seed divergence, checkpoint restart identity, and documented device-count behavior. |
| P0 | `test_integrator_md.py` | One-step reference equations, convergence order, NVE energy drift, harmonic-oscillator period, zero-force motion, Langevin equipartition/temperature distribution, invalid parameters, and strain masks. |
| P0 | `test_integrator_llg.py` | Constant-field precession, damping toward the field, norm preservation over long runs, zero-field stability, SIB convergence/nonconvergence, stochastic equilibrium, midpoint convention, and coupled-field ordering. |
| P1 | `test_simulation.py` | NVE/NVT/NPT/minimization lifecycle; active vs frozen fields; convergence and max-iteration reporting; missing integrators; pressure setup; repeated runs; reporter counters; empty systems. |
| P1 | `test_reporter.py` | Parent-directory creation, valid intervals, headers/schema, scalar values, volume convention, append/restart behavior, dump loadability, I/O failure propagation, process-zero/per-rank behavior. |
| P1 | `test_parallel_multidevice.py` | Real energy, force, and integrator parity on 1/2/4 devices; expected `NamedSharding`; kernel not accidentally replicated; nondivisible shapes rejected; neighbor rolls and Ewald FFT run correctly. |
| P1 | multi-host Slurm test | Initialize JAX before use; run a tiny two-node model; compare with one-device reference; verify collective order, I/O uniqueness, checkpoint/restart, and clean shutdown. Keep this scheduled, not a login-node test. |
| P1 | packaging tests | Build sdist/wheel from a clean archive; inspect package list; install into a clean environment; import `openferro`, every engine/integrator, and run a tiny energy/force calculation. Run `twine check`. |
| P1 | documentation tests | `sphinx-build -W`, linkcheck, doctest/copy-paste smoke tests for snippets, and notebook execution in a small CPU mode. |
| P1 | example smoke tests | Parameterized tiny lattice/steps, working-directory independence, no pre-existing output directory, deterministic seed, and CPU-only execution for every maintained example family. |
| P1 | performance regressions | Synchronized setup/compile/steady-state timings, peak memory, interaction breakdown, and strong/weak scaling. Store raw JSON and compare only on controlled hardware. |
| P2 | property/error tests | Hypothesis-style shapes/parameters where useful, NaN/Inf handling, invalid IDs/types, PyTree parameters, serialization round trips, and error-message quality. |

Test infrastructure recommendations:

- Add `pyproject.toml` pytest configuration, `conftest.py` fixtures for tiny lattices/fields and x64 mode, and markers such as `unit`, `gpu`, `multigpu`, `multihost`, `slow`, and `benchmark`.
- Keep deterministic unit tests CPU-small; schedule accelerator tests separately through Slurm.
- Enable `jax_enable_x64` for finite-difference/reference tests and test float32 tolerances separately.
- Track coverage by module and scientific feature, not just a global percentage. A public engine should have analytical energy, force finite-difference, JIT, and dtype coverage.
- Convert profiling scripts into importable benchmark helpers plus thin CLIs; do not let them masquerade as correctness tests.

## 5. Documentation gaps and inaccuracies

### DOC-01 — Installation requirements contradict each other (**P0**)

`setup.py` requires Python `>=3.13` and JAX `>=0.10.0`; `docs/source/installation.rst` instructs Python 3.10 and JAX 0.4+, and `.readthedocs.yaml` builds with Python 3.10. Read the Docs therefore cannot install the package as configured.

**Solution:** decide and test a support matrix, express it once in package metadata, reuse it in installation/CI/Read the Docs, and give current official CPU/CUDA installation commands with a tested Perlmutter/Della environment note.

### DOC-02 — The custom-engine guide is empty and misspelled (**P0/P1**)

`guide_custum_engine.rst` contains only a title and “TODO.” This is central to the project's modularity claim.

**Solution:** rename with a redirect, then document engine signatures, scalar-energy requirement, parameter PyTrees/arrays, force sign, self/mutual/triple registration, JIT purity, shapes, units, sharding, explicit forces, finite-difference testing, and a complete runnable example.

### DOC-03 — The logic guide is unfinished and contains broken code (**P1**)

The custom-interaction and simulation sections are under construction. Snippets use `mode == 'gaussian'`, `jnp.ones(l1,l2,l3)` instead of a shape tuple, and a nonexistent `Square2D`; field descriptions and dimensional claims disagree with the implementation.

**Solution:** turn examples into tested snippets, complete the interaction/integrator/reporter workflow, and have docs CI execute them.

### DOC-04 — There is no authoritative supported-feature/stability matrix (**P1**)

README/docs mix implemented, partially implemented, experimental, and stub features. The API page omits `ferroelectric_superlatt`; multiferroic coverage is fragmented; autodoc exposes incomplete/private behavior as if stable.

**Solution:** publish a table by lattice, field, engine, integrator, boundary condition, dtype, device mode, and validation level (`stable`, `experimental`, `not implemented`). Curate explicit public members and link every stable feature to tests and references.

### DOC-05 — Multi-GPU and multi-node operation is undocumented (**P1**)

Missing topics include distributed initialization order, launch commands, mesh/topology choice, dimension divisibility, sharding of fields/kernels/random arrays, process-safe reporters, checkpoint/restart, profiling/synchronization, and common NCCL/XLA failures.

**Solution:** add separate single-node and multi-node guides with tested Slurm scripts for both target systems, expected logs, topology checks, limitations, and a correctness-first smoke command.

### DOC-06 — Units and scientific conventions are not centralized (**P0/P1**)

Users cannot reliably determine units and conventions for every field, mass, time step, force, pressure, magnetic moment/field, elastic coefficient, Ewald prefactor, Voigt shear, exchange bond count, or model parameter.

**Solution:** create a units/conventions chapter and annotate every public parameter. Include equations and mappings to code parameter order, energy/force sign, neighbor counting, strain tensor conversion, and source references.

### DOC-07 — Model configurations lack schema and provenance (**P0/P1**)

`model_configs/README.md` is minimal. JSON files have no machine-checked schema/version, explicit units, code/version compatibility, exact citations/DOIs, conversion formulas, uncertainty/valid ranges, or checksums.

**Solution:** define a versioned schema and validated loader; add units and provenance for every parameter; include a minimal reference observable/test for each material; fail on unknown/missing fields rather than indexing raw dictionaries throughout examples.

### DOC-08 — Examples are not reproducible workflows (**P1**)

Examples assume a particular current directory, relative JSON paths, pre-existing `output/` directories, fixed large lattice sizes/step counts, GPU layouts, and implicit seeds. Several useful submit scripts are local/untracked rather than documented assets.

**Solution:** resolve resources relative to `__file__`, create output parents safely, add CLI arguments and a tiny `--smoke` mode, record config/commit/environment/seed, and maintain reviewed Slurm scripts. State which examples are scientific validation versus demonstrations.

### DOC-09 — Performance claims are not reproducible (**P1**)

“Highly efficient” multi-GPU and “over 100X” GPU/CPU claims lack commit, workload, CPU/GPU definitions, precision, software stack, compile/warmup policy, peak memory, repetitions, error bars, and raw results. Current local scaling data is clearly sublinear.

**Solution:** replace broad claims with dated, reproducible benchmark reports and raw JSON; show compile and steady state, strong and weak scaling, memory limits, and the exact command/configuration.

### DOC-10 — Troubleshooting dismisses potentially material compile warnings (**P1**)

The FAQ treats constant-folding warnings as safe to ignore, even though large captured Ewald constants can materially affect startup time and memory.

**Solution:** explain how to distinguish a harmless warning from pathological compilation, how to enable compile logs/profiles, and which kernel-caching/sharding workaround is supported.

### DOC-11 — Release/project documentation is incomplete (**P2**)

There is no `CHANGELOG`, `CONTRIBUTING.md` with test/style workflow, `SECURITY.md`, code of conduct, `CITATION.cff`, released DOI, formal API stability/deprecation policy, or release procedure. README says a paper is forthcoming and tells users only to cite the repository.

**Solution:** add the standard project files, a citation entry and archival release DOI when available, semantic versioning/deprecation policy, and a release checklist tied to clean artifact tests.

### DOC-12 — Documentation quality is not gated (**P1/P2**)

There is no warnings-as-errors Sphinx build, linkcheck, doctest, spelling/style check, or notebook execution. RST includes Markdown-style links and visible spelling/grammar errors (`Bravias`, `funcitons`, `graident`, etc.).

**Solution:** pin a compatible docs environment, build with `-W`, run linkcheck and executable snippets in CI, and perform one terminology/spelling pass after technical content is corrected.

## 6. Packaging and release engineering

Yes—the packaging is outdated and currently broken for normal wheel users. `setup.py` itself remains supported as a configuration mechanism, but direct `python setup.py ...` workflows are deprecated and modern projects should declare the build backend and project metadata in `pyproject.toml`. See PyPA's [tool recommendations](https://packaging.python.org/en/latest/guides/tool-recommendations/) and [setup.py modernization guide](https://packaging.python.org/en/latest/guides/modernize-setup-py-project/).

### PKG-01 — The wheel omits engines and integrators (**P0, Confirmed**)

`packages=['openferro']` excludes the `openferro.engine` and `openferro.integrator` subpackages. A clean wheel is therefore unusable because root modules import those missing packages. In-tree editable imports or stale `build/lib` content can hide the failure.

**Solution:** use setuptools package discovery (`find_packages()` or `tool.setuptools.packages.find`), build both sdist and wheel from a clean archive, install each in a fresh isolated environment outside the repository, and import/run all public subpackages.

### PKG-02 — There is no `pyproject.toml` (**P0/P1**)

Build requirements and project metadata depend on ambient tooling. There is also no centralized configuration for pytest or future lint/docs tools.

**Solution:** add a minimal PEP 517/518 build-system table and PEP 621 project metadata, retaining a thin `setup.py` only if an actual compatibility need remains. PyPA's [pyproject guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) provides the current structure.

### PKG-03 — Test tooling is a runtime dependency (**P1**)

`pytest` is in `install_requires`, so every user installs it. Conversely, documentation/build/dev tools are not declared in reproducible optional groups.

**Solution:** keep only runtime libraries in core dependencies; put tests/docs/build/lint tools in optional dependency groups such as `test`, `docs`, and `dev`. Decide whether GPU JAX extras belong in documentation rather than forcing a backend.

### PKG-04 — Supported Python/JAX versions are inconsistent and unbounded by evidence (**P0/P1**)

Metadata, docs, Read the Docs, environments, and recent commit messages disagree. `jax>=0.10.0` permits future breaking releases without a tested upper policy, while no CI proves a minimum version.

**Solution:** choose the actual supported Python/JAX/jaxlib/NumPy matrix, test minimum and latest versions, document backend-specific installation, and update constraints deliberately through compatibility PRs.

### PKG-05 — Distribution metadata is incomplete (**P1/P2**)

The version is hard-coded as `0.1.0`, while the package exposes no `__version__`; there are no tags. Metadata lacks README/long description, declared license expression, project URLs, classifiers, keywords, and normalized author email. There is no release/changelog linkage.

**Solution:** single-source the version, expose it through `importlib.metadata`, tag releases, declare the MIT license and metadata, and add URLs/classifiers/readme. Automate version/artifact consistency checks.

### PKG-06 — Builds are not reproducible by process (**P0/P1**)

Ignored stale `build/` content changed one wheel's contents during this audit. No CI verifies artifact contents or prevents repository-local imports from masking missing files.

**Solution:** always build in an isolated clean checkout, inspect archives, install into a new environment whose working directory is elsewhere, run import/smoke tests, and publish only those exact checked artifacts. Never use stale in-tree `build/` as input.

### PKG-07 — Package-data and distribution scope are undefined (**P1**)

It is unclear whether `model_configs`, runnable examples, docs, and notebook assets are meant to ship in the sdist/wheel. Users need a stable way to obtain model data used by public examples.

**Solution:** explicitly choose what belongs in the library wheel, use `importlib.resources` for shipped data, include source examples/docs/configs in the sdist as intended, and test their presence. Large/generated outputs must remain excluded.

### PKG-08 — There is no release CI (**P0/P1**)

No workflow builds artifacts, runs the supported matrix, builds docs, checks metadata, or performs an isolated installation.

**Solution:** add CI for CPU unit/reference tests, package build/inspection/install, `twine check`, and docs. Use scheduled/manual Slurm jobs for GPU/multi-node gates and attach their result metadata to releases.

## 7. API, architecture, examples, and maintainability

### ENG-01 — Wildcard imports make behavior and API accidental (**P1/P2**)

`system.py` uses wildcard imports from fields/interactions/engines and then calls `jax.block_until_ready` without importing `jax`; it works only because `jax` leaks through a wildcard import. `openferro.__init__` also exports broad, undocumented symbol sets.

**Solution:** use explicit imports in implementation modules, define `__all__` for the public package, and preserve renamed public symbols through deliberate deprecation aliases.

### ENG-02 — Base classes are concrete no-ops (**P2**)

Base `calc_energy`, `calc_force`, `set_integrator`, `Simulation._step`, and `Simulation.run` methods use `pass`, allowing silent `None` behavior if instantiated or called.

**Solution:** use `abc.ABC`/`@abstractmethod` or raise `NotImplementedError` with a clear contract. Keep experimental stubs out of the stable namespace.

### ENG-03 — Structural metadata is stored as device arrays (**P1/P2**)

`lattice.size` and derived counts are JAX arrays even though they control Python shapes, loops, logging, and serialization. This causes concretization/serialization friction and unnecessary transfers.

**Solution:** store immutable sizes as tuples of Python integers and expose JAX versions only inside kernels when needed. Apply the same rule to IDs, shapes, replica counts, and byte estimates.

### ENG-04 — Public methods depend on private attributes (**P2**)

Simulation/engine code inspects `field._sharding` and other internal state directly, while setters do not consistently preserve invariants/sharding. This makes future changes risky.

**Solution:** add read-only public properties and centralize state updates in validated methods; consider immutable state PyTrees for JIT paths and lightweight configuration objects for Python orchestration.

### ENG-05 — Interaction parameters are unnecessarily restricted (**P2**)

`set_parameters` accepts only a list, NumPy array, or JAX array, rejecting scalars, tuples, mappings, and useful PyTrees despite engines having heterogeneous parameters.

**Solution:** define whether parameters are a numeric array or a registered typed PyTree. Validate per-engine schemas rather than using one weak generic check.

### ENG-06 — Reporters do not implement their own directory comment (**P1**)

Reporter initialization comments “make the directory if not exists” but never does so. Fresh-clone examples writing `output/...` can fail. A broad bare `except` when looking up global strain hides unrelated errors, and `log_interval=0` causes modulo-by-zero.

**Solution:** validate intervals, create parent directories explicitly, catch only the expected missing-field exception, fail loudly on I/O problems, and make append/overwrite/resume semantics explicit.

### ENG-07 — System construction does not validate interaction compatibility early (**P1**)

Most built-in adders accept any existing field ID and defer shape/type/lattice errors until JIT trace or execution, where messages are harder to interpret.

**Solution:** give engines declarative requirements (field dimension/type, lattice class/geometry, parameter shape/units), validate before registering, and leave the system unchanged on failure.

### ENG-08 — Model configuration is raw dictionary plumbing (**P1/P2**)

Examples index JSON dictionaries directly, with no schema, defaults, units, version migration, or cross-parameter checks. Typographical and unit mistakes reach JIT kernels unchecked.

**Solution:** introduce small typed configuration dataclasses/PyTrees with schema validation and explicit conversion into engine arrays; record the validated config in logs/checkpoints.

### ENG-09 — Naming and documentation quality need a compatibility-aware cleanup (**P2**)

Examples include `Thermo_Reporter`, `Field_Reporter`, lowercase interaction classes, `get_short_range_3rdnn`, `2ednn`, and `guide_custum_engine`. Pure renaming would break users, but leaving accidental names indefinitely enlarges maintenance cost.

**Solution:** define a stable PEP 8 API, keep deprecated aliases for at least one documented release, emit warnings, and test both the new API and compatibility layer.

### ENG-10 — Observability is ad hoc (**P2**)

Profiling is controlled by booleans and per-interaction logging, while compile/setup/I/O/collective phases are not represented in structured metrics.

**Solution:** add optional structured timing events and metadata export around kernel setup, compilation, force terms, integrator, reporting, checkpoint, and synchronization. Keep instrumentation out of JIT kernels and near-zero overhead when disabled.

## 8. Recommended implementation roadmap

### Milestone A — Restore scientific and release safety

- Fix PKG-01/02/04/06 and add clean artifact installation tests.
- Fix COR-01 through COR-08, COR-14/15/18/20, including regression tests.
- Quarantine or clearly label engines affected by COR-09 through COR-12 until derivations/reference tests are complete.
- Make the supported feature matrix honest and reconcile Python/JAX/docs versions.

**Exit gate:** clean sdist/wheel installs outside the repository; all advertised basic fields/interactions work; analytical/finite-difference tests pass; no unresolved scientific TODO is labeled stable.

### Milestone B — Build the scientific validation harness

- Add engine energy/force/reference tests and lattice/neighbor-count tests.
- Validate MD, Langevin, NPT, and LLG/SIB against solvable systems and published equations.
- Add typed/unit-aware model configs with one reference observable per material.
- Establish deterministic persistent RNG behavior.

**Exit gate:** every stable Hamiltonian/integrator has a reference, invariant, force, JIT, and dtype test; stochastic continuation and restart are reproducible.

### Milestone C — Establish and improve the performance baseline

- Measure setup, compilation, steady force/step, communication, reporting, and peak memory independently.
- Implement explicit Ewald force, cached/sharded kernels, and then evaluate whole-force fusion and `lax.scan`.
- Profile neighbor rolls and SIB after the Ewald work rather than assuming the same bottleneck for every model.

**Exit gate:** reproducible 1/2/4-GPU strong and weak scaling data, memory model within a documented tolerance, and numerical parity at every device count.

### Milestone D — Make multi-host runs operationally safe

- Add bootstrap/distributed initialization and topology-aware meshes.
- Construct arrays from process-local data.
- Add process-safe reporters and atomic versioned checkpoints.
- Run scheduled two-node correctness, restart, and scaling tests on Perlmutter/Della.

**Exit gate:** a documented two-node job can start, run, checkpoint near wall time, restart, produce one valid output set, and match a smaller reference within declared tolerances.

### Milestone E — Documentation and release

- Complete custom-engine, workflow, units, model provenance, distributed, restart, and performance documentation.
- Convert snippets/notebooks/examples into executable smoke tests.
- Add project governance/citation/changelog/release files and warnings-as-errors docs CI.

**Exit gate:** a new user can install the wheel, run a CPU smoke example, understand feature stability/units, and reproduce a documented GPU benchmark without relying on repository-local knowledge.

## 9. Definition of done for a production-ready release

A release intended for scientific large-scale use should not be tagged until all of the following are true:

- Clean wheel and sdist contain/import every intended module and pass a tiny simulation outside the checkout.
- All stable engines have analytical/reference energy tests and float64 finite-difference force tests.
- All stable integrators pass conservation/equilibrium/norm/convergence tests appropriate to their ensemble.
- No stable engine contains unresolved coefficient, counting, or unit TODOs.
- Ewald rejects unsupported geometry or passes general-cell reference tests.
- RNG streams persist across repeated calls and checkpoints; the distributed reproducibility contract is tested.
- One-, two-, and four-device runs agree numerically and demonstrate expected shardings without unintended kernel replication.
- A two-host Slurm smoke test covers initialization, FFT/forces, reporter safety, checkpoint, restart, and shutdown.
- Benchmarks synchronize a real output and separate setup, compile, warmup, steady state, I/O, and peak memory.
- Sphinx builds with warnings as errors; code examples execute; install docs match package metadata and CI.
- Version, license, citation, changelog, API stability, and release artifacts are complete and internally consistent.
