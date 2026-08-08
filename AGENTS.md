# OpenFerro Agent Guide

OpenFerro is a JAX-based Python package for on-lattice atomistic dynamics. Keep changes simple, readable, and close to the existing design: lattices define geometry, fields hold state, interactions wrap Hamiltonian terms, engines compute energies, and simulations drive integrators and reporting.

## Repository Map

- `openferro/`: package source.
- `openferro/system.py`: central `System` object; owns the lattice, fields, interactions, force updates, and energy calculations.
- `openferro/field.py`: field state, masses, velocities, forces, sharding, and integrator selection.
- `openferro/lattice.py`: 3D Bravais lattice classes and neighbor-shell rollers.
- `openferro/interaction.py`: self, mutual, and triple interaction wrappers. Energy engines are required; force engines may be explicit or derived with JAX autodiff.
- `openferro/engine/`: Hamiltonian energy engines for elastic, ferroelectric, magnetic, multiferroic, superlattice, and Ewald terms.
- `openferro/integrator/`: molecular-dynamics and LLG integrators.
- `openferro/simulation.py`: minimization, NVE, NVT, and NPT simulation loops.
- `openferro/reporter.py`: thermo and field-output reporters.
- `openferro/parallelism.py`: JAX device mesh and sharding helpers.
- `examples/`: runnable examples and analysis scripts.
- `model_configs/`: JSON material/model parameters used by examples.
- `docs/source/`: Sphinx documentation.
- `tests/`: lightweight tests and profiling scripts.

## Development Principles

- Prefer the existing public flow: create a lattice, create a `System`, add fields, add interactions, set field integrators, then run a simulation.
- Keep APIs small and explicit. Avoid adding abstraction unless it removes real duplication or matches an existing pattern.
- Preserve scientific meaning in names and units. If a value is in atomic units, eV, Angstrom, bar, Kelvin, or Voigt notation, make that clear in names, docs, or nearby comments.
- Write code that is easy to inspect. Short helper functions are fine; clever JAX one-liners are not helpful when they hide the physics.
- Do not rewrite unrelated modules while touching one interaction, engine, field, or integrator.

## Python And JAX Style

- Follow the local style: standard imports first, then `numpy`, `jax`, `jax.numpy as jnp`, then `openferro` imports.
- Use `jax.numpy` for arrays that participate in JIT, autodiff, forces, or simulation state. Use `numpy` mainly for Python-side setup and simple constants.
- Keep JIT-compiled energy and force engines functionally pure. Avoid file I/O, logging, mutation, and Python-side dynamic control flow inside JIT paths.
- Keep array shapes explicit. Field values are commonly shaped `(l1, l2, l3, dim)`; global strain uses 6-component Voigt notation.
- Use `jax.random.PRNGKey` and pass keys deliberately when randomness is needed. Do not hide random state in globals.
- When adding an energy engine, use the existing signature style: field values first, then `parameters`. Return a scalar energy.
- When adding explicit force engines, return arrays with the same shape as the target field values. Remember forces are negative energy gradients.
- Be careful with multi-device changes. Preserve sharding on values, masses, velocities, forces, and random arrays when working near `DeviceMesh`.

## Documentation And Comments

- Use NumPy-style docstrings for public classes and methods, matching the current modules.
- Comments should explain the physics, units, indexing, or non-obvious JAX constraints. Do not narrate simple Python statements.
- Update `docs/source/api.rst` or the relevant `.rst` page when adding public modules, classes, or user-facing behavior.
- Keep README-level text broad. Put detailed API and workflow documentation in `docs/source/`.

## Examples And Data

- Keep examples runnable from their own directory unless there is a clear reason otherwise.
- Put reusable material parameters in JSON under `model_configs/` or the example directory, following the existing format.
- Avoid committing generated logs, large dumps, profiling output, or local run artifacts. Existing `dev/` files are development/profiling artifacts, not style templates.

## Tests And Validation

There is no broad test harness or formatter configuration in the repo. Use the smallest validation that matches the change.

- Install locally with `pip install -e .` after dependency setup.
- Run the current unit test with `python -m pytest tests/unit_tests/test_ewald.py`.
- For import-level checks, use `python -c "import openferro as of; print(of.System)"`.
- Build docs from `docs/` with `make html` after installing `docs/requirements.txt`.
- For GPU or multi-GPU behavior, prefer the examples in `examples/Profiling_GPU/`; do not assume GPU tests are available in every environment.

## Change Hygiene

- Keep changes focused and readable.
- Preserve existing user files and generated outputs unless the task explicitly asks to modify them.
- If a numerical change affects physical behavior, add or update a focused test or example check when practical.
- Before finishing, report what was changed and which validation commands were run or skipped.

## Development Environment
This code will be developed on Perlmutter supercomputer, and sometimes Della supercomputer. Codex CLI will run on login node. Use `hostname` to check the supercomputer you are on.

Activate the development environment on Perlmutter:
```bash
module load python
conda activate of_dev
```

For Della, activate the development environment:
```bash
module load anaconda3/2025.12
conda activate openferro
```

Never run heavy CPU/GPU simulations on login nodes. On login nodes, only do lightweight actions such as `git pull`, editing small files, inspecting logs, or submitting/monitoring Slurm jobs.
You should use salloc to activate an interactive session. However, salloc is forbidden if you are in sandbox mode. Get out of the sandbox to use salloc.

For quick validation that requires only one GPU, you can use the following command to activate an interactive session on Perlmutter:
```bash
salloc -N1 -n32 -t 04:00:00 -C gpu -q shared_interactive --gres=gpu:1 -A m5025
```
This will give you a login node with 32 CPU cores and 1 NVIDIA A100 GPU. This is typically sufficient for most of the tests.

On Della, you can use the following command to activate an interactive session:
```bash
gputest
```
This allocate one A100 GPU on a single node.

For development that requires all 4 NVIDIA A100 GPUs on a single node, activate a multi-GPU sesseion on Perlmutter:
```bash
salloc -N1 -t 04:00:00 -C gpu -q interactive -A m5025
```

On Della, you can use the following command to activate a multi-GPU session:
```bash
salloc -N 1 -n 32 --gres=gpu:4 -t 1:0:0
```

If you need two nodes to test multi-node parallelism using all 8 NVIDIA A100 GPUs, activate a multi-node session:
```bash
salloc --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 -t 01:00:00 -C gpu -q interactive -A m5025
```

On Della, you can use the following command to activate a multi-node session:
```bash
salloc --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 -t 01:00:00
```