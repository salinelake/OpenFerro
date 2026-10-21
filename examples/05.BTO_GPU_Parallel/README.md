# Single-node GPU scaling for BaTiO3

This example benchmarks the BaTiO3 NPT system from `../01.BTO_Cooling` at a
fixed temperature of 270 K. It retains the complete Hamiltonian, including the
Ewald dipole-dipole interaction, and uses the same masses, pressure, time step,
and Langevin relaxation times. Structural minimization and trajectory output
are omitted so the measurement covers the simulation kernel.

The benchmark measures all combinations of cubic lattice size L=120, 240, 360,
480, and 600 with 1, 2, 3, and 4 A100 GPUs on one node. Each value is the median
of three 100-step measurements following compilation and three warmup steps.
One JAX process controls the visible GPUs with OpenFerro's default `1 x N` mesh.

On Perlmutter, request a full GPU node and run:

```bash
salloc -N1 -t 04:00:00 -C gpu -q interactive -A m5218
cd examples/05.BTO_GPU_Parallel
bash run_strong_scaling.bash
```

The launcher creates a unique `results/JOB_ID_TIMESTAMP/` directory containing:

- `strong_scaling.csv`: timings, peak memory, and run metadata.

## Reference results

The checked-in [benchmark results](benchmark_results.csv) and
[plot](benchmark.png) were measured on an NVIDIA A100-SXM4-40GB node in
Perlmutter allocation `56494987` with JAX 0.11.0.

| L | 1 GPU [ms/step] | 2 GPUs | 3 GPUs | 4 GPUs | 4-GPU efficiency | Peak GiB/GPU, 1 -> 4 GPUs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 120 | 4.235 | 9.521 | 13.217 | 15.953 | 6.6% | 0.35 -> 0.19 |
| 240 | 28.409 | 27.906 | 20.530 | 17.669 | 40.2% | 2.63 -> 1.20 |
| 480 | 225.216 | 191.478 | 140.116 | 118.383 | 47.6% | 19.36 -> 8.90 |

![Single-node GPU scaling results](benchmark.png)

L=120 is too small to amortize sharding and collective communication. L=240
and L=480 reach 1.61x and 1.90x speedup on four GPUs, respectively, while
reducing peak per-GPU memory by about 54%.

Parallel efficiency is `(1-GPU time) / (N-GPU time * N)`. GPU preallocation is
disabled, and memory is the peak reported by JAX's allocator on the busiest
GPU. The plot uses a logarithmic memory axis because the three workloads span
two orders of magnitude. JAX's measurement excludes CUDA driver and context
memory reported by `nvidia-smi`.

For a shorter diagnostic run, set the timing controls explicitly:

```bash
BTO_STEPS=10 BTO_REPEATS=1 bash run_strong_scaling.bash
```

## Maximal system size

`max_size.py` finds the largest mesh-aligned cubic cell that completes one
NPT step on the visible GPUs, then times that cell. Probe attempts run in
child processes so an out-of-memory failure does not poison the parent.
On a full GPU node:

```bash
salloc -N1 -t 04:00:00 -C gpu -q interactive -A m5218
cd examples/05.BTO_GPU_Parallel
bash run_max_size.bash
```

The launcher writes `max_size.csv` with one timed maximum per GPU count.
To time a chosen size without searching:

```bash
srun --gpus-per-task=4 --exact python max_size.py --size 720 \
    --output results/RUN/max_size.csv
```

Plot a completed strong-scaling CSV from `plot_strong_scaling.ipynb`.
