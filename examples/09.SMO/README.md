# Coupled spin--lattice heating in strained SrMnO3

This example heats 5% biaxially tensile SrMnO3 from a homogeneous [110]
ferroelectric (FE) and ferromagnetic (FM) state. The ferroelectric
effective Hamiltonian and the displacement-dependent exchange come from
Edstrom and Ederer, [Phys. Rev. Materials 2, 104409
(2018)](https://doi.org/10.1103/PhysRevMaterials.2.104409). All published
numbers are stored in [`reference_data.json`](reference_data.json) and
mapped in `model.py`.

At this strain the paper's Figure 8 combines two separate calculations:

$$
(\mathrm{FM,FE})
\xrightarrow{\sim 100\ \mathrm{K}}
(\mathrm{PM,FE})
\xrightarrow{\sim 200\ \mathrm{K}}
(\mathrm{PM,FE\ domains}).
$$

The published $J(u)$ curve exists only along a homogeneous [110] polar
path, so this example stays below the ferroelectric-domain regime. The
default heating grid is 20--160 K. The ferroelectric Curie temperature,
cooling from the domain phase, and the full strain--temperature diagram
are not part of this workflow.

## Two controlled calculations

Both drivers build the same unified Hamiltonian with
`build_smo_coupled_system`. They differ only in which fields evolve:

| Script | Local mode | Local strain | Spin | Role |
| --- | --- | --- | --- | --- |
| `run_frozen.py` | frozen at the relaxed [110] FE endpoint | frozen | stochastic LLG | Restricted reproduction of the paper's FM--PM transition |
| `run_coupled.py` | Langevin | Langevin | stochastic LLG | Prediction of the bond-local extension |

The global strain is clamped at the 5% snapshot

$$
\boldsymbol\eta=(0.05,\ 0.05,\ -0.0285,\ 0,\ 0,\ 0)
$$

because `SimulationNVTLangevin` does not integrate `gstrain`. Both runs
start from

$$
\mathbf u_i=u_\mathrm{eq}(1,1,0)/\sqrt{2},\qquad
\mathbf S_i=(0,0,1),
$$

with $u_\mathrm{eq}=0.23$ Angstrom. Heating is sequential: the final
state at one temperature initializes the next.

The frozen calculation is a restricted-model reproduction. The coupled
calculation is a first-principles-informed extension, not a reproduction
of either SrMnO3 reference. A statistically insignificant shift in the
magnetic $T_\mathrm{C}$ is a valid outcome.

## Unified Hamiltonian

The published FE Hamiltonian was parameterized with G-type antiferromagnetism
as the magnetic reference. The unified model therefore subtracts that
reference so it is not counted twice:

$$
H(\mathbf u,\mathbf v,\boldsymbol\eta,\mathbf S)
=H_\mathrm{FE}^{G}(\mathbf u,\mathbf v,\boldsymbol\eta)
+H_\mathrm{spin}(\mathbf S;\mathbf u)
-H_\mathrm{spin}(\mathbf S^{G};\mathbf u).
$$

With $\mathbf S=\mathbf S^G$, the energy and the structural forces reduce
to $H_\mathrm{FE}^{G}$. With the structural fields frozen, the subtracted
term is a spin-independent constant, so the magnetic statistics are those
of the corresponding frozen-structure Heisenberg model.

The first-neighbor exchange is bond-local. For each unique nearest-neighbor
bond,

$$
s_{ij}=\frac{|\mathbf u_i|^2+|\mathbf u_j|^2}{2u_\mathrm{eq}^2},\qquad
H_1=-\sum_{\langle ij\rangle_1}
J_{1,ij}(s_{ij})\,(\mathbf S_i\cdot\mathbf S_j+1).
$$

$J_1(s)$ is a cubic polynomial in $s$, fitted to the paper's homogeneous
[110] curve and then anchored at the relaxed FE endpoint
($J_1^{x,y}=3.013$ meV, $J_1^z=13.266$ meV). The $+1$ makes both the
energy and the local-mode force vanish for exact G-type spins. The
second-neighbor couplings are the three relaxed-FE values, held constant
because no public $J_2(u)$ curve exists:

$$
H_2=-\sum_{\langle ij\rangle_2}
J_{2,ij}\,(\mathbf S_i\cdot\mathbf S_j-1).
$$

The ferroelectric sector is the usual onsite polynomial, three-shell
short-range coupling, Ewald dipole--dipole interaction, and homogeneous
plus inhomogeneous strain terms.

## How to run

Launch from this directory. The drivers write `output_frozen/` and
`output_coupled/` relative to the working directory, and `analyze.py`
reads those directories next to the script.

The default $16^3$ heating schedule is a short demonstration: 0.0002 ps
steps, 5000 equilibration steps (1 ps) and 10000 sampling steps (2 ps) at
each temperature, with Langevin $\tau=0.1$ ps and Gilbert damping
$\alpha=0.5$. It does not establish production convergence.

```bash
python run_frozen.py
python run_coupled.py
python analyze.py --size 16
```

For a login-node execution check, request a $4^3$ cell and two sampling
steps at 20 K:

```bash
python run_frozen.py --size 4 --temperatures 20 \
    --equilibration-steps 1 --sampling-steps 2 --report-interval 1 \
    --output-dir /tmp/openferro-smo-frozen-smoke
python run_coupled.py --size 4 --temperatures 20 \
    --equilibration-steps 1 --sampling-steps 2 --report-interval 1 \
    --output-dir /tmp/openferro-smo-coupled-smoke
```

Production sampling belongs on a GPU node. The included Perlmutter
template runs the two cases in parallel on two GPUs at $L=64$ with
50000 sampling steps (10 ps) per temperature. Replace the account if
needed:

```bash
sbatch -A PROJECT submit.sh
```

`--size`, `--seed`, `--temperatures`, `--equilibration-steps`,
`--sampling-steps`, `--report-interval`, and `--output-dir` are available
on both drivers. Coupled runs also expose `--polar-dt-ps`,
`--local-strain-dt-ps`, `--langevin-tau-ps`, `--spin-dt-ps`, and
`--spin-alpha`.

## Outputs

Each temperature writes a thermo log and field-average logs under
`output_frozen/L{size}/` or `output_coupled/L{size}/`. Coupled runs also
write `local_mode_{T}K_avg.log`. Full spin (and, when coupled, local-mode)
snapshots are dumped at the same interval.

`analyze.py` reports the time-averaged magnetization magnitude

$$
|M|=\langle|\langle\mathbf S\rangle_\mathrm{sites}|\rangle_t
$$

and the fluctuation susceptibility

$$
\chi=N\bigl(\langle|m|^2\rangle-\langle|m|\rangle^2\bigr)/(k_B T)
$$

in eV$^{-1}$. It also overlays `evaluate_exchange` on the published
$J_1(u/u_\mathrm{eq})$ points and, for coupled runs, the mean
$|\langle u_x\rangle|$ and $|\langle u_y\rangle|$.

`magnetic.png` includes plus or minus one standard error for magnetization
and susceptibility. The magnetization error uses ten contiguous time-block
means; the susceptibility error uses a delete-one-block jackknife of the full
fluctuation estimator. For the $L=64$ submission template, each block spans
1 ps. These finite-trajectory estimates do not establish sampling convergence
near the transition. Regenerate that figure with `python analyze.py --size 64`.
