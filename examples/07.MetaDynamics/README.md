# Analytically solvable two-dimensional metadynamics

This example exercises `MetadynamicsNVT` against an analytically known surface
on a `(2, 2, 2)` simple-cubic lattice with a three-component `dipole` field.
The collective variables are the total dipoles

$$
S_x=\sum_r p_{r,x}, \qquad S_y=\sum_r p_{r,y}.
$$

They are simulation-owned scalar functions, not interactions in the physical
Hamiltonian. Their only force contribution is the chain-rule force from the
metadynamics bias and optional walls.

The runtime bias is accumulated on a fixed ``201 x 201`` grid from ``-2.4`` to
``2.4`` in both CVs.  Its spacing is ``0.024``, equal to one fifth of the hill
width.  Cubic interpolation makes the per-step bias cost independent of the
number of deposited hills; HILLS still records the exact centers for the
independent reconstruction below.

## Exact free energy

The registered toy Hamiltonian is

$$
H=A[(S_x^2-S_0^2)^2+(S_y^2-S_0^2)^2]
+\frac{K_\perp}{2}\sum_r[(p_{r,x}-\bar p_x)^2
+(p_{r,y}-\bar p_y)^2+p_{r,z}^2],
$$

where $\bar p_x=S_x/N$ and $\bar p_y=S_y/N$. At fixed $S_x,S_y$, all
orthogonal modes have a Gaussian integral independent of the CV values.
Consequently,

$$
F(S_x,S_y)=A[(S_x^2-S_0^2)^2+(S_y^2-S_0^2)^2]+C.
$$

The four minima are $(\pm S_0,\pm S_0)$, the axis saddle is $A S_0^4$ above
the minima, and the central barrier is $2A S_0^4$. Dipole components use the
arbitrary unit $P$ in `config.json`; $A$ is in eV/$P^4$ and $K_\perp$ in
eV/$P^2$.
 
## Quick smoke run

From this directory, after activating the OpenFerro development environment:

```bash
python run.py --quick --output-dir output/quick
OPENFERRO_METAD_OUTPUT=output/quick jupyter lab analyze.ipynb
```

Quick mode checks execution, finite output, and the HILLS layout. It is too
short to validate free-energy accuracy, so the notebook is expected to report
`NOT CONVERGED`.

## Full validation target

Do not run the full trajectory on a login node. Use the documented one-GPU
interactive allocation, then run

```bash
python run.py --output-dir output/full
OPENFERRO_METAD_OUTPUT=output/full jupyter lab analyze.ipynb
```

Open the notebook and run all cells. It plots the hill-center trajectory,
independently reconstructs the fixed-height Gaussian sum from HILLS, and
compares $-V_{\mathrm{meta}}$ with the exact surface inside fixed, predeclared
comparison bounds. It reports surface error, all four minimum locations, axis
and central barriers, basin visitation, selected 300/400/500 ps prefixes, and
the final result. The plots and numerical outputs are also saved under
`output/full/analysis`.

Classical fixed-height metadynamics does not converge monotonically, so these
are finite-run integration targets rather than an assertion of exact
asymptotic convergence.

The deterministic defaults run 1,000,000 steps with a 0.002 ps timestep, for
a total of 2 ns. A 0.001 eV hill is deposited every 200 steps, or 0.4 ps. At
300 K this is about $0.039k_BT$ per hill and $0.097k_BT/\mathrm{ps}$; the hill
interval is four times the 0.1 ps thermostat relaxation time.
