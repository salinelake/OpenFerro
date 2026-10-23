# BaTiO3 metadynamics in a cubic-symmetry chamber

This example applies three-dimensional fixed-height metadynamics to the
BaTiO3 effective Hamiltonian used by `01.BTO_Cooling`. The Cartesian
polarization is represented by

$$
q_1=P_x-P_y,\qquad q_2=P_y-P_z,\qquad q_3=P_z.
$$

Thus, $q_i\ge0$ is the fundamental chamber
$P_x\ge P_y\ge P_z\ge0$. The inverse transformation is

$$
P_z=q_3,\qquad P_y=q_2+q_3,\qquad
P_x=q_1+q_2+q_3.
$$

The transformation has unit Jacobian. The cubic, tetragonal, orthorhombic,
and rhombohedral representatives occur respectively at

$$
(q_1,q_2,q_3)=(0,0,0),\quad(P,0,0),\quad
(0,P/\sqrt{2},0),\quad(0,0,P/\sqrt{3}).
$$

The CV engines calculate the instantaneous polarization as

$$
\mathbf P(\mathbf u,\boldsymbol\eta)
=\frac{Z^*e}{V(\boldsymbol\eta)}\sum_r\mathbf u_r,
$$

where $V(\boldsymbol\eta)=V_0\det(I+\boldsymbol\eta)$ uses OpenFerro's
engineering-Voigt strain convention. Each CV therefore depends on both
`dipole` and `gstrain`. They belong to the simulation and are not interactions
in the physical Hamiltonian. Autodiff propagates their bias force to both the
soft modes and the global strain.

## Buffered symmetry walls

For every coordinate, the lower-wall contribution is

$$
U_i(q_i)=\kappa_i
\left[\max\left(0,\frac{-\epsilon-q_i}{w_i}\right)\right]^2.
$$

It is exactly zero throughout the target chamber and in the buffer
$-\epsilon\le q_i<0$. With the defaults, the wall starts at
$-0.02\ \mathrm{C/m^2}$ and costs $0.2$ eV after a further
$0.02\ \mathrm{C/m^2}$ violation. Upper walls keep the trajectory inside the
finite bias grid. Sensitivity to both the buffer and wall strength must be
checked before interpreting a production result.

## Ensemble and workflow

`run.py` first minimizes and equilibrates the cell at $-48$ kbar, then
continues directly with `MetadynamicsNPT` at the same pressure and temperature.
The homogeneous strain remains dynamical throughout metadynamics. Although an
instantaneous symmetry-broken configuration can have anisotropic strain, the
Hamiltonian and hydrostatic-pressure ensemble retain cubic symmetry. The
result is therefore the constant-pressure surface $G(P_x,P_y,P_z;p,T)$.

The default demonstrative system is $8^3$ at 250 K. It deposits a 0.002 eV
hill every 500 steps with a 0.002 ps timestep: one hill per ps, or about
$0.093k_BT$ per ps. The three-dimensional bias uses a fixed $129^3$ grid, so
the per-step interpolation cost does not grow with the number of hills. The
2 ns default is only a starting point; basin visitation and free-energy
stability must be checked for every temperature and cell size.

Run the smoke mode on a login node:

```bash
python run.py --quick --output-dir output/quick
```

Run production calculations on a GPU compute node, using a fresh output
directory at each temperature:

```bash
python run.py --temperature 250 --output-dir output/250K
python analyze.py --output-dir output/250K
```

Do not carry the accumulated bias between temperatures. `HILLS` contains
$(q_1,q_2,q_3)$ at each deposition, already normalized using the instantaneous
volume. The field log contains the average soft-mode displacement and the
thermo log contains the evolving strain and volume. `analyze.py` plots the CV
and reconstructed polarization trajectories and shows $-V_\mathrm{meta}$ on the
three chamber-boundary planes. In fixed-height metadynamics these slices are
finite-run estimates,
not guaranteed converged free energies. `analysis_summary.json` reports how
often deposited hills leave the nominal chamber and how often a lower wall is
active; the latter fraction should remain small.

Pointwise free energies can be extended to the other cubic domains by signed
permutations of $(P_x,P_y,P_z)$. Basin-integrated phase weights additionally
need the symmetry multiplicities 6, 12, and 8 for the $\langle100\rangle$,
$\langle110\rangle$, and $\langle111\rangle$ families.

## Figures

Run `analyze.ipynb` with the `of_dev` kernel to reconstruct the saved
`T*/HILLS` runs and generate the FES convergence and four-panel 3D symmetry
figures. Install its plotting dependencies with
`python -m pip install matplotlib`. The notebook works from this
directory, `examples/`, or the repository root.

The default temperatures are 180, 220, 250, and 310 K. The symmetry figure
shows the full signed-permutation extension of the nominal chamber as
energy-colored 3D scatter clouds, matching the style in `analyze.ipynb`, at a
common energy cutoff, $\Delta F\leq200\,\mathrm{meV}$. Four separate convergence figures show
directional cuts along [111], [110], [100], and [100] at 180, 220, 250, and
310 K, respectively. Every panel includes the cumulative HILLS profiles at
0.5, 1.5, 2.5, 3.5, and 4.5 ns. Each curve has its own minimum shifted to
zero, and all y-axes span 0–200 meV. Each panel uses a separate polarization
window around its low-energy region, configured in `PANELS`.
These finite-run estimates remain subject to sampling and wall effects.

The convergence figures are saved as `bto_metadynamics_convergence_180K.png`,
`bto_metadynamics_convergence_220K.png`,
`bto_metadynamics_convergence_250K.png`, and
`bto_metadynamics_convergence_310K.png`. 