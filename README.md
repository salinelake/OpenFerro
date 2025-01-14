<p align="center" >
  <img width="60%" src="/docs/openferro_logo.png" />
</p>

A universal framework for on-lattice atomistic dynamics simulation

# About OpenFerro
OpenFerro is a Python package for on-lattice atomistic dynamics simulation. OpenFerro is based on [JAX](https://github.com/google/jax), a high-performance linear algebra package supporting auto-differentiation and GPU acceleration.
OpenFerro is designed to minimize the effort required to build on-lattice Hamiltonian models, and to perform molecular dynamics (MD) and Landau-Lifshitz-Gilbert simulations. 
 
# Highlighted features
* **GPU supports**, highly efficient for large-scale simulations.
* **auto-differentiable**, will have native support for enhanced sampling and Hamiltonian optimization.
* **highly modularized**, easy to implement new interaction terms without looking into the codebase.
* **highly flexible**, supports simultaneous simulation of $R^d$ and SO(3) local order parameters. Fields with other symmetries can also be implemented.


# Installation
See documentation for installation instructions.

# Credits
There will be a paper in the near future explaining the technical details of OpenFerro.

# OpenFerro in a nutshell

## Crystalline system and Lattice Hamiltonian
A crystalline system is a periodic arrangement of atoms or molecules in space. In OpenFerro, a crystalline system is defined by a Bravias lattice, local order parameter, global variables of the lattice (such as global strain) and a Hamiltonian describing the energy of the system.

**Bravias lattice** A Bravias lattice is specified by a set of basis vectors. For example, a 3D Bravais lattice is an infinite array of discrete points described by 
$$
\mathbf{R} = i \mathbf{a}_1 + j \mathbf{a}_2 + k \mathbf{a}_3
$$
where $i, j, k$ are integers, and $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ are the basis vectors.

**Local order parameters** Local order parameters describe the state of each lattice site. They can be vectors in $R^d$ (e.g. atomic displacements, electric dipoles) or elements of SO(3) (e.g. fixed-magnitude magnetic moments, molecular orientations). Denote $\mathbf{u}_{n}$ the local order parameter at site $n=(i, j, k)$. OpenFerro stores the local order parameters in a 4D tensor, with the shape of $(N_1, N_2, N_3, d)$, where $N_1, N_2, N_3$ are the number of lattice sites in the three directions, and $d$ is the dimension of the local order parameter. The fixed topology of the local order parameters make dynamical simulation much more efficient than generic molecular dynamics where the topology is not fixed.

**Global variables** Global variables describe global properties of the lattice. In OpenFerro, the default global variable is the global strain tensor in the [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation): $\eta = (\eta_1, \eta_2, \eta_3, \eta_4, \eta_5, \eta_6)=(\eta_{xx}, \eta_{yy}, \eta_{zz}, 2\eta_{yz}, 2\eta_{xz}, 2\eta_{xy})$. where $\eta_{xx}, \eta_{yy}, \eta_{zz}$ are the normal strains, and $\eta_{yz}, \eta_{xz}, \eta_{xy}$ are the shear strains.

**Lattice Hamiltonian** A lattice Hamiltonian $E$, as a function of all local order parameters  and all global variables, is an energy function defined on the crystalline system. It typically consists of:
- On-site terms describing the local energetics at each site
- Interaction terms between different sites (e.g. dipole-dipole interactions)
- Global terms like elastic energy
- Interaction between system variables and external fields (e.g. electric field, magnetic field)

OpenFerro provides a flexible framework to construct lattice Hamiltonians by combining these energy terms. The energy terms are implemented as Python classes with a unified interface, making it easy to add new types of interactions.

Examples:
- Perovskite ferroelectric materials
(1) BaTiO3: local dipole fields $\mathbf{u}_{n}\in R^3$, local strain $\eta^{\text{loc}}_{n} \in R^6$, global strain $\eta \in R^6$.
See [Physical Review B 52.9 (1995): 6301](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.52.6301) for details of the Hamiltonian.

- Magnetic materials
(1) Toy Ising model: spin fields $\mathbf{u}_{n}\in SO(3)$. 
$$
E = \sum_{n\sim m} J \mathbf{u}_n \cdot \mathbf{u}_m
$$
Here, $n\sim m$ means a nearest-neighbor pair. $J$ is the exchange coupling.
(2) Bcc Iron: local dipole fields $\mathbf{s}_{n}\in SO(3)$. 
$$
E = \sum_{n,m} J_{nm} \mathbf{s}_n \cdot \mathbf{s}_m
$$
Here $n,m$ are summed over all pairs. $J_{nm}$ is the exchange coupling between site $n$ and $m$. The exchange interaction can be cut off at a certain distance. For example, if the exchange interaction is cut off at the fourth shell. We have $J_{nm}\in (J_1, J_2, J_3, J_4)$. 
See [PRL 95, 087207 (2005)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.95.087207) for details of the Hamiltonian.
- Multiferroic materials
(1) BiFeO3: see [PRL 99, 227602 (2007)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.99.227602) for one possible realization of the Hamiltonian.

## Simulate the dynamics
With the lattice, local order parameters, global variables and Hamiltonian defined, we can compute the equilibrium properties of the system by sampling configurations from the Boltzmann distribution. There are many algorithms to sample the Boltzmann distribution. Two major ones are molecular dynamics (MD) and Monte Carlo (MC). Monte Carlo sampling relies on constucting a Markovian process (not an actual dynamical physical process) that samples the Boltzmann distribution. Molecular dynamics, on the other hand, is associated with the Born-Oppenheimer dynamics of an atomic system, which corresponds to the actual physical process when all atomic degrees of freedom are considered. This gives access to dynamical aspects (e.g. spectroscopic properties, domain dynamics, spin wave) of an atomic system.

OpenFerro, up to this point, support only molecular dynamics simulation, since it is a more general approach to study the dynamics of a physical system. Enhanced sampling methods (e.g. metadynamics, umbrella sampling) will be supported in the future. In the long term, kinetic Monte Carlo may also be supported to overcome the limitation of time scale.

**Equations of motion**
OpenFerro deals with coarse-grained representation of atomic systems. The molecular dynamics simulated by OpenFerro will be different from the [generic molecular dynamics](https://en.wikipedia.org/wiki/Molecular_dynamics) (considering all atomic degrees of freedom) formulated under the Born-Oppenheimer approximation. The existence of $SO(3)$ fields further complicates the problem because the general molecular dynamics equations are not applicable to $SO(3)$ fields. So we need to have two sets of equations of motion for $R^d$ and $SO(3)$ fields.

- Equation of motion for $R^d$ fields
For $R^d$ local order parameters and unconstrained global variables, the equations of motion are:
**NVE ensemble**: the microcanonical ensemble is sampled by the Newton's equation of motion, driven by the potential energy $E$:
$$
\frac{d^2 \mathbf{u}_{n}}{dt^2} = -\nabla_{\mathbf{u}_n} E(\mathbf{u}_1, \cdots, \mathbf{u}_N)
$$

- Equation of motion for $SO(3)$ fields ([Landau-Lifshitz-Gilbert equation](https://en.wikipedia.org/wiki/Landau%E2%80%93Lifshitz%E2%80%93Gilbert_equation))


<!-- ### Generic molecular dynamics
A generic molecular dynamics simulation tracks the position of all atoms (position vector $R_i$ associated to atom-$i$) from an atomic system. 
- NVE ensemble: the microcanonical ensemble is sampled by the Newton's equation of motion, driven by the potential energy $E$:
$$
m_i\frac{d^2 \mathbf{R}_{i}}{dt^2} = -\nabla_{i} E(\mathbf{R_1}, \cdots, \mathbf{R_N})
$$

- NVT ensemble: the canonical ensemble can be sampled by the Langevin equation, driven by the potential energy $E$, the friction force and a random force:
$$
m_i\frac{d^2 \mathbf{R}_{i}}{dt^2} = -\nabla_{i} E(\mathbf{R_1}, \cdots, \mathbf{R_N}) - \gamma \frac{d \mathbf{R}_{i}}{dt} + \sqrt{2\gamma k_B T} \xi_i
$$
where $\gamma$ is the friction coefficient, $k_B$ is the Boltzmann constant, $T$ is the temperature, and $\xi_i$ is a random force with zero mean and variance $2\gamma k_B T$.

### On-lattice atomisticdynamics

### Landau-Lifshitz-Gilbert equation -->




# Use OpenFerro

## Overview
A schematic of OpenFerro is shown below.
<p align="center" >
  <img width="60%" src="/docs/overview.png" />
</p>

## Quick start
See /tutorials for Jupyter-notebook tutorials of OpenFerro, where we cover both the basic usage and more advanced features. 

## Examples
See /examples for more examples. 

## Documentation
See [documentation](https://openferro.readthedocs.io/en/latest/) for more details.

<!-- ## Units
OpenFerro's internal unit system is the same as the 'metal' unit system used in LAMMPS.

mass = grams/mole

distance = Angstroms

time = picoseconds

energy = eV

velocity = Angstroms/picosecond

force = eV/Angstrom

torque = eV

temperature = Kelvin

pressure = bars

dynamic viscosity = Poise

charge = multiple of electron charge (1.0 is a proton)

dipole = charge*Angstroms

electric field = volts/Angstrom

density = gram/cm^dim

Magnetic dipole moment = Bohr magneton

Magnetic field = eV / Bohr magneton

 -->
