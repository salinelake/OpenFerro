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
See [documentation](https://openferro.readthedocs.io/en/latest/installation.html) for installation instructions.


# OpenFerro in a nutshell

## Crystalline system and Lattice Hamiltonian
A crystalline system is a periodic arrangement of atoms or molecules in space. In OpenFerro, a crystalline system is defined by a Bravias lattice, local order parameter, global variables of the lattice (such as global strain) and a Hamiltonian describing the energy of the system.

#### Bravias lattice
A Bravias lattice is specified by a set of basis vectors. For example, a 3D Bravais lattice is an infinite array of discrete points described by $\mathbf{R} = n_1 \mathbf{a}_1 + n_2 \mathbf{a}_2 + n_3 \mathbf{a}_3$, where $n_1, n_2, n_3$ are integers, and $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ are the basis vectors.

#### Local order parameters
Local order parameters describe the state of each lattice site. They can be vectors in $R^d$ (e.g. atomic displacements, electric dipoles) or elements of SO(3) (e.g. fixed-magnitude magnetic moments, molecular orientations). For a 3D lattice with N sites, the total degree of freedom of $R^d$ local order parameters is $Nd$, and the total degree of freedom of $SO(3)$ local order parameters is $3N$. 

#### Global variables
Global variables describe global properties of the lattice. In OpenFerro, the default global variable is the global strain tensor in the [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation): $\eta = (\eta_1, \eta_2, \eta_3, \eta_4, \eta_5, \eta_6)=(\eta_{xx}, \eta_{yy}, \eta_{zz}, 2\eta_{yz}, 2\eta_{xz}, 2\eta_{xy})$. where $\eta_{xx}, \eta_{yy}, \eta_{zz}$ are the normal strains, and $\eta_{yz}, \eta_{xz}, \eta_{xy}$ are the shear strains.

#### Lattice Hamiltonian
A lattice Hamiltonian $E$, as a function of all local order parameters  and all global variables, is an energy function defined on the crystalline system. It typically consists of:
- On-site terms describing the local energetics at each site
- Interaction terms between different sites (e.g. dipole-dipole interactions)
- Global terms like elastic energy
- Interaction between system variables and external fields (e.g. electric field, magnetic field)

OpenFerro provides a flexible framework to construct lattice Hamiltonians by combining these energy terms. The energy terms are implemented as Python classes with a unified interface, making it easy to add new types of interactions.

##### Example 1: Perovskite ferroelectric materials: BaTiO3
_Variables_: Local dipole fields $u_{n}\in R^3$, local strain $\eta^{loc}_{n} \in R^6$, global strain $\eta \in R^6$.
_Hamiltonian_: See [Physical Review B 52.9 (1995): 6301](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.52.6301) for details.

##### Example 2: Magnetic materials: Bcc Iron
_Variables_: Local dipole fields $s_{n}\in SO(3)$.
_Hamiltonian_: $E = \sum_{n,m} J_{nm} s_n \cdot s_m$. Here $n,m$ are summed over all pairs. $J_{nm}$ is the exchange coupling between site $n$ and $m$. The exchange interaction can be cut off at a certain distance. For example, if the exchange interaction is cut off at the fourth shell. We have $J_{nm}\in (J_1, J_2, J_3, J_4)$.  See [PRL 95, 087207 (2005)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.95.087207) for details of the Hamiltonian.

## Simulate the dynamics
With the lattice, local order parameters, global variables and Hamiltonian defined, we can compute the equilibrium properties of the system by sampling configurations from the Boltzmann distribution. There are many algorithms to sample the Boltzmann distribution. Two major ones are molecular dynamics (MD) and Monte Carlo (MC).  
OpenFerro, up to this point, support only molecular dynamics simulation, since it is a more general approach to study dynamical aspects (e.g. spectroscopic properties, domain dynamics, spin wave) of a physical system. Enhanced sampling methods (e.g. metadynamics, umbrella sampling) will be supported in the future. In the long term, kinetic Monte Carlo may also be supported to overcome the limitation of time scale.

### Equations of motion
OpenFerro use NVE/NVT/NPT molecular dynamics equations of motion to simulate the dynamics of the $R^d$ local order parameters and global variables. On an equal footing, OpenFerro use Landau-Lifshitz-Gilbert (LLG) equation of motion to simulate the dynamics of the $SO(3)$ local order parameters. See the [documentation](https://openferro.readthedocs.io/en/latest/theory-dynamics.html) for details. 

References:
- MD: Rapaport, Dennis C. The art of molecular dynamics simulation. Cambridge university press, 2004.
- LLG: Eriksson, Olle, et al. Atomistic spin dynamics: foundations and applications. Oxford university press, 2017.
- MD + LLG: Wang, Dawei, Jeevaka Weerasinghe, and L. Bellaiche. "Atomistic molecular dynamic simulations of multiferroics." Physical Review Letters 109.6 (2012): 067203.

### Integrator
Supported integrators (see [documentation](https://openferro.readthedocs.io/en/latest/core_components.html#integrators.html) for details):
- MD: Leapfrog (NVE), Mid-point Langevin (NVT, NPT, see J. Phys. Chem. A 2019, 123, 28, 6056-6079)
- LLG: Semi-implicit B (SIB) scheme (see J. Phys.: Condens. Matter 22 (2010) 176001)

# Use OpenFerro

## Overview
A schematic of OpenFerro is shown below. See [documentation](https://openferro.readthedocs.io/en/latest/core_components.html) for introduction to core components of OpenFerro.
<p align="center" >
  <img width="60%" src="/docs/overview.png" />
</p>

## Quick start
See [documentation](https://openferro.readthedocs.io/en/latest/quickstart.html) for a quick start, where we cover both the basic usage and more advanced features. 

## Examples
See examples in [examples](https://github.com/salinelake/OpenFerro/tree/main/examples). 

## Snapshot
<p align="center" >
  <img width="80%" src="/docs/domain.png" />
</p>

# Credits
There will be a paper in the near future explaining the technical details of OpenFerro. Before that, please cite this repository for any use of OpenFerro.

The initial development of OpenFerro is done by Pinchen Xie with support from LBNL. 

# Model configuration
At this point, only a few publicly availabel model configurations are provided in the [model_configs](https://github.com/salinelake/OpenFerro/tree/main/model_configs) directory. We welcome contributions from the community to add more model configurations. 

# Contributing
We welcome contributions from the community. Raise an issue or submit a pull request. Also feel free to contact pinchenxie@lbl.gov for any questions.
