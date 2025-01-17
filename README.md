<p align="center" >
  <img width="60%" src="/docs/openferro_logo.png" />
</p>

A universal framework for on-lattice atomistic dynamics simulation

# About OpenFerro
OpenFerro is a Python package for on-lattice atomistic dynamics simulation. OpenFerro is based on [JAX](https://github.com/google/jax), a high-performance linear algebra package supporting auto-differentiation and GPU acceleration.
OpenFerro is designed to minimize the effort required to build on-lattice Hamiltonian models, and to perform molecular dynamics (MD) and Landau-Lifshitz-Gilbert simulations. 
 
# Highlighted features
* **Multi-GPU Acceleration**, highly efficient for large-scale simulations.
* **Auto-differentiable**, will have native support for enhanced sampling and Hamiltonian optimization.
* **Modularized**, easy to implement new interaction term as a plug-in python function. No need to modify source code. OpenFerro handles the rest, including graident calculation and GPU acceleration.
* **Flexible**, supports simultaneous simulation of $R^d$ and SO(3) local order parameters. Fields with other symmetries can also be implemented.


# Installation
See [documentation](https://openferro.readthedocs.io/en/latest/installation.html) for installation instructions.

# OpenFerro in a nutshell

## Crystalline system and Lattice Hamiltonian
A crystalline system is a periodic arrangement of atoms or molecules in space. In OpenFerro, a crystalline system is defined by a Bravias lattice, local order parameter, global variables of the lattice (such as global strain) and a Hamiltonian describing the energy of the system. See [documentation](https://openferro.readthedocs.io/en/latest/theory-lattice-model.html) for more details than the outline below.

#### Bravias lattice
A Bravias lattice is specified by a set of basis vectors. For example, a 3D Bravais lattice is an infinite array of discrete points described by $\mathbf{R} = n_1 \mathbf{a}_1 + n_2 \mathbf{a}_2 + n_3 \mathbf{a}_3$, where $n_1, n_2, n_3$ are integers, and $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ are the basis vectors.

#### Local order parameters
<!-- there should be a figure here,   a few atomic unit cell -> local order parameter -> energy -->

Local order parameters describe the state of each lattice site. They can be vectors in $R^d$ (e.g. atomic displacements, electric dipoles) or elements of SO(3) (e.g. fixed-magnitude magnetic moments, molecular orientations).  

#### Global variables
Global variables describe global properties of the lattice. In OpenFerro, the default global variable is the global strain tensor in the [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation): $\eta = (\eta_1, \eta_2, \eta_3, \eta_4, \eta_5, \eta_6)$.

#### Lattice Hamiltonian
A lattice Hamiltonian $E$, as a function of all local order parameters  and all global variables, is an energy function defined on the crystalline system. It typically consists of terms like dipole-dipole interaction, dipole-strain interaction, elastic energy, etc. 
OpenFerro provides a flexible framework to construct lattice Hamiltonians by combining these energy terms. The energy terms are implemented as Python classes with a unified interface, making it easy to add new types of interactions.

##### Relevant materials: 

- Ferroelectric materials: perovskites like [BaTiO3](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.52.6301), PbTiO3, etc.
- Magnetic materials:  [bcc iron](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.95.087207), etc.
- Heterostructures
- ...


## Simulate the dynamics
OpenFerro simulates dynamical evolution of local order parameters with molecular dynamics (MD) and Landau-Lifshitz-Gilbert (LLG) equations of motion. 
See the [documentation](https://openferro.readthedocs.io/en/latest/theory-dynamics.html) for equations of motion used in NVE, NVT, NPT, and structure-optimization simulations. The isothermal condition is maintained by the second fluctuation-dissipation theorem.

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

The code for this superlattices simulation will be added to the [examples](https://github.com/salinelake/OpenFerro/tree/main/examples) soon.


# Benchmark
Running OpenFerro on a GPU node can bring over 100X speedup compared to a CPU node. See [example](https://github.com/salinelake/OpenFerro/tree/main/examples/Profiling_GPU) for details.

<p align="center" >
  <img width="30%" src="/examples/Profiling_GPU/GPU_benchmark.png" />
</p>  



# Troubleshooting
See [FAQ](https://openferro.readthedocs.io/en/latest/faq.html) for frequently asked questions.

# Credits
There will be a paper in the near future explaining the technical details of OpenFerro. Before that, please cite this repository for any use of OpenFerro.

The initial development of OpenFerro is done by Pinchen Xie with support from LBNL. 

# Model configuration
At this point, only a few publicly available model configurations are provided in the [model_configs](https://github.com/salinelake/OpenFerro/tree/main/model_configs) directory. We welcome contributions from the community to add more model configurations. 

# Contributing
We welcome contributions from the community. Raise an issue or submit a pull request. Also feel free to contact pinchenxie@lbl.gov for any questions.
