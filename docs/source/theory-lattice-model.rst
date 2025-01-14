Crystalline system and Lattice Hamiltonian
==========================================

A crystalline system is a periodic arrangement of atoms or molecules in space. In OpenFerro, a crystalline system is defined by a Bravias lattice, local order parameter, global variables of the lattice (such as global strain) and a Hamiltonian describing the energy of the system.

Bravias lattice
---------------
A Bravias lattice is specified by a set of basis vectors. For example, a 3D Bravais lattice is an infinite array of discrete points described by 

.. math::

   \mathbf{R} = i \mathbf{a}_1 + j \mathbf{a}_2 + k \mathbf{a}_3

where :math:`i, j, k` are integers, and :math:`\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3` are the basis vectors.

Local order parameters
----------------------
Local order parameters describe the state of each lattice site. They can be vectors in :math:`R^d` (e.g. atomic displacements, electric dipoles) or elements of SO(3) (e.g. fixed-magnitude magnetic moments, molecular orientations). Denote :math:`\mathbf{u}_{n}` the local order parameter at site :math:`n=(i, j, k)`. OpenFerro stores the local order parameters in a 4D tensor, with the shape of :math:`(N_1, N_2, N_3, d)`, where :math:`N_1, N_2, N_3` are the number of lattice sites in the three directions, and :math:`d` is the dimension of the local order parameter. The fixed topology of the local order parameters make dynamical simulation much more efficient than generic molecular dynamics where the topology is not fixed.

Global variables
----------------
Global variables describe global properties of the lattice. In OpenFerro, the default global variable is the global strain tensor in the [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation): :math:`\eta = (\eta_1, \eta_2, \eta_3, \eta_4, \eta_5, \eta_6)=(\eta_{xx}, \eta_{yy}, \eta_{zz}, 2\eta_{yz}, 2\eta_{xz}, 2\eta_{xy})`. where :math:`\eta_{xx}, \eta_{yy}, \eta_{zz}` are the normal strains, and :math:`\eta_{yz}, \eta_{xz}, \eta_{xy}` are the shear strains.

Lattice Hamiltonian
-------------------
A lattice Hamiltonian :math:`E`, as a function of all local order parameters  and all global variables, is an energy function defined on the crystalline system. It typically consists of:
- On-site terms describing the local energetics at each site
- Interaction terms between different sites (e.g. dipole-dipole interactions)
- Global terms like elastic energy
- Interaction between system variables and external fields (e.g. electric field, magnetic field)

OpenFerro provides a flexible framework to construct lattice Hamiltonians by combining these energy terms. The energy terms are implemented as Python classes with a unified interface, making it easy to add new types of interactions.

Examples
--------

- **Perovskite ferroelectric materials:  BaTiO3**

-- Variables:
local dipole fields :math:`\mathbf{u}_{n}\in R^3`, local strain :math:`\eta^{\text{loc}}_{n} \in R^6`, global strain :math:`\eta \in R^6`.

-- Hamiltonian:
See [Physical Review B 52.9 (1995): 6301](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.52.6301) for details of the Hamiltonian.

- **Magnetic materials: Toy Ising model**

-- Variables:
spin fields :math:`\mathbf{u}_{n}\in SO(3)`. 

-- Hamiltonian:

.. math::

   E = \sum_{n\sim m} J \mathbf{s}_n \cdot \mathbf{s}_m

Here, :math:`n\sim m` means a nearest-neighbor pair. :math:`J` is the exchange coupling.

- **Magnetic materials: Bcc Iron**

-- Variables:
local dipole fields :math:`\mathbf{s}_{n}\in SO(3)`. 

-- Hamiltonian:

.. math::

   E = \sum_{n,m} J_{nm} \mathbf{s}_n \cdot \mathbf{s}_m

Here :math:`n,m` are summed over all pairs. :math:`J_{nm}` is the exchange coupling between site :math:`n` and :math:`m`. The exchange interaction can be cut off at a certain distance. For example, if the exchange interaction is cut off at the fourth shell. We have :math:`J_{nm}\in (J_1, J_2, J_3, J_4)`. 
See [PRL 95, 087207 (2005)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.95.087207) for details of the Hamiltonian.

- **Multiferroic materials: BiFeO3**

See [PRL 99, 227602 (2007)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.99.227602) for one possible realization of the Hamiltonian.

