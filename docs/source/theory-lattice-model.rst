Crystalline system and lattice Hamiltonian
==========================================

A crystalline system is a periodic arrangement of atoms or molecules in space.
In OpenFerro, its coarse-grained representation consists of a Bravais lattice,
local order parameters, optional global variables such as strain, and a
Hamiltonian describing the energy.

Bravais lattice
---------------
A Bravais lattice is specified by a set of primitive vectors. For example, a
three-dimensional Bravais lattice is an infinite array of points described by

.. math::

   \mathbf{R} = i \mathbf{a}_1 + j \mathbf{a}_2 + k \mathbf{a}_3

where :math:`i, j, k` are integers, and :math:`\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3` are the basis vectors.

Local order parameters
----------------------
Local order parameters describe the state of each lattice site. OpenFerro
stores real-valued fields in :math:`\mathbb{R}^d` and represents a fixed-
magnitude orientation with the three-component ``SO3`` field API. Examples
include atomic displacements, electric dipoles, atomistic spins, and molecular
orientations. A field with site value :math:`\mathbf{u}_n` has array shape
:math:`(N_1, N_2, N_3, d)`. Its fixed lattice topology enables regular JAX
array operations rather than particle-neighbor reconstruction.

Global variables
----------------
Global variables describe collective properties of the lattice. OpenFerro's
global strain uses engineering `Voigt notation
<https://en.wikipedia.org/wiki/Voigt_notation>`_:
:math:`\eta=(\eta_{xx},\eta_{yy},\eta_{zz},2\eta_{yz},2\eta_{xz},2\eta_{xy})`.
The factors of two on shear components are part of the public scientific
convention; see :doc:`scientific_conventions`.

Lattice Hamiltonian
-------------------
A lattice Hamiltonian :math:`E` is a scalar function of the system's local and
global fields. It typically contains:

- On-site terms describing the local energetics at each site
- Interaction terms between different sites (e.g. dipole-dipole interactions)
- Global terms like elastic energy
- Interaction between system variables and external fields (e.g. electric field, magnetic field)

OpenFerro combines these terms through self, mutual, and triple interaction
wrapper classes. The wrappers hold parameters and call scalar energy-engine
functions. Users normally provide only a pure JAX energy function; the wrapper
derives its force engine as the negative automatic-differentiation gradient.

Examples
--------

- **Perovskite ferroelectric materials:  BaTiO3**

-- Variables:
local dipole fields :math:`\mathbf{u}_{n}\in R^3`, local strain :math:`\eta^{\text{loc}}_{n} \in R^6`, global strain :math:`\eta \in R^6`.

-- Hamiltonian:
See `Physical Review B 52, 6301 (1995)
<https://journals.aps.org/prb/abstract/10.1103/PhysRevB.52.6301>`_ for details
of the Hamiltonian.

- **Magnetic materials: simple-cubic classical Heisenberg model**

-- Variables:
spin fields :math:`\mathbf{u}_{n}\in SO(3)`. 

-- Hamiltonian:

.. math::

   E = -\sum_{\langle n,m\rangle} J \mathbf{s}_n \cdot \mathbf{s}_m

Here, :math:`\langle n,m\rangle` is a unique undirected nearest-neighbor
displacement bond. Positive :math:`J` is ferromagnetic. The field contains
continuous fixed-magnitude vectors, not discrete Ising variables.

- **Magnetic materials: Bcc Iron**

-- Variables:
local spin fields :math:`\mathbf{s}_{n}\in SO(3)`.

-- Hamiltonian:

.. math::

   E = -\sum_{\langle n,m\rangle} J_{nm} \mathbf{s}_n \cdot \mathbf{s}_m

OpenFerro's engine sums unique displacement bonds. The bcc Fe source instead
uses ordered unit-spin pairs, so its four published shell values require the
conversion in :doc:`scientific_conventions`.
See `Physical Review Letters 95, 087207 (2005)
<https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.95.087207>`_ for
details of the Hamiltonian.

- **Multiferroic materials: BiFeO3**

See `Physical Review Letters 99, 227602 (2007)
<https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.99.227602>`_ for one
possible realization of the Hamiltonian.
