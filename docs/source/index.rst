.. OpenFerro documentation master file, created by
   sphinx-quickstart on Mon Jan 13 04:09:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpenFerro Documentation
======================

OpenFerro is a Python package for on-lattice atomistic dynamics simulation. Built on `JAX <https://github.com/google/jax>`_, OpenFerro provides a high-performance framework for simulating lattice Hamiltonian models with GPU acceleration and auto-differentiation capabilities.

Key Features
-----------

* **GPU Support**: Highly efficient for large-scale simulations with multi-GPU parallelization
* **Auto-differentiable**: Native support for enhanced sampling and Hamiltonian optimization
* **Modular Design**: Easy implementation of new interaction terms without deep codebase knowledge
* **Flexible Architecture**: Supports simultaneous simulation of R^n and SO(3) local order parameters

Installation
------------

OpenFerro requires Python 3.8 or later. Install via pip:

.. code-block:: bash

   conda create -n openferro python=3.10
   conda activate openferro
   
   git clone https://github.com/salinelake/OpenFerro.git
   cd OpenFerro
   pip install .

Core Components
--------------

System
^^^^^^
The central class managing the simulation, handling fields and their interactions.

Fields
^^^^^^
Supported field types:

* Rn - For continuous vector fields
* SO3 - For spin/magnetic systems
* LocalStrain3D - For local strain fields
* GlobalStrain - For global lattice deformation

Interactions
^^^^^^^^^^^
Built-in interaction types:

* Dipole-dipole interactions with Ewald summation
* Short-range interactions up to 3rd nearest neighbors
* Elastic interactions
* Magnetic exchange interactions

Integrators
^^^^^^^^^^
Available integrators for time evolution:

* Gradient Descent
* Molecular Dynamics
* Landau-Lifshitz-Gilbert

Units
-----

OpenFerro uses the 'metal' unit system (same as LAMMPS):

* mass = grams/mole
* distance = Angstroms
* time = picoseconds
* energy = eV
* temperature = Kelvin
* magnetic moment = Bohr magneton
* magnetic field = eV/Bohr magneton

Examples
--------

Check the ``examples/`` directory for various simulation examples:

* BaTiO3 cooling simulation
* bcc Fe heating
* Simple cubic Ising model
* PbTiO3/SrTiO3 superlattice

Contributing
-----------

OpenFerro is open source under the MIT license. Contributions are welcome!

* Source Code: https://github.com/salinelake/OpenFerro
* Issue Tracker: https://github.com/salinelake/OpenFerro/issues

License
-------

This project is licensed under the MIT License.

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   api
