.. OpenFerro documentation master file, created by
   sphinx-quickstart on Mon Jan 13 04:09:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpenFerro Documentation
=======================

OpenFerro is a Python package for on-lattice atomistic dynamics simulation. Built on `JAX <https://github.com/google/jax>`_, OpenFerro provides a high-performance framework for simulating lattice Hamiltonian models with GPU acceleration and auto-differentiation capabilities.

.. warning::
   OpenFerro is a research alpha. Review :doc:`feature_status` before using
   available Hamiltonians or integrators for scientific production.

Key Features
------------

* **JAX Execution**: CPU/GPU backends with experimental multi-device sharding
* **Auto-differentiable**: Forces derived from scalar custom energy functions
* **Modular Design**: Custom interactions use small, pure energy functions
* **Flexible Fields**: R^n and fixed-magnitude SO(3) local order parameters

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   feature_status
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory-lattice-model
   theory-dynamics

.. toctree::
   :maxdepth: 2
   :caption: Use OpenFerro
   
   logic
   core_components
   guide_custum_engine
   ewald_performance

.. toctree::
   :maxdepth: 2
   :caption: Others

   units
   credits
   faq

Examples
--------

Check the ``examples/`` directory for various simulation examples:

* BaTiO3 cooling simulation
* bcc Fe heating
* Simple cubic Ising model
* PbTiO3/SrTiO3 superlattice

Contributing
------------

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
