API Reference
=============

.. important::
   API availability does not imply scientific validation. Consult
   :doc:`feature_status` before selecting an engine or integrator.


System
------

.. automodule:: openferro.system
   :members:
   :undoc-members:
   :show-inheritance:

Fields
------

.. automodule:: openferro.field
   :members:
   :undoc-members:
   :show-inheritance:

Lattice
-------

.. automodule:: openferro.lattice
   :members:
   :undoc-members:
   :show-inheritance:

Hamiltonian
-----------

.. warning::
   Built-in engines outside the promoted scope in :doc:`feature_status` remain
   experimental and require independent validation.

Elastic
^^^^^^^

.. automodule:: openferro.engine.elastic
   :members:
   :undoc-members:
   :show-inheritance:

Ferroelectric
^^^^^^^^^^^^^

.. automodule:: openferro.engine.ferroelectric
   :members:
   :undoc-members:
   :show-inheritance:


Ferroelectric Superlattice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: openferro.engine.ferroelectric_superlatt
   :members:
   :undoc-members:
   :show-inheritance:

Multiferroic
^^^^^^^^^^^^

.. automodule:: openferro.engine.multiferroic
   :members:
   :undoc-members:
   :show-inheritance:

Magnetic
^^^^^^^^

.. automodule:: openferro.engine.magnetic
   :members:
   :undoc-members:
   :show-inheritance:

Ewald
^^^^^

.. automodule:: openferro.engine.ewald
   :members:
   :undoc-members:
   :show-inheritance:


Integrators
-----------

.. warning::
   Only the classes and field aliases marked stable in :doc:`feature_status`
   are covered by the Milestone B reference suite.

Base
^^^^

.. automodule:: openferro.integrator.base
   :members:
   :undoc-members:
   :show-inheritance:

Molecular Dynamics
^^^^^^^^^^^^^^^^^^

.. automodule:: openferro.integrator.md
   :members:
   :undoc-members:
   :show-inheritance:

Landau-Lifshitz-Gilbert
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: openferro.integrator.llg
   :members:
   :undoc-members:
   :show-inheritance:

Simulation
----------

.. automodule:: openferro.simulation
   :members:
   :undoc-members:
   :show-inheritance:

Interaction (Abstract)
----------------------

.. automodule:: openferro.interaction
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

Reporters
^^^^^^^^^

.. automodule:: openferro.reporter
   :members:
   :undoc-members:
   :show-inheritance:

Units
^^^^^

.. automodule:: openferro.units
   :members:
   :undoc-members:
   :show-inheritance:

Helper Functions
^^^^^^^^^^^^^^^^

.. automodule:: openferro.utilities
   :members:
   :undoc-members:
   :show-inheritance:

Parallelism
^^^^^^^^^^^

.. automodule:: openferro.parallelism
   :members:
   :undoc-members:
   :show-inheritance:
