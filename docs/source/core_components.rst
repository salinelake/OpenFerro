Introduction
------------

(This page is currently under construction. Others are more or less complete.)
Before you get to know the core components of OpenFerro, we recommend you to follow the hands-on tutorial_ first. Then the purpose of most of the core components will be clear.

.. _tutorial: https://github.com/salinelake/OpenFerro/blob/main/tutorials/quickstart.ipynb

Lattice
-------

The 
Here is how we can define a 

System
------

The central class managing the simulation, handling fields and their interactions.

Fields
------
Supported field types:

* Rn - For R^n vector fields
* SO3 - For SO(3) vector fields
* LocalStrain3D - For local strain fields
* GlobalStrain - For global lattice deformation

Interactions
------------
Built-in interaction types:

* Dipole-dipole interactions with Ewald summation
* Short-range interactions up to 3rd nearest neighbors
* Elastic interactions
* Magnetic exchange interactions

Simulations
-----------

