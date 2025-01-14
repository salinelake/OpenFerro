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
^^^^^^^^^^^^
Built-in interaction types:

* Dipole-dipole interactions with Ewald summation
* Short-range interactions up to 3rd nearest neighbors
* Elastic interactions
* Magnetic exchange interactions

Integrators
^^^^^^^^^^^
Available integrators for time evolution:

* Gradient Descent
* Molecular Dynamics
* Landau-Lifshitz-Gilbert