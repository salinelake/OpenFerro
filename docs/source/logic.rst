Logic of OpenFerro
==================

OpenFerro simulation is structured in an intuitive way. Before reading into the details, we recommend you to follow the hands-on tutorial_ first. 

A typical OpenFerro simulation program is structured as follows:

Define the lattice 
------------------
The following types of 3D/2D Bravias lattices are supported in OpenFerro:

* 3D Simple cubic (`openferro.lattice.SimpleCubic3D`)
* 3D Face-centered cubic (`openferro.lattice.FaceCenteredCubic3D`, not implemented yet)
* 3D Body-centered cubic (`openferro.lattice.BodyCenteredCubic3D`)
* 3D Hexagonal (`openferro.lattice.Hexagonal3D`, not implemented yet)
* 2D Square (`openferro.lattice.Square2D`)

A bravais lattice in OpenFerro is specified by the size of the lattice (the number of unit cells in each direction: :math:`l_1, l_2, l_3`) and the primitive vectors (:math:`a_1, a_2, a_3`).
So a lattice is created by, e.g.,

.. code-block:: python

   import jax.numpy as jnp
   l1,l2,l3 = 10,10,10  ## size of the lattice
   a1 = jnp.array([1.0,0.0,0.0])  ## primitive vector 1
   a2 = jnp.array([0.0,1.0,0.0])  ## primitive vector 2
   a3 = jnp.array([0.0,0.0,1.0])  ## primitive vector 3
   lat = of.BodyCenteredCubic3D(l1,l2,l3, a1,a2,a3)

No matter what type of 3D lattice, the d-dimensional local order parameters 
are always stored in an array of shape :math:`(l_1, l_2, l_3, d)`. 
Same strategy applies to 2D lattices.

So how do different classes of lattices differ from each other? In OpenFerro, each lattice class has attributes that specify the topology of the lattice, by indicating the displacement vectors from a site to all its n-th shell neighbors. For example, a `SimpleCubic3D` object has the attribute `first_shell_roller` that indicates the displacement vectors from a site to its six nearest neighbors. This will come in handy when you want to define translationally invariant short-range interactions. If you are developing new interaction terms in the Hamiltonian, take advantages of these roller funcitons (similar to `np.roll` in numpy but with preset displacement specifiers)! 

Define the Physical System
--------------------------
In OpenFerro, a `system` is associated with a lattice object. It should be created by, 

.. code-block:: python

   system = of.System(lat)

A `system` object has a dictionary of fields (we will use the term "field" and "local order parameter" interchangeably) that live on the lattice sites, as well as global variables associated with the lattice. 
Upon initialization, the dictonary is empty. So you need to add fields (jump to the next section for what is a field in OpenFerro) to the system by, e.g.,

.. code-block:: python

   toy_field = system.add_field(ID="Bianca", ftype="Rn", dim=3, mass = 1.0)
   spin_field = system.add_field(ID="spin", ftype="SO3")

Arbitrary number of fields can be added to the system with their unique IDs. 
The first line creates a 3D real-valued vector field (on-lattice) with :math:`l_1\times l_2\times l_3\times 3` degrees of freedom. The pointer to this field is stored in a dictionary in the `system` object with the key "Bianca". It can be retrieved by `system.get_field_by_ID("Bianca")`. 

Apparently, to have a classical mechanical descirption of the system, you also need to specify how these fields interact with themselves and with each other, given by the potential energy functional in the Hamiltonian. We call each term in the potential energy functional an `interaction`. 
There are two ways to add interactions to the system:

Add interaction to the system with built-in energy engine, through preset methods in system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This approach applies to widely-adopted Hamiltonian terms in the ferroelectric/magnetic materials literature. 
For example, the following code adds a Ising-like nearest neighbor interaction to the system for the 3D field with ID `Bianca` we just created:

.. code-block:: python

   J0 = 1.0
   system.add_isotropic_exchange_interaction_1st_shell(ID="Bianca-NN", field_ID="Bianca", coupling=J0)

Let's break down the arguments:

- `ID="Bianca-NN"`: the unique ID of the interaction, can not be the same as other interactions in the system.
- `field_ID="Bianca"`: the ID of the field to which the interaction is applied.
- `coupling=J0`: the interaction parameters.

Let the 3D vector :math:`u_{i}` denote the value of the field at site :math:`i`. This interaction term is given by :math:`\sum_{i\sim j} -J_0 u_{i}\cdot u_{j}`, where :math:`i\sim j` means :math:`i` and :math:`j` are nearest neighbors. 

A complete set of preset methods for adding interactions to the system can be found in the `system` module.

Add interaction to the system with custom energy engine, through a general interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This approach is the general way to add interactions to the system. Currently, there are three types of interactions supported in OpenFerro:

(Under construction)



What is a field in OpenFerro?
-----------------------------
A `Field` object in OpenFerro is associated with a lattice. It can be initialized without a `system` through (the `add_field` method is a shortcut that automatically associates the field to a system), e.g.,

.. code-block:: python

   toy_field = of.field.FieldRn(lattice=lat, ID="Bianca", dim=3)

Each `Field` object has a `_value` attribute that stores the values of the field as a jax.numpy array. 
Here, the instance of `FieldRn` stores a :math:`l_1\times l_2\times l_3\times 3` array as the values of the field (default value is 0). If you see the local order parameter on each site as a virtual particle, this is the position of the particle.  So naturally, a `Field` object also stores the mass associated with the field as a :math:`l_1\times l_2\times l_3\times 1` array. Mass is allowed to be different for each site. You can set mass by, e.g.,

.. code-block:: python

   toy_field.set_mass(mass=1.0)

or 

.. code-block:: python

   toy_field.set_mass(mass=jnp.ones(l1,l2,l3))

These will set the mass of the field on each site to 1.0. Note that the mass array won't be created until you set the mass. This is to save memory when the mass is not needed. 

Similarly, a `Field` object can store the velocity of the field, and the force acting on the field, all as :math:`l_1\times l_2\times l_3\times d` arrays. The velocity can be initialized with a finite-temperature distribution by, e.g.,

.. code-block:: python

   toy_field.init_velocity(mode == 'gaussian', temperature=1.0)

Currently, the following types of fields are supported:

* Abstract field (`openferro.field.Field`)
* Real-valued, general vector field (`openferro.field.FieldRn`)
* Real-valued scalar field (`openferro.field.FieldScalar`, alias of `FieldRn` with :math:`d=1`)
* Real-valued 3D vector field (`openferro.field.FieldR3`, alias of `FieldRn` with :math:`d=3`)
* SO(3)-valued field (`openferro.field.FieldSO3`)
* Real-valued, 6D vector field ('openferro.field.LocalStrain3D', with helper functions to calculate the local strain tensor)


TODO: Below is under construction

Setup simulation
------------------

Specify integrators
^^^^^^^^^^^^^^^^^^^

Specify reporters
^^^^^^^^^^^^^^^^^^

What is not covered here
------------------------



.. _tutorial: https://github.com/salinelake/OpenFerro/blob/main/tutorials/quickstart.ipynb


