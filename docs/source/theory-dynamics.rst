Simulate the dynamics
=====================


OpenFerro evolves coarse-grained lattice fields rather than all atomistic
coordinates. Unconstrained real-valued fields use molecular-dynamics-style
equations, while fixed-magnitude orientation fields use Landau-Lifshitz
dynamics. The package does not currently provide a Monte Carlo driver.

Effective masses and equations of motion are model choices: an OpenFerro
trajectory represents the declared coarse-grained dynamics and should not be
identified automatically with an all-atom Born-Oppenheimer trajectory.

In the following, we introduce the equations of motion (EOM) for different ensembles. We will use `local order parameters` and `field` interchangeably.
Without loss of generality, we will assume a lattice system (site index :math:`n\in[1,N]`) with a :math:`R^d` field :math:`u=(u_n)`, a :math:`SO(3)` field :math:`s=(s_n)`, and a global strain tensor :math:`\eta`.
Additional unconstrained fields follow the same orchestration. Simulation with
more than one coupled :math:`SO(3)` field is not implemented because SIB needs
a simultaneous implicit stage. The system Hamiltonian is given by
:math:`E(u,s,\eta)`. The effective mass of the field :math:`u` is :math:`m_u`.
The :math:`SO(3)` field is massless [#]_. The fixed scalar magnitude associated
with it is :math:`M`. The mass of the global strain tensor is :math:`m_\eta`.


NVE ensemble
------------

For NVE (microcanonical ensemble) simulation, the strain tensor is fixed. The dynamics preserves the energy. The EOM is given by

.. math::
   m_u \frac{d^2 u_n}{dt^2} = -\frac{\partial E(u,s,\eta)}{\partial u_n}

.. math::
   \frac{ds_n}{dt} = -\gamma_r s_n \times B_n

where :math:`B_n=-\frac{\partial E(u,s,\eta)}{\partial s_n}` is the effective magnetic field. :math:`\gamma_r` is the renormalized gyromagnetic ratio.

NVT ensemble
------------

For NVT (canonical ensemble) simulation, the strain tensor is fixed. Langevin
damping and stochastic noise obey the documented fluctuation-dissipation
relation. The EOM is given by

.. math::
   m_u \frac{d^2 u_n}{dt^2} = -\frac{\partial E(u,s,\eta)}{\partial u_n} - \gamma m_u \frac{d u_n}{dt} + \sqrt{2\gamma m_u k_B T} \xi_n

Here :math:`\gamma` is the friction coefficient, :math:`k_B` is the Boltzmann constant, :math:`T` is the temperature, and :math:`\xi_n` is a random force with zero mean and unit variance, i.e. a white noise.

.. math::
   \frac{ds_n}{dt} = -\gamma_r s_n \times (B_n+b_n) - \gamma_r \alpha s_n \times (s_n \times (B_n+b_n))

Here :math:`\alpha` is the Gilbert damping constant, which controls the damping effect of the :math:`SO(3)` field.
:math:`b_n` is a stochastic force satisfying  :math:`\langle b_{i,\alpha}(t) b_{j,\beta}(t') \rangle = 2D\delta_{ij} \delta_{\alpha\beta} \delta(t-t')`. :math:`i,j\in[1,N]` are the site indices, :math:`\alpha,\beta\in[1,d]` are the component indices. :math:`\delta` is the Dirac delta function. The constant :math:`D` is given by :math:`D=\frac{\alpha k_B T}{(1+\alpha^2)\gamma_r M}`.

NPT ensemble
------------

For NPT (isothermal-isobaric ensemble) simulation, the strain tensor is variable. Let :math:`P` be the target hydrostatic pressure. 
In OpenFerro, the local order parameters do not scale with global strain. So one can simply deal with the strain tensor as a heavy virtual particle. The EOM is given by

.. math::
   m_u \frac{d^2 u_n}{dt^2} = -\frac{\partial E(u,s,\eta)}{\partial u_n} - \gamma m_u \frac{d u_n}{dt} + \sqrt{2\gamma m_u k_B T} \xi_n

.. math::
   \frac{ds_n}{dt} = -\gamma_r s_n \times (B_n+b_n) - \gamma_r \alpha s_n \times (s_n \times (B_n+b_n))

.. math::
   m_\eta \frac{d^2 \eta}{dt^2} = -\frac{\partial (E(u,s,\eta) + P V)}{\partial \eta} - \gamma_\eta m_\eta \frac{d \eta}{dt} + \sqrt{2\gamma_\eta m_\eta k_B T} \xi_\eta

Here, :math:`V=V_0\det(I+\eta)` is the default system volume, with engineering
Voigt shear components divided by two when constructing the symmetric tensor
:math:`\eta`. :math:`V_0` is the reference volume set by the user. A selectable
linearized compatibility mode uses
:math:`V=V_0(1+\eta_{xx}+\eta_{yy}+\eta_{zz})`. :math:`\xi_\eta` is a random
force with zero mean and unit variance and is not correlated with :math:`\xi_n`.

The discrete leapfrog/LFMiddle velocity timing, SIB midpoint stages, and
restart state are specified in :doc:`scientific_conventions`.

Classical metadynamics
----------------------

``MetadynamicsNVT`` adds a classical fixed-height metadynamics bias to the NVT
Langevin driver.  A collective variable is a pure scalar function with the
same signature as an energy engine, ``engine(field1, ..., parameters)``.  CV
functions belong to the simulation and are not registered as system
interactions, so an observable such as a field sum contributes no physical
Hamiltonian energy or force by itself.

For example, a total dipole component can be configured without adding it to
the Hamiltonian:

.. code-block:: python

   def total_dipole_z(dipole, parameters):
       return parameters[0] * jnp.sum(dipole[..., 2])


   simulation = MetadynamicsNVT(
       system,
       collective_variables=[{
           "id": "total_dipole_z",
           "field_ids": "dipole",
           "engine": total_dipole_z,
           "parameters": [1.0],
       }],
       pace=100,
       sigma=0.02,
       height=0.001,
       grid_min=-1.5,
       grid_max=1.5,
       upper_walls={"at": 1.0, "kappa": 0.1},
   )

For :math:`1\leq d\leq3` collective variables and deposited centers
:math:`c^{(k)}`, the bias is

.. math::
   V_{\mathrm{meta}}(s) = \sum_k h \exp\left[-\frac{1}{2}
   \sum_{i=1}^d \frac{(s_i-c_i^{(k)})^2}{\sigma_i^2}\right].

The fixed height :math:`h` is in eV.  Each width :math:`\sigma_i` uses the
native units of its CV.  Optional upper and lower walls use

.. math::
   V_{\mathrm{upper}}(s) = \sum_i \kappa_i
   \left[\frac{s_i-a_i+o_i}{\epsilon_i}\right]_+^{e_i},

.. math::
   V_{\mathrm{lower}}(s) = \sum_i \kappa_i
   \left[\frac{a_i+o_i-s_i}{\epsilon_i}\right]_+^{e_i},

where :math:`[z]_+=\max(z,0)`.  ``KAPPA`` is in eV; ``AT``, ``EPS``, and
``OFFSET`` use the corresponding CV units.  JAX differentiates each CV only
to assemble the additional field force

.. math::
   F_{\mathrm{bias}}^{(x)} = -\sum_i
   \frac{\partial V_{\mathrm{bias}}}{\partial s_i}
   \frac{\partial s_i}{\partial x}.

A hill is deposited at the updated state after every ``pace`` accepted outer
MD steps and affects the next force evaluation.  Calls made internally by an
implicit SO(3) solver do not deposit hills.  At deposition, the Gaussian is
accumulated on a fixed grid spanning ``grid_min`` through ``grid_max``.  The
bias and its derivative are evaluated with tensor-product cubic interpolation,
so their cost does not grow with the number of hills.  ``grid_bin`` gives the
number of intervals in each direction; when omitted, OpenFerro chooses a
spacing no larger than :math:`\sigma_i/5`.  The grid must cover the sampled CV
range.  Values outside it use the boundary bias, so walls should be placed
inside the grid to return the trajectory to the represented region. Active
walls configured outside the grid are rejected.

The fixed grid has :math:`\prod_i(n_i+1)` values, making it suitable for the
supported one to three CVs but increasingly memory-intensive with dimension.
Adaptive widths, multiple walkers, restart import, and well-tempered
metadynamics are not implemented.

The optional HILLS file records step, exact CV centers, widths, and height.
It remains independent of the interpolated runtime grid and can reconstruct
the deposited Gaussian sum directly.  Repeated
``run`` calls on one object continue its history, but a fresh simulation
refuses to overwrite an existing HILLS file.  ``Thermo_Reporter`` continues to
report the physical system potential; use ``calc_total_bias`` or
``calc_biased_potential_energy`` for bias-aware energies.  The
``examples/07.MetaDynamics`` workflow compares a two-CV run with an exactly
known toy free-energy surface.

Structure Optimization
----------------------

Structure optimization finds a local minimum-energy configuration. OpenFerro
0.2 provides gradient-descent minimization with the EOM

.. math::
   m_u \frac{d u_n}{dt} = -\frac{\partial E(u,s,\eta)}{\partial u_n}


.. math::
   \frac{ds_n}{dt} = -\gamma_r s_n \times B_n - \gamma_r \alpha s_n \times (s_n \times B_n)

For flexible simulation cell, the strain tensor is optimized through

.. math::
   m_\eta \frac{d \eta}{dt} = -\frac{\partial (E(u,s,\eta) + P V)}{\partial \eta}

``MDMinimize`` evaluates forces at the initial state and after every accepted
update. Convergence and force-based reporting therefore describe the stored
field values, and a state already below the force tolerance returns with zero
optimization iterations.


**References**

- MD: Rapaport, Dennis C. The art of molecular dynamics simulation. Cambridge university press, 2004.

- LLG: Eriksson, Olle, et al. Atomistic spin dynamics: foundations and applications. Oxford university press, 2017.

- MD + LLG: Wang, Dawei, Jeevaka Weerasinghe, and L. Bellaiche. "Atomistic molecular dynamic simulations of multiferroics." Physical Review Letters 109.6 (2012): 067203.


.. [#] Massive :math:`SO(3)` field is more general (associated with inertial effect in spin dynamics) but much less common in literature. So currently we only support massless :math:`SO(3)` field, which can be simulated by the standard Landau-Lifshitz-Gilbert equation.
