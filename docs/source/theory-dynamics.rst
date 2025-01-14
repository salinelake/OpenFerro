Simulate the dynamics
=====================


With the lattice, local order parameters, global variables and Hamiltonian defined, we can compute the equilibrium properties of the system by sampling configurations from the Boltzmann distribution. There are many algorithms to sample the Boltzmann distribution. Two major ones are molecular dynamics (MD) and Monte Carlo (MC). Monte Carlo sampling relies on constucting a Markovian process (not an actual dynamical physical process) that samples the Boltzmann distribution. Molecular dynamics, on the other hand, is associated with the Born-Oppenheimer dynamics of an atomic system, which corresponds to the actual physical process when all atomic degrees of freedom are considered. This gives access also to dynamical aspects (e.g. spectroscopic properties, domain dynamics, spin wave) of an atomic system. OpenFerro, up to this point, support only molecular dynamics simulation [#]_.

That being said, apparently, OpenFerro does not deal with all atomic degrees of freedom. Instead, OpenFerro deals with coarse-grained representation of atomic systems in terms of those local order parameters. So the generic, particle-based `molecular dynamics <https://en.wikipedia.org/wiki/Molecular_dynamics>`_ is not directly applicable here. However, we can treat those local order parameters, as well as global variables, as virtual particles, to fit in the generic framework of molecular dynamics, such that basic conclusions and algorithms of molecular dynamics can be applied. 

In the following, we introduce the equations of motion (EOM) for different ensembles. We will use `local order parameters` and `field` interchangeably.
Without loss of generality, we will assume a lattice system (site index :math:`n\in[1,N]`) with a :math:`R^d` field :math:`u=(u_n)`, a :math:`SO(3)` field :math:`s=(s_n)`, and a global strain tensor :math:`\eta`.
The extension to more fields is straightforward. The system Hamiltonian is given by :math:`E(u,s,\eta)`. The effective mass of the field :math:`u` is :math:`m_u`. The :math:`SO(3)` field is massless [#]_. The fixed scalar magnitude associated with the :math:`SO(3)` field is :math:`M`. The mass of the global strain tensor is :math:`m_\eta`.


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

For NVT (canonical ensemble) simulation, the strain tensor is fixed. The isothermal condition is enforced by the second fluctuation-dissipation theorem. The EOM is given by

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

Here, :math:`V=(1+\eta_{xx}+\eta_{yy}+\eta_{zz}) V_0` is the volume of the system. :math:`V_0` is the reference volume of the unit cell, which is set by the user. :math:`\xi_\eta` is a random force with zero mean and unit variance. Note that :math:`\xi_\eta` is not correlated with :math:`\xi_n`. Do not confuse the two. 

Structure Optimization
----------------------

Structure optimization is a process of finding the minimum energy configuration of a system. In OpenFerro, we support structure optimization with vanilla gradient descent method. The EOM is given by

.. math::
   m_u \frac{d u_n}{dt} = -\frac{\partial E(u,s,\eta)}{\partial u_n}


.. math::
   \frac{ds_n}{dt} = -\gamma_r s_n \times B_n - \gamma_r \alpha s_n \times (s_n \times B_n)

For flexible simulation cell, the strain tensor is optimized through

.. math::
   m_\eta \frac{d \eta}{dt} = -\frac{\partial (E(u,s,\eta) + P V)}{\partial \eta}


**References**

- MD: Rapaport, Dennis C. The art of molecular dynamics simulation. Cambridge university press, 2004.

- LLG: Eriksson, Olle, et al. Atomistic spin dynamics: foundations and applications. Oxford university press, 2017.

- MD + LLG: Wang, Dawei, Jeevaka Weerasinghe, and L. Bellaiche. "Atomistic molecular dynamic simulations of multiferroics." Physical Review Letters 109.6 (2012): 067203.


.. [#] Enhanced sampling methods (e.g. metadynamics, umbrella sampling) will be supported in the future. In the long term, kinetic Monte Carlo may also be supported to overcome the limitation of time scale.

.. [#] Massive :math:`SO(3)` field is more general (associated with inertial effect in spin dynamics) but much less common in literature. So currently we only support massless :math:`SO(3)` field, which can be simulated by the standard Landau-Lifshitz-Gilbert equation.
