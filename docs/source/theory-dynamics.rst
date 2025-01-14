Simulate the dynamics
=====================


With the lattice, local order parameters, global variables and Hamiltonian defined, we can compute the equilibrium properties of the system by sampling configurations from the Boltzmann distribution. There are many algorithms to sample the Boltzmann distribution. Two major ones are molecular dynamics (MD) and Monte Carlo (MC). Monte Carlo sampling relies on constucting a Markovian process (not an actual dynamical physical process) that samples the Boltzmann distribution. Molecular dynamics, on the other hand, is associated with the Born-Oppenheimer dynamics of an atomic system, which corresponds to the actual physical process when all atomic degrees of freedom are considered. This gives access to dynamical aspects (e.g. spectroscopic properties, domain dynamics, spin wave) of an atomic system.

OpenFerro, up to this point, support only molecular dynamics simulation, since it is a more general approach to study the dynamics of a physical system. Enhanced sampling methods (e.g. metadynamics, umbrella sampling) will be supported in the future. In the long term, kinetic Monte Carlo may also be supported to overcome the limitation of time scale.

**Equations of motion**
OpenFerro deals with coarse-grained representation of atomic systems. The molecular dynamics simulated by OpenFerro will be different from the [generic molecular dynamics](https://en.wikipedia.org/wiki/Molecular_dynamics) (considering all atomic degrees of freedom) formulated under the Born-Oppenheimer approximation. The existence of $SO(3)$ fields further complicates the problem because the general molecular dynamics equations are not applicable to $SO(3)$ fields. So we need to have two sets of equations of motion for $R^d$ and $SO(3)$ fields.

- Equation of motion for $R^d$ fields
For $R^d$ local order parameters and unconstrained global variables, the equations of motion are:
**NVE ensemble**: the microcanonical ensemble is sampled by the Newton's equation of motion, driven by the potential energy $E$:
$$
\frac{d^2 \mathbf{u}_{n}}{dt^2} = -\nabla_{\mathbf{u}_n} E(\mathbf{u}_1, \cdots, \mathbf{u}_N)
$$

- Equation of motion for $SO(3)$ fields ([Landau-Lifshitz-Gilbert equation](https://en.wikipedia.org/wiki/Landau%E2%80%93Lifshitz%E2%80%93Gilbert_equation))


.. ### Generic molecular dynamics
.. A generic molecular dynamics simulation tracks the position of all atoms (position vector $R_i$ associated to atom-$i$) from an atomic system. 
.. - NVE ensemble: the microcanonical ensemble is sampled by the Newton's equation of motion, driven by the potential energy $E$:
.. $$
.. m_i\frac{d^2 \mathbf{R}_{i}}{dt^2} = -\nabla_{i} E(\mathbf{R_1}, \cdots, \mathbf{R_N})
.. $$

.. - NVT ensemble: the canonical ensemble can be sampled by the Langevin equation, driven by the potential energy $E$, the friction force and a random force:
.. $$
.. m_i\frac{d^2 \mathbf{R}_{i}}{dt^2} = -\nabla_{i} E(\mathbf{R_1}, \cdots, \mathbf{R_N}) - \gamma \frac{d \mathbf{R}_{i}}{dt} + \sqrt{2\gamma k_B T} \xi_i
.. $$
.. where $\gamma$ is the friction coefficient, $k_B$ is the Boltzmann constant, $T$ is the temperature, and $\xi_i$ is a random force with zero mean and variance $2\gamma k_B T$.

.. ### On-lattice atomisticdynamics

.. ### Landau-Lifshitz-Gilbert equation