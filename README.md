JAX-based framework for Lattice Hamiltonian simulation

# About OpenFerro
OpenFerro is a Python package for Lattice Hamiltonian simulation. OpenFerro is largely based on [JAX](https://github.com/google/jax), a high-performance linear algebra package supporting auto-differentiation and painless GPU executation.
OpenFerro is designed to minimize the effort required to build on-lattice Hamiltonian models, and to perform molecular dynamics (MD) simulations. 
 

# Highlighted features
* **GPU supports**, making it highly efficient for large-scale simulations.
* **highly modularized**, easy to implement new interaction terms in an lattice Hamiltonian model, benefitted from auto-differentiation.

# Milestones
(1, 90% completed) Ferroelectric effective Hamiltonian 

Ref: Zhong, W., David Vanderbilt, and K. M. Rabe. "First-principles theory of ferroelectric phase transitions for perovskites: The case of BaTiO3." Physical Review B 52.9 (1995): 6301.

(2) Landau-Lifshitz-Gilbert equation 

Ref: Eriksson, Olle, et al. Atomistic spin dynamics: foundations and applications. Oxford university press, 2017.


(3) Multiferroic 

Ref: Kornev, I. A., Lisenkov, S., Haumont, R., Dkhil, B., & Bellaiche, L. (2007). Finite-temperature properties of multiferroic BiFeO3. Physical review letters, 99(22), 227602.

# Credits

# OpenFerro in a nutshell

# Installation
Clone the package and pip install.

# Use OpenFerro

## Units

OpenFerro do not process unit conversion. The unit system is the same as the 'metal' unit system used in LAMMPS.

mass = grams/mole

distance = Angstroms

time = picoseconds

energy = eV

velocity = Angstroms/picosecond

force = eV/Angstrom

torque = eV

temperature = Kelvin

pressure = bars

dynamic viscosity = Poise

charge = multiple of electron charge (1.0 is a proton)

dipole = charge*Angstroms

electric field = volts/Angstrom

density = gram/cm^dim
