JAX-based framework for Lattice Hamiltonian simulation

# About OpenFerro
OpenFerro is a Python package for Lattice Hamiltonian simulation. OpenFerro is largely based on [JAX](https://github.com/google/jax), a high-performance linear algebra package supporting auto-differentiation and painless GPU executation.
OpenFerro is designed to minimize the effort required to build on-lattice Hamiltonian models, and to perform molecular dynamics (MD) simulations. 
 

# Highlighted features
* **GPU supports**, making it highly efficient for large-scale simulations.
* **highly modularized**, easy to implement new interaction terms in an lattice Hamiltonian model, benefitted from auto-differentiation.

# Credits

# OpenFerro in a nutshell

# Download and install

# Use OpenFerro

## Units

OpenFerro do not process unit conversion. The internal unit system is the same as the 'metal' unit system used in LAMMPS.

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