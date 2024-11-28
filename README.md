<p align="center" >
  <img width="60%" src="/docs/openferro_logo.png" />
</p>

A universal framework for on-lattice atomistic dynamics simulation

# About OpenFerro
OpenFerro is a Python package for on-lattice atomistic dynamics simulation. OpenFerro is largely based on [JAX](https://github.com/google/jax), a high-performance linear algebra package supporting auto-differentiation and painless GPU acceleration.
OpenFerro is designed to minimize the effort required to build on-lattice Hamiltonian models, and to perform molecular dynamics (MD) and Landau-Lifshitz-Gilbert simulations. 
 

# Highlighted features
* **GPU supports**, making it highly efficient for large-scale simulations.
* **highly modularized**, easy to implement new interaction terms in a lattice Hamiltonian model, benefitted from auto-differentiation.

# Credits

# OpenFerro in a nutshell

## Crystalline system and Lattice Hamiltonian

## Generic molecular dynamics and Landau-Lishiftz-Gilbert equations

# Installation
Clone the package and pip install.

# Use OpenFerro

## Code structure

## Examples

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

Magnetic dipole moment = Bohr magneton

Magnetic field = eV / Bohr magneton


