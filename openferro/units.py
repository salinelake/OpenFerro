"""
Module containing classes that define the internal units.

Notes
-----
This file is part of OpenFerro.
"""

import numpy as np

class Constants(object):
    """
    Class containing fundamental physical constants and unit conversions. This class specifies the internal units of OpenFerro.
    """

    """base units"""
    Angstrom = 1.0  # Angstrom unit (1.0)
    length = Angstrom  # length unit of OpenFerro
    eV = 1.0  # eV unit (1.0)
    energy = eV  # energy unit of OpenFerro
    ps = 1.0  # ps unit (1.0)
    time = ps  # time unit of OpenFerro
    elementary_charge = 1.0  # elementary charge unit (1.0)
    charge = elementary_charge  # charge unit of OpenFerro
    Kelvin = 1.0  # Kelvin unit (1.0)
    temperature = Kelvin  # temperature unit of OpenFerro

    """derived units"""
    kb = 8.6173303e-5  # Boltzmann constant in eV/K
    amu = 0.0001035  # Atomic mass unit in eV/(A/ps)^2
    me = 0.00054858 * amu  # Electron mass in eV/(A/ps)^2
    epsilon0 = 5.526349406e-3  # Vacuum permittivity in e^2(eV*Angstrom)^-1

    ## length units
    meter = 1e10 * Angstrom  # Meter in Angstroms
    cm = 1e8 * Angstrom  # Centimeter in Angstroms
    mm = 1e7 * Angstrom  # Millimeter in Angstroms
    um = 1e4 * Angstrom  # Micrometer in Angstroms
    nm = 10.0 * Angstrom  # Nanometer in Angstroms
    pm = 1e-3 * Angstrom  # Picometer in Angstroms
    bohr_radius = 0.529177210  # Bohr radius in Angstrom

    ## time units
    second = 1e12 * ps  # Second in ps
    ms = 1e9 * ps  # Millisecond in ps
    us = 1e6 * ps  # Microsecond in ps
    ns = 1e3 * ps  # Nanosecond in ps
    fs = 1e-3 * ps  # Femtosecond in ps
    
    ## pressure units
    bar = 1e-6 / 1.602176621  # Pressure unit in eV/Angstrom^3
    
    ## electric units
    Coulomb = 1 / 1.602176634e-19  # Coulomb in elementary charge
    Ampere = Coulomb / second  # Electric current unit (C/s) in e/ps
    Voltage = eV / elementary_charge 
    V_Angstrom = Voltage / Angstrom  # Electric field unit in V/Angstrom
    
    ## energy units
    Joule = 6.241509e18  # Joule in eV
    Ry = 13.605693123  # Rydberg in eV
    mRy = 1e-3 * Ry  # Milli-Rydberg in eV
    hbar = 6.582119569e-4  # Reduced Planck constant in eV*ps
    hartree = 27.211386245988  # Hartree in eV
    
    ## magnetic unit
    """
    we will use Bohr magneton based unit system for magnetic interactions. This is not entirely consistent with the other base units. 
    For example, here mu0 will be in unit of eV * A^3 / muB^2, instead of eV * ps^2 / A * e^-2
    Derivation:
    _speed_of_light = 2997924.58   # Speed of light in A/ps
    _mu0 = 1 / _speed_of_light**2 / epsilon0  # Vacuum permeability in eV*ps^2/A*e^-2
    _muB = elementary_charge * hbar / 2 / me  # Bohr magneton in e * A^2 / ps
    mu0 = _mu0 * _muB**2 / Angstrom**3  # Vacuum permeability in eV * A^3 / muB^2
    """
    muB = 1.0  # Bohr magneton unit (1.0) 
    mu0 = 0.0006764429231627333 ## eV * A^3 / muB^2
    Tesla = 5.7883818060e-5  # Magnetic field unit in eV/muB
    electron_g_factor = 2.00231930436  # Electron g-factor (dimensionless)
    electron_gyro_ratio = electron_g_factor / hbar  # Electron gyromagnetic ratio in muB/eV/ps

    # ## magnetic units that are consistent with the other base units. Not used because bohr magneton is more common for magnetic dipole. 
    # muB = elementary_charge * hbar / 2 / me  # Bohr magneton in e * A^2 / ps
    # speed_of_light = 2997924.58   # Speed of light in A/ps
    # mu0 = 1 / speed_of_light**2 / epsilon0  # Vacuum permeability in eV*ps^2/A*e^-2
    # Tesla = Joule / Ampere / meter**2
    # electron_g_factor = 2.00231930436  # Electron g-factor (dimensionless)
    # electron_gyro_ratio = electron_g_factor / hbar  # Electron gyromagnetic ratio in muB/eV/ps


# class Constants_AtomicUnit(object):
#     """
#     Class containing fundamental physical constants and unit conversions. This class specifies the internal units of OpenFerro.
#     """

#     """base units"""
#     bohr_radius = 1.0
#     length = bohr_radius

#     hartree = 1.0
#     energy = hartree
    
#     electron_mass = 1.0
#     mass = electron_mass
    
#     elementary_charge = 1.0  # elementary charge unit (1.0)
#     charge = elementary_charge  

#     hbar = 1.0
#     time = hbar / hartree
    

#     """derived units"""
#     epsilon0 = 1/4/np.pi  # Vacuum permittivity
#     speed_of_light = 137   # Speed of light in A/ps
#     mu0 = 1 / speed_of_light**2 / epsilon0  # Vacuum permeability
#     amu = 1822.89
#     me = mass
#     pass

class AtomicUnits_to_InternalUnits(object):
    """
    Class defining atomic units, given in terms of the internal units of OpenFerro.
    """
    time = 2.418884326505e-5  # Atomic time in ps
    length = Constants.bohr_radius  # length unit of Atomic unit
    energy = Constants.hartree  # energy unit of Atomic unit
    mass = Constants.me  # mass unit of Atomic unit
    velocity = length / time  # velocity unit of Atomic unit
    force = energy / length  # force unit of Atomic unit
    electric_dipole_moment = Constants.elementary_charge * Constants.bohr_radius  # Atomic dipole moment in e*Angstrom

