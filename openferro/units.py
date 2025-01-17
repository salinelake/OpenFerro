"""
Module containing classes that define the internal units.

Notes
-----
This file is part of OpenFerro.
"""

import numpy as np

class Constants(object):
    """
    Class containing fundamental physical constants and unit conversions.
    """
    kb = 8.6173303e-5  # Boltzmann constant in eV/K
    amu = 0.0001035  # Atomic mass unit in eV/(A/ps)^2
    epsilon0 = 5.526349406e-3  # Vacuum permittivity in e^2(eV*Angstrom)^-1
    elementary_charge = 1.0  # Elementary charge in units of electron charge

    ## length units
    Angstrom = 1.0  # Length unit (1.0)
    nm = 10.0 * Angstrom  # Nanometer in Angstroms
    
    ## time units
    ps = 1  # Time unit (1.0)
    fs = 1e-3 * ps  # Femtosecond in ps
    ns = 1e3 * ps  # Nanosecond in ps
    
    ## pressure units
    bar = 1e-6 / 1.602176621  # Pressure unit in eV/Angstrom^3
    
    ## electric field units
    V_Angstrom = 1.0  # Electric field unit in V/Angstrom
    
    ## energy units
    eV = 1  # Energy unit (1.0)
    Joule = 6.241509e18  # Joule in eV
    Ry = 13.605693123  # Rydberg in eV
    mRy = 1e-3 * Ry  # Milli-Rydberg in eV
    hbar = 6.582119569e-4  # Reduced Planck constant in eV*ps
    
    ## magnetic units
    muB = 1.0  # Bohr magneton unit (1.0)
    Tesla = 5.7883818060e-5  # Magnetic field unit in eV/muB
    electron_g_factor = 2.00231930436  # Electron g-factor (dimensionless)
    electron_gyro_ratio = electron_g_factor / hbar  # Electron gyromagnetic ratio in muB/eV/ps
