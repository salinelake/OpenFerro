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

    Attributes
    ----------
    kb : float
        Boltzmann constant in eV/K
    amu : float
        Atomic mass unit in eV/(A/ps)^2
    epsilon0 : float
        Vacuum permittivity in e^2(eV*Angstrom)^-1
    elementary_charge : float
        Elementary charge in units of electron charge
    Angstrom : float
        Length unit (1.0)
    nm : float
        Nanometer in Angstroms
    ps : float
        Time unit (1.0)
    fs : float
        Femtosecond in ps
    ns : float
        Nanosecond in ps
    bar : float
        Pressure unit in eV/Angstrom^3
    V_Angstrom : float
        Electric field unit in V/Angstrom
    eV : float
        Energy unit (1.0)
    Joule : float
        Joule in eV
    Ry : float
        Rydberg in eV
    mRy : float
        Milli-Rydberg in eV
    hbar : float
        Reduced Planck constant in eV*ps
    muB : float
        Bohr magneton unit (1.0)
    Tesla : float
        Magnetic field unit in eV/muB
    electron_g_factor : float
        Electron g-factor (dimensionless)
    electron_gyro_ratio : float
        Electron gyromagnetic ratio in muB/eV/ps
    """
    ## physical constants
    kb = 8.6173303e-5  
    amu = 0.0001035 
    epsilon0 = 5.526349406e-3  
    elementary_charge = 1.0

    ## length units
    Angstrom = 1.0
    nm = 10.0 * Angstrom
    
    ## time units
    ps = 1
    fs = 1e-3 * ps
    ns = 1e3 * ps
    
    ## pressure units
    bar = 1e-6 / 1.602176621
    
    ## electric field units
    V_Angstrom = 1.0
    
    ## energy units
    eV = 1
    Joule = 6.241509e18
    Ry = 13.605693123
    mRy = 1e-3 * Ry
    hbar = 6.582119569e-4
    
    ## magnetic units
    muB = 1.0   # Bohr magneton
    Tesla = 5.7883818060e-5
    electron_g_factor = 2.00231930436
    electron_gyro_ratio = electron_g_factor / hbar
