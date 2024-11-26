"""
Classes which define the internal units. 
"""
# This file is part of OpenFerro.

import numpy as np

class Constants(object):
    """Class whose members are fundamental constants.
    """
    ## physical constants
    kb = 8.6173303e-5  # eV/K
    amu = 0.0001035 # atomic mass unit 1 amu = eV/(A/ps)^2
    epsilon0 = 5.526349406e-3  # e^2(eV*Angstrom)^-1
    elementary_charge = 1.0 # electron charge

    ## length units
    Angstrom = 1.0
    nm = 10.0 * Angstrom
    
    ## time units
    ps = 1
    fs = 1e-3 * ps
    ns = 1e3 * ps
    
    ## pressure units
    bar = 1e-6 / 1.602176621 # eV/Angstrom^3
    
    ## electric field units
    V_Angstrom = 1.0 # V/Angstrom
    
    ## energy units
    eV = 1
    Joule = 6.241509e18 # eV
    Ry = 13.605693123 # eV
    mRy = 1e-3 * Ry # eV
    hbar = 6.582119569e-4 # eV*ps
    
    ## magnetic units
    muB = 1.0   # Bohr magneton
    Tesla = 5.7883818060e-5 # eV/muB
    electron_g_factor = 2.00231930436 # dimensionless
    electron_gyro_ratio = electron_g_factor / hbar # muB/eV/ps
