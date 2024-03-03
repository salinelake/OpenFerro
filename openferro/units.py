"""
Classes which define the internal units. 
"""
# This file is part of OpenFerro.


import numpy as np

class Constants(object):
    """Class whose members are fundamental constants.

    Attributes:
        kb: Boltzmann constant.
        amu: Atomic mass unit.
    """

    kb = 8.6173303e-5  # eV/K
    amu = 1 # 1 amu = 1 g/mol
    epsilon0 = 5.526349406e-3  # e^2(eV*Angstrom)^-1
    bar = 1e-6 / 1.602176621 # eV/Angstrom^3
