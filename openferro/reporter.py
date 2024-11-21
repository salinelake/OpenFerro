"""
Reporter of the simulation.
"""
# This file is part of OpenFerro.

class Thermo_Reporter:
    def __init__(self, system):
        self.system = system
        raise NotImplementedError("Reporter is not implemented yet.")
    
class Dump_Reporter:
    def __init__(self, system):
        self.system = system
        raise NotImplementedError("Reporter is not implemented yet.")