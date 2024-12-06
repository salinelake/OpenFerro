"""
Classes to define the Bravais lattices
"""
# This file is part of OpenFerro.

import numpy as np
import jax.numpy as jnp

class BravaisLattice3D:
    """
    A class to represent a 3D Bravais lattice
    """
    def __init__(self, l1, l2, l3, a1=None, a2=None, a3=None, pbc=True):
        self.name = 'BravaisLattice3D'
        self.dim = 3
        self.size = jnp.array([l1, l2, l3])
        self.a1 = jnp.array([1.0, 0.0, 0.0]) if a1 is None else a1
        self.a2 = jnp.array([0.0, 1.0, 0.0]) if a2 is None else a2
        self.a3 = jnp.array([0.0, 0.0, 1.0]) if a3 is None else a3
        self.pbc = pbc
        if pbc is False:
            raise NotImplementedError("Non-periodic boundary conditions have not been implemented for 3D Bravais lattices.")
        
    def __repr__(self):
        return f"Bravais lattice with size {self.size} and primitive vectors {self.a1}, {self.a2}, {self.a3}"    
    
    ## total number of sites
    @property
    def nsites(self):
        return jnp.prod(self.size)


    @property
    def unit_volume(self):
        """
        Returns the volume of the unit cell
        """
        return jnp.abs(jnp.dot(jnp.cross(self.a1, self.a2), self.a3))
    
    @property
    def ref_volume(self):
        return self.unit_volume * jnp.prod(self.size)

    @property
    def latt_vec(self):
        """
        Returns lattice vectors.
        """
        return jnp.stack([self.a1, self.a2, self.a3], axis=0)
    
    @property
    def reciprocal_latt_vec(self):
        """
        Calculates reciprocal lattice vectors.
        """
        coef = 2 * jnp.pi / self.ref_volume
        b1 = coef * np.cross(self.a2,self.a3)  
        b2 = coef * np.cross(self.a3,self.a1) 
        b3 = coef * np.cross(self.a1,self.a2) 
        return jnp.stack([b1, b2, b3], axis=0)

class SimpleCubic3D(BravaisLattice3D):
    """
    A class to represent a simple cubic lattice. Coordination number is 6. First shell has 6 neighbours. Second shell has 12 neighbours. Third shell has 8 neighbours.
    """
    def __init__(self, l1, l2, l3, a1=None, a2=None, a3=None, pbc=True):
        super().__init__(l1, l2, l3, a1, a2, a3, pbc)
        self.name = 'SimpleCubic3D'
        self.first_shell_roller = [ x for x in self._1st_shell_roller()]
        self.second_shell_roller = [ x for x in self._2nd_shell_roller()]
        self.third_shell_roller = [ x for x in self._3rd_shell_roller()]

    def _1st_shell_roller(self):
        """
        Return a list of rolling functions for moving a site to all sites in the first shell of the lattice. 
        The number of rolling functions is half the shell coordination number.
        This will come useful when we want to calculate the interaction between a site and its neighbours.
        """
        roller = [
            lambda x: jnp.roll(x, 1, axis=0),
            lambda x: jnp.roll(x, 1, axis=1),
            lambda x: jnp.roll(x, 1, axis=2)
        ]
        return roller
    
    def _2nd_shell_roller(self):
        """
        Return a list of rolling functions for moving a site to all sites in the second shell of the lattice. 
        The number of rolling functions is half the shell coordination number.
        This will come useful when we want to calculate the interaction between a site and its neighbours.
        """
        roller = [
            lambda x: jnp.roll(x, (1, 1), axis=(0,1)),
            lambda x: jnp.roll(x, (-1,1), axis=(0,1)),

            lambda x: jnp.roll(x, (1, 1), axis=(0,2)),
            lambda x: jnp.roll(x, (-1,1), axis=(0,2)),

            lambda x: jnp.roll(x, (1, 1), axis=(1,2)),
            lambda x: jnp.roll(x, (-1,1), axis=(1,2)),
        ]
        return roller
    
    def _3rd_shell_roller(self):
        """
        Return a list of rolling functions for moving a site to all sites in the third shell of the lattice. 
        The number of rolling functions is half the shell coordination number.
        This will come useful when we want to calculate the interaction between a site and its neighbours.
        """
        roller = [
            lambda x: jnp.roll(x, (1, 1, 1), axis=(0,1,2)),
            lambda x: jnp.roll(x, (-1, 1, 1), axis=(0,1,2)),
            lambda x: jnp.roll(x, (1, -1, 1), axis=(0,1,2)),
            lambda x: jnp.roll(x, (-1, -1, 1), axis=(0,1,2))
        ]
        return roller

class BodyCenteredCubic3D(SimpleCubic3D):
    """
    A class to represent a body-centered cubic lattice. Coordination number is 8. 
    First shell has 8 neighbours. Second shell has 6 neighbours. Third shell has 12 neighbours. Fourth shell has 24 neighbours.
    """
    def __init__(self, l1, l2, l3, a1=None, a2=None, a3=None, pbc=True):
        super().__init__(l1, l2, l3, a1, a2, a3, pbc)
        self.name = 'BodyCenteredCubic3D'
        self.a1 = 0.5 * jnp.array([-1.0, 1.0, 1.0]) if a1 is None else a1
        self.a2 = 0.5 * jnp.array([ 1.0,-1.0, 1.0]) if a2 is None else a2
        self.a3 = 0.5 * jnp.array([ 1.0, 1.0,-1.0]) if a3 is None else a3
        self.first_shell_roller = [ x for x in self._1st_shell_roller()]
        self.second_shell_roller = [ x for x in self._2nd_shell_roller()]
        self.third_shell_roller = [ x for x in self._3rd_shell_roller()]
        self.fourth_shell_roller = [ x for x in self._4th_shell_roller()]

    def _1st_shell_roller(self):
        """
        Return a list of rolling functions for moving a site to all sites in the first shell of the lattice.
        The number of rolling functions is half the shell coordination number.
        This will come useful when we want to calculate the interaction between a site and its neighbours.
        """
        roller = [
            lambda x: jnp.roll(x, 1, axis=0),  # (-0.5, 0.5, 0.5)
            lambda x: jnp.roll(x, 1, axis=1),  # (0.5, -0.5, 0.5)
            lambda x: jnp.roll(x, 1, axis=2),  # (0.5, 0.5, -0.5)
            lambda x: jnp.roll(x, (1, 1, 1), axis=(0,1,2)),  # (0.5, 0.5, 0.5)
        ]
        return roller

    def _2nd_shell_roller(self):
        """
        Return a list of rolling functions for moving a site to all sites in the second shell of the lattice.
        The number of rolling functions is half the shell coordination number.
        This will come useful when we want to calculate the interaction between a site and its neighbours.
        """
        roller = [
            lambda x: jnp.roll(x, (1,1), axis=(1,2)),   # (1,0,0)
            lambda x: jnp.roll(x, (1,1), axis=(0,2)),   # (0,1,0)
            lambda x: jnp.roll(x, (1,1), axis=(0,1)),   # (0,0,1)
        ]
        return roller
    
    def _3rd_shell_roller(self):
        """
        Return a list of rolling functions for moving a site to all sites in the third shell of the lattice.
        The number of rolling functions is half the shell coordination number.
        This will come useful when we want to calculate the interaction between a site and its neighbours.
        """
        roller = [
            lambda x: jnp.roll(x, (1,2,1), axis=(0,1,2)),   # (1,0,1)
            lambda x: jnp.roll(x, (2,1,1), axis=(0,1,2)),   # (0,1,1)
            lambda x: jnp.roll(x, (1,1,2), axis=(0,1,2)),   # (1,1,0)
            lambda x: jnp.roll(x, (-1,1), axis=(0,2)),      # (-1,0,1)
            lambda x: jnp.roll(x, (1,-1), axis=(1,2)),      # (0,-1,1)
            lambda x: jnp.roll(x, (1,-1), axis=(0,1)),      # (-1,1,0)
        ]
        return roller

    def _4th_shell_roller(self):
        """
        Return a list of rolling functions for moving a site to all sites in the fourth shell of the lattice.
        The number of rolling functions is half the shell coordination number.
        This will come useful when we want to calculate the interaction between a site and its neighbours.
        """
        roller = [
            lambda x: jnp.roll(x, (1,2,2), axis=(0,1,2)),  # (1.5,0.5,0.5)
            lambda x: jnp.roll(x, (2,1), axis=(1,2)),  # (1.5,-0.5,0.5)
            lambda x: jnp.roll(x, (1,2), axis=(1,2)),  # (1.5,0.5,-0.5)
            lambda x: jnp.roll(x, (-1,1,1), axis=(0,1,2)), # (1.5,-0.5,-0.5)

            lambda x: jnp.roll(x, (2,1,2 ), axis=(0,1,2)),  # (0.5, 1.5,0.5)
            lambda x: jnp.roll(x, (1,2), axis=(0,2)),  # (0.5, 1.5,-0.5)
            lambda x: jnp.roll(x, (2,1), axis=(0,2)),  # (-0.5, 1.5,0.5)
            lambda x: jnp.roll(x, (1,-1,1), axis=(0,1,2)),  # (-0.5, 1.5,-0.5)

            lambda x: jnp.roll(x, (2,2,1), axis=(0,1,2)),  # (0.5, 0.5, 1.5)
            lambda x: jnp.roll(x, (2,1), axis=(0,1)),  # (-0.5, 0.5, 1.5)
            lambda x: jnp.roll(x, (1,2), axis=(0,1)),  # (0.5, -0.5, 1.5)
            lambda x: jnp.roll(x, (1,1,-1), axis=(0,1,2)),  # (-0.5, -0.5, 1.5)
        ]
        return roller

class FaceCenteredCubic3D(BodyCenteredCubic3D):
    """
    A class to represent a face-centered cubic lattice. Coordination number is 12. First shell has 12 neighbours. Second shell has 6 neighbours. Third shell has 24 neighbours.
    """
    def __init__(self, l1, l2, l3, a1=None, a2=None, a3=None, pbc=True):
        super().__init__(l1, l2, l3, a1, a2, a3, pbc)
        self.name = 'FaceCenteredCubic3D'
        self.a1 = jnp.array(0.5 *[1.0, 1.0, 0.0]) if a1 is None else a1
        self.a2 = jnp.array(0.5 *[1.0, 0.0, 1.0]) if a2 is None else a2
        self.a3 = jnp.array(0.5 *[0.0, 1.0, 1.0]) if a3 is None else a3
        raise NotImplementedError("Face-centered cubic lattice has not been implemented yet.")

class Hexagonal3D(BravaisLattice3D):
    """
    A class to represent a hexagonal lattice.  
    """
    def __init__(self, l1, l2, l3, a1=None, a2=None, a3=None, pbc=True):
        super().__init__(l1, l2, l3, a1, a2, a3, pbc)
        self.name = 'Hexagonal3D'
        self.a1 = jnp.array([jnp.sqrt(3)/2, 0.5, 0.0]) if a1 is None else a1
        self.a2 = jnp.array([jnp.sqrt(3)/2, -0.5, 0.0]) if a2 is None else a2
        self.a3 = jnp.array([0.0, 0.0, 1.0]) if a3 is None else a3

class BravaisLattice2D:
    """
    A class to represent a 2D Bravais lattice
    """
    def __init__(self, l1, l2, a1=None, a2=None, pbc=True):
        self.name = 'BravaisLattice2D'
        self.dim = 2
        self.size = jnp.array([l1, l2])
        self.a1 = jnp.array([1.0, 0.0]) if a1 is None else a1
        self.a2 = jnp.array([0.0, 1.0]) if a2 is None else a2
        self.pbc = pbc
        if pbc is False:
            raise NotImplementedError("Non-periodic boundary conditions have not been implemented for 2D Bravais lattices.")
        
    def __repr__(self):
        return f"Bravais lattice with size {self.size} and primitive vectors {self.a1}, {self.a2}"    

    @property
    def unit_area(self):
        """
        Returns the area of the unit cell
        """
        return jnp.abs(jnp.cross(self.a1, self.a2))
    
    @property
    def ref_area(self):
        return self.unit_area * jnp.prod(self.size)

    @property
    def latt_vec(self):
        """
        Returns lattice vectors.
        """
        return jnp.stack([self.a1, self.a2], axis=0)
    
    @property
    def reciprocal_latt_vec(self):
        """
        Calculates reciprocal lattice vectors.
        """
        coef = 2 * jnp.pi / (self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0])
        b1 = coef * jnp.array([self.a2[1], -self.a2[0]])
        b2 = coef * jnp.array([-self.a1[1], self.a1[0]])
        return jnp.stack([b1, b2], axis=0)
    
class SimpleSquare2D(BravaisLattice2D):
    """
    A class to represent a simple square lattice. Coordination number is 4. First shell has 4 neighbours. Second shell has 4 neighbours. Third shell has 4 neighbours.
    """
    def __init__(self, l1, l2, a1=None, a2=None, pbc=True):
        super().__init__(l1, l2, a1, a2, pbc)
        self.name = 'SimpleSquare2D'
        self.first_shell_roller = [ x for x in self._1st_shell_roller()]
        self.second_shell_roller = [ x for x in self._2nd_shell_roller()]
        self.third_shell_roller = [ x for x in self._3rd_shell_roller()]

    def _1st_shell_roller(self):
        """
        Return a list of rolling functions for moving a site to all sites in the first shell of the lattice.
        The number of rolling functions is half the shell coordination number.
        This will come useful when we want to calculate the interaction between a site and its neighbours.
        """
        roller = [lambda x: jnp.roll(x, 1, axis=i) for i in range(2)]
        return roller
    
    def _2nd_shell_roller(self):
        """
        Return a list of rolling functions for moving a site to all sites in the second shell of the lattice.
        The number of rolling functions is half the shell coordination number.
        This will come useful when we want to calculate the interaction between a site and its neighbours.
        """
        roller = [
            lambda x: jnp.roll(x, (1,1), axis=(0,1)),
            lambda x: jnp.roll(x, (-1,1), axis=(0,1)),
        ]
        return roller
    
    def _3rd_shell_roller(self):
        """
        Return a list of rolling functions for moving a site to all sites in the third shell of the lattice.
        The number of rolling functions is half the shell coordination number.
        This will come useful when we want to calculate the interaction between a site and its neighbours.
        """
        roller = [lambda x: jnp.roll(x, 2, axis=i) for i in range(2)]
        return roller
    
class Hexagonal2D(SimpleSquare2D):
    """
    A class to represent a hexagonal lattice. Coordination number is 6. 
    """
    def __init__(self, l1, l2, a1=None, a2=None):
        super().__init__(l1, l2, a1, a2)
        self.name = 'Hexagonal2D'
        self.a1 = jnp.array([1.0, 0.0]) if a1 is None else a1
        self.a2 = jnp.array([-0.5, jnp.sqrt(3)/2]) if a2 is None else a2
        raise NotImplementedError("Hexagonal lattice has not been implemented yet.")