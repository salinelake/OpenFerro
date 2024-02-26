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
    def __init__(self, l1, l2, l3, a1=None, a2=None, a3=None):
        self.size = jnp.array([l1, l2, l3])
        self.a1 = jnp.array([1.0, 0.0, 0.0]) if a1 is None else a1
        self.a2 = jnp.array([0.0, 1.0, 0.0]) if a2 is None else a2
        self.a3 = jnp.array([0.0, 0.0, 1.0]) if a3 is None else a3
    def __repr__(self):
        return f"Bravais lattice with size {self.size} and primitive vectors {self.a1}, {self.a2}, {self.a3}"    
    
    @property
    def ref_volume(self):
        """
        Returns the volume of the unit cell
        """
        return jnp.dot(jnp.cross(self.a1, self.a2), self.a3)
    
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

    def get_neighbours(self, i, j, k, pbc=False):  # check
        """
        Returns the neighbours of a site at position (i, j, k) 
        """
        neighbours = []
        if pbs is True:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    for dk in [-1, 0, 1]:
                        if di == 0 and dj == 0 and dk == 0:
                            continue
                        neighbours.append([(i + di) % self.size[0], (j + dj) % self.size[1], (k + dk) % self.size[2]])
        else:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    for dk in [-1, 0, 1]:
                        if di == 0 and dj == 0 and dk == 0:
                            continue
                        if i + di >= 0 and i + di < self.size[0] and j + dj >= 0 and j + dj < self.size[1] and k + dk >= 0 and k + dk < self.size[2]:
                            neighbours.append([i + di, j + dj, k + dk])


class BravaisLattice2D:
    """
    A class to represent a 2D Bravais lattice
    """
    def __init__(self, l1, l2, a1=None, a2=None):
        self.size = jnp.array([l1, l2])
        self.a1 = jnp.array([1.0, 0.0]) if a1 is None else a1
        self.a2 = jnp.array([0.0, 1.0]) if a2 is None else a2
    def __repr__(self):
        return f"Bravais lattice with size {self.size} and primitive vectors {self.a1}, {self.a2}"    

    @property
    def ref_area(self):
        """
        Returns the area of the unit cell
        """
        return jnp.abs(jnp.cross(self.a1, self.a2))
    
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

    
    def get_neighbours(self, i, j, pbc=False):  # check
        """
        Returns the neighbours of a site at position (i, j) 
        """
        neighbours = []
        if pbs is True:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    neighbours.append([(i + di) % self.size[0], (j + dj) % self.size[1]])
        else:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    if i + di >= 0 and i + di < self.size[0] and j + dj >= 0 and j + dj < self.size[1]:
                        neighbours.append([i + di, j + dj])
        return neighbours