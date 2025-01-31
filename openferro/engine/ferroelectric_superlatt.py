"""
Functions that define a ferroelectric term in the Hamiltonian. They will be added into <class interaction> for automatic differentiation.
"""
# This file is part of OpenFerro.

from openferro.engine.ferroelectric import *

def get_self_energy_onsite_on_AmBnLattice(lattice, m, n):
    """
    Returns the engine of the self-energy of a field on an a AmBn superlattice. The stacking direction is along the z-axis.

    Parameters
    ----------
    lattice : object
        The lattice object containing size information
    m : int
        Size of A layer
    n : int 
        Size of B layer

    Returns
    -------
    function
        Energy engine function

    Notes
    -----
    m+n has to be the size of the simulation cell along the z-axis.
    """
    l1, l2, l3 = lattice.size
    if l3 != m+n:
        raise ValueError("The size of the lattice along the z-axis must be equal to m+n.")
    if m==1 or n==1:
        raise ValueError("m and n should be larger than 1.")
    
    def energy_engine(field, parameters):
        """
        Returns the isotropic self-energy of a 3D field.
        See Eq.(2-3) in [Zhong, W., David Vanderbilt, and K. M. Rabe. Physical Review B 52.9 (1995): 6301.] for meaning of the parameters.

        Parameters
        ----------
        field : jnp.array
            The field to calculate the energy
        parameters : jax.numpy array
            The parameters of the energy function

        Returns
        -------
        jnp.array
            The energy of the field
        """
        para_A = parameters[:3]
        para_B = parameters[3:]
        para_I = (para_A + para_B)/2.0  # interface

        field2_A = field[:, :, :m-1,:] ** 2
        field2_B = field[:, :, m:-1,:] ** 2
        field2_I = field[:, :, [m-1,-1],:] ** 2 # interface

        energy = 0
        for f, p  in zip([field2_A, field2_B, field2_I], [para_A, para_B, para_I]):
            energy += p[0] * jnp.sum(f)
            energy += p[1] * jnp.sum( (f.sum(axis=-1))**2 )
            energy += p[2] * jnp.sum(
                f[...,0]*f[...,1] + f[...,1]*f[...,2] + f[...,2]*f[...,0]
                )

        return energy
    return energy_engine

def get_short_range_1stnn_on_AmBnLattice(lattice, m, n):
    """
    Returns the engine of the short-range interaction of nearest neighbors for a field on an a AmBn superlattice.

    Parameters
    ----------
    lattice : object
        The lattice object containing size information
    m : int
        Size of A layer
    n : int
        Size of B layer

    Returns
    -------
    function
        Energy engine function

    Notes
    -----
    The stacking direction is along the z-axis.
    m+n has to be the size of the simulation cell along the z-axis.
    """
    l1, l2, l3 = lattice.size
    if l3 != m+n:
        raise ValueError("The size of the lattice along the z-axis must be equal to m+n.")
    if m==1 or n==1:
        raise ValueError("m and n should be larger than 1.")
    
    def energy_engine(field, parameters):
        """
        Returns the short-range interaction of nearest neighbors for a R^3 field defined on a isotropic lattice with periodic boundary conditions.

        Parameters
        ----------
        field : jnp.array
            The field to calculate the energy
        parameters : jax.numpy array
            The parameters of the energy function

        Returns
        -------
        jnp.array
            The energy of the field
        """
        para_A = parameters[:2]
        para_B = parameters[2:4]
        para_I = (para_A + para_B)/2.0

        field_A = field[:, :, :m-1, :]
        field_B = field[:, :, m:-1, :]
        field_I = field[:, :, [m-1,-1], :]
        energy = 0
        ## x-axis and y-axis
        for axis in [0,1]:
            for f, p in zip([field_A, field_B, field_I], [para_A, para_B, para_I]):
                f_shifted = jnp.roll(f, 1, axis=axis)
                energy += p[0] * jnp.sum(f * f_shifted)
                energy += (p[1] - p[0]) * jnp.sum(f[..., axis] * f_shifted[..., axis])
        ## z-axis
        ## A-type interaction, including A-I interfacial interaction
        field_1 = jnp.concatenate([field[:,:,[-1]], field_A], axis=2)
        field_2 = jnp.concatenate([field_A, field[:,:,[m-1]]], axis=2)
        energy += para_A[0] * jnp.sum(field_1 * field_2)
        energy += (para_A[1] - para_A[0]) * jnp.sum(field_1[..., 2] * field_2[..., 2])
        ## B-type interaction, including B-I interfacial interaction
        field_1 = jnp.concatenate([field[:,:,[m-1]], field_B], axis=2)
        field_2 = jnp.concatenate([field_B, field[:,:,[-1]]], axis=2)
        energy += para_B[0] * jnp.sum(field_1 * field_2)
        energy += (para_B[1] - para_B[0]) * jnp.sum(field_1[..., 2] * field_2[..., 2])
        return energy
    return energy_engine

def get_short_range_2ednn_on_AmBnLattice(lattice, m, n):
    """
    Returns the engine of the short-range interaction of second nearest neighbors for a field on an a AmBn superlattice.

    Parameters
    ----------
    lattice : object
        The lattice object containing size information
    m : int
        Size of A layer
    n : int
        Size of B layer

    Returns
    -------
    function
        Energy engine function

    Notes
    -----
    The stacking direction is along the z-axis.
    m+n has to be the size of the simulation cell along the z-axis.
    """
    l1, l2, l3 = lattice.size
    if l3 != m+n:
        raise ValueError("The size of the lattice along the z-axis must be equal to m+n.")
    if m==1 or n==1:
        raise ValueError("m and n should be larger than 1.")
    
    def energy_engine(field, parameters):
        """
        Returns the short-range interaction of nearest neighbors for a R^3 field defined on a lattice with periodic boundary conditions.

        Parameters
        ----------
        field : jnp.array
            The field to calculate the energy
        parameters : jax.numpy array
            The parameters of the energy function

        Returns
        -------
        jnp.array
            The energy of the field
        """
        para_A = parameters[:3]  # j3, j4, j5
        para_B = parameters[3:6] 
        para_I = (para_A + para_B)/2.0
        
        field_A = field[:,:,:m-1,:]
        field_B = field[:,:,m:-1,:]
        field_I = field[:,:, [m-1,-1],:]
        energy = 0

        ## x-y plane
        axis_pair = (0,1)
        for f, p in zip([field_A, field_B, field_I], [para_A, para_B, para_I]):
            j3, j4, j5 = p
            f1 = jnp.roll( f, (1, 1), axis=axis_pair) 
            f2 = jnp.roll( f, (1,-1), axis=axis_pair) 

            # Uni-axis interactions
            energy += j3 * jnp.sum(f * (f1 + f2))
            energy += (j4 - j3) * jnp.sum(f[..., 3 - axis_pair[0] - axis_pair[1]] * (f1 + f2)[..., 3 - axis_pair[0] - axis_pair[1]])
            # Orthogonal-axis interactions
            energy += j5 * jnp.sum(f[..., [axis_pair[0], axis_pair[1]]] * (f1 - f2)[..., [axis_pair[1], axis_pair[0]]])

        ## x-z plane and y-z plane
        for axis_pair in [(0,2), (1,2)]:
            ## A-type interaction, including A-I interfacial interaction
            j3, j4, j5 = para_A
            f = jnp.concatenate([field[:,:,[-1]], field_A], axis=2)
            f1 = jnp.roll(jnp.concatenate([field_A, field[:,:,[m-1]]], axis=2), -1, axis=axis_pair[0])
            f2 = jnp.roll(jnp.concatenate([field_A, field[:,:,[m-1]]], axis=2), 1, axis=axis_pair[0])
            # Uni-axis interactions
            energy += j3 * jnp.sum(f * (f1 + f2))
            energy += (j4 - j3) * jnp.sum(f[..., 3 - axis_pair[0] - axis_pair[1]] * (f1 + f2)[..., 3 - axis_pair[0] - axis_pair[1]])
            # Orthogonal-axis interactions, be careful with the sign
            energy += j5 * jnp.sum(f[..., [axis_pair[0], axis_pair[1]]] * (f1 - f2)[..., [axis_pair[1], axis_pair[0]]])
            
            ## B-type interaction, including B-I interfacial interaction
            j3, j4, j5 = para_B
            f = jnp.concatenate([field[:,:,[m-1]], field_B], axis=2)
            f1 = jnp.roll(jnp.concatenate([field_B, field[:,:,[-1]]], axis=2), -1, axis=axis_pair[0])
            f2 = jnp.roll(jnp.concatenate([field_B, field[:,:,[-1]]], axis=2), 1, axis=axis_pair[0])
            # Uni-axis interactions
            energy += j3 * jnp.sum(f * (f1 + f2))
            energy += (j4 - j3) * jnp.sum(f[..., 3 - axis_pair[0] - axis_pair[1]] * (f1 + f2)[..., 3 - axis_pair[0] - axis_pair[1]])
            # Orthogonal-axis interactions, be careful with the sign 
            energy += j5 * jnp.sum(f[..., [axis_pair[0], axis_pair[1]]] * (f1 - f2)[..., [axis_pair[1], axis_pair[0]]])
        return energy
    return energy_engine


def get_short_range_3rdnn_on_AmBnLattice(lattice, m, n):
    """
    Returns the engine of the short-range interaction of second nearest neighbors for a field on an a AmBn superlattice.

    Parameters
    ----------
    lattice : object
        The lattice object containing size information
    m : int
        Size of A layer
    n : int
        Size of B layer

    Returns
    -------
    function
        Energy engine function

    Notes
    -----
    The stacking direction is along the z-axis.
    m+n has to be the size of the simulation cell along the z-axis.
    """
    l1, l2, l3 = lattice.size
    if l3 != m+n:
        raise ValueError("The size of the lattice along the z-axis must be equal to m+n.")
    if m==1 or n==1:
        raise ValueError("m and n should be larger than 1.")
    c_0 = jnp.eye(3)
    c_1 = jnp.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        ])
    c_2 = jnp.array([
        [ 0, -1,  1],
        [-1,  0, -1],
        [ 1, -1,  0],
        ])    
    c_3 = jnp.array([
        [ 0, -1, -1],
        [-1,  0,  1],
        [-1,  1,  0],
        ])
    c_4 = jnp.array([
        [ 0,  1, -1],
        [ 1,  0, -1],
        [-1, -1,  0],
        ])
    
    def energy_engine(field, parameters):
        """
        Returns the engine of short-range interaction of third nearest neighbors for a R^3 field defined on a superlattice

        Parameters
        ----------
        field : jnp.array
            The field to calculate the energy
        parameters : jax.numpy array
            The parameters of the energy function

        Returns
        -------
        jnp.array
            The energy of the field
        """
        field_A = field[:,:,:m-1,:]
        field_B = field[:,:,m:-1,:]
        field_I1 = field[:,:,[m-1],:]
        field_I2 = field[:,:,[-1],:]
        energy = 0 

        ##### A-type interaction #####
        j6, j7 = parameters[:2]
        ## get R_ij_alpha * R_ij_beta for different i-j displacement
        r_1 = c_1 * j7 + c_0 * j6
        r_2 = c_2 * j7 + c_0 * j6
        r_3 = c_3 * j7 + c_0 * j6
        r_4 = c_4 * j7 + c_0 * j6
        ## sum up the interaction
        f1 = jnp.concatenate([field_I2, field_A], axis=2)
        f2 = jnp.concatenate([field_A, field_I1], axis=2)
        energy += jnp.sum(f1 * jnp.dot( jnp.roll( f2, ( 1, 1), axis=(0,1)), r_1))
        energy += jnp.sum(f1 * jnp.dot( jnp.roll( f2, ( 1,-1), axis=(0,1)), r_2))
        energy += jnp.sum(f1 * jnp.dot( jnp.roll( f2, (-1, 1), axis=(0,1)), r_3))
        energy += jnp.sum(f1 * jnp.dot( jnp.roll( f2, (-1,-1), axis=(0,1)), r_4))

        ##### B-type interaction #####
        j6, j7 = parameters[2:4]
        ## get R_ij_alpha * R_ij_beta for different i-j displacement
        r_1 = c_1 * j7 + c_0 * j6
        r_2 = c_2 * j7 + c_0 * j6
        r_3 = c_3 * j7 + c_0 * j6
        r_4 = c_4 * j7 + c_0 * j6
        ## sum up the interaction
        f1 = jnp.concatenate([field_I1, field_B], axis=2)
        f2 = jnp.concatenate([field_B, field_I2], axis=2)
        energy += jnp.sum(f1 * jnp.dot( jnp.roll( f2, ( 1, 1), axis=(0,1)), r_1))
        energy += jnp.sum(f1 * jnp.dot( jnp.roll( f2, ( 1,-1), axis=(0,1)), r_2))
        energy += jnp.sum(f1 * jnp.dot( jnp.roll( f2, (-1, 1), axis=(0,1)), r_3))
        energy += jnp.sum(f1 * jnp.dot( jnp.roll( f2, (-1,-1), axis=(0,1)), r_4))
        return energy
    return energy_engine


def get_homo_strain_dipole_interaction_on_AmBnLattice(lattice, m, n):
    """
    Returns the engine of the homogeneous strain dipole interaction for an AmBn superlattice.

    Parameters
    ----------
    lattice : object
        The lattice object containing size information
    m : int
        Size of A layer
    n : int
        Size of B layer

    Returns
    -------
    function
        Energy engine function

    Notes
    -----
    The stacking direction is along the z-axis.
    m+n has to be the size of the simulation cell along the z-axis.
    """
    l1, l2, l3 = lattice.size
    if l3 != m+n:
        raise ValueError("The size of the lattice along the z-axis must be equal to m+n.")
    if m==1 or n==1:
        raise ValueError("m and n should be larger than 1.")
    
    def energy_engine(global_strain, dipole_field, parameters):
        """
        Returns the homogeneous strain dipole interaction energy.

        Parameters
        ----------
        global_strain : jnp.array
            Shape=(6), the global strain of a supercell
        dipole_field : jnp.array
            Shape=(nx, ny, nz, 3), the dipole field
        parameters : jax.numpy array
            The parameters of the energy function containing:
            'B1xx': float, elastic constant B1xx
            'B1yy': float, elastic constant B1yy
            'B4yz': float, elastic constant B4yz
            'offset': float, offset of the dipole field

        Returns
        -------
        jnp.array
            The homogeneous strain dipole interaction energy
        """
        para_A = parameters[:3] # B1xx, B1yy, B4yz
        para_B = parameters[3:]
        para_I = (para_A + para_B)/2.0  # interface

        field_A = dipole_field[:, :, :m-1,:] 
        field_B = dipole_field[:, :, m:-1,:] 
        field_I = dipole_field[:, :, [m-1,-1],:] # interface
        gs = global_strain
        
        energy = 0
        for f, p  in zip([field_A, field_B, field_I], [para_A, para_B, para_I]):
            B1xx, B1yy, B4yz = p            
            energy +=  (B1xx*gs[0] + B1yy*(gs[1]+gs[2])) * (f[...,0]**2).sum() * 0.5
            energy +=  (B1xx*gs[1] + B1yy*(gs[0]+gs[2])) * (f[...,1]**2).sum() * 0.5
            energy +=  (B1xx*gs[2] + B1yy*(gs[0]+gs[1])) * (f[...,2]**2).sum() * 0.5
            energy +=  B4yz*gs[5] * (f[...,0] * f[...,1]).sum()
            energy +=  B4yz*gs[4] * (f[...,0] * f[...,2]).sum()
            energy +=  B4yz*gs[3] * (f[...,1] * f[...,2]).sum()
        return energy
    return energy_engine
 

def get_inhomo_strain_dipole_interaction_on_AmBnLattice(lattice, m, n):
    """
    Returns the engine of the inhomogeneous strain dipole interaction for an AmBn superlattice.

    Parameters
    ----------
    lattice : object
        The lattice object containing size information
    m : int
        Size of A layer
    n : int
        Size of B layer

    Returns
    -------
    function
        Energy engine function

    Notes
    -----
    The stacking direction is along the z-axis.
    m+n has to be the size of the simulation cell along the z-axis.
    """
    l1, l2, l3 = lattice.size
    if l3 != m+n:
        raise ValueError("The size of the lattice along the z-axis must be equal to m+n.")
    if m==1 or n==1:
        raise ValueError("m and n should be larger than 1.")
    get_local_strain = jit(LocalStrain3D.get_local_strain)
    def energy_engine(local_displacement, dipole_field, parameters):
        """
        Returns the inhomogeneous strain dipole interaction energy.

        Parameters
        ----------
        local_displacement : jnp.array
            Shape=(nx, ny, nz, 3), the local displacement field
        dipole_field : jnp.array
            Shape=(nx, ny, nz, 3), the dipole field
        parameters : jax.numpy array
            The parameters of the energy function containing:
            'B1xx': float, elastic constant B1xx
            'B1yy': float, elastic constant B1yy
            'B4yz': float, elastic constant B4yz

        Returns
        -------
        jnp.array
            The inhomogeneous strain dipole interaction energy
        """
        para_A = parameters[:3]
        para_B = parameters[3:]
        para_I = (para_A + para_B)/2.0
        local_strain = get_local_strain(local_displacement)  # (l1, l2, l3, 6)
        ls_A = local_strain[:,:,:m-1]
        ls_B = local_strain[:,:,m:-1]
        ls_I = local_strain[:,:,[m-1,-1]]
        dp_A = dipole_field[:,:,:m-1]
        dp_B = dipole_field[:,:,m:-1]
        dp_I = dipole_field[:,:,[m-1,-1]]
        energy = 0
        for ls, dp, para in zip([ls_A, ls_B, ls_I], [dp_A, dp_B, dp_I], [para_A, para_B, para_I]):
            B1xx, B1yy, B4yz = para
            energy += 0.5 * ((B1xx * ls[...,0] + B1yy * (ls[...,1] + ls[...,2])) * dp[...,0]**2 ).sum()
            energy += 0.5 * ((B1xx * ls[...,1] + B1yy * (ls[...,0] + ls[...,2])) * dp[...,1]**2 ).sum()
            energy += 0.5 * ((B1xx * ls[...,2] + B1yy * (ls[...,0] + ls[...,1])) * dp[...,2]**2 ).sum()
            energy +=  (B4yz * ls[...,5] * dp[...,0] * dp[...,1]).sum()
            energy +=  (B4yz * ls[...,4] * dp[...,0] * dp[...,2]).sum()
            energy +=  (B4yz * ls[...,3] * dp[...,1] * dp[...,2]).sum()
        return energy
    return energy_engine
