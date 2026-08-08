"""
Classes which specify the physical system. 

Notes
-----
This file is part of OpenFerro.
"""

from time import time as timer
import logging
import numpy as np
import jax.numpy as jnp
from openferro.field import *
from openferro.interaction import *
from openferro.units import Constants
## import force engines
from openferro.engine.elastic import *
from openferro.engine.ferroelectric import *
from openferro.engine.magnetic import *
from openferro.engine.ewald import build_dipole_dipole_ewald
## import parallelism modules
from openferro.parallelism import DeviceMesh


_DEFAULT_FIELD_MASS = object()
_PRESSURE_VOLUME_ENGINES = {
    "determinant": pV_energy,
    "linearized_small_strain": pV_energy_linearized,
}
_PRESSURE_VOLUME_FUNCTIONS = {
    "determinant": deformed_volume,
    "linearized_small_strain": linearized_volume,
}


class System:
    """
    A class to define a physical system.
    
    A system is a lattice with fields and a Hamiltonian. Fields are added to the system by the user. 
    Interactions are added to the system by the user. Pointers to fields and interactions are stored in dictionaries.

    Attributes
    ----------
    lattice : BravaisLattice3D
        The lattice of the system
    """
    def __init__(self, lattice ):
        self.lattice = lattice
        self.pbc = lattice.pbc
        self._fields_dict = {}
        self._self_interaction_dict = {}
        self._mutual_interaction_dict = {}
        self._triple_interaction_dict = {}
        self._pressure_volume_mode = None
        
    def __repr__(self):
        return f"System with lattice {self.lattice} and fields {self._fields_dict.keys()}"
    
    """
    Methods for fields
    """

    def get_field_by_ID(self, ID):
        """
        Get a field by ID.

        Parameters
        ----------
        ID : str
            ID of the field

        Returns
        -------
        Field
            The field with the given ID

        Raises
        ------
        ValueError
            If field with given ID does not exist
        """
        if ID in self._fields_dict:
            return self._fields_dict[ID]
        else:
            raise ValueError('Field with the ID {} does not exist.'.format(ID))

    def get_all_fields(self):
        """
        Get all fields in the system.

        Returns
        -------
        list
            All fields in the system
        """
        return [self._fields_dict[ID] for ID in self._fields_dict.keys()]

    def get_all_SO3_fields(self):
        return [field for field in self.get_all_fields() if isinstance(field, FieldSO3)]
    
    def get_all_non_SO3_fields(self):
        return [field for field in self.get_all_fields() if not isinstance(field, FieldSO3)]

    def calc_volume(self):
        """Calculate the current supercell volume.

        Returns the reference volume for a fixed-cell system. Variable-cell
        systems use the pressure-volume convention selected when global strain
        was added.

        Returns
        -------
        float or jax.Array
            Supercell volume in Angstrom cubed.
        """
        if "gstrain" not in self._fields_dict:
            return self.lattice.ref_volume
        mode = self._pressure_volume_mode or "determinant"
        volume_function = _PRESSURE_VOLUME_FUNCTIONS[mode]
        strain = self.get_field_by_ID("gstrain").get_values()
        return volume_function(strain, self.lattice.ref_volume)

    def move_fields_to_multi_devs(self, mesh: DeviceMesh):
        """
        Move all fields to given devices for parallelization.

        Parameters
        ----------
        mesh : DeviceMesh
            The device mesh to move the fields to
        """
        for ID in self._fields_dict.keys():
            self._fields_dict[ID].to_multi_devs(mesh)

    def add_field(self, ID, ftype='scalar', dim=None, value=None, mass=_DEFAULT_FIELD_MASS):
        """
        Add a predefined field to the system.

        Parameters
        ----------
        ID : str
            ID of the field
        ftype : str, optional
            Type of field: ``scalar``, ``R3``, ``Rn``, ``SO3``, or
            ``LocalStrain3D``. The default is ``scalar``.
        dim : int, optional
            Dimension of the field. Only used for Rn fields
        value : array-like, optional
            Initial value of the field. Will be broadcasted to the shape of the field
        mass : float or array-like, optional
            Mass of the field. When omitted, Rn-like fields use 1.0 and SO(3)
            fields remain massless.

        Returns
        -------
        Field
            The field with the given ID

        Raises
        ------
        ValueError
            If field with this ID already exists or if ID is reserved
        ValueError
            If field type is unknown
        """
        if ID in self._fields_dict:
            raise ValueError("Field with this ID already exists. Pick another ID")
        if ID == 'gstrain':
            raise ValueError("The ID 'gstrain' is reserved for global strain field. Please pick another ID.")

        if ftype in ('scalar', 'FieldScalar'):
            field = FieldScalar(self.lattice, ID)
            default_value = 0.0
        elif ftype in ('R3', 'FieldR3'):
            field = FieldR3(self.lattice, ID)
            default_value = jnp.zeros(3)
        elif ftype == 'Rn':
            if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim <= 0:
                raise ValueError("Rn field dimension must be a positive integer.")
            field = FieldRn(self.lattice, ID, int(dim))
            default_value = jnp.zeros(dim)
        elif ftype == 'SO3':
            field = FieldSO3(self.lattice, ID)
            default_value = jnp.array([0.0, 0.0, 1.0])
        elif ftype == 'LocalStrain3D':
            field = LocalStrain3D(self.lattice, ID)
            default_value = jnp.zeros(3)
        else:
            raise ValueError(
                f"Unknown field type {ftype!r}. Expected scalar, R3, Rn, SO3, "
                "or LocalStrain3D."
            )

        initial_value = default_value if value is None else value
        try:
            initial_values = jnp.broadcast_to(jnp.asarray(initial_value), field.get_values().shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Initial value for field {ID!r} cannot broadcast to {field.get_values().shape}."
            ) from exc
        field.set_values(initial_values)

        if mass is _DEFAULT_FIELD_MASS:
            mass = None if isinstance(field, FieldSO3) else 1.0
        if mass is not None:
            field.set_mass(mass)

        self._fields_dict[ID] = field
        return field

    def add_global_strain(self, value=None, mass=1, pressure_volume="determinant"):
        """
        Add a global strain to the system. Allow variable cell simulation.

        Parameters
        ----------
        value : array-like, optional
            Initial value of the global strain
        mass : float, optional
            Effective mass of the global strain for the barostat
        pressure_volume : {"determinant", "linearized_small_strain"}, optional
            Volume convention for pressure energy and reporting. The default
            uses ``det(I + epsilon)``; the linearized option is retained for
            controlled comparison and compatibility.

        Returns
        -------
        Field
            The global strain field

        Raises
        ------
        ValueError
            If reserved IDs already exist or value is not a 6D vector
        """
        ID = 'gstrain'
        if ID in self._fields_dict:
            raise ValueError("Global strain field 'gstrain' already exists.")
        if 'pV' in self.interaction_dict:
            raise ValueError("Pressure interaction 'pV' already exists.")
        if pressure_volume not in _PRESSURE_VOLUME_ENGINES:
            raise ValueError(
                "pressure_volume must be 'determinant' or "
                "'linearized_small_strain'."
            )

        init = jnp.zeros(6) if value is None else jnp.asarray(value)
        if init.shape != (6,):
            raise ValueError("Global strain must be a 6D vector with shape (6,).")

        field = GlobalStrain(self.lattice, ID)
        field.set_values(init)
        if mass is not None:
            field.set_mass(mass)

        self._fields_dict[ID] = field
        try:
            self.add_pressure(0.0, volume_mode=pressure_volume)
        except Exception:
            self._fields_dict.pop(ID, None)
            self._self_interaction_dict.pop('pV', None)
            self._pressure_volume_mode = None
            raise
        return field
    
    """
    Methods for interactions
    """
    
    @property
    def interaction_dict(self):
        """
        Get all interactions in the system.

        Returns
        -------
        dict
            All interactions in the system
        """
        return {**self._self_interaction_dict, **self._mutual_interaction_dict, **self._triple_interaction_dict}

    def get_interaction_by_ID(self, interaction_ID):
        """
        Get an interaction by ID.

        Parameters
        ----------
        interaction_ID : str
            ID of the interaction

        Returns
        -------
        Interaction
            The interaction with the given ID

        Raises
        ------
        ValueError
            If interaction with given ID does not exist
        """
        if interaction_ID in self._self_interaction_dict:
            return self._self_interaction_dict[interaction_ID]
        elif interaction_ID in self._mutual_interaction_dict:
            return self._mutual_interaction_dict[interaction_ID]
        elif interaction_ID in self._triple_interaction_dict:
            return self._triple_interaction_dict[interaction_ID]
        else:
            raise ValueError("Interaction with ID {} does not exist. Existing interactions: {}".format(
                interaction_ID, self.interaction_dict.keys()))
 
    def _add_interaction_sanity_check(self, ID):
        """
        Sanity check for adding an interaction.

        Parameters
        ----------
        ID : str
            ID of the interaction to check

        Raises
        ------
        ValueError
            If ID is reserved or already exists
        """
        if ID == 'pV':
            raise ValueError("The interaction ID 'pV' is internal. The term pV in the Hamiltonian will be added automatically when you add a global strain.")
        if ID in self.interaction_dict:
            raise ValueError("Interaction with ID {} already exists. Pick another ID.".format(ID))
        return

    """
    Methods for adding pre-defined interactions to the Hamiltonian
    """
    ## electric dipole-type interactions
    def add_dipole_dipole_interaction(self, ID, field_ID, prefactor=1.0, enable_jit=True):
        """
        Add the long-range dipole-dipole interaction term to the Hamiltonian.

        Parameters
        ----------
        ID : str
            ID of the interaction
        field_ID : str
            ID of the field
        prefactor : float, optional
            Prefactor of the interaction
        enable_jit : bool, optional
            Whether to use JIT compilation

        Returns
        -------
        Interaction
            The created interaction
        """
        self._add_interaction_sanity_check(ID)
        field = self.get_field_by_ID(field_ID)
        interaction = self_interaction(field_ID)
        energy_engine, UkGG = build_dipole_dipole_ewald(
            field.lattice,
            dtype=field.get_values().dtype,
            sharding=field._sharding,
        )
        interaction.set_engine_data(UkGG)
        interaction.set_energy_engine(energy_engine, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        interaction.set_parameters(jnp.array([prefactor]))
        self._self_interaction_dict[ID] = interaction
        return interaction

    def add_dipole_onsite_interaction(self, ID, field_ID, K2, alpha, gamma, enable_jit=True):
        self._add_interaction_sanity_check(ID)
        interaction = self_interaction(field_ID)
        interaction.set_energy_engine(self_energy_onsite_isotropic, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        interaction.set_parameters(jnp.array([K2, alpha, gamma]))
        self._self_interaction_dict[ID] = interaction
        return interaction

    def add_dipole_interaction_1st_shell(self, ID, field_ID, j1, j2, enable_jit=True):
        self._add_interaction_sanity_check(ID)
        interaction = self_interaction(field_ID)
        interaction.set_energy_engine(short_range_1stnn_isotropic, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        interaction.set_parameters(jnp.array([j1, j2]))
        self._self_interaction_dict[ID] = interaction
        return interaction
    
    def add_dipole_interaction_2nd_shell(self, ID, field_ID, j3, j4, j5, enable_jit=True):
        self._add_interaction_sanity_check(ID)
        interaction = self_interaction(field_ID)
        interaction.set_energy_engine(short_range_2ednn_isotropic, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        interaction.set_parameters(jnp.array([j3, j4, j5]))
        self._self_interaction_dict[ID] = interaction
        return interaction
    
    def add_dipole_interaction_3rd_shell(self, ID, field_ID, j6, j7, enable_jit=True):
        self._add_interaction_sanity_check(ID)
        interaction = self_interaction(field_ID)
        energy_engine = get_short_range_3rdnn_isotropic()
        interaction.set_energy_engine(energy_engine, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        interaction.set_parameters(jnp.array([j6, j7]))
        self._self_interaction_dict[ID] = interaction
        return interaction
    
    def add_dipole_efield_interaction(self, ID, field_ID, E, enable_jit=True):
        self._add_interaction_sanity_check(ID)
        interaction = self_interaction(field_ID)
        interaction.set_energy_engine(dipole_efield_interaction, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        interaction.set_parameters(jnp.array(E))
        self._self_interaction_dict[ID] = interaction
        return interaction

    ## elastic-type interactions
    def add_homo_elastic_interaction(self, ID, field_ID, B11, B12, B44, enable_jit=True):
        N = float(self.lattice.nsites)
        self._add_interaction_sanity_check(ID)
        interaction = self_interaction(field_ID)
        interaction.set_energy_engine(homo_elastic_energy, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        interaction.set_parameters(jnp.array([B11, B12, B44, N]))
        self._self_interaction_dict[ID] = interaction
        return interaction
    
    def add_inhomo_elastic_interaction(self, ID, field_ID, B11, B12, B44, enable_jit=True):
        self._add_interaction_sanity_check(ID)
        interaction = self_interaction(field_ID)
        interaction.set_energy_engine(inhomo_elastic_energy, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        interaction.set_parameters(jnp.array([B11, B12, B44]))
        self._self_interaction_dict[ID] = interaction
        return interaction

    def add_pressure(self, pressure, volume_mode="determinant"):
        """
        Add a pressure term (pV) to the Hamiltonian.

        The ID of the interaction is reserved as 'pV'. V is the volume of the system, 
        which is calculated from the reference lattice vectors and the global strain.

        Parameters
        ----------
        pressure : float
            Pressure in bars
        volume_mode : {"determinant", "linearized_small_strain"}, optional
            Volume convention used by the pressure energy.

        Returns
        -------
        Interaction
            The pV interaction

        Raises
        ------
        ValueError
            If pV term already exists or if gstrain field is invalid
        """
        if volume_mode not in _PRESSURE_VOLUME_ENGINES:
            raise ValueError(
                "volume_mode must be 'determinant' or "
                "'linearized_small_strain'."
            )
        _pres = pressure * Constants.bar  # bar -> eV/Angstrom^3
        ## interaction ID sanity check
        ID = 'pV'
        if ID in self.interaction_dict:
            raise ValueError("pV term already exists in the Hamiltonian.")
        ## field ID sanity check
        field_ID = 'gstrain'
        field = self.get_field_by_ID(field_ID)
        if not isinstance(field, GlobalStrain):
            raise ValueError("I find a field with ID <gstrain>, but it is not a global strain field. Please rename the field or remove it. Then add the global strain field to the system.")
        ## add the interaction
        interaction = self_interaction(field_ID)
        energy_engine = _PRESSURE_VOLUME_ENGINES[volume_mode]
        interaction.set_energy_engine(energy_engine=energy_engine, enable_jit=True)
        interaction.create_force_engine(enable_jit=True)
        parameters = jnp.array([_pres, self.lattice.ref_volume]) 
        interaction.set_parameters(parameters)
        self._self_interaction_dict[ID] = interaction
        self._pressure_volume_mode = volume_mode
        return interaction

    ## elastic-dipole interactions
    def add_homo_strain_dipole_interaction(self, ID, field_1_ID, field_2_ID, B1xx, B1yy, B4yz,  enable_jit=True):
        self._add_interaction_sanity_check(ID)
        interaction = mutual_interaction(field_1_ID, field_2_ID)
        interaction.set_energy_engine(homo_strain_dipole_interaction, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        interaction.set_parameters(jnp.array([B1xx, B1yy, B4yz]))
        self._mutual_interaction_dict[ID] = interaction
        return interaction
    
    def add_inhomo_strain_dipole_interaction(self, ID, field_1_ID, field_2_ID, B1xx, B1yy, B4yz, enable_jit=True):
        self._add_interaction_sanity_check(ID)
        interaction = mutual_interaction(field_1_ID, field_2_ID)
        energy_engine = get_inhomo_strain_dipole_interaction(enable_jit=enable_jit)
        interaction.set_energy_engine(energy_engine, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        interaction.set_parameters(jnp.array([B1xx, B1yy, B4yz]))
        self._mutual_interaction_dict[ID] = interaction
        return interaction


    ## atomistic spin-type interactions
    def add_cubic_anisotropy_interaction(self, ID, field_ID, K1, K2, enable_jit=True):
        """
        Add the cubic anisotropy interaction term.

        Parameters
        ----------
        ID : str
            ID of the interaction
        field_ID : str
            ID of the field
        K1 : float
            First anisotropy constant
        K2 : float
            Second anisotropy constant
        enable_jit : bool, optional
            Whether to use JIT compilation

        Returns
        -------
        Interaction
            The created interaction
        """
        self._add_interaction_sanity_check(ID)
        interaction = self_interaction(field_ID)
        interaction.set_energy_engine(cubic_anisotropy_energy, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        interaction.set_parameters(jnp.array([K1, K2]))
        self._self_interaction_dict[ID] = interaction
        return interaction

    def _add_isotropic_exchange_interaction_by_rollers(
        self,
        ID,
        field_ID,
        coupling,
        rollers,
        enable_jit=True,
        bond_counting="unique",
    ):
        """
        Add ``E = -sum_<ij> J_ij m_i dot m_j`` to the Hamiltonian.

        Parameters
        ----------
        ID : str
            ID of the interaction
        field_ID : str
            ID of the field
        coupling : float
            Coupling constant
        rollers : list
            List of rolling functions for specifying the neighbouring relationship
        enable_jit : bool, optional
            Whether to use JIT compilation
        bond_counting : {"unique", "ordered"}, optional
            ``"unique"`` uses each half-shell roller once. ``"ordered"``
            reproduces the legacy factor-of-two convention.

        Returns
        -------
        Interaction
            The created interaction
        """
        self._add_interaction_sanity_check(ID)
        energy_engine = get_isotropic_exchange_energy_engine(
            rollers, bond_counting=bond_counting
        )
        interaction = self_interaction(field_ID)
        interaction.set_energy_engine(energy_engine, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        interaction.set_parameters(jnp.array([coupling]))
        self._self_interaction_dict[ID] = interaction
        return interaction

    def add_isotropic_exchange_interaction_1st_shell(
        self,
        ID,
        field_ID,
        coupling,
        enable_jit=True,
        bond_counting="unique",
    ):
        """
        Add the first shell isotropic exchange interaction term.

        The first shell is defined in lattice class.

        Parameters
        ----------
        ID : str
            ID of the interaction
        field_ID : str
            ID of the field
        coupling : float
            Coupling constant
        enable_jit : bool, optional
            Whether to use JIT compilation
        bond_counting : {"unique", "ordered"}, optional
            Pair-counting convention. The default counts each bond once;
            ``"ordered"`` restores the pre-Milestone-B factor of two.

        Returns
        -------
        Interaction
            The created interaction
        """
        interaction = self._add_isotropic_exchange_interaction_by_rollers(
            ID,
            field_ID,
            coupling,
            self.lattice.first_shell_roller,
            enable_jit=enable_jit,
            bond_counting=bond_counting,
        )
        return interaction

    def add_isotropic_exchange_interaction_2nd_shell(
        self,
        ID,
        field_ID,
        coupling,
        enable_jit=True,
        bond_counting="unique",
    ):
        """
        Add the second shell isotropic exchange interaction term.

        The second shell is defined in lattice class.

        Parameters
        ----------
        ID : str
            ID of the interaction
        field_ID : str
            ID of the field
        coupling : float
            Coupling constant
        enable_jit : bool, optional
            Whether to use JIT compilation
        bond_counting : {"unique", "ordered"}, optional
            Pair-counting convention. The default counts each bond once;
            ``"ordered"`` restores the pre-Milestone-B factor of two.

        Returns
        -------
        Interaction
            The created interaction
        """
        interaction = self._add_isotropic_exchange_interaction_by_rollers(
            ID,
            field_ID,
            coupling,
            self.lattice.second_shell_roller,
            enable_jit=enable_jit,
            bond_counting=bond_counting,
        )
        return interaction
    
    def add_isotropic_exchange_interaction_3rd_shell(
        self,
        ID,
        field_ID,
        coupling,
        enable_jit=True,
        bond_counting="unique",
    ):
        """
        Add the third shell isotropic exchange interaction term.

        The third shell is defined in lattice class.

        Parameters
        ----------
        ID : str
            ID of the interaction
        field_ID : str
            ID of the field
        coupling : float
            Coupling constant
        enable_jit : bool, optional
            Whether to use JIT compilation
        bond_counting : {"unique", "ordered"}, optional
            Pair-counting convention. The default counts each bond once;
            ``"ordered"`` restores the pre-Milestone-B factor of two.

        Returns
        -------
        Interaction
            The created interaction
        """
        interaction = self._add_isotropic_exchange_interaction_by_rollers(
            ID,
            field_ID,
            coupling,
            self.lattice.third_shell_roller,
            enable_jit=enable_jit,
            bond_counting=bond_counting,
        )
        return interaction
    
    def add_isotropic_exchange_interaction_4th_shell(
        self,
        ID,
        field_ID,
        coupling,
        enable_jit=True,
        bond_counting="unique",
    ):
        """
        Add the fourth shell isotropic exchange interaction term.

        The fourth shell is defined in lattice class.

        Parameters
        ----------
        ID : str
            ID of the interaction
        field_ID : str
            ID of the field
        coupling : float
            Coupling constant
        enable_jit : bool, optional
            Whether to use JIT compilation
        bond_counting : {"unique", "ordered"}, optional
            Pair-counting convention. The default counts each bond once;
            ``"ordered"`` restores the pre-Milestone-B factor of two.

        Returns
        -------
        Interaction
            The created interaction
        """
        interaction = self._add_isotropic_exchange_interaction_by_rollers(
            ID,
            field_ID,
            coupling,
            self.lattice.fourth_shell_roller,
            enable_jit=enable_jit,
            bond_counting=bond_counting,
        )
        return interaction
    """
    Methods for adding custom interactions to the Hamiltonian. Energy engines should be provided by the user.
    """
    def add_self_interaction(self, ID, field_ID, energy_engine, parameters=None, enable_jit=True):
        """
        Add a custom self-interaction term to the Hamiltonian.

        Parameters
        ----------
        ID : str
            ID of the interaction
        field_ID : str
            ID of the field
        energy_engine : callable
            A function that takes the field as input and returns the interaction energy
        parameters : array-like, optional
            Parameters for the interaction
        enable_jit : bool, optional
            Whether to use JIT compilation

        Returns
        -------
        Interaction
            The created interaction
        """
        self._add_interaction_sanity_check(ID)
        interaction = self_interaction( field_ID)
        interaction.set_energy_engine(energy_engine, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        if parameters is not None:
            interaction.set_parameters(parameters)
        self._self_interaction_dict[ID] = interaction
        return interaction
    
    def add_mutual_interaction(self, ID, field_1_ID, field_2_ID, energy_engine,  parameters=None, enable_jit=True):
        """
        Add a custom mutual interaction term to the Hamiltonian.

        Parameters
        ----------
        ID : str
            ID of the interaction
        field_1_ID : str
            ID of the first field
        field_2_ID : str
            ID of the second field
        energy_engine : callable
            A function that takes the fields as input and returns the interaction energy
        parameters : array-like, optional
            Parameters for the interaction
        enable_jit : bool, optional
            Whether to use JIT compilation

        Returns
        -------
        Interaction
            The created interaction
        """
        self._add_interaction_sanity_check(ID)
        interaction = mutual_interaction( field_1_ID, field_2_ID)
        interaction.set_energy_engine(energy_engine, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        if parameters is not None:
            interaction.set_parameters(parameters)
        self._mutual_interaction_dict[ID] = interaction
        return interaction
    
    def add_triple_interaction(self, ID, field_1_ID, field_2_ID, field_3_ID, energy_engine, parameters=None, enable_jit=True):
        """
        Add a custom triple interaction term to the Hamiltonian.

        Parameters
        ----------
        ID : str
            ID of the interaction
        field_1_ID : str
            ID of the first field
        field_2_ID : str
            ID of the second field
        field_3_ID : str
            ID of the third field
        energy_engine : callable
            A function that takes the fields as input and returns the interaction energy
        parameters : array-like, optional
            Parameters for the interaction
        enable_jit : bool, optional
            Whether to use JIT compilation

        Returns
        -------
        Interaction
            The created interaction
        """
        self._add_interaction_sanity_check(ID)
        interaction = triple_interaction(field_1_ID, field_2_ID, field_3_ID)
        interaction.set_energy_engine(energy_engine, enable_jit=enable_jit)
        interaction.create_force_engine(enable_jit=enable_jit)
        if parameters is not None:
            interaction.set_parameters(parameters)
        self._triple_interaction_dict[ID] = interaction
        return interaction

    """
    Methods for energy and force calculation
    """

    def calc_energy_by_ID(self, interaction_ID):
        """
        Calculate the energy of an interaction by ID.

        Parameters
        ----------
        interaction_ID : str
            ID of the interaction

        Returns
        -------
        float
            The energy of the interaction

        Raises
        ------
        ValueError
            If interaction with given ID does not exist
        """
        if interaction_ID in self._self_interaction_dict:
            interaction = self.get_interaction_by_ID(interaction_ID)
            field = self.get_field_by_ID(interaction.field_ID)
            energy = interaction.calc_energy(field)
        elif interaction_ID in self._mutual_interaction_dict:
            interaction = self.get_interaction_by_ID(interaction_ID)
            field1 = self.get_field_by_ID(interaction.field_1_ID)
            field2 = self.get_field_by_ID(interaction.field_2_ID)
            energy = interaction.calc_energy(field1, field2)
        elif interaction_ID in self._triple_interaction_dict:
            interaction = self.get_interaction_by_ID(interaction_ID)
            field1 = self.get_field_by_ID(interaction.field_1_ID)
            field2 = self.get_field_by_ID(interaction.field_2_ID)
            field3 = self.get_field_by_ID(interaction.field_3_ID)
            energy = interaction.calc_energy(field1, field2, field3)
        else:
            raise ValueError("Interaction with ID {} does not exist. Existing interactions: {}".format(
                interaction_ID, self.interaction_dict.keys()))
        return energy            

    def calc_force_by_ID(self, interaction_ID):
        """
        Calculate the gradient force from an interaction by ID.

        Parameters
        ----------
        interaction_ID : str
            ID of the interaction

        Returns
        -------
        array
            The gradient force of the interaction

        Raises
        ------
        ValueError
            If interaction with given ID does not exist
        """
        if interaction_ID in self._self_interaction_dict:
            interaction = self.get_interaction_by_ID(interaction_ID)
            field = self.get_field_by_ID(interaction.field_ID)
            force = interaction.calc_force(field)
        elif interaction_ID in self._mutual_interaction_dict:
            interaction = self.get_interaction_by_ID(interaction_ID)
            field1 = self.get_field_by_ID(interaction.field_1_ID)
            field2 = self.get_field_by_ID(interaction.field_2_ID)
            force = interaction.calc_force(field1, field2)
        elif interaction_ID in self._triple_interaction_dict:
            interaction = self.get_interaction_by_ID(interaction_ID)
            field1 = self.get_field_by_ID(interaction.field_1_ID)
            field2 = self.get_field_by_ID(interaction.field_2_ID)
            field3 = self.get_field_by_ID(interaction.field_3_ID)
            force = interaction.calc_force(field1, field2, field3)
        else:
            raise ValueError("Interaction with ID {} does not exist. Existing interactions: {}".format(
                interaction_ID, self.interaction_dict.keys()))
        return force

    def calc_total_self_energy(self):
        """
        Calculate the total self-interaction energy.

        Returns
        -------
        float
            Total self-interaction energy
        """
        energy = 0.0
        for interaction_ID in self._self_interaction_dict:
            energy += self.calc_energy_by_ID(interaction_ID)
            # e = self.calc_energy_by_ID(interaction_ID)
            # logging.info('Energy from {}: {}'.format(interaction_ID, e))
        return energy

    def calc_total_mutual_interaction(self):
        """
        Calculate the total mutual interaction energy.

        Returns
        -------
        float
            Total mutual interaction energy
        """
        energy = 0.0
        for interaction_ID in self._mutual_interaction_dict:
            energy += self.calc_energy_by_ID(interaction_ID)
            # e = self.calc_energy_by_ID(interaction_ID)
            # logging.info('Energy from {}: {}'.format(interaction_ID, e))
        return energy

    def calc_total_triple_interaction(self):
        """
        Calculate the total triple interaction energy.

        Returns
        -------
        float
            Total triple interaction energy
        """
        energy = 0.0
        for interaction_ID in self._triple_interaction_dict:
            energy += self.calc_energy_by_ID(interaction_ID)
        return energy

    def calc_total_potential_energy(self):
        """
        Calculate the total potential energy of the system.

        Calculate the total potential energy of the system. 
        Total potential energy is the sum of self-interaction energy, mutual interaction energy, and triple interaction energy.
        """
        return self.calc_total_self_energy() + self.calc_total_mutual_interaction() + self.calc_total_triple_interaction()

    def calc_total_kinetic_energy(self):
        """
        Calculate the total kinetic energy of the system. 
        """
        kinetic_energy = 0.0
        for field in self.get_all_fields():
            kinetic_energy += field.get_kinetic_energy()
        return kinetic_energy

    def calc_temp_by_ID(self, ID):
        """
        Calculate the temperature of a field.
        """
        field = self.get_field_by_ID(ID)
        return field.get_temperature()

    def calc_excess_stress(self):
        '''
        Get instantaneous stress - applied stress (e.g. from hydrostatic pressure)
        '''
        field = self.get_field_by_ID('gstrain')
        return field.get_excess_stress()

    """
    Methods for updating the gradient force 
    """

    def update_force_from_self_interaction(self, profile=False):
        """
        update the gradient force felt by each field from self interactions.
        """
        for interaction_ID in self._self_interaction_dict:
            if profile:
                t0 = timer()
            interaction = self._self_interaction_dict[interaction_ID]
            field = self.get_field_by_ID(interaction.field_ID)
            force = interaction._accumulate_force(field, field.get_force())
            field.set_force(force)
            if profile:
                jax.block_until_ready(field.get_force())
                logging.info('Time for updating force from {}: {:.8f}s'.format(interaction_ID, timer()-t0))
                # print("energy from %s:" % interaction_ID, interaction.calc_energy(field))
        return
    
    def update_force_from_mutual_interaction(self, profile=False):
        """
        update the gradient force felt by each field from mutual interactions.
        """
        for interaction_ID in self._mutual_interaction_dict:
            if profile:
                t0 = timer()
            interaction = self.get_interaction_by_ID(interaction_ID)
            field1 = self.get_field_by_ID(interaction.field_1_ID)
            field2 = self.get_field_by_ID(interaction.field_2_ID)
            force1, force2 = interaction._accumulate_force(
                field1, field2, field1.get_force(), field2.get_force()
            )
            field1.set_force(force1)
            field2.set_force(force2)
            if profile:
                jax.block_until_ready(field2.get_force())
                logging.info('Time for updating force from {}: {:.8f}s'.format(interaction_ID, timer()-t0))
        return
    
    def update_force_from_triple_interaction(self, profile=False):
        """
        update the gradient force felt by each field from triple interactions.
        """
        for interaction_ID in self._triple_interaction_dict:
            if profile:
                t0 = timer()
            interaction = self.get_interaction_by_ID(interaction_ID)
            field1 = self.get_field_by_ID(interaction.field_1_ID)
            field2 = self.get_field_by_ID(interaction.field_2_ID)
            field3 = self.get_field_by_ID(interaction.field_3_ID)
            force1, force2, force3 = interaction._accumulate_force(
                field1,
                field2,
                field3,
                field1.get_force(),
                field2.get_force(),
                field3.get_force(),
            )
            field1.set_force(force1)
            field2.set_force(force2)
            field3.set_force(force3)
            if profile:
                jax.block_until_ready(field3.get_force())
                logging.info('Time for updating force from {}: {:.8f}s'.format(interaction_ID, timer()-t0))
        return

    def update_force(self, profile=False):
        """
        update the gradient force felt by each field from all interactions.
        """
        ## zero force
        for field in self.get_all_fields():
            if profile:
                t0 = timer()
            field.zero_force()
            if profile:
                jax.block_until_ready(field.get_force())
                logging.info('Time for zeroing force of {}: {:.8f}s'.format(field.ID, timer()-t0))

        ## update force from all interactions
        self.update_force_from_self_interaction(profile=profile)
        self.update_force_from_mutual_interaction(profile=profile)
        self.update_force_from_triple_interaction(profile=profile)
        return
     
class RingPolymerSystem(System):
    """
    A class to define a ring polymer system for path-integral molecular dynamics simulations. 
    """
    def __init__(self, lattice, nbeads=1):
        super().__init__(lattice)
        self.nbeads = nbeads
        raise NotImplementedError("Ring polymer system is not implemented yet.")
