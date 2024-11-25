"""
Classes which define the "interaction" between fields. Each interaction is associated with a term in the Hamiltonian. 
Each interaction stores a function "self.energy_engine" that calculates the energy of the interaction and a function "force engine" that calculates the force of the interaction.
Only the energy engine is required. The force engine is optional. If the force engine is not set, the force will be calculated by automatic differentiation of the energy engine.
"""
# This file is part of OpenFerro.

import numpy as np
import jax.numpy as jnp
from jax import grad, jit

class interaction_base:
    """
    The base class to specify the interaction between fields.
    """
    def __init__(self, parameters=None):
        self.parameters = parameters
        self.energy_engine = None
        self.force_engine = None
    def set_parameters(self, parameters):
        """
        Set the parameters of the interaction. 
        Args:
            parameters (list-like or jax array): the parameters of the interaction
        """
        ## turning list or numpy array to jax array
        if isinstance(parameters, jnp.ndarray):
            paras = parameters
        elif isinstance(parameters, np.ndarray) or isinstance(parameters, list):
            paras = jnp.array(parameters)
        else:
            raise ValueError("Parameters must be a numpy array, a list, or a jax array")
        self.parameters = paras
    def get_parameters(self):
        """
        Get the parameters of the interaction.
        Returns:
            parameters (jax array): the parameters of the interaction
        """
        return self.parameters
    def set_energy_engine(self, energy_engine, enable_jit=True):
        """
        Set the energy engine of the interaction.
        Args:
            energy_engine (function): the energy engine of the interaction. It should take the values of the fields as input and return the energy as output.
            enable_jit (bool): whether to enable jit for the energy engine
        """
        if enable_jit:
            self.energy_engine = jit(energy_engine)
        else:
            self.energy_engine = energy_engine
    def calc_energy(self):
        pass
    def calc_force(self):
        pass


class self_interaction(interaction_base):
    """
    A class to specify the self-interaction of a field.
    """
    def __init__(self, field_ID, parameters=None):
        super().__init__( parameters)
        self.field_ID = field_ID
    def create_force_engine(self, enable_jit=True):
        """
        Derive the force engine of the interaction from the energy engine through automatic differentiation.
        Args:
            enable_jit (bool): whether to enable jit for the force engine
        """
        if self.energy_engine is None:
            raise ValueError("Energy engine is not set. Set energy engine before creating force engine.")
        if enable_jit:
            self.force_engine =  jit(grad(self.energy_engine, argnums=0 )) 
        else:
            self.force_engine =  grad(self.energy_engine, argnums=0 )
        return

    def calc_energy(self, field):
        """
        Calculate the energy of the interaction for a given field.
        Args:
            field (Field): the field to calculate the energy
        Returns:
            energy (float): the energy of the interaction
        """
        field_values = field.get_values()
        return self.energy_engine(field_values, self.parameters)
    def calc_force(self, field):
        """
        Calculate the force of the interaction for a given field.
        Args:
            field (Field): the field to calculate the force
        Returns:
            force (jax array): the gradient of the energy with respect to the field. It has the same shape as the field.
        """
        field_values = field.get_values()
        gradient = self.force_engine(field_values, self.parameters)
        return -gradient

class mutual_interaction(interaction_base):
    """
    A class to specify the  mutual interaction between two fields.
    """
    def __init__(self, field_1_ID, field_2_ID, parameters=None):
        super().__init__( parameters)
        self.field_1_ID = field_1_ID
        self.field_2_ID = field_2_ID
    def create_force_engine(self, enable_jit=True):
        """
        Derive the force engine of the interaction from the energy engine through automatic differentiation.
        Args:
            enable_jit (bool): whether to enable jit for the force engine
        """
        if self.energy_engine is None:
            raise ValueError("Energy engine is not set. Set energy engine before creating force engine.")
        if enable_jit:
            self.force_engine =  jit(grad(self.energy_engine, argnums=(0, 1) )) 
        else:
            self.force_engine =  grad(self.energy_engine, argnums=(0, 1) )
    def calc_energy(self, field1, field2):
        """
        Calculate the energy of the interaction for a given pair of fields.
        Args:
            field1 (Field): the first field
            field2 (Field): the second field
        Returns:
            energy (float): the energy of the interaction
        """
        f1 = field1.get_values()
        f2 = field2.get_values()
        return self.energy_engine(f1, f2, self.parameters)
    def calc_force(self, field1, field2):
        """
        Calculate the force of the interaction for a given pair of fields.
        Args:
            field1 (Field): the first field
            field2 (Field): the second field
        Returns:
            force (tuple of jax arrays): the gradient of the energy with respect to the fields. It has the same shape as the fields.
        """
        f1 = field1.get_values()
        f2 = field2.get_values()
        gradient = self.force_engine(f1, f2, self.parameters)
        return (- gradient[0], - gradient[1])

class triple_interaction:
    """
    A class to specify the  mutual interaction between three fields.
    """
    def __init__(self, field_1_ID, field_2_ID, field_3_ID, parameters=None):
        super().__init__( parameters)
        self.field_1_ID = field_1_ID
        self.field_2_ID = field_2_ID
        self.field_3_ID = field_3_ID
    def create_force_engine(self, enable_jit=True):
        """
        Derive the force engine of the interaction from the energy engine through automatic differentiation.
        Args:
            enable_jit (bool): whether to enable jit for the force engine
        """
        if self.energy_engine is None:
            raise ValueError("Energy engine is not set. Set energy engine before creating force engine.")
        if enable_jit:
            self.force_engine =  jit(grad(self.energy_engine, argnums=(0, 1, 2) )) 
        else:
            self.force_engine =  grad(self.energy_engine, argnums=(0, 1, 2) )
    def calc_energy(self, field1, field2, field3):
        """
        Calculate the energy of the interaction for a given triple of fields.
        Args:
            field1 (Field): the first field
            field2 (Field): the second field
            field3 (Field): the third field
        Returns:
            energy (float): the energy of the interaction
        """
        f1 = field1.get_values()
        f2 = field2.get_values()
        f3 = field3.get_values()
        return self.energy_engine(f1, f2, f3, self.parameters)
    def calc_force(self, field1, field2, field3):
        """
        Calculate the force of the interaction for a given triple of fields.
        Args:
            field1 (Field): the first field
            field2 (Field): the second field
            field3 (Field): the third field
        Returns:
            force (tuple of jax arrays): the gradient of the energy with respect to the fields. It has the same shape as the fields.
        """
        f1 = field1.get_values()
        f2 = field2.get_values()
        f3 = field3.get_values()
        gradient = self.force_engine(f1, f2, f3, self.parameters)
        return (- gradient[0], - gradient[1], - gradient[2])
 