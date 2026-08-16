"""
Classes which define the time evolution of physical systems.

Notes
-----
This file is part of OpenFerro.
"""

import logging
from functools import partial
from itertools import product
from numbers import Integral, Real
from pathlib import Path
from time import time as timer

import numpy as np
import jax
import jax.numpy as jnp

from openferro.units import Constants
from openferro.field import GlobalStrain
from openferro.reporter import Thermo_Reporter, Field_Reporter

class Simulation:
    """
    The base class to define a simulation.
    
    A simulation controls the time evolution of a system. This class does not implement any time 
    integration algorithm. Each field object has its own integrator. The class only calls the step 
    method of each integrator and controls output.

    Parameters
    ----------
    system : System
        The physical system to simulate
    seed : int, optional
        Seed for the simulation-owned random stream, by default 42
    key : array_like, optional
        JAX ``PRNGKey`` state. When supplied, it takes precedence over ``seed``.
    """
    def __init__(self, system, seed=42, key=None):
        self.system = system
        self.reporters = []
        self.reset_random_key(seed=seed, key=key)

    def reset_random_key(self, seed=42, key=None):
        """Reset the simulation-owned random stream.

        Parameters
        ----------
        seed : int, optional
            Seed used to create a JAX ``PRNGKey``, by default 42
        key : array_like, optional
            Existing JAX ``PRNGKey`` state. When supplied, it takes precedence
            over ``seed``.
        """
        if key is not None:
            key = jnp.asarray(key)
            if key.shape != (2,) or key.dtype != jnp.uint32:
                raise ValueError("key must be a uint32 JAX PRNGKey with shape (2,).")
            self._random_key = key
            return
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer.")
        self._random_key = jax.random.PRNGKey(int(seed))

    def get_random_key(self):
        """Return the current random key for restart or checkpoint storage."""
        return self._random_key

    def _next_random_keys(self, count):
        keys = jax.random.split(self._random_key, count + 1)
        self._random_key = keys[0]
        return keys[1:]

    @staticmethod
    def _validate_so3_field_count(fields):
        if len(fields) > 1:
            raise NotImplementedError(
                "Coupled dynamics for multiple SO(3) fields require a "
                "simultaneous SIB solve and are not supported."
            )
    
    def clear_reporters(self):
        self.reporters = []

    def add_thermo_reporter(self, file='thermo.log', log_interval=100, global_strain=False, excess_stress=False, volume=False, potential_energy=False, kinetic_energy=False, temperature=False):
        """
        Add a reporter to output global thermodynamic information.

        Parameters
        ----------
        file : str, optional
            Output file name, by default 'thermo.log'
        log_interval : int, optional
            Number of steps between outputs, by default 100
        global_strain : bool, optional
            Whether to output global strain, by default False
        excess_stress : bool, optional
            Whether to output excess stress, by default False
        volume : bool, optional
            Whether to output volume, by default False
        potential_energy : bool, optional
            Whether to output potential energy, by default False
        kinetic_energy : bool, optional
            Whether to output kinetic energy, by default False
        temperature : bool, optional
            Whether to output temperature, by default False
        """
        self.reporters.append(Thermo_Reporter(file, log_interval, global_strain, excess_stress, volume, potential_energy, kinetic_energy, temperature))
    
    def add_field_reporter(self, file_prefix, field_ID, log_interval=100, field_average=True, dump_field=False):
        """
        Add a reporter to dump the values of a given field.

        Parameters
        ----------
        file_prefix : str
            Prefix for output files
        field_ID : str
            ID of field to report
        log_interval : int, optional
            Number of steps between outputs, by default 100
        field_average : bool, optional
            Whether to output field averages, by default True
        dump_field : bool, optional
            Whether to dump full field values, by default False
        """
        self.reporters.append(Field_Reporter(file_prefix, field_ID, log_interval, field_average, dump_field))

    def initialize_reporters(self):
        """Initialize all reporters."""
        for reporter in self.reporters:
            reporter.initialize(self.system)

    def remove_all_reporters(self):
        self.reporters = []

    def reset_reporters(self):
        """Reset the counters of all reporters."""
        for reporter in self.reporters:
            reporter.counter = -1
    
    def step_reporters(self):
        """Step all reporters."""
        for reporter in self.reporters:
            reporter.step(self.system)
    

    def init_velocity(self, mode='zero', temp=None):
        keys = self._next_random_keys(len(self.all_fields))
        for field, key in zip(self.all_fields, keys):
            field.init_velocity(mode=mode, temperature=temp, key=key)

    def _step(self):
        """
        Update the system by one time step. To be implemented by subclasses.
        """
        pass
    
    def run(self):
        """
        Run the simulation for a given number of steps or until convergence. To be implemented by subclasses.
        """
        pass
        
class MDMinimize(Simulation):
    """
    Class for energy minimization using molecular dynamics.

    Parameters
    ----------
    system : System
        The physical system to minimize
    max_iter : int, optional
        Maximum number of iterations, by default 100
    tol : float, optional
        Force tolerance for convergence, by default 1e-5

    Notes
    -----
    Forces are evaluated at the initial state and after every accepted update.
    ``iterations`` counts accepted updates, so an initially converged system
    finishes with zero iterations.
    """
    def __init__(self, system, max_iter=100, tol=1e-5 ):
        super().__init__(system)
        self.max_iter = max_iter
        self.tol = tol
        self.all_fields = self.system.get_all_fields()
        self.converged = False
        self.iterations = 0
        self.max_force_by_field = {}
            
    def _step(self, variable_cell):
        """
        Update the fields by one step using their current forces.

        Parameters
        ----------
        variable_cell : bool
            Whether to allow cell parameters to vary
        """
        SO3_fields = self.system.get_all_SO3_fields()
        non_SO3_fields = self.system.get_all_non_SO3_fields()
        if len(non_SO3_fields) > 0:
            for field in non_SO3_fields:
                if (variable_cell is False) and isinstance(field, GlobalStrain):
                    continue
                field.integrator.step(field)
        if len(SO3_fields) > 0:
            # Implicit SIB stages evaluate forces through this callback.
            for field in SO3_fields:
                field.integrator.step(field, force_updater=self.system.update_force)

    def _update_force_and_check_convergence(self, active_fields):
        """Evaluate forces at the current state and apply the force tolerance."""
        self.system.update_force()
        self.max_force_by_field = {
            field.ID: float(jax.device_get(jnp.max(jnp.abs(field.get_force()))))
            for field in active_fields
        }
        return all(
            force < self.tol for force in self.max_force_by_field.values()
        )
            
    def run(self, variable_cell=True, pressure=None):
        """
        Run the minimization.

        Parameters
        ----------
        variable_cell : bool, optional
            Whether to allow cell parameters to vary, by default True
        pressure : float, optional
            External pressure in bar, required if variable_cell=True

        Raises
        ------
        ValueError
            If pressure not specified for variable cell minimization
            If pressure specified for fixed cell minimization
            If integrator not set for any field
        """
        self._validate_so3_field_count(self.system.get_all_SO3_fields())
        active_fields = [
            field for field in self.all_fields
            if variable_cell or not isinstance(field, GlobalStrain)
        ]
        ## sanity check
        if variable_cell:
            if pressure is None:
                raise ValueError('Please specify pressure for variable-cell structural minimization')
            else:
                # self.system.get_interaction_by_ID('pV').set_parameter_by_ID(
                #     'p', pressure * Constants.bar)  # bar -> eV/Angstrom^3
                pV_param = self.system.get_interaction_by_ID('pV').get_parameters()
                pV_param_new = [pressure * Constants.bar, pV_param[1]]
                self.system.get_interaction_by_ID('pV').set_parameters(pV_param_new)
            for field in active_fields:
                if field.integrator is None:
                    raise ValueError('Please set the integrator for the field %s for variable-cell structural minimization' % type(field))
        else:
            if pressure is not None:
                raise ValueError('Specifying pressure is not allowed for fixed-cell structural minimization')
            for field in active_fields:
                if field.integrator is None:
                    raise ValueError('Please set the integrator for the field %s for fixed-cell structural minimization' % type(field))
        ## structural relaxation
        self.initialize_reporters()
        self.converged = False
        self.iterations = 0
        self.max_force_by_field = {}
        self.converged = self._update_force_and_check_convergence(active_fields)
        if self.converged:
            return

        for i in range(self.max_iter):
            self._step(variable_cell)
            self.iterations = i + 1
            self.converged = self._update_force_and_check_convergence(
                active_fields
            )
            self.step_reporters()
            if self.converged:
                break
        if not self.converged:
            logging.warning(
                "Minimization did not converge after %d iterations; maximum forces: %s",
                self.iterations,
                self.max_force_by_field,
            )

class SimulationNVE(Simulation):
    """
    Class for NVE (microcanonical) molecular dynamics simulation.

    Parameters
    ----------
    system : System
        The physical system to simulate
    seed : int, optional
        Seed for the simulation-owned random stream, by default 42
    key : array_like, optional
        Existing JAX ``PRNGKey`` state, by default None
    """
    def __init__(self, system, seed=42, key=None):
        super().__init__(system, seed=seed, key=key)
        ## get all fields, excluding the global strain field
        self.SO3_fields = self.system.get_all_SO3_fields()
        self.non_SO3_fields = [field for field in self.system.get_all_non_SO3_fields() if not isinstance(field, GlobalStrain)]
        self.all_fields = self.SO3_fields + self.non_SO3_fields
        self.nfields = len(self.all_fields)

    def _update_force(self, profile=False):
        """Update all field forces at the current state."""
        self.system.update_force(profile=profile)

    def _step(self, profile=False):
        """
        Update the field by one step.

        Parameters
        ----------
        profile : bool, optional
            Whether to profile timing, by default False
        """
        if len(self.non_SO3_fields) > 0:
            ## update the force for all fields. 
            ## Force will not be updated again while integrating each non-SO3 field with simple explicit integrator. 
            self._update_force(profile=profile)
            for field in self.non_SO3_fields:
                if profile:
                    t0 = timer()
                field.integrator.step(field)
                if profile:
                    jax.block_until_ready(field.get_values())
                    logging.info('Time for updating field {}: {:.8f}s'.format(type(field), timer()-t0))
        if len(self.SO3_fields) > 0:
            ## Force updater will be passed to the integrator of each SO3 fields because implicit methods are used.
            ## So the force will not be updated here. 
            for field in self.SO3_fields:
                if profile:
                    t0 = timer()
                field.integrator.step(
                    field,
                    force_updater=partial(self._update_force, profile=profile),
                )
                if profile:
                    jax.block_until_ready(field.get_values())
                    logging.info('Time for updating field {}: {:.8f}s'.format(type(field), timer()-t0))
    def run(self, nsteps=1, profile=False):
        """
        Run the simulation.

        Parameters
        ----------
        nsteps : int, optional
            Number of steps to run, by default 1
        profile : bool, optional
            Whether to profile timing, by default False

        Raises
        ------
        ValueError
            If integrator not set for any field
        """
        self._validate_so3_field_count(self.SO3_fields)
        ## sanity check
        for field in self.all_fields:
            if field.integrator is None:
                raise ValueError('Please set the integrator for the field %s before running the simulation' % type(field))
        ## run the simulation
        self.initialize_reporters()
        for i in range(nsteps):
            self._step(profile=profile)
            self.step_reporters()

class SimulationNVTLangevin(SimulationNVE):
    """
    Class for NVT molecular dynamics simulation using Langevin dynamics.

    Parameters
    ----------
    system : System
        The physical system to simulate
    seed : int, optional
        Seed for the simulation-owned random stream, by default 42
    key : array_like, optional
        Existing JAX ``PRNGKey`` state, by default None
    """
    def __init__(self, system, seed=42, key=None):
        super().__init__(system, seed=seed, key=key)

    def _step(self, keys, profile=False):
        """
        Update the field by one step.

        Parameters
        ----------
        keys : array_like
            Random keys for Langevin dynamics
        profile : bool, optional
            Whether to profile timing, by default False
        """
        keys_SO3 = keys[:len(self.SO3_fields)]
        keys_non_SO3 = keys[len(self.SO3_fields):]
        if len(self.non_SO3_fields) > 0:
            self._update_force(profile=profile)
            for field, subkey in zip(self.non_SO3_fields, keys_non_SO3):
                if profile:
                    t0 = timer()
                field.integrator.step(subkey, field)
                if profile:
                    jax.block_until_ready(field.get_values())
                    logging.info('Time for updating field {}: {:.8f}s'.format(type(field), timer()-t0))
        if len(self.SO3_fields) > 0:
            for field, subkey in zip(self.SO3_fields, keys_SO3):
                if profile:
                    t0 = timer()
                field.integrator.step(
                    subkey,
                    field,
                    force_updater=partial(self._update_force, profile=profile),
                )
                if profile:
                    jax.block_until_ready(field.get_values())
                    logging.info('Time for updating field {}: {:.8f}s'.format(type(field), timer()-t0))
        return

    def run(self, nsteps=1, profile=False, seed=None):
        """
        Run the simulation.

        Parameters
        ----------
        nsteps : int, optional
            Number of steps to run, by default 1
        profile : bool, optional
            Whether to profile timing, by default False
        seed : int, optional
            Reset the random stream before this run. When omitted, the stream
            continues from previous initialization and run calls.

        Raises
        ------
        ValueError
            If integrator not set for any field
        """
        self._validate_so3_field_count(self.SO3_fields)
        ## sanity check
        for field in self.all_fields:
            if field.integrator is None:
                raise ValueError('Please set the integrator for the field %s before running the simulation' % type(field))
        if seed is not None:
            self.reset_random_key(seed=seed)
        ## run the simulation
        self.initialize_reporters()
        for id_step in range(nsteps):
            if profile:
                t0 = timer()
            subkeys = self._next_random_keys(self.nfields)
            self._step(subkeys, profile)
            self.step_reporters()
            if profile:
                logging.info('Total time for one step: {:.8f}s'.format(timer()-t0))
        return


class MetadynamicsNVT(SimulationNVTLangevin):
    """Fixed-height metadynamics with NVT Langevin dynamics.

    Collective variables are simulation-owned scalar functions. They follow
    the energy-engine signature ``engine(field1, ..., parameters)`` but are
    not added to the system Hamiltonian.

    Parameters
    ----------
    system : System
        The physical system to simulate.
    collective_variables : sequence of dict
        One to three CV definitions. Each definition contains ``id``,
        ``field_ids``, ``engine``, and optional ``parameters`` entries.
    pace : int
        Number of outer MD steps between Gaussian hill depositions.
    sigma : float or array_like
        Gaussian widths in the native units of the CVs.
    height : float
        Fixed Gaussian height in eV.
    grid_min, grid_max : float or array_like
        Lower and upper bounds of the bias grid in the native CV units.
    grid_bin : int or array_like, optional
        Number of grid intervals in each CV direction. If omitted, each
        spacing is at most one fifth of the corresponding Gaussian width.
    upper_walls, lower_walls : dict, optional
        Wall definitions with required ``at`` and ``kappa`` entries and
        optional ``exp``, ``eps``, and ``offset`` entries.
    hills_file : path_like, optional
        HILLS output path. Set to ``None`` to disable output.
    seed : int, optional
        Seed for the simulation-owned random stream, by default 42.
    key : array_like, optional
        Existing JAX ``PRNGKey`` state, by default None.
    """

    _cv_keys = {"id", "field_ids", "engine", "parameters"}
    _wall_keys = {"at", "kappa", "exp", "eps", "offset"}

    def __init__(
        self,
        system,
        collective_variables,
        pace,
        sigma,
        height,
        *,
        grid_min,
        grid_max,
        grid_bin=None,
        upper_walls=None,
        lower_walls=None,
        hills_file="HILLS",
        seed=42,
        key=None,
    ):
        super().__init__(system, seed=seed, key=key)
        self._collective_variables, initial_values = (
            self._prepare_collective_variables(collective_variables)
        )
        self.cv_dimension = len(self._collective_variables)

        if isinstance(pace, bool) or not isinstance(pace, Integral) or pace <= 0:
            raise ValueError("pace must be a positive integer.")
        if (
            isinstance(height, bool)
            or not isinstance(height, Real)
            or not np.isfinite(height)
            or height <= 0.0
        ):
            raise ValueError("height must be a finite, strictly positive scalar.")

        self.pace = int(pace)
        self.height = float(height)
        self.sigma = self._vector_parameter(
            "sigma", sigma, self.cv_dimension, positive=True
        )
        self._sigma_host = np.asarray(jax.device_get(self.sigma))
        self.grid_min = self._vector_parameter(
            "grid_min", grid_min, self.cv_dimension
        )
        self.grid_max = self._vector_parameter(
            "grid_max", grid_max, self.cv_dimension
        )
        grid_range = np.asarray(jax.device_get(self.grid_max - self.grid_min))
        if np.any(grid_range <= 0.0):
            raise ValueError(
                "grid_max must be greater than grid_min in every dimension."
            )
        if grid_bin is None:
            sigma_values = np.asarray(jax.device_get(self.sigma))
            grid_bin = np.maximum(
                2, np.ceil(grid_range / (0.2 * sigma_values)).astype(int)
            )
        grid_bin = self._integer_vector_parameter(
            "grid_bin", grid_bin, self.cv_dimension
        )
        self.grid_bin = tuple(grid_bin.tolist())
        grid_shape = tuple(value + 1 for value in self.grid_bin)
        self.grid_spacing = (self.grid_max - self.grid_min) / jnp.asarray(
            self.grid_bin, dtype=self.grid_min.dtype
        )
        axes = [
            jnp.linspace(self.grid_min[i], self.grid_max[i], grid_shape[i])
            for i in range(self.cv_dimension)
        ]
        mesh = jnp.meshgrid(*axes, indexing="ij")
        self._grid_points = jnp.stack(
            [coordinate.reshape(-1) for coordinate in mesh], axis=1
        )
        self._bias_grid = jnp.zeros(grid_shape, dtype=initial_values[0].dtype)
        self._add_grid_hill_engine = jax.jit(self._add_grid_hill)
        self._grid_value_and_grad_engine = jax.jit(
            jax.value_and_grad(self._interpolate_grid, argnums=1)
        )
        self.upper_walls = self._prepare_wall("upper_walls", upper_walls)
        self.lower_walls = self._prepare_wall("lower_walls", lower_walls)
        if self.upper_walls is not None:
            upper_start = self.upper_walls["at"] - self.upper_walls["offset"]
            active = self.upper_walls["kappa"] > 0.0
            outside = active & (upper_start > self.grid_max)
            if bool(jax.device_get(jnp.any(outside))):
                raise ValueError("Every active upper wall must begin inside the grid.")
        if self.lower_walls is not None:
            lower_start = self.lower_walls["at"] + self.lower_walls["offset"]
            active = self.lower_walls["kappa"] > 0.0
            outside = active & (lower_start < self.grid_min)
            if bool(jax.device_get(jnp.any(outside))):
                raise ValueError("Every active lower wall must begin inside the grid.")

        if hills_file is None:
            self.hills_file = None
        else:
            try:
                self.hills_file = Path(hills_file)
            except TypeError as exc:
                raise TypeError("hills_file must be path-like or None.") from exc
        self._hills_initialized = False
        self._metadynamics_step = 0
        self._cv_dtype = initial_values[0].dtype
        self._hill_centers = []

    @staticmethod
    def _vector_parameter(name, value, size, *, positive=False, minimum=None):
        """Normalize a scalar or length-sized vector on the Python side."""
        try:
            raw = np.asarray(value)
            if np.issubdtype(raw.dtype, np.bool_):
                raise TypeError
            values = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric.") from exc
        if values.shape == ():
            values = np.full(size, values.item())
        elif values.shape != (size,):
            raise ValueError(f"{name} must be a scalar or have shape ({size},).")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        if positive and np.any(values <= 0.0):
            raise ValueError(f"{name} must contain strictly positive values.")
        if minimum is not None and np.any(values < minimum):
            raise ValueError(f"{name} values must be at least {minimum}.")
        return jnp.asarray(values)

    @staticmethod
    def _integer_vector_parameter(name, value, size):
        """Normalize a positive scalar or length-sized integer vector."""
        values = np.asarray(value)
        if values.shape == ():
            values = np.full(size, values.item())
        elif values.shape != (size,):
            raise ValueError(f"{name} must be a scalar or have shape ({size},).")
        if (
            np.issubdtype(values.dtype, np.bool_)
            or not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.complexfloating)
            or not np.all(np.isfinite(values))
            or not np.all(values == np.floor(values))
        ):
            raise TypeError(f"{name} must contain positive integers.")
        values = values.astype(int)
        if np.any(values < 2):
            raise ValueError(f"{name} must contain integers of at least two.")
        return values

    @staticmethod
    def _add_grid_hill(grid, grid_points, center, sigma, height):
        displacement = (grid_points - center[None, :]) / sigma
        hill = height * jnp.exp(-0.5 * jnp.sum(jnp.square(displacement), axis=1))
        return grid + hill.reshape(grid.shape)

    @staticmethod
    def _interpolate_grid(grid, values, grid_min, grid_spacing):
        """Interpolate a one- to three-dimensional grid with cubic splines."""
        grid_max = grid_min + grid_spacing * jnp.asarray(
            [size - 1 for size in grid.shape]
        )
        position = (jnp.clip(values, grid_min, grid_max) - grid_min) / grid_spacing
        lower = jnp.floor(position).astype(int)
        lower = jnp.minimum(
            lower, jnp.asarray([size - 2 for size in grid.shape])
        )
        fraction = position - lower
        fraction2 = jnp.square(fraction)
        fraction3 = fraction2 * fraction
        weights = jnp.stack(
            (
                -0.5 * fraction + fraction2 - 0.5 * fraction3,
                1.0 - 2.5 * fraction2 + 1.5 * fraction3,
                0.5 * fraction + 2.0 * fraction2 - 1.5 * fraction3,
                -0.5 * fraction2 + 0.5 * fraction3,
            ),
            axis=1,
        )
        value = jnp.asarray(0.0, dtype=grid.dtype)
        for corner in product(range(4), repeat=values.shape[0]):
            weight = jnp.prod(
                weights[jnp.arange(values.shape[0]), jnp.asarray(corner)]
            )
            index = tuple(
                jnp.clip(lower[i] + corner[i] - 1, 0, grid.shape[i] - 1)
                for i in range(values.shape[0])
            )
            value += weight * grid[index]
        return value

    def _prepare_collective_variables(self, definitions):
        if not isinstance(definitions, (list, tuple)):
            raise TypeError("collective_variables must be a sequence of dictionaries.")
        if not 1 <= len(definitions) <= 3:
            raise ValueError("collective_variables must contain between one and three CVs.")

        records = []
        initial_values = []
        used_ids = set()
        for definition in definitions:
            if not isinstance(definition, dict):
                raise TypeError("Each collective variable definition must be a dictionary.")
            unknown = set(definition) - self._cv_keys
            missing = {"id", "field_ids", "engine"} - set(definition)
            if unknown:
                raise ValueError(f"Unknown collective variable keys: {sorted(unknown)}.")
            if missing:
                raise ValueError(f"Missing collective variable keys: {sorted(missing)}.")

            cv_id = definition["id"]
            if not isinstance(cv_id, str):
                raise TypeError("Collective variable id must be a string.")
            if not cv_id or any(character.isspace() for character in cv_id):
                raise ValueError("Collective variable id must be nonempty and contain no whitespace.")
            if cv_id in used_ids:
                raise ValueError(f"Duplicate collective variable id {cv_id!r}.")
            used_ids.add(cv_id)

            field_ids = definition["field_ids"]
            if isinstance(field_ids, str):
                field_ids = (field_ids,)
            elif isinstance(field_ids, (list, tuple)):
                field_ids = tuple(field_ids)
            else:
                raise TypeError("field_ids must be a field ID or a sequence of field IDs.")
            if not 1 <= len(field_ids) <= 3:
                raise ValueError("Each collective variable must use one to three fields.")
            if any(not isinstance(field_id, str) for field_id in field_ids):
                raise TypeError("Every field ID must be a string.")
            if len(set(field_ids)) != len(field_ids):
                raise ValueError("A collective variable cannot repeat a field ID.")
            fields = tuple(self.system.get_field_by_ID(field_id) for field_id in field_ids)

            engine = definition["engine"]
            if not callable(engine):
                raise TypeError(f"Collective variable engine {cv_id!r} must be callable.")
            parameters = definition.get("parameters")
            if parameters is not None:
                try:
                    parameters = jnp.asarray(parameters)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"Collective variable parameters for {cv_id!r} must be numeric."
                    ) from exc
                if not np.issubdtype(parameters.dtype, np.number):
                    raise TypeError(
                        f"Collective variable parameters for {cv_id!r} must be numeric."
                    )
                if bool(jax.device_get(jnp.any(~jnp.isfinite(parameters)))):
                    raise ValueError(
                        f"Collective variable parameters for {cv_id!r} must be finite."
                    )

            value_engine = jax.jit(engine)
            field_values = tuple(field.get_values() for field in fields)
            try:
                value = jnp.asarray(value_engine(*field_values, parameters))
            except Exception as exc:
                raise ValueError(
                    f"Collective variable {cv_id!r} could not be evaluated."
                ) from exc
            if value.shape != () or jnp.issubdtype(value.dtype, jnp.complexfloating):
                raise ValueError(
                    f"Collective variable {cv_id!r} must return one real scalar."
                )
            if not bool(jax.device_get(jnp.isfinite(value))):
                raise ValueError(f"Collective variable {cv_id!r} must be finite.")

            field_argnums = tuple(range(len(fields)))
            value_and_grad_engine = jax.jit(
                jax.value_and_grad(engine, argnums=field_argnums)
            )
            try:
                _, gradients = value_and_grad_engine(*field_values, parameters)
            except Exception as exc:
                raise ValueError(
                    f"Collective variable {cv_id!r} could not be differentiated."
                ) from exc
            for field, gradient in zip(fields, gradients):
                if gradient.shape != field.get_values().shape:
                    raise ValueError(
                        f"Gradient of collective variable {cv_id!r} has the wrong shape."
                    )
                if gradient.dtype != field.get_values().dtype:
                    raise ValueError(
                        f"Gradient of collective variable {cv_id!r} has the wrong dtype."
                    )
                if bool(jax.device_get(jnp.any(~jnp.isfinite(gradient)))):
                    raise ValueError(
                        f"Gradient of collective variable {cv_id!r} must be finite."
                    )
                if field._sharding is not None and gradient.sharding != field._sharding:
                    raise ValueError(
                        f"Gradient of collective variable {cv_id!r} must preserve sharding."
                    )

            records.append(
                {
                    "id": cv_id,
                    "fields": fields,
                    "parameters": parameters,
                    "value_engine": value_engine,
                    "value_and_grad_engine": value_and_grad_engine,
                }
            )
            initial_values.append(value)
        return records, initial_values

    def _prepare_wall(self, name, definition):
        if definition is None:
            return None
        if not isinstance(definition, dict):
            raise TypeError(f"{name} must be a dictionary or None.")
        unknown = set(definition) - self._wall_keys
        missing = {"at", "kappa"} - set(definition)
        if unknown:
            raise ValueError(f"Unknown {name} keys: {sorted(unknown)}.")
        if missing:
            raise ValueError(f"Missing {name} keys: {sorted(missing)}.")
        size = self.cv_dimension
        return {
            "at": self._vector_parameter(f"{name}.at", definition["at"], size),
            "kappa": self._vector_parameter(
                f"{name}.kappa", definition["kappa"], size, minimum=0.0
            ),
            "exp": self._vector_parameter(
                f"{name}.exp", definition.get("exp", 2.0), size, minimum=2.0
            ),
            "eps": self._vector_parameter(
                f"{name}.eps", definition.get("eps", 1.0), size, positive=True
            ),
            "offset": self._vector_parameter(
                f"{name}.offset", definition.get("offset", 0.0), size
            ),
        }

    def calc_collective_variables(self):
        """Return the current collective-variable values with shape ``(d,)``."""
        values = []
        for cv in self._collective_variables:
            field_values = tuple(field.get_values() for field in cv["fields"])
            values.append(cv["value_engine"](*field_values, cv["parameters"]))
        return jnp.stack(values)

    def _collective_values_and_gradients(self):
        values = []
        gradients = []
        for cv in self._collective_variables:
            field_values = tuple(field.get_values() for field in cv["fields"])
            value, gradient = cv["value_and_grad_engine"](
                *field_values, cv["parameters"]
            )
            values.append(value)
            gradients.append(gradient)
        return jnp.stack(values), gradients

    def _metadynamics_bias_and_derivative(self, values):
        return self._grid_value_and_grad_engine(
            self._bias_grid, values, self.grid_min, self.grid_spacing
        )

    @staticmethod
    def _wall_bias_and_derivative(values, wall, direction):
        if wall is None:
            return jnp.asarray(0.0, dtype=values.dtype), jnp.zeros_like(values)
        scale = (direction * (values - wall["at"]) + wall["offset"]) / wall["eps"]
        active_scale = jnp.maximum(scale, 0.0)
        energy = jnp.sum(wall["kappa"] * active_scale ** wall["exp"])
        derivative = direction * wall["kappa"] * wall["exp"] / wall["eps"]
        derivative *= jnp.where(
            scale > 0.0, active_scale ** (wall["exp"] - 1.0), 0.0
        )
        return energy, derivative

    def _bias_components_and_derivative(self, values):
        metadynamics, derivative = self._metadynamics_bias_and_derivative(values)
        upper, upper_derivative = self._wall_bias_and_derivative(
            values, self.upper_walls, 1.0
        )
        lower, lower_derivative = self._wall_bias_and_derivative(
            values, self.lower_walls, -1.0
        )
        return metadynamics, upper + lower, derivative + upper_derivative + lower_derivative

    def calc_metadynamics_bias(self):
        """Return the Gaussian-hill bias energy at the current state."""
        values = self.calc_collective_variables()
        return self._metadynamics_bias_and_derivative(values)[0]

    def calc_wall_bias(self):
        """Return the upper- and lower-wall bias energy at the current state."""
        values = self.calc_collective_variables()
        return self._bias_components_and_derivative(values)[1]

    def calc_total_bias(self):
        """Return the total metadynamics and wall bias energy."""
        values = self.calc_collective_variables()
        metadynamics, walls, _ = self._bias_components_and_derivative(values)
        return metadynamics + walls

    def calc_biased_potential_energy(self):
        """Return the physical system potential plus the simulation-owned bias."""
        return self.system.calc_total_potential_energy() + self.calc_total_bias()

    def get_hill_centers(self):
        """Return a copy of the deposited hill centers."""
        if not self._hill_centers:
            return jnp.empty((0, self.cv_dimension), dtype=self._cv_dtype)
        return jnp.asarray(np.stack(self._hill_centers), dtype=self._cv_dtype)

    def _update_force(self, profile=False):
        super()._update_force(profile=profile)
        if (
            not self._hill_centers
            and self.upper_walls is None
            and self.lower_walls is None
        ):
            return
        if profile:
            t0 = timer()
        values, gradients = self._collective_values_and_gradients()
        _, _, cv_derivative = self._bias_components_and_derivative(values)
        for coefficient, cv, cv_gradients in zip(
            cv_derivative, self._collective_variables, gradients
        ):
            for field, gradient in zip(cv["fields"], cv_gradients):
                field.set_force(field.get_force() - coefficient * gradient)
        if profile:
            jax.block_until_ready(cv_derivative)
            logging.info(
                "Time for updating metadynamics bias force: %.8fs", timer() - t0
            )

    def _initialize_hills_file(self):
        if self._hills_initialized or self.hills_file is None:
            self._hills_initialized = True
            return
        if self.hills_file.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing HILLS file {self.hills_file}."
            )
        cv_ids = [cv["id"] for cv in self._collective_variables]
        fields = ["step", *cv_ids, *(f"sigma_{cv_id}" for cv_id in cv_ids), "height"]
        with self.hills_file.open("x", encoding="utf-8") as stream:
            stream.write("#! FIELDS " + " ".join(fields) + "\n")
            stream.write(f"#! SET pace {self.pace}\n")
        self._hills_initialized = True

    def _write_hill(self, center):
        if self.hills_file is None:
            return
        center = np.asarray(center)
        row = [self._metadynamics_step, *center, *self._sigma_host, self.height]
        with self.hills_file.open("a", encoding="utf-8") as stream:
            stream.write(" ".join(f"{value:.16g}" for value in row) + "\n")

    def _deposit_hill(self):
        center = self.calc_collective_variables()
        self._bias_grid = self._add_grid_hill_engine(
            self._bias_grid,
            self._grid_points,
            center,
            self.sigma,
            self.height,
        )
        center = np.asarray(jax.device_get(center))
        self._hill_centers.append(center)
        self._write_hill(center)

    def run(self, nsteps=1, profile=False, seed=None):
        """Run NVT metadynamics for a number of outer MD steps.

        Hills are deposited after steps divisible by ``pace`` and affect the
        following force evaluation. Repeated calls continue the deposition
        counter and random stream.
        """
        self._validate_so3_field_count(self.SO3_fields)
        for field in self.all_fields:
            if field.integrator is None:
                raise ValueError(
                    "Please set the integrator for the field %s before running the simulation"
                    % type(field)
                )
        self._initialize_hills_file()
        if seed is not None:
            self.reset_random_key(seed=seed)
        self.initialize_reporters()
        for _ in range(nsteps):
            if profile:
                t0 = timer()
            subkeys = self._next_random_keys(self.nfields)
            self._step(subkeys, profile)
            self._metadynamics_step += 1
            if self._metadynamics_step % self.pace == 0:
                self._deposit_hill()
            self.step_reporters()
            if profile:
                logging.info("Total time for one step: %.8fs", timer() - t0)


class SimulationNPTLangevin(SimulationNVTLangevin):
    """
    Class for NPT molecular dynamics simulation using Langevin dynamics.

    Parameters
    ----------
    system : System
        The physical system to simulate
    pressure : float, optional
        External pressure in bar, by default 0.0
    seed : int, optional
        Seed for the simulation-owned random stream, by default 42
    key : array_like, optional
        Existing JAX ``PRNGKey`` state, by default None
    """
    def __init__(self, system, pressure=0.0, seed=42, key=None):
        super().__init__(system, seed=seed, key=key)
        ## set pressure
        self.pressure = pressure
        pV_param = self.system.get_interaction_by_ID('pV').get_parameters()
        pV_param_new = [pressure * Constants.bar, pV_param[1]]
        self.system.get_interaction_by_ID('pV').set_parameters(pV_param_new)
        ## get all fields, including the global strain field
        self.SO3_fields = self.system.get_all_SO3_fields()
        self.non_SO3_fields = self.system.get_all_non_SO3_fields()
        self.all_fields = self.SO3_fields + self.non_SO3_fields
        self.nfields = len(self.all_fields)
