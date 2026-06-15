"""
Classes for multi-GPU parallelism.

Notes
-----
This file is part of OpenFerro.
"""

import logging
import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec


class DeviceMesh:
    def __init__(self, devices=None, num_rows=None, num_cols=None):
        """
        Initialize the single-host multi-device parallelism. Get the mesh of the devices.

        Parameters
        ----------
        devices : array-like, optional
            List of devices to use. If None, uses all available devices
        num_rows : int, optional
            Number of rows in device mesh. If None, automatically determined
        num_cols : int, optional
            Number of columns in device mesh. If None, automatically determined

        Raises
        ------
        ValueError
            If no devices are available
            If num_rows * num_cols does not match number of devices
        """
        if devices is None:
            devices = jax.devices()
        devices = np.asarray(devices, dtype=object)
        num_devices = devices.size
        if num_devices == 0:
            raise ValueError("No devices are available for parallelism.")

        num_rows, num_cols = self._resolve_mesh_shape(
            devices, num_devices, num_rows, num_cols
        )

        logging.info("The number of devices is {}".format(num_devices))
        logging.info(
            "The configuration of the devices is ({} x {})".format(
                num_rows, num_cols
            )
        )

        devices = devices.reshape(num_rows, num_cols)
        self.mesh = Mesh(devices=devices, axis_names=('x', 'y'))

    @staticmethod
    def _resolve_mesh_shape(devices, num_devices, num_rows, num_cols):
        if num_rows is None and num_cols is None:
            if devices.ndim == 2:
                return devices.shape
            if devices.ndim != 1:
                raise ValueError("devices must be a one- or two-dimensional array.")
            for i in range(int(np.sqrt(num_devices)), 0, -1):
                if num_devices % i == 0:
                    return i, num_devices // i
        elif num_rows is None:
            num_cols = int(num_cols)
            if num_cols <= 0:
                raise ValueError("num_cols must be a positive integer.")
            if num_devices % num_cols != 0:
                raise ValueError("The number of devices does not match the configuration.")
            return num_devices // num_cols, num_cols
        elif num_cols is None:
            num_rows = int(num_rows)
            if num_rows <= 0:
                raise ValueError("num_rows must be a positive integer.")
            if num_devices % num_rows != 0:
                raise ValueError("The number of devices does not match the configuration.")
            return num_rows, num_devices // num_rows

        num_rows = int(num_rows)
        num_cols = int(num_cols)
        if num_rows <= 0 or num_cols <= 0:
            raise ValueError("num_rows and num_cols must be positive integers.")
        if num_rows * num_cols != num_devices:
            raise ValueError("The number of devices does not match the configuration.")
        return num_rows, num_cols

    def partition_sharding(self):
        """
        Produce a NamedSharding object to distribute a value across devices, partitioning along the x and y axes.

        Returns
        -------
        NamedSharding
            Sharding object for partitioning values across devices
        """
        sharding = NamedSharding(self.mesh, PartitionSpec('x', 'y'))
        return sharding

    def replicate_sharding(self):
        """
        Produce a NamedSharding object to replicate a value across devices.

        Used for broadcasting values that do not need to be partitioned.

        Returns
        -------
        NamedSharding
            Sharding object for replicating values across devices
        """
        sharding = NamedSharding(self.mesh, PartitionSpec())
        return sharding
