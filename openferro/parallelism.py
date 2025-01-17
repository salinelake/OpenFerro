"""
Classes for multi-GPU parallelism.

Notes
-----
This file is part of OpenFerro.
"""

import logging
import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding

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
            If only one device is available
            If num_rows * num_cols does not match number of devices
        """
        if devices is None:
            devices = np.array(jax.devices())
        else:
            devices = np.array(devices)
        num_devices = len(devices)
        if num_devices == 1:
            num_rows = 1
            num_cols = 1
            raise ValueError("Only one device is available. No parallelism is applied.")
        if num_rows is None or num_cols is None:
            for i in range(int(np.sqrt(num_devices)), 0, -1):
                if num_devices % i == 0:
                    num_rows = i
                    num_cols = num_devices // i
                    break
        else:
            num_rows = int(num_rows)
            num_cols = int(num_cols)
            if num_rows * num_cols != num_devices:
                raise ValueError("The number of devices does not match the configuration.")
        logging.info('The number of devices is {}'.format(num_devices))
        logging.info('The configuration of the devices is ({} x {})'.format(num_rows, num_cols))
        devices = devices.reshape(num_rows, num_cols)
        # Create a Mesh object to distribute a value across devices:
        self.mesh = Mesh(devices=devices, axis_names=('x', 'y'))

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