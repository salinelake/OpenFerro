Quickstart
==========


Tutorial
--------

Follow this jupyter-notebook tutorial_ to learn the basic usage of OpenFerro.

Running OpenFerro on CPU
------------------------

No matter if you are working on a CPU-only machine or a GPU-enabled machine, if you want to run OpenFerro on CPU, specify the environment variable ``JAX_PLATFORMS`` before you run your code.

.. code-block:: bash

   export JAX_PLATFORMS=cpu

JAX automatically determine how many CPU cores it will utilize.

Running OpenFerro on GPU 
------------------------

If you have the GPU-version of JAX installed and you are working on a GPU-enabled machine, OpenFerro will automatically use GPU to accelerate the simulation. 
Try run tutorial_ and see if the GPU has been utilized.

GPU parallelization
-------------------

You only need to add three lines to your code to enable multi-GPU parallelization. See here_ for a complete example.

.. code-block:: python

   from openferro.parallelism import DeviceMesh
   # if you have 4 GPUs, you can set num_rows=2, num_cols=2. If you have 2 GPUs, you can set num_rows=1, num_cols=2.
   gpu_mesh = DeviceMesh(num_rows=2, num_cols=2)   
   # Move all fields to GPUs after you have defined the system.
   system.move_fields_to_multi_devs(gpu_mesh)      

.. _tutorial: https://github.com/salinelake/OpenFerro/blob/main/tutorials/quickstart.ipynb
.. _here: https://github.com/salinelake/OpenFerro/blob/main/examples/Profiling_GPU/npt_GPU.ipynb
