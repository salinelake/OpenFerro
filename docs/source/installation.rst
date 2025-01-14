Installation
============

OpenFerro is available on `GitHub <https://github.com/salinelake/OpenFerro>`_.

First, create a new conda environment and activate it:

.. code-block:: bash

   conda create -n openferro python=3.10
   conda activate openferro

Then, we will install JAX. OpenFerro requires JAX 0.4.0 or later. See [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for more details.

- CPU-only (linux, macos, windows):

.. code-block:: bash

   pip install -U jax

- GPU (NVIDIA, CUDA 12)

.. code-block:: bash

   pip install -U "jax[cuda12]"

Last, let us install OpenFerro:

.. code-block:: bash

   git clone https://github.com/salinelake/OpenFerro.git
   cd OpenFerro
   pip install .


Try importing OpenFerro in Python command line:

.. code-block:: python

   import openferro as of

Congratulations! You have successfully installed OpenFerro.