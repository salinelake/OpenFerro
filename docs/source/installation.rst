Installation
============

OpenFerro is available on PyPI and can also be installed from source. The
package metadata is the authoritative compatibility contract.

Supported versions
------------------

OpenFerro 0.3 supports Python 3.13 and 3.14, JAX versions from 0.10 up to (but
not including) 0.12, and NumPy versions from 2.0 up to (but not including) 3.
GPU and multi-host execution remain experimental; see
:doc:`feature_status`.

Create an environment
---------------------

For a fresh Conda environment:

.. code-block:: bash

   conda create -n openferro python=3.13 pip
   conda activate openferro

CPU installation
----------------

Install from PyPI:

.. code-block:: bash

   python -m pip install openferro

NVIDIA GPU installation
-----------------------

GPU support comes from JAX's CUDA extra. On Linux with NVIDIA driver 580 or
newer, install OpenFerro and JAX's CUDA 13 support in one command:

.. code-block:: bash

   python -m pip install openferro "jax[cuda13]"

For drivers 525–579, use CUDA 12 instead:

.. code-block:: bash

   python -m pip install openferro "jax[cuda12]"

Pip resolves compatible OpenFerro, JAX, and NumPy versions and installs the
CUDA libraries. A separate CUDA toolkit installation is unnecessary; the
machine still needs a compatible NVIDIA driver. Consult the official
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
for platform and hardware requirements.

Verify GPU access on a GPU node, inside a scheduler allocation on HPC systems:

.. code-block:: bash

   JAX_PLATFORMS=cuda python -c "import jax; import openferro as of; print(jax.devices()); print(of.System)"

The device list should contain a ``CudaDevice``. Setting ``JAX_PLATFORMS=cuda``
makes this check fail if the CUDA backend cannot initialize.

Source and development installation
------------------------------------

Clone the repository and install from its root:

.. code-block:: bash

   git clone https://github.com/salinelake/OpenFerro.git
   cd OpenFerro
   python -m pip install .

For editable development with the unit-test dependencies and analysis plotting:

.. code-block:: bash

   python -m pip install -e ".[test]" matplotlib

Matplotlib is used by the analysis scripts and plotting tests. To enable GPU
support in a source installation, include ``"jax[cuda13]"`` or
``"jax[cuda12]"`` in the same pip command.

Validation
----------

Verify the public import:

.. code-block:: bash

   python -c "import openferro as of; print(of.System)"

From a source checkout with the development dependencies installed, run the
lightweight CPU tests:

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m pytest tests/unit_tests -q \
       -m "not slow and not stochastic and not gpu"

The ``slow``, ``stochastic``, and ``gpu`` markers select additional execution
and numerical checks; the short CPU run above is not the complete suite.

To build and validate clean source and wheel artifacts, install the packaging
extra and run the packaging marker:

.. code-block:: bash

   python -m pip install -e ".[test,package]"
   JAX_PLATFORMS=cpu python -m pytest tests/packaging -q

The artifact test builds from a temporary clean source tree and imports each
installed artifact from outside the repository, preventing ignored ``build/``
contents from masking missing packages. The initial artifact build deliberately
disables isolation, so the ``package`` extra includes every requirement from
``[build-system].requires`` as well as ``build`` and ``twine``. Artifact smoke
installs use normal PEP 517 build isolation, ensuring that the sdist can
bootstrap the build backend declared in its own metadata. Install the extra
before running the test.
