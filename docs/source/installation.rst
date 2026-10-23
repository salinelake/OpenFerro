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

CPU installation
----------------

Create and activate a Python 3.13 or 3.14 environment, then install from PyPI:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install openferro

For a source installation, clone the repository and install from its root:

.. code-block:: bash

   git clone https://github.com/salinelake/OpenFerro.git
   cd OpenFerro
   python -m pip install .

For editable development with the unit-test dependencies:

.. code-block:: bash

   python -m pip install -e ".[test]"

NVIDIA GPU installation
-----------------------

Install the JAX accelerator build appropriate for the system before installing
OpenFerro. For CUDA installed through pip, the current JAX commands are:

.. code-block:: bash

   # CUDA 13
   python -m pip install --upgrade "jax[cuda13]>=0.10,<0.12"

   # CUDA 12 alternative
   # python -m pip install --upgrade "jax[cuda12]>=0.10,<0.12"

   python -m pip install --no-deps openferro

Consult the official `JAX installation guide
<https://docs.jax.dev/en/latest/installation.html>`_ for driver, CUDA, cuDNN,
NCCL, platform, and local-toolkit requirements. Do not assume that a successful
CPU import validates the GPU backend.

Validation
----------

Verify the public import and run the lightweight tests:

.. code-block:: bash

   python -c "import openferro as of; print(of.System)"
   JAX_PLATFORMS=cpu python -m pytest tests/unit_tests -q

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
