from setuptools import setup, find_packages

DESCRIPTION = 'JAX-based framework for Lattice Hamiltonian simulation'
setup(
    name='OpenFerro',
    version='0.1.0',
    author="Pinchen Xie",
    author_email="<pinchenxie@lbl.gov>",
    # packages=find_packages(),
    packages=['openferro'],
    description=DESCRIPTION,
    python_requires=">=3.8",
    install_requires=[
        'numpy>=1.24.0',
        'jax>=0.4.0',
    ],
)
