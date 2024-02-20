from setuptools import setup, find_packages

DESCRIPTION = '(Continuous) Ab initio Generalized Langevin equations '
setup(
    name='AIGLE',
    version='0.1.0',
    author="Pinchen Xie",
    author_email="<pinchenx@math.princeton.edu>",
    # packages=find_packages(),
    pakages=['AIGLE', 'AIGLE/plugins'],
    description=DESCRIPTION,
    python_requires=">=3.8",
    install_requires=[
        'numpy>=1.24.0',
        'torch>=2.0.0',
        # 'openmm>=8.0.0'
    ],
    # entry_points={
    #     'console_scripts': [
    #         'aigle = AIGLE.module1:main',
    #     ],
    # },
)