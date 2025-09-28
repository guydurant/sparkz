from setuptools import setup, find_packages

setup(
    name='sparkz',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[                 # if it's on PyPI
        'pdbfixer @ git+https://github.com/openmm/pdbfixer.git',
        "ligandmpnn @ git+https://github.com/guydurant/LigandMPNN.git",
        'boltz==2.1.1',
    ],
    entry_points={
        'console_scripts': [
            'sparkz=cli:main',
            'screwzfix=screwzfix.main:main',
        ],
    },
)