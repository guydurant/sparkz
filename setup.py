from setuptools import setup, find_packages

setup(
    name='sparkz',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'rdkit',
        'torchvision==0.21.0',
        'boltz @ git+https://github.com/guydurant/boltz.git',                    # if it's on PyPI
        'pdbfixer @ git+https://github.com/openmm/pdbfixer.git'
    ],
    entry_points={
        'console_scripts': [
            'sparkz=cli:main',
            'screwz=inpainting.main:main',
        ],
    },
)