from distutils.core import setup

setup(
    name='rotor_acoustics_tonal_noise',
    version='0',
    packages=[
        'rotor_acoustics_tonal_noise',
    ],
    install_requires=[
        'openmdao',
        'csdl',
        'csdl_om',
        ],
)