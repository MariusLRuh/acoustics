from distutils.core import setup

setup(
    name='acoustics_tonal_noise',
    version='0',
    packages=[
        'acoustics_tonal_noise',
    ],
    install_requires=[
        'openmdao',
        'csdl',
        'csdl_om',
        ],
)