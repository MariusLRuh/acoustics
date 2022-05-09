# Tonal noise acoustics models for rotor analysis

# Installation 

This acoustics package requires the following packages to be installed before it can be used:

* [csdl](https://lsdolab.github.io/csdl/docs/tutorial/install) (including csdl_om)


Please follow the installation instructions provided in the above links. Once these packages are installed you can proceed as follows with the installation of lsdo_rotor:

* Clone this repository via ``git clonehttps://github.com/MariusLRuhacoustics.git`` or download it as a .zip file.
  If the repository is cloned successfully, future versions of this acoustics package can be downloaded via `git pull`.
* Execute the next two commands to install lsdo_rotor
  * ``cd acoustics_tonal_noise``
  * ``pip install -e .``
* If the installation is successful, check that the run.py file executes by typing
  * ``cd acoustics_tonal_noise``
  * ``python run.py``