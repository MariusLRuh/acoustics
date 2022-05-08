import numpy as np 
import openmdao.api as om
from csdl import Model 
try:
    from csdl_om import Simulator
except:
    raise ModuleNotFoundError("This run file requires a backend for CSDL")

from functions.get_acoustics_dictionary import get_acoustics_parameters
from functions.polar_plot import polar_plot
from core.core_acoustics_model import CoreAcousticsModel


rotor_radius   = np.array([0.6096,0.6096,0.6096])                 # in (m)
hub_radius     = 17.70                                            # percent of rotor radius
num_blades     = 2 # np.array([2,3,4])
altitude       = 10                                               # in (m)
RPM            = np.array([2150,2150,2150])
M_inf          = np.array([0,0,0])                                # free stream Mach number             
x_position     = np.array([3.167,3.167,3.167])                    # in (m)
y_position     = np.array([0,0,0])                                # in (m)
z_position     = np.array([1.829,1.829,1.829])                    # in (m)


t_c_ratio      = np.loadtxt('txt_files/test_t_c.txt')
dT_dr          = np.loadtxt('txt_files/test_dT_dr.txt')
dQ_dr          = np.loadtxt('txt_files/test_dQ_dr.txt')
chord          = np.loadtxt('txt_files/test_chord.txt')
twist          = 4 * np.ones((len(chord),)) * np.pi / 180


directivity    = 1
max_freq_mode   = 1
num_evaluations = 1
num_radial      = len(dT_dr)
num_tangential  = 33

# TO DO: change num_tangential name
# TO DO: Group vs component 
# Acoustics vs AcousticsModel
#   - AcousticsModel: subclass of csdl model
#   - Acoustics: subclass of lsdo_kit.solver (conventions for solver, talk to
#     Darshan)
#     TO DO: contact Jiayao regarding naming convention for expanded variables;
#     want acoustics, dynamic inflow, BEM and (U)VLM to follow same convention
# lsdo_vlm
# lsdo_bem
# lsdo_fwh
# lsdo_ph
# lsdo_fdt
# lsdo_uvlm
# lsdo_vpm
# TO DO:
shape = (num_evaluations,num_radial, num_tangential)


acoustics_dict = get_acoustics_parameters(directivity,max_freq_mode,num_blades, altitude)


acoustics_model = Model()

group = CoreAcousticsModel(
    acoustics_dict = acoustics_dict, # acoustics = acoustics (instance of subclass of lsdo_kit.solver)
    num_evaluations = num_evaluations, # no empty space left and right of equal sign 
    num_radial = num_radial,
    num_tangential = num_tangential,
)
acoustics_model.add(group,'core_acoustics_group')
# pylint 

sim = Simulator(acoustics_model)

for i in range(num_evaluations):
    sim['rotational_speed'][i] = RPM[i]/60
    sim['rotor_radius'][i] = rotor_radius[i]
    sim['dr'][i] = ((rotor_radius[i])-(hub_radius * 1e-2 * rotor_radius[i]))/ (num_radial -1)

    sim['hub_radius'][i]   = hub_radius * 1e-2 * rotor_radius[i]
   

    sim['x_position'][i] = x_position[i]
    sim['y_position'][i] = y_position[i]
    sim['z_position'][i] = z_position[i]
    sim['dS'][i] = (x_position[i]**2 + y_position[i]**2 + z_position[i]**2)**0.5

    sim['dT'][i,:] = dT_dr
    sim['dQ'][i,:] = dQ_dr
    sim['thickness_to_chord_ratio'][i,:] = t_c_ratio
    sim['chord'][i,:] = chord
    sim['twist'][i,:] = twist
    sim['M_inf'][i] = M_inf[i]



sim.run()
print('\n')
# print(sim['rotor_radius'])
# print(sim['_rotor_radius'])

# print(sim['gr_test'])
print(sim['fr_test'],'fr_test')
# exit()


GD_tonal_noise = sim['SPL_tonal_Gutin_Deming']
BM_tonal_noise = sim['SPL_tonal_Barry_Magliozzi']
theta = sim['_theta'][0,0,:]


# print(sim['fr_test'][0,:,0])

GD_gill = np.loadtxt('txt_files/GD_tonal_Gill_output.txt')
BM_gill = np.loadtxt('txt_files/BM_tonal_Gill_output.txt')
# polar_plot(theta,GD_tonal_noise,GD_gill,BM_tonal_noise,BM_gill)


