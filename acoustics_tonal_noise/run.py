import numpy as np 
import openmdao.api as om
from csdl import Model 
try:
    from csdl_om import Simulator
except:
    raise ModuleNotFoundError("This run file requires a backend for CSDL")


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


t_c_ratio      = np.genfromtxt('txt_files/test_t_c.txt')
num_radial= len(t_c_ratio)
directivity = 0
mode = 1
num_nodes = 1
num_azimuthal = 33

dT_dr          = np.loadtxt('txt_files/test_dT_dr.txt')
dQ_dr          = np.loadtxt('txt_files/test_dQ_dr.txt')
chord          = np.loadtxt('txt_files/test_chord.txt')
twist          = 4 * np.ones((len(chord),)) * np.pi / 180


# TO DO: 
#   - how to specify/ define directivity/observer location from central run file
#   - single or multiple observer location 
#   - where to specify? 
#   - Reference frame



# TO DO: change num_azimuthal name

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
shape = (num_nodes,num_radial,num_azimuthal)

thrust_origin = np.array([[1.829,0,3.167]])

class RunModel(Model):
    def define(self):
         # Inputs not changing across conditions (segments)
        self.create_input(name='propeller_radius', shape=(1, ), units='m', val=0.6096)
        self.create_input(name='chord_profile', shape=(num_radial,), units='m', val=chord)
        self.create_input(name='twist_profile', shape=(num_radial,), units='rad', val=twist)
        self.create_input(name='thickness_to_chord_ratio', shape=(num_radial,), units='rad', val=t_c_ratio)
        # pitch_cp = self.create_input(name='pitch_cp', shape=(4,), units='rad', val=np.array([8.60773973e-01,6.18472835e-01,3.76150609e-01,1.88136239e-01]))#np.linspace(35,10,4)*np.pi/180)
        # self.add_design_variable('pitch_cp', lower=5*np.pi/180,upper=60*np.pi/180)
        self.create_input(name='thrust_origin', shape=(num_nodes,3), val=np.tile(thrust_origin,(num_nodes,1)))

        self.create_input('omega', shape=(num_nodes, 1), units='rpm', val=2.150)
        self.create_input('M_inf',shape=(num_nodes), val=0)
        self.create_input('dT', shape=(num_nodes,num_radial), val=np.tile(dT_dr,(num_nodes,1)))
        self.create_input('dQ', shape=(num_nodes,num_radial), val=np.tile(dQ_dr,(num_nodes,1)))


        self.create_input(name='z', shape=(num_nodes,  1), units='m', val=0)
        
        
        
        self.add(CoreAcousticsModel(
            #  acoustics_dict=acoustics_dict, # acoustics = acoustics (instance of subclass of lsdo_kit.solver)
            name='acoustics',
            num_blades=num_blades,
            mode=mode,
            directivity=directivity,
            num_nodes=num_nodes, # no empty space left and right of equal sign 
            num_radial=num_radial,
            num_azimuthal=num_azimuthal,
        ), name='noise_model')

# acoustics_model.visualize_sparsity(recursive=True)
sim = Simulator(RunModel())
sim.run()
# exit()
print('\n')

# print(sim['tonal_plus_broadband'])



GD_tonal_noise = sim['SPL_tonal_Gutin_Deming']
BM_tonal_noise = sim['SPL_tonal_Barry_Magliozzi']
theta = sim['_theta'][0,0,:]
print(GD_tonal_noise)
print(BM_tonal_noise)

# GD_gill = np.loadtxt('txt_files/GD_tonal_Gill_output.txt')
# BM_gill = np.loadtxt('txt_files/BM_tonal_Gill_output.txt')
# polar_plot(theta,GD_tonal_noise,GD_gill,BM_tonal_noise,BM_gill)


