import numpy as np 
import openmdao.api as om
from csdl import Model 

# from csdl_om import Simulator
from python_csdl_backend import Simulator
# from python_csdl_backend import Simulator

from functions.polar_plot import polar_plot
from core.core_acoustics_model import CoreAcousticsModel


rotor_radius   = np.array([0.6096,0.6096,0.6096])                 # in (m)
hub_radius     = 17.70                                            # percent of rotor radius
num_blades     = 2 # np.array([2,3,4])
altitude       = 0                                               # in (m)
RPM            = np.array([2150,2150,2150])
M_inf          = np.array([0,0,0])                                # free stream Mach number             
x_position     = np.array([3.167,3.167,3.167])                    # in (m)
y_position     = np.array([0,0,0])                                # in (m)
z_position     = np.array([1.829,1.829,1.829])                    # in (m)


t_c_ratio      = np.genfromtxt('txt_files/test_t_c.txt')
num_radial= len(t_c_ratio)
directivity = 0
mode = 3
num_nodes = 1
num_azimuthal = 1

dT_dr = np.loadtxt('txt_files/test_dT_dr.txt').reshape(1,num_radial,1)
dQ_dr = np.loadtxt('txt_files/test_dQ_dr.txt').reshape(1,num_radial,1)
chord = np.loadtxt('txt_files/test_chord.txt').reshape(num_radial,1)
twist = (4 * np.ones((len(chord),)) * np.pi / 180).reshape(num_radial,1)


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

thrust_origin = np.array([[1.829,0,3.16757]])
thrust_vector = np.array([[0,0,-1]])

class RunModel(Model):
    def define(self):
         # Inputs not changing across conditions (segments)
        self.create_input(name='propeller_radius', shape=(1, ), units='m', val=0.6096)
        self.add_design_variable('propeller_radius')
        self.create_input(name='chord_profile', shape=(num_radial,1), units='m', val=chord)
        self.add_design_variable('chord_profile')
        self.create_input(name='twist_profile', shape=(num_radial,1), units='rad', val=twist)
        self.add_design_variable('twist_profile')
        self.create_input(name='thickness_to_chord_ratio', shape=(num_radial,), units='rad', val=t_c_ratio)
        self.add_design_variable('thickness_to_chord_ratio')
        # pitch_cp = self.create_input(name='pitch_cp', shape=(4,), units='rad', val=np.array([8.60773973e-01,6.18472835e-01,3.76150609e-01,1.88136239e-01]))#np.linspace(35,10,4)*np.pi/180)
        # self.add_design_variable('pitch_cp', lower=5*np.pi/180,upper=60*np.pi/180)
        self.create_input(name='thrust_origin', shape=(num_nodes,3), val=np.tile(thrust_origin,(num_nodes,1)))
        self.create_input(name='thrust_vector', shape=(num_nodes,3), val=np.tile(thrust_vector,(num_nodes,1)))

        self.create_input('omega', shape=(num_nodes, 1), units='rpm', val=2150)
        self.add_design_variable('omega')
        self.create_input('M_inf',shape=(num_nodes), val=0)
        self.add_design_variable('M_inf')
        self.create_input('dT', shape=(num_nodes,num_radial,num_azimuthal), val=np.tile(dT_dr,(num_nodes,num_azimuthal)))
        self.add_design_variable('dT')
        self.create_input('dQ', shape=(num_nodes,num_radial,num_azimuthal), val=np.tile(dQ_dr,(num_nodes,num_azimuthal)))
        self.add_design_variable('dQ')

        self.create_input(name='z', shape=(num_nodes,  1), units='m', val=76)
        self.add_design_variable('z')
        
        
        
        self.add(CoreAcousticsModel(
            #  acoustics_dict=acoustics_dict, # acoustics = acoustics (instance of subclass of lsdo_kit.solver)
            name='acoustics',
            num_blades=num_blades,
            mode=mode,
            directivity=directivity,
            num_nodes=num_nodes, # no empty space left and right of equal sign 
            num_radial=num_radial,
            num_azimuthal=num_azimuthal,
            mission_segment='hover'
        ), name='noise_model')

# acoustics_model.visualize_sparsity(recursive=True)
sim = Simulator(RunModel())
sim.run()

# sim.prob.check_partials(compact_print=True, form='central')
# sim.prob.check_totals(of='BM_bessel_output_mode_1',wrt='BM_bessel_input_mode_2')
# sim.prob.check_totals(of='BM_bessel_output_mode_1',wrt='BM_bessel_input_mode_3')
# sim.prob.check_totals(of='BM_bessel_output_mode_2',wrt='BM_bessel_input_mode_3')
# sim.prob.check_totals(of='BM_bessel_output_mode_2',wrt='BM_bessel_input_mode_1')
# sim.prob.check_totals(of='BM_bessel_output_mode_3',wrt='BM_bessel_input_mode_2')
# sim.prob.check_totals(of='BM_bessel_output_mode_3',wrt='BM_bessel_input_mode_1')


# sim.prob.check_totals(compact_print=False)
# exit()
print('\n')
# sim.prob.check_totals(of='BM_bessel_output_mode_3',wrt='BM_bessel_input_mode_3')
# sim.prob.check_totals(of='BM_bessel_output_mode_2',wrt='BM_bessel_input_mode_2')
# sim.prob.check_totals(of='BM_bessel_output_mode_1',wrt='BM_bessel_input_mode_1')

# sim.prob.check_partials(compact_print=True)
# sim.prob.check_totals(of='GD_bessel_output_mode_1',wrt='GD_bessel_input_mode_2')
# sim.prob.check_totals(of='GD_bessel_output_mode_1',wrt='GD_bessel_input_mode_3')
# sim.prob.check_totals(of='GD_bessel_output_mode_2',wrt='GD_bessel_input_mode_3')
# sim.prob.check_totals(of='GD_bessel_output_mode_2',wrt='GD_bessel_input_mode_1')
# sim.prob.check_totals(of='GD_bessel_output_mode_3',wrt='GD_bessel_input_mode_2')
# sim.prob.check_totals(of='GD_bessel_output_mode_3',wrt='GD_bessel_input_mode_1')


# sim.prob.check_totals(compact_print=False)
# exit()
print('\n')
# sim.check_totals(of='GD_bessel_output_mode_1',wrt='GD_bessel_input_mode_1')
# sim.prob.check_totals(of='GD_bessel_output_mode_1',wrt='GD_bessel_input_mode_1', form='backward')
# sim.prob.check_totals(of='GD_bessel_output_mode_2',wrt='GD_bessel_input_mode_2')
# sim.prob.check_totals(of='GD_bessel_output_mode_1',wrt='GD_bessel_input_mode_1')

# print(sim['tonal_plus_broadband'])



# GD_tonal_noise = sim['SPL_tonal_Gutin_Deming']
BM_tonal_noise = sim['SPL_tonal_Barry_Magliozzi']
total_tonal_noise = sim['total_tonal_noise']
total_noise = sim['tonal_plus_broadband_noise']
theta = sim['_theta'][0,0,:]
# print(GD_tonal_noise)
print(BM_tonal_noise)
print(total_tonal_noise)
print(total_noise)

# GD_gill = np.loadtxt('txt_files/GD_tonal_Gill_output.txt')
# BM_gill = np.loadtxt('txt_files/BM_tonal_Gill_output.txt')
# polar_plot(theta,GD_tonal_noise,GD_gill,BM_tonal_noise,BM_gill)


