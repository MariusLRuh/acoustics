import numpy as np
from csdl import Model
import csdl

from acoustics_parameters import AcousticsParameters
from core.GD_bessel_explicit_comp import GDBesselExplicitComp

class GutinDemingGroup(Model):

    def initialize(self):
        self.parameters.declare('acoustics_dict', types=AcousticsParameters)
        self.parameters.declare('shape', types=tuple)

    def define(self):
        acoustics_dict = self.parameters['acoustics_dict']
        shape = self.parameters['shape']
        max_frequency_mode = acoustics_dict['mode']

        a = acoustics_dict['speed_of_sound']
        # print(a,'SPEED of SOUND')
        dir = acoustics_dict['directivity']
        B = acoustics_dict['num_blades']
        rho = acoustics_dict['density']

        x = self.declare_variable('_x_position', shape = shape)
        y = self.declare_variable('_y_position', shape = shape)
        z = self.declare_variable('_z_position', shape = shape)

        dT = self.declare_variable('_dT', shape = shape)
        dQ = self.declare_variable('_dQ', shape = shape)

        R  = self.declare_variable('_radius', shape =shape)
        dr = self.declare_variable('_dr', shape = shape)
        
        Omega = self.declare_variable('_angular_speed', shape = shape)

        chord = self.declare_variable('_chord', shape = shape)
        t_c   = self.declare_variable('_thickness_to_chord_ratio',shape=shape)
        # self.print_var(t_c)


        if dir == 0:
            ds = (x**2 + y**2 + z**2)**0.5
            theta = csdl.arctan((z**2 + y**2)**0.5 / x)
            bessel_input_list = []
            for i in range(max_frequency_mode):
                frequency_mode = i+1
                bessel_input = B * frequency_mode * Omega * R * csdl.sin(theta) / a
                bessel_input_list.append(bessel_input)
                input_string = 'GD_bessel_input_mode_{}'.format(i+1)
                self.register_output(input_string, bessel_input_list[i])


        elif dir == 1:
            ds = (x**2 + y**2 + z**2)**0.5
            theta = self.declare_variable('_theta', shape = shape)
            bessel_input_list = []
            for i in range(max_frequency_mode):
                frequency_mode = i+1
                bessel_input = B * frequency_mode * Omega * R * csdl.sin(theta) / a
                bessel_input_list.append(bessel_input)
                input_string = 'GD_bessel_input_mode_{}'.format(i+1)
                self.register_output(input_string, bessel_input_list[i])

        bessel_function = csdl.custom(*bessel_input_list,op = GDBesselExplicitComp(
            shape = shape,
            acoustics_dict = acoustics_dict,
        ))
        
        # bessel_output = self.create_output('bessel_output_all_modes', shape = (max_frequency_mode, shape[0],shape[1],shape[2]))
        
        

        p_ref = 2*10**-5
        if dir == 0:
            Omega_2 = self.declare_variable('rotational_speed', shape = (shape[0],)) * 2 * np.pi
            ds_2 = self.declare_variable('dS', shape = (shape[0],))
 
            SPL_tonal = self.create_output('SPL_tonal_Gutin_Deming', shape = (max_frequency_mode, shape[0]))
            SPL_T = self.create_output('SPL_thickness_Gutin_Deming', shape = (max_frequency_mode, shape[0]))
            SPL_L = self.create_output('SPL_loading_Gutin_Deming', shape = (max_frequency_mode, shape[0]))
            
            for i in range(max_frequency_mode):
                bessel_output[i,:,:,:] = csdl.reshape(bessel_function[i],new_shape =(1, shape[0],shape[1],shape[2]))
                
                fr = dT * csdl.cos(theta) - dQ * a / (Omega * R**2) * bessel_function[i]
                gr = chord * t_c * bessel_function[i]
                
                fr_sum = csdl.sum(fr * dr, axes = (1,2)) / shape[2]
                gr_sum = csdl.sum(gr * dr, axes = (1,2)) /shape[2]
                PmL = (i+1) * B * Omega_2 / (2 * np.sqrt(2) * np.pi * a * ds_2) * fr_sum
                PmT = (-rho * ((i+1) * B * Omega_2)**2 * B / (3 * 2**0.5) * np.pi * ds_2) * gr_sum
                
                SPL_tonal[i,:] = csdl.reshape(10 * csdl.log10((PmL**2 + PmT**2) / p_ref**2) ,new_shape = (1,shape[0]))
                SPL_T[i,:] = csdl.reshape(10 * csdl.log10((PmT**2) / p_ref**2) ,new_shape = (1,shape[0]))
                SPL_L[i,:] = csdl.reshape(10 * csdl.log10((PmL**2) / p_ref**2), new_shape = (1,shape[0]))
                
        elif dir == 1:
            Omega_2 = self.declare_variable('rotational_speed', shape = (shape[0],)) * 2 * np.pi
            Omega_2_exp = csdl.expand(Omega_2,(shape[0],shape[2]), 'i->ik')
            ds_2 = self.declare_variable('dS', shape = (shape[0],))
            ds_2_exp = csdl.expand(ds_2, (shape[0],shape[2]),'i->ik')
            
            SPL_tonal = self.create_output('SPL_tonal_Gutin_Deming', shape = (max_frequency_mode, shape[0], shape[2]))
            SPL_T = self.create_output('SPL_thickness_Gutin_Deming', shape = (max_frequency_mode, shape[0], shape[2]))
            SPL_L = self.create_output('SPL_loading_Gutin_Deming', shape = (max_frequency_mode, shape[0], shape[2]))
            for i in range(max_frequency_mode):
                # bessel_output[i,:,:,:] = csdl.reshape(bessel_function[i],new_shape =(1, shape[0],shape[1],shape[2]))
                
                # print(bessel_function.shape,'bessl_shape')
                if ((max_frequency_mode == 1) and (shape[0] == 1)):
                    fr = (dT * csdl.cos(theta) - dQ * a / (Omega * R**2)) * bessel_function[i,:,:]
                    gr = chord * t_c * chord * bessel_function[i,:,:]
                elif (max_frequency_mode > 1):
                    fr = (dT * csdl.cos(theta) - dQ * a / (Omega * R**2)) * bessel_function[i]
                    gr = chord * t_c * chord * bessel_function[i]
                elif ((max_frequency_mode == 1) and (shape[0]>1)):
                    fr = (dT * csdl.cos(theta) - dQ * a / (Omega * R**2)) * bessel_function # fr (num_evaluation, num_radial,num_tangential)
                    gr = chord * t_c * chord * bessel_function


                fr_sum = csdl.sum(fr * dr, axes = (1,)) # fr_sum (num_evaluations,num_tangential)
                gr_sum = csdl.sum(gr * dr, axes = (1,)) 

                PmL = ((i+1) * B * Omega_2_exp / (2 * np.sqrt(2) * np.pi * a * ds_2_exp)) * fr_sum
                PmT = (-rho * ((i+1) * B * Omega_2_exp)**2 * B / ((3 * 2**0.5) * np.pi * ds_2_exp)) * gr_sum

                tonal = 10 * csdl.log10((PmL**2 + PmT**2) / p_ref**2)
                thickness = 10 * csdl.log10((PmT**2) / p_ref**2)
                loading = 10 * csdl.log10((PmL**2) / p_ref**2)


                SPL_tonal[i,:,:] = csdl.reshape(tonal ,new_shape = (1, shape[0], shape[2]))
                SPL_T[i,:,:] = csdl.reshape(thickness,new_shape = (1,shape[0], shape[2]))
                SPL_L[i,:,:] = csdl.reshape(loading, new_shape = (1,shape[0], shape[2]))


        







