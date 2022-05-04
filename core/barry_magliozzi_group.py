import numpy as np
from csdl import Model
import csdl

from acoustics_parameters import AcousticsParameters
from core.BM_bessel_explicit_comp import BMBesselExplicitComp

class BarryMagliozziGroup(Model):

    def initialize(self):
        self.parameters.declare('acoustics_dict', types=AcousticsParameters)
        self.parameters.declare('shape', types=tuple)

    def define(self):
        acoustics_dict = self.parameters['acoustics_dict']
        shape = self.parameters['shape']
        max_frequency_mode = acoustics_dict['mode']

        a = 343#acoustics_dict['speed_of_sound']
        print(a,'SPEED of SOUND')
        dir = acoustics_dict['directivity']
        B = acoustics_dict['num_blades']
        rho = acoustics_dict['density']

        x = self.declare_variable('_x_position', shape = shape)
        y = self.declare_variable('_y_position', shape = shape)
        z = self.declare_variable('_z_position', shape = shape)

        dT = self.declare_variable('_dT', shape = shape)
        dQ = self.declare_variable('_dQ', shape = shape)

        M_inf = self.declare_variable('_M_inf',shape = shape)

        R  = self.declare_variable('_radius', shape =shape)
        dr = self.declare_variable('_dr', shape = shape)
        
        Omega = self.declare_variable('_angular_speed', shape = shape)
        chord = self.declare_variable('_chord', shape = shape)
        twist = self.declare_variable('_twist', shape = shape)
        t_c   = self.declare_variable('_thickness_to_chord_ratio',shape=shape)

        h = chord * t_c 
        Ax = 0.6853 * chord * h



        if dir == 0:
            ds = (x**2 + y**2 + z**2)**0.5
            Y  = (y**2 + z**2)**0.5
            S0 = (x**2 + (1-M_inf**2)*Y**2)**0.5
            bessel_input_list = []
            for i in range(max_frequency_mode):
                frequency_mode = i+1
                bessel_input = B * frequency_mode * Omega * Y * R / (a * S0)
                bessel_input_list.append(bessel_input)
                input_string = 'BM_bessel_input_mode_{}'.format(i+1)
                self.register_output(input_string, bessel_input_list[i])


        elif dir == 1:
            ds = (x**2 + y**2 + z**2)**0.5
            Y  = (y**2 + z**2)**0.5
            S0 = (x**2 + (1-M_inf**2)*Y**2)**0.5
            bessel_input_list = []
            input_0 = (B * 1 - 1) * Omega * Y * R / (a * S0)
            self.register_output('BM_bessel_input_mode_0', input_0)
            bessel_input_list.append(input_0)
            for i in range(max_frequency_mode):
                frequency_mode = i+1
                bessel_input = B * frequency_mode * Omega * Y * R / (a * S0)
                bessel_input_list.append(bessel_input)
                input_string = 'BM_bessel_input_mode_{}'.format(i+1)
                self.register_output(input_string, bessel_input_list[i+1])
        
            input_m_plus_one = (B * max_frequency_mode + 1) * Omega * Y * R / (a * S0)
            self.register_output('BM_bessel_input_mode_m_plus_one', input_m_plus_one)
            bessel_input_list.append(input_m_plus_one)
        
        
        bessel_function = csdl.custom(*bessel_input_list,op = BMBesselExplicitComp(
            shape = shape,
            acoustics_dict = acoustics_dict,
        ))
        
        # bessel_output = self.create_output('bessel_output_all_modes', shape = (max_frequency_mode, shape[0],shape[1],shape[2]))
        
        

        p_ref = 2*10**-5
        if dir == 0:
            Omega_2 = self.declare_variable('rotational_speed', shape = (shape[0],)) * 2 * np.pi
            ds_2 = self.declare_variable('dS', shape = (shape[0],))
 
            SPL_tonal = self.create_output('SPL_tonal_Barry_Magliozzi', shape = (max_frequency_mode, shape[0]))
            SPL_T = self.create_output('SPL_thickness_Barry_Magliozzi', shape = (max_frequency_mode, shape[0]))
            SPL_L = self.create_output('SPL_loading_Barry_Magliozzi', shape = (max_frequency_mode, shape[0]))
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
            theta = self.declare_variable('_theta', shape = shape)
            theta_2 = self.declare_variable('theta', shape = (shape[2],))
            theta_2_exp = csdl.expand(theta_2, (shape[0], shape[2]),'k->ik')
            ds_2 = self.declare_variable('dS', shape = (shape[0],))
            ds_2_exp = csdl.expand(ds_2, (shape[0],shape[2]),'i->ik')
            x = ds * csdl.cos(theta)
            x_2 = ds_2_exp * csdl.cos(theta_2_exp)
            z = ds * csdl.sin(theta)
            z_2 = ds_2_exp * csdl.sin(theta_2_exp)
            Omega_2 = self.declare_variable('rotational_speed', shape = (shape[0],)) * 2 * np.pi
            Omega_2_exp = csdl.expand(Omega_2,(shape[0],shape[2]), 'i->ik')
            M_inf_2 = self.declare_variable('M_inf', shape = (shape[0],))
            M_inf_2_exp = csdl.expand(M_inf_2, (shape[0], shape[2]) ,'i->ik')
            Y_2 = (z_2**2)**0.5
            S0_2 = (x_2**2 + (1-M_inf_2_exp**2)*Y_2**2)**0.5
            
            SPL_tonal = self.create_output('SPL_tonal_Barry_Magliozzi', shape = (max_frequency_mode, shape[0], shape[2]))
            SPL_T = self.create_output('SPL_thickness_Barry_Magliozzi', shape = (max_frequency_mode, shape[0], shape[2]))
            SPL_L = self.create_output('SPL_loading_Barry_Magliozzi', shape = (max_frequency_mode, shape[0], shape[2]))
            for i in range(max_frequency_mode):
                # bessel_output[i,:,:,:] = csdl.reshape(bessel_function[i],new_shape =(1, shape[0],shape[1],shape[2]))
                order = (i+1)*B

                # fr = (R / (chord * csdl.cos(twist))) * csdl.sin(order * chord *csdl.cos(twist) / 2 / R) \
                #     * ((M_inf + x / S0) * Omega * dT / (a * (1-M_inf**2))  - dQ / R**2 ) \
                #     * (bessel_function[i+1] + (1 - M_inf**2) * Y * R / (2 *S0**2) * (bessel_function[i] - bessel_function[i+2]))
                fr = (bessel_function[i+1] + (1 - M_inf**2) * Y * R / (2 *S0**2) * (bessel_function[i] - bessel_function[i+2]))
                self.register_output('fr_test', fr)
                gr = Ax * (bessel_function[i+1] + (1 - M_inf**2) * Y * R * (bessel_function[i] - bessel_function[i+2])/ (2 *S0**2))
                

                fr_sum = csdl.sum(fr * dr, axes = (1,)) 
                gr_sum = csdl.sum(gr * dr, axes = (1,)) 

                PmL = (1/(2**0.5 * np.pi*S0_2))*  fr_sum
                PmT = (-rho * ((i+1) * B * Omega_2_exp)**2 * B**3 / (2 * 2**0.5 * np.pi * (1-M_inf_2_exp**2)**2) * (S0_2 + M_inf_2_exp * x_2)**2 / S0_2**3) * gr_sum

                tonal = 10 * csdl.log10((PmL**2 + PmT**2) / p_ref**2)
                thickness = 10 * csdl.log10((PmT**2) / p_ref**2)
                loading = 10 * csdl.log10((PmL**2) / p_ref**2)


                SPL_tonal[i,:,:] = csdl.reshape(tonal ,new_shape = (1, shape[0], shape[2]))
                SPL_T[i,:,:] = csdl.reshape(thickness,new_shape = (1,shape[0], shape[2]))
                SPL_L[i,:,:] = csdl.reshape(loading, new_shape = (1,shape[0], shape[2]))


        







