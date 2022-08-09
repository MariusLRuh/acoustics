import numpy as np
from csdl import Model
import csdl

from acoustics_tonal_noise.acoustics_parameters import AcousticsParameters
from acoustics_tonal_noise.core.BM_bessel_custom_explicit_operation import BMBesselCustomExplicitOperation

class BarryMagliozziModel(Model):

    def initialize(self):
        self.parameters.declare('acoustics_dict', types=AcousticsParameters)
        self.parameters.declare('shape', types=tuple)

    def define(self):
        acoustics_dict = self.parameters['acoustics_dict']
        shape = self.parameters['shape']
        max_frequency_mode = acoustics_dict['mode']

        # print('MAX FREQUENCY MODE', max_frequency_mode)

        num_nodes = shape[0]
        num_radial = shape[1]
        num_azimuthal = shape[2]

        dir = acoustics_dict['directivity']
        B = acoustics_dict['num_blades']
        
        rho = self.declare_variable('density', shape=(num_nodes,))
        rho_exp = csdl.expand(rho,(num_nodes,num_azimuthal), 'i->ik')
        a = self.declare_variable('speed_of_sound', shape=(num_nodes,))
        a_exp = csdl.expand(a, shape,'i->ijk')
        
        x = self.declare_variable('_x_position', shape=shape)
        # self.print_var(x)
        y = self.declare_variable('_y_position', shape=shape)
        # self.print_var(y)
        z = self.declare_variable('_z_position', shape=shape)
        # self.print_var(z)

        dT = self.declare_variable('_dT', shape=shape)
        # self.print_var(dT)
        dQ = self.declare_variable('_dQ', shape=shape)
        # self.print_var(dQ)
        M_inf = self.declare_variable('_M_inf',shape=shape)
        # self.print_var(M_inf)
        R  = self.declare_variable('_radius', shape =shape)
        dr = self.declare_variable('_dr', shape=shape)
        
        Omega = self.declare_variable('_angular_speed', shape=shape) + 1e-7
        # self.print_var(Omega)
        chord = self.declare_variable('_chord', shape=shape)
        twist = self.declare_variable('_twist', shape=shape)
        # self.print_var(twist)
        t_c   = self.declare_variable('_thickness_to_chord_ratio',shape=shape)
        # self.print_var(t_c)

        h = chord * t_c 
        Ax = 0.6853 * chord * h



        if dir == 0:
            ds = (x**2 + y**2 + z**2)**0.5 # y and z are in rotor-plane and x is rotor-axis 
            Y  = (y**2 + z**2)**0.5
            S0 = (x**2 + (1-M_inf**2)*Y**2)**0.5
            theta = csdl.arctan((z**2 + y**2)**0.5 / x)
            bessel_input_list = []
            input_0 = (B * 1 - 1) * Omega * Y * R / (a_exp * S0)
            self.register_output('BM_bessel_input_mode_0', input_0)
            bessel_input_list.append(input_0)
            for i in range(max_frequency_mode):
                frequency_mode = i+1
                bessel_input = B * frequency_mode * Omega * Y * R / (a_exp * S0)
                bessel_input_list.append(bessel_input)
                input_string = 'BM_bessel_input_mode_{}'.format(i+1)
                self.register_output(input_string, bessel_input_list[i+1])
            input_m_plus_one = (B * max_frequency_mode + 1) * Omega * Y * R / (a_exp * S0)
            self.register_output('BM_bessel_input_mode_{}'.format(max_frequency_mode+1), input_m_plus_one)
            bessel_input_list.append(input_m_plus_one)


        elif dir == 1:
            ds = (x**2 + y**2 + z**2)**0.5
            theta = self.declare_variable('_theta', shape=shape)
            x = ds * csdl.cos(theta)
            z = ds * csdl.sin(theta)
            Y  = (y**2 + z**2)**0.5
            S0 = (x**2 + (1-M_inf**2)*Y**2)**0.5
            bessel_input_list = []
            input_0 = (B * 1 - 1) * Omega * Y * R / (a_exp * S0)
            self.register_output('BM_bessel_input_mode_0', input_0)
            bessel_input_list.append(input_0)
            for i in range(max_frequency_mode):
                frequency_mode = i+1
                bessel_input = B * frequency_mode * Omega * Y * R / (a_exp * S0)
                bessel_input_list.append(bessel_input)
                input_string = 'BM_bessel_input_mode_{}'.format(i+1)
                self.register_output(input_string, bessel_input_list[i+1])
        
            input_m_plus_one = (B * max_frequency_mode + 1) * Omega * Y * R / (a_exp * S0)
            self.register_output('BM_bessel_input_mode_{}'.format(max_frequency_mode+1), input_m_plus_one)
            bessel_input_list.append(input_m_plus_one)
        
        
        bessel_function = csdl.custom(*bessel_input_list,op = BMBesselCustomExplicitOperation(
            shape=shape,
            acoustics_dict = acoustics_dict,
        ))
        # self.print_var(bessel_function[2])
        # bessel_output = self.create_output('bessel_output_all_modes', shape = (max_frequency_mode, num_nodes,shape[1],num_azimuthal))
        

        

        p_ref = 2*10**-5
        if dir == 0:
            x2 = self.declare_variable('x_position', shape=(num_nodes,1))
            # self.print_var(x2)
            y2 = self.declare_variable('y_position', shape=(num_nodes,1))
            # self.print_var(y2)
            z2 = self.declare_variable('z_position', shape=(num_nodes,1))
            # self.print_var(z2)
            M_inf2 = csdl.expand(self.declare_variable('M_inf', shape=(1,)), (num_nodes,1))
            # self.print_var(M_inf2)
            Omega_2 = self.declare_variable('rotational_speed', shape=(num_nodes,1)) * 2 * np.pi + 1e-7
            # self.print_var(Omega_2)
            rho_2 = self.declare_variable('density', shape=(num_nodes,1))
            # self.print_var(rho_2)


            ds_2 = (x2**2 + y2**2 + z2**2)**0.5 # y and z are in rotor-plane and x is rotor-axis 
            Y_2  = (y2**2 + z2**2)**0.5
            S0_2 = (x2**2 + (1-M_inf2**2)*Y_2**2)**0.5
            theta_2 = csdl.arctan((z2**2 + y2**2)**0.5 / x2)

            # Blade passing frequency BPF
            BPF = B * csdl.reshape(Omega_2,(1,num_nodes))

            SPL_tonal = self.create_output('SPL_tonal_Barry_Magliozzi', shape=(max_frequency_mode, num_nodes))
            SPL_tonal_2 = self.create_output('SPL_tonal_Barry_Magliozzi_2', shape=(max_frequency_mode, num_nodes))
            SPL_T = self.create_output('SPL_thickness_Barry_Magliozzi', shape=(max_frequency_mode, num_nodes))
            SPL_L = self.create_output('SPL_loading_Barry_Magliozzi', shape=(max_frequency_mode, num_nodes))
            counter = 0
            for i in range(max_frequency_mode):
                # print(counter)
                # bessel_output[i,:,:,:] = csdl.reshape(bessel_function[i],new_shape =(1, num_nodes,shape[1],num_azimuthal))
                order = (i+1)*B
                
                f = (i+1) * BPF 
                Af = A_weighting(self,f)
                
                # self.print_var(csdl.exp_a(10,Af/10))


                fr = (R / (chord * csdl.cos(twist))) * csdl.sin(order * chord *csdl.cos(twist) / 2 / R) \
                    * ((M_inf + x / S0) * Omega * dT / (a_exp * (1-M_inf**2))  - dQ / R**2 ) \
                    * (bessel_function[i+1] + (1 - M_inf**2) * Y * R / (2 *S0**2) * (bessel_function[i] - bessel_function[i+2]))
                gr = Ax * (bessel_function[i+1] + (1 - M_inf**2) * Y * R * (bessel_function[i] - bessel_function[i+2])/ (2 *S0**2))

                fr_sum = csdl.sum(fr * dr, axes = (1,)) 
                gr_sum = csdl.sum(gr * dr, axes = (1,)) 
                # print('FR SUM',fr_sum.shape)
                PmL = (1/(2**0.5 * np.pi*S0_2))*  fr_sum
                PmT = rho_2 * ((i+1) * Omega_2)**2 * B**3 * (S0_2 + M_inf2 * x2)**2 / (2 * 2**0.5 * np.pi * (1-M_inf2**2)**2 * S0_2**3) * gr_sum
                tonal = 10 * csdl.log10((PmL**2 + PmT**2 +1e-10) / p_ref**2) 
                thickness = 10 * csdl.log10((PmT**2 +1e-10) / p_ref**2)
                loading = 10 * csdl.log10((PmL**2 + 1e-10 ) / p_ref**2)

                

                SPL_tonal[i,:] = csdl.reshape(tonal ,new_shape = (num_azimuthal, num_nodes))
                SPL_T[i,:] = csdl.reshape(thickness,new_shape = (num_azimuthal,num_nodes))
                SPL_L[i,:] = csdl.reshape(loading, new_shape = (num_azimuthal,num_nodes))
                # self.print_var(csdl.exp_a(10,SPL_tonal[i,:]/10))
                SPL_tonal_2[i,:] = csdl.reshape(csdl.exp_a(10,SPL_tonal[i,:]/10) + csdl.exp_a(10,Af/10),new_shape = (num_azimuthal, num_nodes))
                counter += 1

            # print('SPL_tonal_2',SPL_tonal_2.shape)
            total_tonal_noise = csdl.reshape(10 * csdl.log10(csdl.sum(SPL_tonal_2, axes=(0,))),(num_nodes,1))
            self.register_output('total_tonal_noise', total_tonal_noise)
            # self.add_objective('total_tonal_noise')
            # self.print_var(total_tonal_noise)
            
            # Broadband noise
            x_skm = self.declare_variable('x_position', shape=(num_nodes,1))
            y_skm = self.declare_variable('y_position', shape=(num_nodes,1))
            z_skm = self.declare_variable('z_position', shape=(num_nodes,1))
            S0_skm = (x_skm**2 + y_skm**2 + z_skm**2)**0.5
            theta0_skm = csdl.arcsin((z_skm**2)**0.5 / S0_skm)
            R_skm = csdl.expand(self.declare_variable('propeller_radius', shape=(1,)),(num_nodes,1))

            # For A-weighting
            freq_r = self.declare_variable('freq_r',val=np.array([1/2,1,2,4,8,16]))
            S_oct = self.declare_variable('S_oct',np.array([-8.163371957,-4.098174545,-8.114897374,-9.028706323,-13.04470834,-14.12250194])) # Octave band SPL correction [dB]
            St = 0.28 # Strouhal number for the usual range of Re numbers for a helicopter
            t07 = self.declare_variable('t07',shape=(num_nodes,1),val=0.02) # Blade thickness @ 70% of rotor radius [m]
            AoA07 = self.declare_variable('AoA_70',shape=(num_nodes,1),val=3*np.pi/180) # Angle of attack @ 70% of rotor radius [deg]
            c07 = self.declare_variable('c_70',shape=(num_nodes,1),val=0.12) # Chord length @ 70% of rotor radius [m]
            hprime = t07*csdl.cos(AoA07) + c07*csdl.sin(AoA07); #Projected blade thickness [m]
            V07 = 0.7 * R_skm * Omega_2
            fp = csdl.expand(St*V07/hprime, (6,num_nodes,1),'ij->kij')  # Peak frequency; expand from(num_nodes,1) ->(6,num_nodes,1)
            # print(freq_r.shape)
            freq = fp * csdl.expand(freq_r,(6,num_nodes,1),'i->ijk') # Octave band frequency distribution [Hz]

            

            Omega_skm = Omega_2 + 1e-6
            chord_skm = self.declare_variable('chord_profile', shape=(num_radial,1))
            dr_skm = csdl.expand(self.declare_variable('dr', shape=(1,)),(num_radial,1),'i->ji')
            Ab_skm = csdl.expand(B * csdl.sum(chord_skm * dr_skm, axes=(0,)),(num_nodes,1),'i->ji')
            sigma_skm = Ab_skm / np.pi / R_skm**2
            CT_skm = self.declare_variable('CT', shape=(num_nodes,1))


            SPL150_SKM = 10*csdl.log10(1e-10 + (Omega_skm*R_skm)**6*Ab_skm*(CT_skm/sigma_skm)**2) - 42.9
            SPL_SKM = SPL150_SKM + 20*csdl.log10(1e-10 + csdl.sin(theta0_skm)/(S0_skm/150))
            
            if SPL_SKM.shape == (1,1): 
                SPL_SKM_OCT = csdl.expand(csdl.reshape(SPL_SKM,(1,)),(num_nodes,6), 'i->jk') + csdl.expand(S_oct, (num_nodes,6), 'j->ij')
            else:
                SPL_SKM_OCT = csdl.expand(csdl.reshape(SPL_SKM, (num_nodes,)),(num_nodes,6), 'i->ik') + csdl.expand(S_oct, (num_nodes,6), 'j->ij')
            
            SPL_SKM_OCT_dB_temp = self.create_output('SPL_SKM_OCT_dB', shape=(num_nodes,6), val=0)
            

            for i in range(num_nodes):
                for j in range(6):
                    SPL_SKM_OCT_dB_temp[i,j] = csdl.exp_a(10,SPL_SKM_OCT[i,j]/10)



            SPL_SKM_OCT_dB = csdl.reshape(10 * csdl.log10(csdl.sum(SPL_SKM_OCT_dB_temp, axes=(1,))),(num_nodes,1))


            A_f_skm = self.create_output('A_f_skm', shape=(num_nodes,6))
            for i in range(num_nodes):
                for j in range(6):
                    A_f_skm[i,j] = csdl.exp_a(10,csdl.reshape(A_weighting(self,freq[j,i,0]), (1,1))/10)
            A_f_skm_dB = csdl.reshape(10 * csdl.log10(csdl.sum(A_f_skm, axes=(1,))),(num_nodes,1))
            SPL_SKM_A = 10 * csdl.log10(1e-10 + csdl.exp_a(10,SPL_SKM_OCT_dB/10) + csdl.exp_a(10,A_f_skm_dB/10))
            # self.register_output('SPL_SKM',SPL_SKM)

            total_noise = 10 * csdl.log10(1e-10 + csdl.exp_a(10,SPL_SKM_A/10) + csdl.exp_a(10,total_tonal_noise/10)) #  10**(SPL_SKM/10) + 10**(total_tonal_noise/10))
            # self.print_var(total_noise)
            self.register_output('tonal_plus_broadband_noise', total_noise)
            # self.add_objective('tonal_plus_broadband_noise')

            # csdl.exp_a(constant_a, csdl_variable_x)
            
              
        elif dir == 1:
            theta = self.declare_variable('_theta', shape=shape)
            # self.print_var(theta)
            theta_2 = self.declare_variable('theta', shape = (num_azimuthal,))
            # self.print_var(theta_2)
            theta_2_exp = csdl.expand(theta_2, (num_nodes, num_azimuthal),'k->ik')
            ds_2 = self.declare_variable('dS', shape = (num_nodes,))
            
            ds_2_exp = csdl.expand(ds_2, (num_nodes,num_azimuthal),'i->ik')
            # self.print_var(ds_2_exp)
            x = ds * csdl.cos(theta)
            x_2 = ds_2_exp * csdl.cos(theta_2_exp)
            z = ds * csdl.sin(theta)
            z_2 = ds_2_exp * csdl.sin(theta_2_exp)
            Omega_2 = self.declare_variable('rotational_speed', shape = (num_nodes,)) * 2 * np.pi
            # self.print_var(Omega_2)
            Omega_2_exp = csdl.expand(Omega_2,(num_nodes,num_azimuthal), 'i->ik')
            # self.print_var(Omega_2_exp)
            M_inf_2 = self.declare_variable('M_inf', shape = (1,))
            M_inf_2_exp = csdl.expand(M_inf_2, (num_nodes, num_azimuthal) ,'i->jk')
            Y_2 = (z_2**2 )**0.5
            S0_2 = (x_2**2 + (1-M_inf_2_exp**2)*Y_2**2)**0.5
            
            SPL_tonal = self.create_output('SPL_tonal_Barry_Magliozzi', shape = (max_frequency_mode, num_nodes, num_azimuthal))
            SPL_T = self.create_output('SPL_thickness_Barry_Magliozzi', shape = (max_frequency_mode, num_nodes, num_azimuthal))
            SPL_L = self.create_output('SPL_loading_Barry_Magliozzi', shape = (max_frequency_mode, num_nodes, num_azimuthal))
            for i in range(max_frequency_mode):
                # bessel_output[i,:,:,:] = csdl.reshape(bessel_function[i],new_shape =(1, num_nodes,shape[1],num_azimuthal))
                order = (i+1)*B
                # self.print_var(chord)
                fr = (R / (chord * csdl.cos(twist))) * csdl.sin(order * chord *csdl.cos(twist) / 2 / R) \
                    * ((M_inf + x / S0) * Omega * dT / (a_exp * (1-M_inf**2))  - dQ / R**2 ) \
                    * (bessel_function[i+1] + (1 - M_inf**2) * Y * R / (2 *S0**2) * (bessel_function[i] - bessel_function[i+2]))
                # self.print_var(fr)
                # fr = (bessel_function[i+1] + (1 - M_inf**2) * Y * R / (2
                # *S0**2) * (bessel_function[i] - bessel_function[i+2]))
                # print(bessel_function,'bessel shape')
                # self.print_var(bessel_function[i+1])
                # self.print_var(bessel_function[i+2])
                # fr = (bessel_function[i] - bessel_function[i+2])
                # self.register_output('fr_test', fr)
                gr = Ax * (bessel_function[i+1] + (1 - M_inf**2) * Y * R * (bessel_function[i] - bessel_function[i+2])/ (2 *S0**2))
                # self.register_output('gr_test', gr)

                fr_sum = csdl.sum(fr * dr, axes = (1,)) 
                gr_sum = csdl.sum(gr * dr, axes = (1,)) 

                PmL = (1/(2**0.5 * np.pi*S0_2))*  fr_sum
                # self.register_output('PmL',PmL)
                PmT = (rho_exp * ((i+1) * Omega_2_exp)**2 * B**3 / (2 * 2**0.5 * np.pi * (1-M_inf_2_exp**2)**2) * (S0_2 + M_inf_2_exp * x_2)**2 / S0_2**3) * gr_sum
                # self.register_output('PmT',PmT)
                # PmT = -(rho * (i+1)**2 * Omega_2_exp**2 + B**3 * (S0_2 + M_inf_2_exp * x_2)**2 / (2 * 2**0.5 * np.pi * (1 - M_inf_2_exp**2)**2 * S0_2**3)) * gr_sum
                tonal = 10 * csdl.log10((PmL**2 + PmT**2) / p_ref**2)
                thickness = 10 * csdl.log10((PmT**2) / p_ref**2)
                loading = 10 * csdl.log10((PmL**2) / p_ref**2)


                SPL_tonal[i,:,:] = csdl.reshape(tonal ,new_shape = (1, num_nodes, num_azimuthal))
                SPL_T[i,:,:] = csdl.reshape(thickness,new_shape = (1,num_nodes, num_azimuthal))
                SPL_L[i,:,:] = csdl.reshape(loading, new_shape = (1,num_nodes, num_azimuthal))

def A_weighting(self,f):
    shape = f.shape
    RA_f = (12194**2 * f**4 / ((f**2 + 20.6**2) * ((f**2 + 107.7**2) * (f**2 + 737.9**2))**0.5 * (f**2 + 12194**2)))
    RA_1000_temp = 12194**2 * 1000**4 / ((1000**2 + 20.6**2) * ((1000**2 + 107.7**2) * (1000**2 + 737.9**2))**0.5 * (1000**2 + 12194**2))
    RA_1000 = csdl.expand(self.declare_variable('RA_1000', RA_1000_temp), shape)
    A_f = 20 * csdl.log10(RA_f + 1e-10 ) - 20 * csdl.log10(RA_1000)

    return A_f




