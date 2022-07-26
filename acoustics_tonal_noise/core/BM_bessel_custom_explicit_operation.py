import numpy as np
from csdl import Model
import csdl
import openmdao.api as om
import scipy
from scipy.special import jv,jvp 

from acoustics_tonal_noise.acoustics_parameters import AcousticsParameters

class BMBesselCustomExplicitOperation(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('shape', types = tuple)
        self.parameters.declare('acoustics_dict', types = AcousticsParameters)
    
    def define(self):
        shape = self.parameters['shape']
        acoustics_dict = self.parameters['acoustics_dict']
        max_frequency_mode = acoustics_dict['mode']
        
        indices = np.arange(shape[0] * shape[1] * shape[2])

        self.add_input('BM_bessel_input_mode_0', shape=shape)
        self.add_output('BM_bessel_output_mode_0', shape=shape)

        for i in range(max_frequency_mode):
            # Declaring inputs 
            input_string = 'BM_bessel_input_mode_{}'.format(i+1)
            self.add_input(input_string,shape=shape)
            # Declaring outputs
            output_string = 'BM_bessel_output_mode_{}'.format(i+1)
            self.add_output(output_string, shape=shape)
            # Declaring derivatives
            self.declare_derivatives(output_string,input_string, rows=indices, cols=indices)

        self.add_input('BM_bessel_input_mode_m_plus_one', shape=shape)
        self.add_output('BM_bessel_output_mode_m_plus_one', shape=shape)

    def compute(self, inputs, outputs):
        acoustics_dict = self.parameters['acoustics_dict']  
        max_frequency_mode = acoustics_dict['mode']
        B = acoustics_dict['num_blades']
        
        order_0 = 1 * B - 1
        bessel_input_0 = inputs['BM_bessel_input_mode_0']
        outputs['BM_bessel_output_mode_0'] = jv(order_0, bessel_input_0)
        # print(outputs['BM_bessel_output_mode_0'],'BM_bessel_output_mode_0')

        # Computing outputs of Bessel function
        for i in range(max_frequency_mode):
            order = (i+1)*B
            input_string = 'BM_bessel_input_mode_{}'.format(i+1)
            bessel_input  = inputs[input_string]
            # print(bessel_input,'bessel_input_BM')
            output_string = 'BM_bessel_output_mode_{}'.format(i+1)
            bessel_output = jv(order, bessel_input)
            # print(bessel_output)
            
            outputs[output_string] = bessel_output

        order_m_plus_one = max_frequency_mode * B + 1
        bessel_input_m_plus_1 = inputs['BM_bessel_input_mode_m_plus_one']
        outputs['BM_bessel_output_mode_m_plus_one'] = jv(order_m_plus_one, bessel_input_m_plus_1)
        # print(outputs['BM_bessel_output_mode_m_plus_one'],'BM_bessel_output_mode_m_plus_one')
        

    # TO DO: "Cross" derivatives 
    def compute_derivatives(self, inputs, derivatives):
        shape = self.parameters['shape']
        acoustics_dict = self.parameters['acoustics_dict']
        max_frequency_mode = acoustics_dict['mode']
        B = acoustics_dict['num_blades']

        # Computing derivatives of Bessel function
        for i in range(max_frequency_mode):
            order = (i+1)*B
            input_string = 'BM_bessel_input_mode_{}'.format(i+1)
            bessel_input  = inputs[input_string]
            output_string = 'BM_bessel_output_mode_{}'.format(i+1)
            
            derivatives[output_string,input_string] = jvp(order,bessel_input,n=1).flatten()



            

