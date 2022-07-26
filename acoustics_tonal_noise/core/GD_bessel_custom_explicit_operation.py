import numpy as np
from csdl import Model
import csdl
import openmdao.api as om
import scipy
from scipy.special import jv,jvp 

from acoustics_tonal_noise.acoustics_parameters import AcousticsParameters

class GDBesselCustomExplicitOperation(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('shape', types = tuple)
        self.parameters.declare('acoustics_dict', types = AcousticsParameters)
    
    def define(self):
        shape = self.parameters['shape']
        acoustics_dict = self.parameters['acoustics_dict']
        max_frequency_mode = acoustics_dict['mode']
        
        indices = np.arange(shape[0] * shape[1] * shape[2])

        for i in range(max_frequency_mode):
            i_index = i + 1
            # Declaring inputs 
            input_string = 'GD_bessel_input_mode_{}'.format(i_index)
            self.add_input(input_string,shape=shape)
            # Declaring outputs
            output_string = 'GD_bessel_output_mode_{}'.format(i_index)
            self.add_output(output_string, shape=shape)
            
            # Declaring derivatives
            self.declare_derivatives(output_string,input_string, rows=indices, cols=indices)
        
        # self.declare_derivatives('GD_bessel_output_mode_1','GD_bessel_input_mode_2', rows=indices, cols=indices)
        # self.declare_derivatives('GD_bessel_output_mode_1','GD_bessel_input_mode_3', rows=indices, cols=indices)
        # self.declare_derivatives('GD_bessel_output_mode_2','GD_bessel_input_mode_1', rows=indices, cols=indices)
        # self.declare_derivatives('GD_bessel_output_mode_2','GD_bessel_input_mode_3', rows=indices, cols=indices)
        # self.declare_derivatives('GD_bessel_output_mode_3','GD_bessel_input_mode_1', rows=indices, cols=indices)
        # self.declare_derivatives('GD_bessel_output_mode_3','GD_bessel_input_mode_2', rows=indices, cols=indices)


        # for j in range(max_frequency_mode-1):
        #     j_index = j + 1



    def compute(self, inputs, outputs):
        acoustics_dict = self.parameters['acoustics_dict']  
        max_frequency_mode = acoustics_dict['mode']
        B = acoustics_dict['num_blades']
        
        # Computing outputs of Bessel function
        for i in range(max_frequency_mode):
            order = (i+1)*B
            input_string = 'GD_bessel_input_mode_{}'.format(i+1)
            bessel_input  = inputs[input_string]
            # print(bessel_input,'bessel_input_GD')
            output_string = 'GD_bessel_output_mode_{}'.format(i+1)
            bessel_output = jv(order, bessel_input)
            # print(bessel_output)
            
            outputs[output_string] = bessel_output

    def compute_derivatives(self, inputs, derivatives):
        shape = self.parameters['shape']
        acoustics_dict = self.parameters['acoustics_dict']
        max_frequency_mode = acoustics_dict['mode']
        B = acoustics_dict['num_blades']

        # Computing derivatives of Bessel function
        for i in range(max_frequency_mode):
            order = (i+1)*B
            input_string = 'GD_bessel_input_mode_{}'.format(i+1)
            bessel_input  = inputs[input_string]
            output_string = 'GD_bessel_output_mode_{}'.format(i+1)
            derivatives[output_string,input_string] = jvp(order,bessel_input,n=1).flatten()

        # derivatives['GD_bessel_output_mode_1','GD_bessel_input_mode_2'] = 0
        # derivatives['GD_bessel_output_mode_1','GD_bessel_input_mode_3'] = 0
        # derivatives['GD_bessel_output_mode_2','GD_bessel_input_mode_1'] = 0
        # derivatives['GD_bessel_output_mode_2','GD_bessel_input_mode_3'] = 0
        # derivatives['GD_bessel_output_mode_3','GD_bessel_input_mode_1'] = 0
        # derivatives['GD_bessel_output_mode_3','GD_bessel_input_mode_2'] = 0

        # self.declare_derivatives('GD_bessel_output_mode_1','GD_bessel_input_mode_2', rows=indices, cols=indices)
        # self.declare_derivatives('GD_bessel_output_mode_1','GD_bessel_input_mode_3', rows=indices, cols=indices)
        # self.declare_derivatives('GD_bessel_output_mode_2','GD_bessel_input_mode_1', rows=indices, cols=indices)
        # self.declare_derivatives('GD_bessel_output_mode_2','GD_bessel_input_mode_3', rows=indices, cols=indices)
        # self.declare_derivatives('GD_bessel_output_mode_3','GD_bessel_input_mode_1', rows=indices, cols=indices)
        # self.declare_derivatives('GD_bessel_output_mode_3','GD_bessel_input_mode_2', rows=indices, cols=indices)



            

