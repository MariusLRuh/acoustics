import numpy as np
from csdl import Model
import csdl
import openmdao.api as om
import scipy
from scipy.special import jv,jvp 

from acoustics_parameters import AcousticsParameters

class GDBesselCustomExplicitOperation(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('shape', types = tuple)
        self.parameters.declare('acoustics_dict', types = AcousticsParameters)
    
    def define(self):
        shape = self.parameters['shape']
        acoustics_dict = self.parameters['acoustics_dict']
        max_frequency_mode = acoustics_dict['mode']
        
        for i in range(max_frequency_mode):
            # Declaring inputs 
            input_string = 'GD_bessel_input_mode_{}'.format(i+1)
            self.add_input(input_string,shape = shape)
            # Declaring outputs
            output_string = 'GD_bessel_output_mode_{}'.format(i+1)
            self.add_output(output_string, shape = shape)
            # Declaring derivatives
            self.declare_derivatives(output_string,input_string)


    def compute(self, inputs, outputs):
        acoustics_dict = self.parameters['acoustics_dict']  
        max_frequency_mode = acoustics_dict['mode']
        B = acoustics_dict['num_blades']
        
        # Computing outputs of Bessel function
        for i in range(max_frequency_mode):
            order = (i+1)*B
            input_string = 'GD_bessel_input_mode_{}'.format(i+1)
            bessel_input  = inputs[input_string]
            output_string = 'GD_bessel_output_mode_{}'.format(i+1)
            bessel_output = jv(order, bessel_input)
            print(bessel_output)
            
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
            
            derivatives[output_string,input_string] = jvp(order,bessel_input,n=1)



            

