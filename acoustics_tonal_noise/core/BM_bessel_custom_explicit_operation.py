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

        
        for i in range(max_frequency_mode+2): 
            for j in range(max_frequency_mode+2):
                if i == 0 and j == 0:
                    # print('Mode 0')
                    output_string = 'BM_bessel_output_mode_{}'.format(i)
                    input_string = 'BM_bessel_input_mode_{}'.format(j)
                    self.add_input(input_string, shape=shape)
                    self.add_output(output_string, shape=shape)
                    self.declare_derivatives(output_string,input_string, rows=indices, cols=indices)
                    # print(output_string + ' wrt ' + input_string)
                    
                elif i == (max_frequency_mode+1) and j == (max_frequency_mode+1):
                    # print('Mode {}'.format(i))
                    output_string = 'BM_bessel_output_mode_{}'.format(i)
                    input_string = 'BM_bessel_input_mode_{}'.format(j)
                    self.add_input(input_string, shape=shape)
                    self.add_output(output_string, shape=shape)
                    self.declare_derivatives(output_string,input_string, rows=indices, cols=indices)
                    # print(output_string + ' wrt ' + input_string)
                    
                elif i == j:
                    # print('Mode {}'.format(i))
                    output_string = 'BM_bessel_output_mode_{}'.format(i)
                    input_string = 'BM_bessel_input_mode_{}'.format(j)
                    self.add_input(input_string, shape=shape)
                    self.add_output(output_string, shape=shape)
                    self.declare_derivatives(output_string,input_string, rows=indices, cols=indices)
                    # print(output_string + ' wrt ' + input_string)
                else:
                    output_string = 'BM_bessel_output_mode_{}'.format(i)
                    input_string = 'BM_bessel_input_mode_{}'.format(j)
                    self.declare_derivatives(output_string,input_string, rows=indices, cols=indices)
                    # print(output_string + ' wrt ' + input_string)
                    




    def compute(self, inputs, outputs):
        acoustics_dict = self.parameters['acoustics_dict']  
        max_frequency_mode = acoustics_dict['mode']
        B = acoustics_dict['num_blades']
        

        for i in range(max_frequency_mode+2): 
            for j in range(max_frequency_mode+2):
                if i == 0 and j == 0:
                    # print('Mode 0')
                    output_string = 'BM_bessel_output_mode_{}'.format(i)
                    input_string = 'BM_bessel_input_mode_{}'.format(j)
                    bessel_input  = inputs[input_string]
                    order = B - 1
                    outputs[output_string] = jv(order,bessel_input)
                elif i == (max_frequency_mode+1) and j == (max_frequency_mode+1):
                    # print('Mode {}'.format(i))
                    output_string = 'BM_bessel_output_mode_{}'.format(i)
                    input_string = 'BM_bessel_input_mode_{}'.format(j)
                    bessel_input  = inputs[input_string]
                    order = B * max_frequency_mode + 1
                    outputs[output_string] = jv(order,bessel_input)
                elif i == j:
                    # print('Mode {}'.format(i))
                    output_string = 'BM_bessel_output_mode_{}'.format(i)
                    input_string = 'BM_bessel_input_mode_{}'.format(j)
                    bessel_input  = inputs[input_string]
                    order = B * i
                    outputs[output_string] = jv(order,bessel_input)
        
        # order_0 = 1 * B - 1
        # bessel_input_0 = inputs['BM_bessel_input_mode_0']
        # outputs['BM_bessel_output_mode_0'] = jv(order_0, bessel_input_0)
        # # print(outputs['BM_bessel_output_mode_0'],'BM_bessel_output_mode_0')

        # # Computing outputs of Bessel function
        # for i in range(max_frequency_mode):
        #     order = (i+1)*B
        #     input_string = 'BM_bessel_input_mode_{}'.format(i+1)
        #     bessel_input  = inputs[input_string]
        #     # print(bessel_input,'bessel_input_BM')
        #     output_string = 'BM_bessel_output_mode_{}'.format(i+1)
        #     bessel_output = jv(order, bessel_input)
        #     # print(bessel_output)
            
        #     outputs[output_string] = bessel_output

        # order_4 = max_frequency_mode * B + 1
        # bessel_input_m_plus_1 = inputs['BM_bessel_input_mode_4']
        # outputs['BM_bessel_output_mode_4'] = jv(order_4, bessel_input_m_plus_1)
        # print(outputs['BM_bessel_output_mode_4'],'BM_bessel_output_mode_4')
        

    # TO DO: "Cross" derivatives 
    def compute_derivatives(self, inputs, derivatives):
        shape = self.parameters['shape']
        acoustics_dict = self.parameters['acoustics_dict']
        max_frequency_mode = acoustics_dict['mode']
        B = acoustics_dict['num_blades']

        # # Computing derivatives of Bessel function
        # for i in range(max_frequency_mode):
        #     order = (i+1)*B
        #     input_string = 'BM_bessel_input_mode_{}'.format(i+1)
        #     bessel_input  = inputs[input_string]
        #     output_string = 'BM_bessel_output_mode_{}'.format(i+1)
            
        #     derivatives[output_string,input_string] = jvp(order,bessel_input,n=1).flatten()

        # Computing derivatives of Bessel function
        for i in range(max_frequency_mode+2): 
            for j in range(max_frequency_mode+2):
                if i == 0 and j == 0:
                    # print('Mode 0')
                    output_string = 'BM_bessel_output_mode_{}'.format(i)
                    input_string = 'BM_bessel_input_mode_{}'.format(j)
                    bessel_input  = inputs[input_string]
                    order = B - 1
                    derivatives[output_string,input_string] = jvp(order,bessel_input,n=1).flatten()
                elif i == (max_frequency_mode+1) and j == (max_frequency_mode+1):
                    # print('Mode {}'.format(i))
                    output_string = 'BM_bessel_output_mode_{}'.format(i)
                    input_string = 'BM_bessel_input_mode_{}'.format(j)
                    bessel_input  = inputs[input_string]
                    order = B * max_frequency_mode + 1
                    derivatives[output_string,input_string] = jvp(order,bessel_input,n=1).flatten()
                elif i == j:
                    # print('Mode {}'.format(i))
                    output_string = 'BM_bessel_output_mode_{}'.format(i)
                    input_string = 'BM_bessel_input_mode_{}'.format(j)
                    bessel_input  = inputs[input_string]
                    order = B * i
                    derivatives[output_string,input_string] = jvp(order,bessel_input,n=1).flatten()
                else:
                    output_string = 'BM_bessel_output_mode_{}'.format(i)
                    input_string = 'BM_bessel_input_mode_{}'.format(j)
                    derivatives[output_string,input_string] = 0




                

              


        # derivatives['BM_bessel_output_mode_1','BM_bessel_input_mode_2'] = 0
        # derivatives['BM_bessel_output_mode_1','BM_bessel_input_mode_3'] = 0
        # derivatives['BM_bessel_output_mode_2','BM_bessel_input_mode_1'] = 0
        # derivatives['BM_bessel_output_mode_2','BM_bessel_input_mode_3'] = 0
        # derivatives['BM_bessel_output_mode_3','BM_bessel_input_mode_1'] = 0
        # derivatives['BM_bessel_output_mode_3','BM_bessel_input_mode_2'] = 0

        # derivatives['BM_bessel_output_mode_m_plus_one','BM_bessel_input_mode_0'] = 0
        # derivatives['BM_bessel_output_mode_m_plus_one','BM_bessel_input_mode_1'] = 0
        # derivatives['BM_bessel_output_mode_m_plus_one','BM_bessel_input_mode_2'] = 0
        # derivatives['BM_bessel_output_mode_m_plus_one','BM_bessel_input_mode_3'] = 0
        # derivatives['BM_bessel_output_mode_m_plus_one','BM_bessel_input_mode_m_plus_one'] = jvp((B*max_frequency_mode+1),inputs['BM_bessel_input_mode_m_plus_one'],n=1).flatten()
       
        # derivatives['BM_bessel_output_mode_0','BM_bessel_input_mode_0'] = jvp(B*1-1,inputs['BM_bessel_input_mode_0'], n=1).flatten()
        # derivatives['BM_bessel_output_mode_0','BM_bessel_input_mode_1'] = 0
        # derivatives['BM_bessel_output_mode_0','BM_bessel_input_mode_2'] = 0
        # derivatives['BM_bessel_output_mode_0','BM_bessel_input_mode_3'] = 0
        # derivatives['BM_bessel_output_mode_0','BM_bessel_input_mode_m_plus_one'] = 0

        # self.declare_derivatives('BM_bessel_output_mode_m_plus_one','BM_bessel_input_mode_0', rows=indices, cols=indices)
        # self.declare_derivatives('BM_bessel_output_mode_m_plus_one','BM_bessel_input_mode_1', rows=indices, cols=indices)
        # self.declare_derivatives('BM_bessel_output_mode_m_plus_one','BM_bessel_input_mode_2', rows=indices, cols=indices)
        # self.declare_derivatives('BM_bessel_output_mode_m_plus_one','BM_bessel_input_mode_3', rows=indices, cols=indices)
        # self.declare_derivatives('BM_bessel_output_mode_m_plus_one','BM_bessel_input_mode_m_plus_one', rows=indices, cols=indices)
        
        # self.declare_derivatives('BM_bessel_output_mode_0','BM_bessel_input_mode_0', rows=indices, cols=indices)
        # self.declare_derivatives('BM_bessel_output_mode_0','BM_bessel_input_mode_1', rows=indices, cols=indices)
        # self.declare_derivatives('BM_bessel_output_mode_0','BM_bessel_input_mode_2', rows=indices, cols=indices)
        # self.declare_derivatives('BM_bessel_output_mode_0','BM_bessel_input_mode_3', rows=indices, cols=indices)
        # self.declare_derivatives('BM_bessel_output_mode_0','BM_bessel_input_mode_m_plus_one', rows=indices, cols=indices)

            

