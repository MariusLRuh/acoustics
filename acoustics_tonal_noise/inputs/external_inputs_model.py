import numpy as np
from csdl import Model
import csdl


class ExternalInputsModel(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('num_evaluations', types=int)
        self.parameters.declare('num_radial', types=int)
        self.parameters.declare('num_azimuthal', types=int)

    def define(self):
        num_evaluations = self.parameters['num_evaluations']
        num_radial = self.parameters['num_radial']
        num_azimuthal = self.parameters['num_azimuthal']
        shape = (num_evaluations, num_radial, num_azimuthal)

        # self.create_input('mode',shape = (frequency_mode,))

        self.create_input('x_position', shape = (num_evaluations,))
        self.create_input('y_position', shape = (num_evaluations,))
        self.create_input('z_position', shape = (num_evaluations,))
        self.create_input('dS', shape = (num_evaluations,))

        self.create_input('twist', shape = (num_evaluations, num_radial))
        self.create_input('chord', shape = (num_evaluations, num_radial))
        self.create_input('thickness_to_chord_ratio', shape = (num_evaluations,num_radial))

        self.create_input('dT', shape = (num_evaluations,num_radial))
        self.create_input('dQ', shape = (num_evaluations,num_radial))

        self.create_input('M_inf',shape = (num_evaluations,))

        self.create_input('rotor_radius', shape = (num_evaluations,))
        self.create_input('dr', shape = (num_evaluations,))
        self.create_input('hub_radius', shape = (num_evaluations,))
        self.create_input('rotational_speed', shape = (num_evaluations,))



       
