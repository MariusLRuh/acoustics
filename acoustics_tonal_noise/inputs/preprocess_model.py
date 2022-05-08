import numpy as np
from csdl import Model
import csdl

class PreprocessModel(Model):

    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):
        shape = self.parameters['shape']


        hub_radius = self.declare_variable('_hub_radius', shape=shape)
        rotor_radius = self.declare_variable('_rotor_radius', shape=shape)
        normalized_radius = self.declare_variable('_normalized_radius', shape=shape)
        rotational_speed = self.declare_variable('_rotational_speed', shape = shape)
        t_c = self.declare_variable('_thickness_to_chord_ratio', shape = shape)
        chord = self.declare_variable('_chord', shape = shape)

        radius = hub_radius + (rotor_radius - hub_radius) * normalized_radius
        self.register_output('_radius', radius)

        angular_speed = 2 * np.pi * rotational_speed
        self.register_output('_angular_speed', angular_speed)

        max_blade_thickness = t_c * chord
        self.register_output('_blade_thickness', max_blade_thickness)
