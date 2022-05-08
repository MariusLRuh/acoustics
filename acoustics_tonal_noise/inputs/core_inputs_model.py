import numpy as np
from csdl import Model
import csdl


class CoreInputsModel(Model):
    def initialize(self):
        # self.parameters.declare('frequency_mode', types = int)
        self.parameters.declare('num_evaluations', types=int)
        self.parameters.declare('num_radial', types=int)
        self.parameters.declare('num_tangential', types=int)

    def define(self):
        # frequency_mode = self.parameters['frequency_mode']
        num_evaluations = self.parameters['num_evaluations']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']
        shape = (num_evaluations, num_radial, num_tangential)

        hub_radius = self.declare_variable('hub_radius', shape = (num_evaluations))
        rotor_radius = self.declare_variable('rotor_radius', shape = (num_evaluations))
        dr = self.declare_variable('dr', shape = (num_evaluations))
        rotational_speed = self.declare_variable('rotational_speed', shape = (num_evaluations))
        
        x = self.declare_variable('x_position', shape = (num_evaluations))
        y = self.declare_variable('y_position', shape = (num_evaluations))
        z = self.declare_variable('z_position', shape = (num_evaluations))

        twist = self.declare_variable('twist', shape=(num_evaluations, num_radial,))
        chord = self.declare_variable('chord', shape=(num_evaluations, num_radial,))
        t_c   = self.declare_variable('thickness_to_chord_ratio', shape = (num_evaluations,num_radial))

        dT = self.declare_variable('dT', shape = (num_evaluations,num_radial))
        dQ = self.declare_variable('dQ', shape = (num_evaluations,num_radial))

        M_inf = self.declare_variable('M_inf', shape = (num_evaluations,))


        # expanding variables to shape
        self.register_output('_hub_radius', csdl.expand(hub_radius, shape,'i->ijk'))
        self.register_output('_rotor_radius', csdl.expand(rotor_radius, shape,'i->ijk'))
        self.register_output('_dr', csdl.expand(dr, shape,'i->ijk'))
        self.register_output('_rotational_speed', csdl.expand(rotational_speed, shape,'i->ijk'))
        
        self.register_output('_x_position', csdl.expand(x, shape,'i->ijk'))
        self.register_output('_y_position', csdl.expand(y, shape,'i->ijk'))
        self.register_output('_z_position', csdl.expand(z, shape,'i->ijk'))       

        self.register_output('_twist', csdl.expand(twist, shape, 'ij->ijk'))
        self.register_output('_chord', csdl.expand(chord, shape, 'ij->ijk'))
        self.register_output('_thickness_to_chord_ratio', csdl.expand(t_c, shape, 'ij->ijk'))
        
        self.register_output('_dT', csdl.expand(dT, shape, 'ij->ijk'))
        self.register_output('_dQ', csdl.expand(dQ, shape, 'ij->ijk'))
        
        self.register_output('_M_inf', csdl.expand(M_inf, shape, 'i->ijk'))

        # v = np.linspace(0, np.pi * 2 - np.pi * 2 / num_tangential, num_tangential)
        v = np.linspace(10 * np.pi/180, 170 * np.pi /180, num_tangential)
        self.create_input('theta', val = v )
        theta = np.einsum(
            'ij,k->ijk',
            np.ones((num_evaluations, num_radial)),
            v,
        )
        self.create_input('_theta', val=theta)

        normalized_radial_discretization = 1. / num_radial / 2. \
            + np.linspace(0., 1. - 1. / num_radial, num_radial)

        normalized_radius = np.einsum(
            'ik,j->ijk',
            np.ones((num_evaluations, num_tangential)),
            normalized_radial_discretization,
        )
        self.create_input('_normalized_radius', val=normalized_radius)