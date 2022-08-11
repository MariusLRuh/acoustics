import numpy as np
from csdl import Model
import csdl


class CoreInputsModel(Model):
    def initialize(self):
        # self.parameters.declare('frequency_mode', types = int)
        self.parameters.declare('shape', types=tuple)

    def define(self):
        # frequency_mode = self.parameters['frequency_mode']

        shape = self.parameters['shape']

        num_nodes = shape[0]
        num_radial = shape[1]
        num_azimuthal = shape[2]
        ft2m = 1/3.281
        hub_radius = self.declare_variable('hub_radius', shape=(1,))
        rotor_radius = self.declare_variable('propeller_radius', shape=(1,)) * ft2m / 2
        # self.print_var(rotor_radius)
        dr = self.declare_variable('dr', shape=(1,))
        rotational_speed = self.declare_variable('rotational_speed', shape=(num_nodes))
        
        x = self.declare_variable('x_position', shape=(num_nodes))
        y = self.declare_variable('y_position', shape=(num_nodes))
        z = self.declare_variable('z_position', shape=(num_nodes))

        twist = self.declare_variable('twist_profile', shape=(num_radial,1))
        # self.print_var(twist*180/np.pi)
        chord = self.declare_variable('chord_profile', shape=(num_radial,1))
        # self.print_var(chord)
        t_c   = self.declare_variable('thickness_to_chord_ratio', shape=(num_radial))

        dT = self.declare_variable('dT', shape=(num_nodes,num_radial,num_azimuthal))
        dQ = self.declare_variable('dQ', shape=(num_nodes,num_radial, num_azimuthal))
        # self.print_var(dQ)
        M_inf = self.declare_variable('M_inf', shape=(1,),val=0)


        # expanding variables to shape
        self.register_output('_hub_radius', csdl.expand(hub_radius, shape,'l->ijk'))
        self.register_output('_rotor_radius', csdl.expand(rotor_radius, shape,'l->ijk'))
        self.register_output('_dr', csdl.expand(dr, shape,'l->ijk'))
        self.register_output('_rotational_speed', csdl.expand(rotational_speed, shape,'i->ijk'))
        
        self.register_output('_x_position', csdl.expand(x, shape,'i->ijk'))
        self.register_output('_y_position', csdl.expand(y, shape,'i->ijk'))
        self.register_output('_z_position', csdl.expand(z, shape,'i->ijk'))       

        self.register_output('_twist', csdl.expand(twist, shape, 'jk->ijk'))
        self.register_output('_chord', csdl.expand(chord, shape, 'jk->ijk'))
        self.register_output('_thickness_to_chord_ratio', csdl.expand(t_c, shape, 'j->ijk'))
        
        self.register_output('_dT',dT*1)# csdl.expand(dT, shape, 'ij->ijk'))
        self.register_output('_dQ',dQ*1) # csdl.expand(dQ, shape, 'ij->ijk'))
        
        self.register_output('_M_inf', csdl.expand(M_inf, shape, 'i->ijk'))

        # v = np.linspace(0, np.pi * 2 - np.pi * 2 / num_azimuthal, num_azimuthal)
        v = np.linspace(10 * np.pi/180, 170 * np.pi /180, num_azimuthal)
        self.create_input('theta', val = v )
        theta = np.einsum(
            'ij,k->ijk',
            np.ones((num_nodes, num_radial)),
            v,
        )
        self.create_input('_theta', val=theta)

        normalized_radial_discretization = 1. / num_radial / 2. \
            + np.linspace(0., 1. - 1. / num_radial, num_radial)

        normalized_radius = np.einsum(
            'ik,j->ijk',
            np.ones((num_nodes, num_azimuthal)),
            normalized_radial_discretization,
        )
        self.create_input('_normalized_radius', val=normalized_radius)


# import numpy as np
# from csdl import Model
# import csdl


# class CoreInputsModel(Model):
#     def initialize(self):
#         # self.parameters.declare('frequency_mode', types = int)
#         self.parameters.declare('num_nodes', types=int)
#         self.parameters.declare('num_radial', types=int)
#         self.parameters.declare('num_azimuthal', types=int)

#     def define(self):
#         # frequency_mode = self.parameters['frequency_mode']
#         num_nodes = self.parameters['num_nodes']
#         num_radial = self.parameters['num_radial']
#         num_azimuthal = self.parameters['num_azimuthal']
#         shape = (num_nodes, num_radial, num_azimuthal)

#         hub_radius = self.declare_variable('hub_radius', shape = (num_nodes))
#         rotor_radius = self.declare_variable('rotor_radius', shape = (num_nodes))
#         dr = self.declare_variable('dr', shape = (num_nodes))
#         rotational_speed = self.declare_variable('rotational_speed', shape = (num_nodes))
        
#         x = self.declare_variable('x_position', shape = (num_nodes))
#         y = self.declare_variable('y_position', shape = (num_nodes))
#         z = self.declare_variable('z_position', shape = (num_nodes))

#         twist = self.declare_variable('twist_profile', shape=(num_nodes, num_radial,))
#         chord = self.declare_variable('chord_profile', shape=(num_nodes, num_radial,))
#         t_c   = self.declare_variable('thickness_to_chord_ratio', shape = (num_nodes,num_radial))

#         dT = self.declare_variable('dT', shape = (num_nodes,num_radial))
#         dQ = self.declare_variable('dQ', shape = (num_nodes,num_radial))

#         M_inf = self.declare_variable('M_inf', shape = (num_nodes,))


#         # expanding variables to shape
#         self.register_output('_hub_radius', csdl.expand(hub_radius, shape,'i->ijk'))
#         self.register_output('_rotor_radius', csdl.expand(rotor_radius, shape,'i->ijk'))
#         self.register_output('_dr', csdl.expand(dr, shape,'i->ijk'))
#         self.register_output('_rotational_speed', csdl.expand(rotational_speed, shape,'i->ijk'))
        
#         self.register_output('_x_position', csdl.expand(x, shape,'i->ijk'))
#         self.register_output('_y_position', csdl.expand(y, shape,'i->ijk'))
#         self.register_output('_z_position', csdl.expand(z, shape,'i->ijk'))       

#         self.register_output('_twist', csdl.expand(twist, shape, 'ij->ijk'))
#         self.register_output('_chord', csdl.expand(chord, shape, 'ij->ijk'))
#         self.register_output('_thickness_to_chord_ratio', csdl.expand(t_c, shape, 'ij->ijk'))
        
#         self.register_output('_dT', csdl.expand(dT, shape, 'ij->ijk'))
#         self.register_output('_dQ', csdl.expand(dQ, shape, 'ij->ijk'))
        
#         self.register_output('_M_inf', csdl.expand(M_inf, shape, 'i->ijk'))

#         # v = np.linspace(0, np.pi * 2 - np.pi * 2 / num_azimuthal, num_azimuthal)
#         v = np.linspace(10 * np.pi/180, 170 * np.pi /180, num_azimuthal)
#         self.create_input('theta', val = v )
#         theta = np.einsum(
#             'ij,k->ijk',
#             np.ones((num_nodes, num_radial)),
#             v,
#         )
#         self.create_input('_theta', val=theta)

#         normalized_radial_discretization = 1. / num_radial / 2. \
#             + np.linspace(0., 1. - 1. / num_radial, num_radial)

#         normalized_radius = np.einsum(
#             'ik,j->ijk',
#             np.ones((num_nodes, num_azimuthal)),
#             normalized_radial_discretization,
#         )
#         self.create_input('_normalized_radius', val=normalized_radius)