import numpy as np
from csdl import Model
import csdl


class ExternalInputsModel(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('mission_segment', types=str)
        self.parameters.declare('T_o_name_list', types=list)

    def define(self):
        shape = self.parameters['shape']
        mission_segment = self.parameters['mission_segment']
        T_o_name_list = self.parameters['T_o_name_list']

        num_nodes = shape[0]
        num_radial = shape[1]
        num_azimuthal = shape[2]

        # self.create_input('mode',shape = (frequency_mode,))

        omega = self.declare_variable('omega', shape=(num_nodes, 1), units='rpm/1000') * 1000
        self.print_var(omega)
        thrust_origin = self.declare_variable(T_o_name_list[0], shape=(num_nodes,3))
        # self.print_var((thrust_origin[0,2]**2)**0.5)
        # thrust_vector = self.declare_variable(name='thrust_vector', shape=(num_nodes,3))

        # yz_selection_vector = self.declare_variable(name='yz_selection_vector', val=np.tile(np.array([[1,1,1]]),(num_nodes,1))) - (thrust_vector**2)**0.5
        # yz = yz_selection_vector * thrust_origin
        # # self.print_var(yz)

        # rotor_axis_coordinate = csdl.dot(thrust_vector, thrust_origin, axis=1)
        # xyz_vec = self.create_output('xyz_vec', shape=(num_nodes,1))
        # xyz_vec[0,0] = csdl.reshape(rotor_axis_coordinate, (1,1))

        # self.print_var(xyz_vec)

        # self.print_var(rotor_axis_coordinate)

        altitude = self.declare_variable('z', shape=(num_nodes,1))
        x = self.create_output('x_position', shape=(num_nodes,1))
        y = self.create_output('y_position', shape=(num_nodes,1))
        z = self.create_output('z_position', shape=(num_nodes,1))
        dS = self.create_output('dS', shape=(num_nodes,1))





        # self.create_input('twist', shape=(num_nodes, num_radial))
        # self.create_input('chord', shape=(num_nodes, num_radial))
        # self.create_input('thickness_to_chord_ratio', shape = (num_nodes,num_radial))

        # self.create_input('dT', shape = (num_nodes,num_radial))al
        # self.create_input('dQ', shape = (num_nodes,num_radial))

        # self.create_input('M_inf',shape = (num_nodes,))

        # self.create_input('propeller_radius', shape = (num_nodes,))
        rotor_radius = self.declare_variable(name='propeller_radius', shape=(1,), units='m')
        # self.print_var(rotor_radius)
        R_h = 0.2 * rotor_radius
        self.register_output('hub_radius',R_h)
        dr = (rotor_radius-R_h)/ (num_radial - 1)
        self.register_output('dr',dr)
        n = self.create_output('rotational_speed', shape=(num_nodes,1))
        
        if mission_segment == 'hover':
            for i in range(num_nodes):
                n[i,0] = omega[i,0] / 60
                x[i,0] = altitude[i,0] + (thrust_origin[i,2]**2)**0.5
                # self.print_var(x)
                y[i,0] = thrust_origin[i,1]
                # self.print_var(y)
                z[i,0] = thrust_origin[i,0]
                # self.print_var(z)
                dS[i,0] = (x[i,0]**2 + y[i,0]**2 + z[i,0]**2)**0.5
                # self.print_var(dS)

        elif mission_segment == 'cruise':
            for i in range(num_nodes):
                n[i,0] = omega[i,0] / 60
                x[i,0] = thrust_origin[i,0]
                # self.print_var(x)
                y[i,0] = thrust_origin[i,1]
                # self.print_var(y)
                z[i,0] = thrust_origin[i,2] + altitude[i,0]
                # self.print_var(z)
                dS[i,0] = (x[i,0]**2 + y[i,0]**2 + z[i,0]**2)**0.5
                # self.print_var(dS)
        
        else:
            raise NotImplementedError


# import numpy as np
# from csdl import Model
# import csdl


# class ExternalInputsModel(Model):
#     def initialize(self):
#         self.parameters.declare('shape', types=tuple)
#         self.parameters.declare('num_nodes', types=int)
#         self.parameters.declare('num_radial', types=int)
#         self.parameters.declare('num_azimuthal', types=int)

#     def define(self):
#         num_nodes = self.parameters['num_nodes']
#         num_radial = self.parameters['num_radial']
#         num_azimuthal = self.parameters['num_azimuthal']
#         shape = (num_nodes, num_radial, num_azimuthal)

#         # self.create_input('mode',shape = (frequency_mode,))

#         self.create_input('x_position', shape = (num_nodes,))
#         self.create_input('y_position', shape = (num_nodes,))
#         self.create_input('z_position', shape = (num_nodes,))
#         self.create_input('dS', shape = (num_nodes,))

#         self.create_input('twist', shape = (num_nodes, num_radial))
#         self.create_input('chord', shape = (num_nodes, num_radial))
#         self.create_input('thickness_to_chord_ratio', shape = (num_nodes,num_radial))

#         self.create_input('dT', shape = (num_nodes,num_radial))
#         self.create_input('dQ', shape = (num_nodes,num_radial))

#         self.create_input('M_inf',shape = (num_nodes,))

#         self.create_input('rotor_radius', shape = (num_nodes,))
#         self.create_input('dr', shape = (num_nodes,))
#         self.create_input('hub_radius', shape = (num_nodes,))
#         self.create_input('rotational_speed', shape = (num_nodes,))


        
        





       
