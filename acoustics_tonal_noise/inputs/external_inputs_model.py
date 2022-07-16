import numpy as np
from csdl import Model
import csdl


class ExternalInputsModel(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('num_radial', types=int)
        self.parameters.declare('num_azimuthal', types=int)

    def define(self):
        num_nodes = self.parameters['num_nodes']
        num_radial = self.parameters['num_radial']
        num_azimuthal = self.parameters['num_azimuthal']
        shape = (num_nodes, num_radial, num_azimuthal)

        # self.create_input('mode',shape = (frequency_mode,))

        omega = self.declare_variable('omega', shape=(num_nodes, 1), units='rpm/1000') #* 1000

        thrust_origin = self.declare_variable(name='thrust_origin', shape=(num_nodes,3))
        altitude = self.declare_variable('z', shape=(num_nodes,1))
        x = self.create_output('x_position', shape=(num_nodes,1))
        y = self.create_output('y_position', shape=(num_nodes,1))
        z = self.create_output('z_position', shape=(num_nodes,1))
        dS = self.create_output('dS', shape=(num_nodes,1))



        # self.create_input('twist', shape=(num_nodes, num_radial))
        # self.create_input('chord', shape=(num_nodes, num_radial))
        # self.create_input('thickness_to_chord_ratio', shape = (num_nodes,num_radial))

        # self.create_input('dT', shape = (num_nodes,num_radial))
        # self.create_input('dQ', shape = (num_nodes,num_radial))

        # self.create_input('M_inf',shape = (num_nodes,))

        # self.create_input('propeller_radius', shape = (num_nodes,))
        rotor_radius = self.declare_variable(name='propeller_radius', shape=(1,), units='m')
        self.print_var(rotor_radius)
        R_h = 0.177 * rotor_radius
        self.register_output('hub_radius',R_h)
        dr = ((rotor_radius)-(0.177 * rotor_radius))/ (num_radial - 1)
        self.register_output('dr',dr)
        n = self.create_output('rotational_speed', shape=(num_nodes,1))
        for i in range(num_nodes):
            n[i,0] = omega[i,0] / 60
            x[i,0] = altitude[i,0] + thrust_origin[i,2]
            # self.print_var(x)
            y[i,0] = thrust_origin[i,1]
            # self.print_var(y)
            z[i,0] = thrust_origin[i,0]
            # self.print_var(z)
            dS[i,0] = (x[i,0]**2 + y[i,0]**2 + z[i,0]**2)**0.5
            # self.print_var(dS)

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


        
        





       
