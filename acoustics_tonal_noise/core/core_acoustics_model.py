import numpy as np
from csdl import Model


from acoustics_tonal_noise.acoustics_parameters import AcousticsParameters
from acoustics_tonal_noise.inputs.external_inputs_model import ExternalInputsModel
from acoustics_tonal_noise.inputs.core_inputs_model import CoreInputsModel
from acoustics_tonal_noise.inputs.preprocess_model import PreprocessModel
from acoustics_tonal_noise.core.gutin_deming_model import GutinDemingModel
from acoustics_tonal_noise.core.barry_magliozzi_model import BarryMagliozziModel
from acoustics_tonal_noise.core.hanson_model import HansonModel
from acoustics_tonal_noise.core.broadband_noise.skm import SchlegelKingMullModel

from lsdo_atmos.atmosphere_model import AtmosphereModel
from acoustics_tonal_noise.functions.get_acoustics_dictionary import get_acoustics_parameters

class CoreAcousticsModel(Model):

    def initialize(self):
        self.parameters.declare(name='name', default='acoustics')
        self.parameters.declare('num_nodes', types=int, default=1)
        self.parameters.declare('num_radial', types=int, default=30)
        self.parameters.declare('num_azimuthal', types=int, default=1)
        self.parameters.declare('num_blades', types=int, default=2)
        self.parameters.declare('directivity', types=int, default=0)
        self.parameters.declare('mode', types=int, default=1)
        
        
    

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        num_radial = self.parameters['num_radial']
        num_azimuthal = self.parameters['num_azimuthal']
        num_blades = self.parameters['num_blades']
        directivity = self.parameters['directivity']
        mode = self.parameters['mode']

        shape = (num_nodes,num_radial, num_azimuthal)

        acoustics_dict = get_acoustics_parameters(num_blades,directivity,mode)

        self.add(ExternalInputsModel(
            shape=shape,
            num_nodes = num_nodes,
            num_radial = num_radial,
            num_azimuthal = num_azimuthal,
        ), name = 'external_inputs_model')

        self.add(CoreInputsModel(
            num_nodes = num_nodes,
            num_radial = num_radial,
            num_azimuthal = num_azimuthal,
        ), name = 'core_inputs_model')

        self.add(PreprocessModel(
            shape=shape,
        ), name = 'preprocess_model')

        self.add(AtmosphereModel(
            shape=(num_nodes,1),
        ),name='atmosphere_model')

        self.add(GutinDemingModel(
            acoustics_dict = acoustics_dict,
            shape=shape,
        ), name = 'gutin_deming_model')

        self.add(BarryMagliozziModel(
            acoustics_dict = acoustics_dict,
            shape=shape,
        ), name = 'barry_magliozzi_model')

        # self.add(SchlegelKingMullModel(
        #     shape=shape
        # ),name='skm_model')

        # spl_skm = self.declare_variable('SPL_SKM', shape=shape)
        # spl_BM = self.declare_variable('SPL_tonal_Barry_Magliozzi', shape=shape) 
        # # to do: need to add up modes logarithmically 
        # # to do: torque only due to induced lift (neglect profile drag)

        # tonal_plus_broadband = 10 * csdl.log10(10**(spl_skm/10) + 10**(spl_BM/10))
        # self.register_output('tonal_plus_broadband', tonal_plus_broadband)
        # self.add(HansonModel(
        #     acoustics_dict=acoustics_dict,
        #     shape=shape,
        # ), name='hanson_model')
    
# sim = Simulator(CoreAcousticsModel)
# sim.run()