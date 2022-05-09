import numpy as np
from csdl import Model
import csdl


from acoustics_parameters import AcousticsParameters
from inputs.external_inputs_model import ExternalInputsModel
from inputs.core_inputs_model import CoreInputsModel
from inputs.preprocess_model import PreprocessModel
from core.gutin_deming_model import GutinDemingModel
from core.barry_magliozzi_model import BarryMagliozziModel

class CoreAcousticsModel(Model):

    def initialize(self):
        self.parameters.declare('acoustics_dict', types=AcousticsParameters)
        self.parameters.declare('num_evaluations', types=int)
        self.parameters.declare('num_radial', types=int)
        self.parameters.declare('num_azimuthal', types=int)
    

    def define(self):
        acoustics_dict = self.parameters['acoustics_dict']
        num_evaluations = self.parameters['num_evaluations']
        num_radial = self.parameters['num_radial']
        num_azimuthal = self.parameters['num_azimuthal']

        shape = (num_evaluations,num_radial, num_azimuthal)

        self.add(ExternalInputsModel(
            shape=shape,
            num_evaluations = num_evaluations,
            num_radial = num_radial,
            num_azimuthal = num_azimuthal,
        ), name = 'external_inputs_model')

        self.add(CoreInputsModel(
            num_evaluations = num_evaluations,
            num_radial = num_radial,
            num_azimuthal = num_azimuthal,
        ), name = 'core_inputs_model')

        self.add(PreprocessModel(
            shape=shape,
        ), name = 'preprocess_model')

        self.add(GutinDemingModel(
            acoustics_dict = acoustics_dict,
            shape=shape,
        ), name = 'gutin_deming_model')

        self.add(BarryMagliozziModel(
            acoustics_dict = acoustics_dict,
            shape=shape,
        ), name = 'barry_magliozzi_model')
    