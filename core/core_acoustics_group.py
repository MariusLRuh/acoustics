import numpy as np
from csdl import Model
import csdl


from acoustics_parameters import AcousticsParameters
from inputs.external_inputs_group import ExternalInputsGroup
from inputs.core_inputs_group import CoreInputsGroup
from inputs.preprocess_group import PreprocessGroup
from core.gutin_deming_group import GutinDemingGroup
from core.barry_magliozzi_group import BarryMagliozziGroup

class CoreAcousticsGroup(Model):

    def initialize(self):
        self.parameters.declare('acoustics_dict', types=AcousticsParameters)
        self.parameters.declare('num_evaluations', types=int)
        self.parameters.declare('num_radial', types=int)
        self.parameters.declare('num_tangential', types=int)
    

    def define(self):
        acoustics_dict = self.parameters['acoustics_dict']
        num_evaluations = self.parameters['num_evaluations']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']

        shape = (num_evaluations,num_radial, num_tangential)

        self.add(ExternalInputsGroup(
            shape = shape,
            num_evaluations = num_evaluations,
            num_radial = num_radial,
            num_tangential = num_tangential,
        ), name = 'external_inputs_group')

        self.add(CoreInputsGroup(
            num_evaluations = num_evaluations,
            num_radial = num_radial,
            num_tangential = num_tangential,
        ), name = 'core_inputs_group')

        self.add(PreprocessGroup(
            shape = shape,
        ), name = 'preprocess_group')

        self.add(GutinDemingGroup(
            acoustics_dict = acoustics_dict,
            shape = shape,
        ), name = 'gutin_deming_group')

        # self.add(BarryMagliozziGroup(
        #     acoustics_dict = acoustics_dict,
        #     shape = shape,
        # ), name = 'barry_magliozzi_group')
    