from acoustics_tonal_noise.utils.options_dicitonary import OptionsDictionary


class AcousticsParameters(OptionsDictionary):

    def initialize(self):
        self.declare('num_blades', types=int)
        self.declare('directivity', types=int)
        self.declare('altitude')
        self.declare('density')
        self.declare('speed_of_sound')
        self.declare('mode', types = int)
