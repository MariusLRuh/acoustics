from acoustics_tonal_noise.acoustics_parameters import AcousticsParameters
import numpy as np

def get_acoustics_parameters(num_blades,directivity,mode):
    acoustic_dict = AcousticsParameters(
        num_blades=num_blades,
        directivity=directivity,
        mode=mode,
    )

    return acoustic_dict
       