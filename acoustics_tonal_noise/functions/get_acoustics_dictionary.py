from acoustics_parameters import AcousticsParameters
import numpy as np

def get_acoustics_parameters(directivity,mode,num_blades, altitude):
    L           = 6.5
    R           = 287
    T0          = 288.16
    P0          = 101325
    g0          = 9.81
    T           = T0 - L * altitude * 1e-3
    P           = P0 * (T/T0)**(g0/(L * 1e-3)/R)
    rho         = P/R/T
    a           = np.sqrt(1.4 * 287 * T)
    acoustic_dict = AcousticsParameters(
        num_blades = num_blades,
        directivity = directivity,
        altitude = altitude,
        density = rho,
        speed_of_sound = a,
        mode = mode,
    )

    return acoustic_dict
       