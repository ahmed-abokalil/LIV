import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, HeliocentricTrueEcliptic, ITRS, EarthLocation
import torch as tn
from time import sleep
from datetime import datetime, timedelta, timezone
from math import pi

tn.set_printoptions(precision=10, sci_mode=False)

# Define the location of CMS in terms of longitude, latitude and azimuth

azimuth = 1.7677    
latitude = 0.8082  
longitude = 0.1061   

# Define the Earth's angular velocity (rad/s)
omega_utc = 2*pi/(86164)     # Earth's angular velocity in rad/s at UTC.
omega_siderial = 2*pi/(86400)
# Rotation matrices to go from SCF to CMS frame

# Rotation around the z-axis by phi (due to the azimuthal angle in spherical coordinates):
def R_z(angle):
    return tn.tensor([
        [1, 0, 0, 0],
        [0,  np.cos(angle), -np.sin(angle),0],
        [0, np.sin(angle),  np.cos(angle),0],
        [0,0,0,1]
    ], dtype=tn.float32)


# Rotation around the y-axis by θ (aligning the z-axis with the polar axis):
def R_y(angle):
    return tn.tensor([
        [1, 0, 0, 0],
        [0, np.sin(angle), 0, np.cos(angle)],
        [0, 0, 1, 0],
        [0, -np.cos(angle), 0, np.sin(angle)]
    ], dtype=tn.float32)


# A final rotation around the Z-axis has two purposes: to follow the rotation of the Earth over time and to synchronize with the SCF:
def R_Z(angle):
    return tn.tensor([
        [1, 0 , 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ], dtype=tn.float32)


