from math import pi

def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta), deg_to_rad(phi), deg_to_rad(gamma))

def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta), rad_to_deg(rphi), rad_to_deg(rgamma))

def deg_to_rad(deg):
    return deg * pi / 180.0

def rad_to_deg(rad):
    return rad * 180.0 / pi