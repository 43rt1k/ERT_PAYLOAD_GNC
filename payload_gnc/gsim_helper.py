import numpy as np
import pfh.glidersim as gsim

POS = "r_RM2O"
ORI = "q_b2e"
VEL = "v_RM2e"

X = 0
Y = 1
Z = 2

XY = 2
XYZ = 3

L = 0
R = 1


def get_time_str(time):
    return f" | at {int(time)}s"


def deg2rad(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def q2euler(q):
    return gsim.orientation.quaternion_to_euler(q)
