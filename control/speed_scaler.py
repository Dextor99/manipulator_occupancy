import numpy as np


def scale_velocity(velocity, speed_scale: float):
    return np.asarray(velocity, dtype=float) * float(speed_scale)
