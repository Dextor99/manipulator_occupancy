import numpy as np


def minimum_jerk(q0, q1, duration: float, t: float):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if duration <= 0:
        raise ValueError("duration must be positive")
    s = np.clip(t / duration, 0.0, 1.0)
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    blend_dot = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
    q = q0 + (q1 - q0) * blend
    qd = (q1 - q0) * blend_dot
    return q, qd
