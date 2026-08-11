"""Finite local NUBS sensitivity matrices for fast 6.4 repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planning.nubs_trajectory import NUBSTrajectory6D


@dataclass(frozen=True)
class LocalSensitivity:
    sample_times: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    sq: np.ndarray
    sqd: np.ndarray
    sqdd: np.ndarray
    variable_count: int


def build_local_sensitivity(
    p_inner: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    durations: np.ndarray,
    sample_times: np.ndarray,
    *,
    epsilon: float = 1.0e-5,
    elastic_tail_position: bool = False,
) -> LocalSensitivity:
    points = np.asarray(p_inner, dtype=np.float64)
    times = np.asarray(sample_times, dtype=np.float64)
    base = NUBSTrajectory6D().generate(points, head, tail, durations)
    samples = base.sample(times)
    inner_variable_count = int(points.size)
    variable_count = inner_variable_count + (6 if elastic_tail_position else 0)
    sq = np.zeros((len(times), 6, variable_count), dtype=np.float64)
    sqd = np.zeros_like(sq)
    sqdd = np.zeros_like(sq)
    if variable_count == 0:
        return LocalSensitivity(times, samples.q, samples.qd, samples.qdd, sq, sqd, sqdd, variable_count)
    for variable in range(inner_variable_count):
        row, col = np.unravel_index(variable, points.shape)
        plus = points.copy()
        minus = points.copy()
        plus[row, col] += epsilon
        minus[row, col] -= epsilon
        plus_samples = NUBSTrajectory6D().generate(plus, head, tail, durations).sample(times)
        minus_samples = NUBSTrajectory6D().generate(minus, head, tail, durations).sample(times)
        sq[:, :, variable] = (plus_samples.q - minus_samples.q) / (2.0 * epsilon)
        sqd[:, :, variable] = (plus_samples.qd - minus_samples.qd) / (2.0 * epsilon)
        sqdd[:, :, variable] = (plus_samples.qdd - minus_samples.qdd) / (2.0 * epsilon)
    if elastic_tail_position:
        for joint in range(6):
            variable = inner_variable_count + joint
            plus_tail = np.asarray(tail, dtype=np.float64).copy()
            minus_tail = np.asarray(tail, dtype=np.float64).copy()
            plus_tail[joint, 0] += epsilon
            minus_tail[joint, 0] -= epsilon
            plus_samples = NUBSTrajectory6D().generate(points, head, plus_tail, durations).sample(times)
            minus_samples = NUBSTrajectory6D().generate(points, head, minus_tail, durations).sample(times)
            sq[:, :, variable] = (plus_samples.q - minus_samples.q) / (2.0 * epsilon)
            sqd[:, :, variable] = (plus_samples.qd - minus_samples.qd) / (2.0 * epsilon)
            sqdd[:, :, variable] = (plus_samples.qdd - minus_samples.qdd) / (2.0 * epsilon)
    return LocalSensitivity(times, samples.q, samples.qd, samples.qdd, sq, sqd, sqdd, variable_count)
