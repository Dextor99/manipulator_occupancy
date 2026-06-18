"""Controller variants for Chapter 4.5 replay and online experiments."""
from __future__ import annotations

import dataclasses

import numpy as np

from experiments.exp_44_main import Frame44, RepulsionEvaluator44, normalize
from risk.safety_policy import SafetyPolicy


CONTROLLERS_45 = ("ssm", "apf", "ours_scale", "ours_rep", "ours_full")


@dataclasses.dataclass
class ControlOutput:
    cmd_velocity: np.ndarray
    speed_scale: float
    rep_velocity: np.ndarray
    safety_state: str
    risk_distance: float


class SafeMotionController:
    def __init__(self, policy: SafetyPolicy, joint_limit: float = 0.4):
        self.policy = policy
        self.joint_limit = float(joint_limit)

    def step(
        self,
        ref_velocity: np.ndarray,
        q: np.ndarray,
        frame: Frame44,
        rep_eval: RepulsionEvaluator44,
    ) -> ControlOutput:
        raise NotImplementedError

    def _clip(self, qd: np.ndarray) -> np.ndarray:
        return np.clip(qd, -self.joint_limit, self.joint_limit)

    def _decision(self, distance: float):
        return self.policy.evaluate(distance)


class SSMController(SafeMotionController):
    def step(self, ref_velocity: np.ndarray, q: np.ndarray, frame: Frame44, rep_eval: RepulsionEvaluator44) -> ControlOutput:
        decision = self._decision(frame.ref.d_ref)
        cmd = self._clip(ref_velocity * decision.speed_scale)
        return ControlOutput(cmd, decision.speed_scale, np.zeros_like(ref_velocity), decision.level.value, frame.ref.d_ref)


class APFController(SafeMotionController):
    def step(self, ref_velocity: np.ndarray, q: np.ndarray, frame: Frame44, rep_eval: RepulsionEvaluator44) -> ControlOutput:
        decision = self._decision(frame.ref.d_ref)
        rep = rep_eval.repulsive_velocity("apf", q, frame)
        cmd = self._clip(ref_velocity * decision.speed_scale + rep)
        return ControlOutput(cmd, decision.speed_scale, rep, decision.level.value, frame.ref.d_ref)


class OursScaleController(SafeMotionController):
    def step(self, ref_velocity: np.ndarray, q: np.ndarray, frame: Frame44, rep_eval: RepulsionEvaluator44) -> ControlOutput:
        decision = self._decision(frame.ref.d_ref)
        cmd = self._clip(ref_velocity * decision.speed_scale)
        return ControlOutput(cmd, decision.speed_scale, np.zeros_like(ref_velocity), decision.level.value, frame.ref.d_ref)


class OursRepController(SafeMotionController):
    def step(self, ref_velocity: np.ndarray, q: np.ndarray, frame: Frame44, rep_eval: RepulsionEvaluator44) -> ControlOutput:
        decision = self._decision(frame.ref.d_ref)
        rep = rep_eval.repulsive_velocity("ours", q, frame)
        cmd = self._clip(ref_velocity * decision.speed_scale + rep)
        return ControlOutput(cmd, decision.speed_scale, rep, decision.level.value, frame.ref.d_ref)


class OursFullController(SafeMotionController):
    def step(self, ref_velocity: np.ndarray, q: np.ndarray, frame: Frame44, rep_eval: RepulsionEvaluator44) -> ControlOutput:
        decision = self._decision(frame.ref.d_ref)
        rep = rep_eval.repulsive_velocity("ours", q, frame)
        candidate = ref_velocity * decision.speed_scale + rep
        grad = rep_eval.distance_gradient(q, frame.ref.obs_points, links=None)
        # Velocity-level safety filter: remove the component that decreases D_ref.
        if np.linalg.norm(grad) > 1e-9 and float(np.dot(candidate, grad)) < 0.0:
            g = normalize(grad)
            candidate = candidate - np.dot(candidate, g) * g
        cmd = self._clip(candidate)
        return ControlOutput(cmd, decision.speed_scale, rep, decision.level.value, frame.ref.d_ref)


def make_controller(name: str, policy: SafetyPolicy, joint_limit: float = 0.4) -> SafeMotionController:
    if name == "ssm":
        return SSMController(policy, joint_limit)
    if name == "apf":
        return APFController(policy, joint_limit)
    if name == "ours_scale":
        return OursScaleController(policy, joint_limit)
    if name == "ours_rep":
        return OursRepController(policy, joint_limit)
    if name == "ours_full":
        return OursFullController(policy, joint_limit)
    raise ValueError(f"unknown controller: {name}")
