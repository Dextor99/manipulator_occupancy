"""Velocity-level tracking, scaling and risk filtering used by P4/P7."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class SafetyCommand:
    qd: np.ndarray
    state: str
    speed_scale: float
    state_error: float

class SafetyExecutor:
    def __init__(self,d_stop:float,d_safe:float,tracking_gain:float=1.5,repulsive_gain:float=8.,qd_limit:float=.5,state_error_limit:float=.2):
        if not 0<d_stop<d_safe: raise ValueError("expected 0 < d_stop < d_safe")
        self.d_stop=d_stop; self.d_safe=d_safe; self.kp=tracking_gain; self.kr=repulsive_gain; self.qd_limit=qd_limit; self.state_error_limit=state_error_limit
    def command(self,q, q_ref, qd_ref, distance, risk_gradient):
        error=float(np.linalg.norm(q_ref-q))
        if error>self.state_error_limit: return SafetyCommand(np.zeros_like(q),"state_mismatch_hold",0.,error)
        if distance<=self.d_stop: return SafetyCommand(np.zeros_like(q),"high_hold",0.,error)
        scale=1. if distance>=self.d_safe else (distance-self.d_stop)/(self.d_safe-self.d_stop)
        gradient=np.zeros_like(q) if risk_gradient is None else np.asarray(risk_gradient,float)
        qd=scale*(qd_ref+self.kp*(q_ref-q))-self.kr*gradient
        return SafetyCommand(np.clip(qd,-self.qd_limit,self.qd_limit),"low" if scale==1 else "medium",float(scale),error)

