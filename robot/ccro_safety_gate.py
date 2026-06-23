"""Fail-closed authorization gate for future CCRO-NUBS real-robot trials."""
from __future__ import annotations
from dataclasses import dataclass

REQUIRED_CHECKS=("self_collision_ok","watchdog_ok","communication_ok","timestamp_ok","switch_state_error_ok","dense_validation_ok","emergency_stop_ok","manufacturer_limits_verified")

@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    mode: str
    failed_checks: tuple[str,...]
    reason: str

def evaluate_gate(mode:str, readiness:dict[str,bool], allow_real_robot_commands:bool, operator_phrase:str|None, required_phrase:str)->GateDecision:
    if mode not in {"shadow","safety_layer","low_speed_switch","dry_run"}: raise ValueError("unsupported P7 mode")
    failed=tuple(name for name in REQUIRED_CHECKS if not bool(readiness.get(name,False)))
    if mode in {"dry_run","shadow"}: return GateDecision(True,mode,failed,"no trajectory-switch command is permitted")
    if not allow_real_robot_commands: return GateDecision(False,mode,failed,"allow_real_robot_commands is false")
    if operator_phrase!=required_phrase: return GateDecision(False,mode,failed,"operator approval phrase mismatch")
    required=REQUIRED_CHECKS if mode=="low_speed_switch" else ("watchdog_ok","communication_ok","timestamp_ok","emergency_stop_ok","manufacturer_limits_verified")
    blocking=tuple(x for x in required if not readiness.get(x,False))
    if blocking: return GateDecision(False,mode,blocking,"readiness checks failed")
    return GateDecision(True,mode,(),"explicit authorization and readiness checks passed")

