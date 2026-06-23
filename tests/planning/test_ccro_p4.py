import numpy as np
from planning.safety_executor import SafetyExecutor

def test_high_and_state_mismatch_are_zero_velocity():
    ex=SafetyExecutor(.04,.1,state_error_limit=.2); z=np.zeros(6); ref=np.ones(6)*.1
    assert np.all(ex.command(z,ref,z,.03,z).qd==0)
    assert np.all(ex.command(z,np.ones(6),z,.2,z).qd==0)

def test_safe_tracking_command_is_bounded():
    ex=SafetyExecutor(.04,.1,qd_limit=.3); out=ex.command(np.zeros(6),np.ones(6)*.01,np.ones(6),.2,np.zeros(6))
    assert out.state=="low" and np.max(np.abs(out.qd))<=.3
