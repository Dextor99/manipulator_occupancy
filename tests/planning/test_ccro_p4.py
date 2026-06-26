import numpy as np
import yaml

from planning.fast_sphere_risk import FastSphereRiskEvaluator
from planning.obstacle_forecast import ConstantVelocitySphereForecast
from planning.robot_surface_model import RobotSurfaceModel
from planning.safety_executor import SafetyExecutor

def test_high_and_state_mismatch_are_zero_velocity():
    ex=SafetyExecutor(.04,.1,state_error_limit=.2); z=np.zeros(6); ref=np.ones(6)*.1
    assert np.all(ex.command(z,ref,z,.03,z).qd==0)
    assert np.all(ex.command(z,np.ones(6),z,.2,z).qd==0)

def test_safe_tracking_command_is_bounded():
    ex=SafetyExecutor(.04,.1,qd_limit=.3); out=ex.command(np.zeros(6),np.ones(6)*.01,np.ones(6),.2,np.zeros(6))
    assert out.state=="low" and np.max(np.abs(out.qd))<=.3

def test_fast_sphere_risk_returns_finite_gradient():
    cfg=yaml.safe_load(open("config/ccro_stage4.yaml",encoding="utf-8"))
    model=RobotSurfaceModel(
        cfg["robot"]["urdf_path"],
        cfg["robot"]["joint_names"],
        cfg["surface"]["density_totals"],
        seed=cfg["surface"]["random_seed"],
        min_points_per_link=cfg["surface"]["min_points_per_link"],
        cache_dir=cfg["surface"]["cache_dir"],
        geometry=cfg["surface"]["geometry"],
    )
    q=np.asarray(cfg["trajectory"]["q_start"],float)
    center=np.mean(model.surface(q,"coarse"),axis=0)+np.array([0.04,0.0,0.0])
    forecast=ConstantVelocitySphereForecast(center,np.zeros(3),0.04,1.0)
    evaluator=FastSphereRiskEvaluator(model,d_safe=.11,d_activate=.17,max_spheres_per_link=4)
    risk=evaluator.configuration(q,forecast,0.0,with_gradient=True)
    assert np.isfinite(risk.min_distance)
    assert risk.gradient_q is not None
    assert risk.gradient_q.shape==(6,)
    assert np.all(np.isfinite(risk.gradient_q))
