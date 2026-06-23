import numpy as np
import yaml

from planning.nubs_trajectory import NUBSTrajectory6D
from planning.optimizer import JointLimits
from planning.time_optimizer import VariableTimeNUBSOptimizer


def _case(mode="total"):
    cfg=yaml.safe_load(open("config/ccro_stage1.yaml", encoding="utf-8")); tr=cfg["trajectory"]
    head=NUBSTrajectory6D.make_boundary_state(tr["q_start"]); tail=NUBSTrajectory6D.make_boundary_state(tr["q_goal"])
    durations=np.asarray(tr["segment_durations"],float); robot=cfg["robot"]
    limits=JointLimits.from_arrays(robot["q_min"],robot["q_max"],robot["qd_max"],robot["qdd_max"])
    opt=VariableTimeNUBSOptimizer(head,tail,durations,limits,mode=mode,max_iterations=100)
    points=NUBSTrajectory6D.linear_inner_points(head[:,0],tail[:,0],durations)
    return opt,points,durations


def test_time_parameterization_round_trip():
    for mode in ("total","segment"):
        opt,points,durations=_case(mode); decoded_points,decoded_durations=opt.decode(opt.encode(points,durations))
        np.testing.assert_allclose(decoded_points,points); np.testing.assert_allclose(decoded_durations,durations)


def test_variable_time_gradient_matches_finite_difference():
    for mode in ("total","segment"):
        opt,points,durations=_case(mode); x=opt.encode(points,durations); _,analytic=opt.objective(x); numeric=np.zeros_like(x)
        for i in range(len(x)):
            plus=x.copy(); minus=x.copy(); plus[i]+=1e-6; minus[i]-=1e-6
            numeric[i]=(opt.objective(plus)[0]-opt.objective(minus)[0])/2e-6
        assert np.linalg.norm(analytic-numeric)/np.linalg.norm(numeric) < 2e-3


def test_both_modes_reduce_objective_and_preserve_constraints():
    for mode in ("total","segment"):
        opt,points,_=_case(mode); result=opt.optimize(points)
        assert result.success; assert result.final_cost < result.initial_cost
        assert max(result.trajectory.boundary_errors().values()) < 1e-5
        assert result.trajectory.waypoint_error() < 1e-7
        assert max(result.max_q_violation,result.max_qd_violation,result.max_qdd_violation) <= 1e-8

