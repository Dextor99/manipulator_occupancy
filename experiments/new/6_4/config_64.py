"""Configuration for Chapter 6.4 dynamic CCRO-NUBS virtual loop."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_4"
STAGE4_CONFIG = ROOT / "config" / "ccro_stage4.yaml"

RANDOM_SEED = 20260723
SPEED_GROUPS = (0.15, 0.25, 0.35)
LEAD_TIME_GROUPS = {
    "Long": 7.5,
    "Medium": 6.0,
    "Short": 4.5,
}
POST_SWITCH_CONFLICT_TIME = 1.5
MIN_REPLAY_MOTION_NORM = 1.0e-3

FAST_LOCAL_HORIZON = 1.0
FAST_LOCAL_SEGMENTS = 5
FAST_SAMPLE_DT = 0.10
FAST_REPAIR_STEPS = 3
FAST_REPAIR_STEP_SIZE = 0.035
FAST_REPAIR_STEP_SIZES = (0.008, 0.015, 0.025, 0.035)
FAST_DISTANCE_GRAD_EPS = 2.0e-4
FAST_REPAIR_TOPK_TIMES = 4
FAST_REPAIR_TOPK_POINTS = 8
FAST_REPAIR_ACCEPT_MS = 150.0
FAST_REPAIR_HARD_MAX_MS = 200.0
FAST_G1_REFERENCE_DENSE_MIN_RANGE = (0.04, 0.08)
FAST_CONFLICT_TIME_GROUPS = {
    "Long": 0.90,
    "Medium": 0.60,
    "Short": 0.35,
}
FAST_TAU_START_RANGE = (1.2, 5.8)
D1_MAIN_INSTANCES_PER_SPEED = 10
D2_MAIN_INSTANCES_PER_SPEED = 10
D2_STRESS_INSTANCES_PER_SPEED = 5
INSTANCES_PER_SPEED = 5
CALIBRATION_TRIALS = 10

METHODS = ("reference_only", "ssm", "ssm_apf", "critical_point_nubs", "ccro_nubs")
MAIN_METHODS = ("ssm_apf", "critical_point_nubs", "ccro_nubs")
PILOT_METHODS = ("reference_only", "ssm_apf", "critical_point_nubs", "ccro_nubs")
METHOD_NAMES = {
    "reference_only": "Reference-only",
    "ssm": "SSM",
    "ssm_apf": "SSM+APF",
    "critical_point_nubs": "Critical-point-NUBS",
    "ccro_nubs": "CCRO-NUBS",
}

BODY_LINKS = ("upperArm_Link", "foreArm_Link", "wrist1_Link")
EE_LINKS = ("wrist3_Link", "gripper_base_link", "left_link", "right_link")
CRITICAL_POINT_LINKS = {
    "upper_arm": ("upperArm_Link",),
    "elbow": ("upperArm_Link", "foreArm_Link"),
    "forearm": ("foreArm_Link",),
    "wrist": ("wrist1_Link", "wrist2_Link", "wrist3_Link"),
    "end_effector": ("gripper_base_link", "left_link", "right_link"),
}
CRITICAL_POINTS_PER_REGION = 2
CRITICAL_POINT_RADII = {
    "upper_arm": 0.055,
    "elbow": 0.050,
    "forearm": 0.045,
    "wrist": 0.040,
    "end_effector": 0.035,
}

DT = 0.04
OBSERVATION_DT = 1.0 / 30.0
MAX_TRIAL_TIME = 20.0
FINISH_TOLERANCE = 0.02

D_REPLAN_IN = 0.35
D_REPLAN_OUT = 0.38
D_SLOW = 0.12
D_STOP = 0.08
D_ACCEPT = 0.08
D_ONLINE_ACCEPT = 0.09
D_INITIAL_SAFE = 0.16

OBS_POS_SIGMA = 0.005
OBS_VEL_SIGMA = 0.01
OBS_RADIUS_SIGMA = 0.003
OBS_VEL_ALPHA = 0.85

# G1 deterministic mode: zero observation noise for pipeline verification
G1_OBS_POS_SIGMA = 0.0
G1_OBS_VEL_SIGMA = 0.0
G1_OBS_RADIUS_SIGMA = 0.0

FORECAST_HORIZON = 8.5
EVALUATE_HORIZON = 6.0
EVALUATE_STEPS = 31
REPLAN_INTERVAL = 0.5
MAX_REPLAN_ATTEMPTS = 2
PLANNING_BUDGET = 5.5
PLANNED_SWITCH_DELAY = 6.00
STRESS_SWITCH_DELAY = 3.00
BRIDGE_SLOW_IN = 0.12
BRIDGE_SLOW_OUT = 0.35
PENDING_LIGHT_SLOW_SCALE = 0.65
PENDING_MIN_SLOW_SCALE = 0.35
APF_GAIN = 0.18
APF_MAX_STEP = 0.020
APF_ACTIVATE_DISTANCE = 0.16
SWITCH_IMPROVEMENT_MARGIN = 0.02
SWITCH_TIME_EXTENSION_MARGIN = 2.0
USE_LOCAL_CANDIDATE = True
LOCAL_REPLAN_HORIZON = 4.0
CLEARANCE_DETOUR_STEP = 0.25
CLEARANCE_DETOUR_TRIGGER = 0.11
CLEARANCE_SOFTMIN_TOPK = 8
CLEARANCE_SOFTMIN_BETA = 60.0

OPTIMIZER_MAX_ITERATIONS = 35
OPTIMIZER_SAMPLES_PER_SEGMENT = 6
RISK_SAMPLES_PER_SEGMENT = 4

SURFACE_DENSITY_LOOP = "coarse"
SURFACE_DENSITY_TRUTH = "medium"
SURFACE_DENSITY_VERIFY = "medium"
