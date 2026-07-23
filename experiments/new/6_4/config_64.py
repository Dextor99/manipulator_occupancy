"""Configuration for Chapter 6.4 dynamic CCRO-NUBS virtual loop."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_4"
STAGE4_CONFIG = ROOT / "config" / "ccro_stage4.yaml"

RANDOM_SEED = 20260723
SPEED_GROUPS = (0.06, 0.09, 0.12)
INSTANCES_PER_SPEED = 5
CALIBRATION_TRIALS = 10

METHODS = ("reference_only", "ssm", "ccro_nubs")
METHOD_NAMES = {
    "reference_only": "Reference-only",
    "ssm": "SSM",
    "ccro_nubs": "CCRO-NUBS",
}

BODY_LINKS = ("upperArm_Link", "foreArm_Link", "wrist1_Link")
EE_LINKS = ("wrist3_Link", "gripper_base_link", "left_link", "right_link")

DT = 0.04
OBSERVATION_DT = 1.0 / 30.0
MAX_TRIAL_TIME = 20.0
FINISH_TOLERANCE = 0.02

D_REPLAN_IN = 0.35
D_REPLAN_OUT = 0.38
D_SLOW = 0.12
D_STOP = 0.08
D_ACCEPT = 0.08
D_INITIAL_SAFE = 0.16

OBS_POS_SIGMA = 0.005
OBS_VEL_SIGMA = 0.01
OBS_RADIUS_SIGMA = 0.003
OBS_VEL_ALPHA = 0.85

FORECAST_HORIZON = 8.5
EVALUATE_HORIZON = 6.0
EVALUATE_STEPS = 31
REPLAN_INTERVAL = 0.5
MAX_REPLAN_ATTEMPTS = 2
PLANNING_BUDGET = 3.0
SWITCH_DELAY = 3.00
EXPECTED_SWITCH_DELAY = 2.60
PENDING_SLOW_SCALE = 0.02

OPTIMIZER_MAX_ITERATIONS = 60
OPTIMIZER_SAMPLES_PER_SEGMENT = 8
RISK_SAMPLES_PER_SEGMENT = 5

SURFACE_DENSITY_LOOP = "coarse"
SURFACE_DENSITY_TRUTH = "medium"
SURFACE_DENSITY_VERIFY = "medium"
