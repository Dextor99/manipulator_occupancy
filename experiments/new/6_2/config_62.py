"""Frozen settings for the revised Chapter 6.2 experiments."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "result" / "new"

DYNAMIC_ALARM_DISTANCE = 0.14
BODY_RISK_DISTANCE = 0.08

TOTAL_DURATION = 8.0
DT = 0.05
H_MON = 0.5
PREDICTION_STEP = 0.1

DYNAMIC_SPEEDS = (0.10, 0.20, 0.30)
DYNAMIC_TRIALS_PER_SPEED = 10
CALIBRATION_TRIALS = 10
VISIBLE_PATH_LENGTH_RANGE = (0.40, 0.55)
CROSSING_TIME_RANGE = (5.0, 5.5)
MIN_SCENE_CLEARANCE_RANGE = (0.09, 0.12)
OBSTACLE_RADIUS_RANGE = (0.045, 0.060)
POINT_NOISE_SIGMA = 0.005
POINT_DROPOUT = 0.05
OBSTACLE_POINTS = 360

RANDOM_SEED = 6200

URDF_PATH = ROOT / "urdf" / "aubo_i16_gripper.urdf"
JOINT_NAMES = (
    "shoulder_joint",
    "upperArm_joint",
    "foreArm_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
)

SURFACE_DENSITY_TOTALS = {
    "coarse": 800,
    "medium": 2400,
    "dense": 12000,
}

Q_START = (0.0, -0.60, 1.40, 0.0, 1.00, 0.0)
Q_GOAL = (1.0, 0.10, 0.50, 0.8, 0.30, -0.70)
Q_WAVE = (0.22, -0.16, 0.20, -0.18, 0.14, 0.16)

BODY_CONFIG_TIMES = (2.0, 4.0, 6.0)
BODY_SAMPLES_PER_REGION = 10
BODY_RISK_PER_REGION = 6
BODY_REGIONS = {
    "upper_arm": ("upperArm_Link",),
    "elbow": ("upperArm_Link", "foreArm_Link"),
    "forearm": ("foreArm_Link",),
    "wrist": ("wrist1_Link", "wrist2_Link"),
}

END_EFFECTOR_LINKS = ("wrist3_Link", "gripper_base_link", "left_link", "right_link")
CRITICAL_POINTS_PER_REGION = 2
CRITICAL_POINT_RADII = {
    "upper_arm": 0.055,
    "elbow": 0.050,
    "forearm": 0.045,
    "wrist": 0.040,
}

RUNTIME_WARMUP = 10
RUNTIME_REPEATS = 20
