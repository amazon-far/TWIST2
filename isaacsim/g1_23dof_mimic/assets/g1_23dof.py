"""ArticulationCfg for Unitree G1 23-DoF (Isaac Lab / Isaac Sim 5.1.0).

Joint layout (23 total) — joint names from local USD g1_23dof_rev_1_0.usd:
  Left Leg  [0-5] : left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint,
                    left_knee_joint, left_ankle_pitch_joint, left_ankle_roll_joint
  Right Leg [6-11]: right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint,
                    right_knee_joint, right_ankle_pitch_joint, right_ankle_roll_joint
  Waist     [12]  : waist_yaw_joint
  Left Arm  [13-17]: left_shoulder_pitch_joint, left_shoulder_roll_joint,
                     left_shoulder_yaw_joint, left_elbow_joint, left_wrist_roll_joint
  Right Arm [18-22]: right_shoulder_pitch_joint, right_shoulder_roll_joint,
                     right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint

Removed vs 29-DoF (indices from 29-DoF pkl):
  13=waist_roll, 14=waist_pitch, 20=L_wrist_pitch, 21=L_wrist_yaw,
  27=R_wrist_pitch, 28=R_wrist_yaw
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# ---------------------------------------------------------------------------
# USD path — local 23-DoF model
# ---------------------------------------------------------------------------
USD_PATH = "/home/wego/unitree_model/G1/23dof/usd/g1_23dof_rev_1_0/g1_23dof_rev_1_0.usd"

# ---------------------------------------------------------------------------
# 23-DoF joint ordering (matches pkl KEEP_IDX ordering & unitree SDK names)
# ---------------------------------------------------------------------------
JOINT_NAMES_ORDERED = [
    # Left Leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right Leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (yaw only)
    "waist_yaw_joint",
    # Left Arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    # Right Arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

# Default joint positions (mirrors legged_gym g1_mimic_distill_config.py)
DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint":  -0.20,
    "left_hip_roll_joint":    0.00,
    "left_hip_yaw_joint":     0.00,
    "left_knee_joint":        0.40,
    "left_ankle_pitch_joint": -0.20,
    "left_ankle_roll_joint":  0.00,

    "right_hip_pitch_joint":  -0.20,
    "right_hip_roll_joint":   0.00,
    "right_hip_yaw_joint":    0.00,
    "right_knee_joint":       0.40,
    "right_ankle_pitch_joint": -0.20,
    "right_ankle_roll_joint":  0.00,

    "waist_yaw_joint":        0.00,

    "left_shoulder_pitch_joint":  0.00,
    "left_shoulder_roll_joint":   0.40,
    "left_shoulder_yaw_joint":    0.00,
    "left_elbow_joint":           1.20,
    "left_wrist_roll_joint":      0.00,

    "right_shoulder_pitch_joint": 0.00,
    "right_shoulder_roll_joint":  -0.40,
    "right_shoulder_yaw_joint":   0.00,
    "right_elbow_joint":          1.20,
    "right_wrist_roll_joint":     0.00,
}

# ---------------------------------------------------------------------------
# Articulation config
# ---------------------------------------------------------------------------
G1_23DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos=DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Legs + waist_yaw (high torque)
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint",
                "waist_yaw_joint",
            ],
            effort_limit_sim=300.0,
            stiffness={
                ".*_hip_yaw_joint":   100.0,
                ".*_hip_roll_joint":  100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint":      150.0,
                "waist_yaw_joint":    150.0,
            },
            damping={
                ".*_hip_yaw_joint":   2.0,
                ".*_hip_roll_joint":  2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint":      4.0,
                "waist_yaw_joint":    4.0,
            },
            armature={
                ".*_hip_yaw_joint":   0.0103,
                ".*_hip_roll_joint":  0.0251,
                ".*_hip_pitch_joint": 0.0103,
                ".*_knee_joint":      0.0251,
                "waist_yaw_joint":    0.0103,
            },
        ),
        # Ankle joints (low torque)
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=40.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.003597,
        ),
        # Arm joints (medium torque)
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=80.0,
            stiffness=40.0,
            damping=5.0,
            armature={
                ".*_shoulder_pitch_joint": 0.003597,
                ".*_shoulder_roll_joint":  0.003597,
                ".*_shoulder_yaw_joint":   0.003597,
                ".*_elbow_joint":          0.003597,
                ".*_wrist_roll_joint":     0.00425,
            },
        ),
    },
)
"""G1 23-DoF articulation config (12 leg + 1 waist_yaw + 10 arm)."""

# ---------------------------------------------------------------------------
# Index mapping: 29-DoF pkl → 23-DoF (mirrors server_motion_lib.py:61-63)
# ---------------------------------------------------------------------------
KEEP_IDX_29_TO_23 = list(range(0, 13)) + list(range(15, 20)) + list(range(22, 27))

NUM_DOFS = 23
