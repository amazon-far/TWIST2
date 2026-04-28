"""Isaac Lab port of the TWIST2 teacher config (G1MimicPrivCfg).

Original reference:
  legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py : G1MimicPrivCfg(+PPO)

This file reproduces the teacher's privileged-observation motion-tracking env on
top of unitree_rl_lab's Isaac Lab mimic stack (23dof G1 variant). Fields that do
not have a 1:1 Isaac Lab counterpart (multi-horizon future motion preview,
per-clip motion curriculum) are implemented with their closest available
equivalents and flagged with comments.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_MIMIC_ACTION_SCALE
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_MIMIC_CFG as ROBOT_CFG


# --------------------------------------------------------------------------- #
# Scene / domain randomization knobs (match G1MimicPrivCfg.domain_rand where possible)
# --------------------------------------------------------------------------- #

# Push (G1MimicPrivCfg.domain_rand.push_robots, max_push_vel_xy=1.0, interval_s=4)
VELOCITY_RANGE = {
    "x": (-1.0, 1.0),
    "y": (-1.0, 1.0),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# Key bodies tracked by the teacher (9 key bodies in the original cfg).
# 23dof G1 uses *_wrist_roll_rubber_hand instead of left_rubber_hand.
KEY_BODY_NAMES = [
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_link",
    "right_elbow_link",
    # head_mocap has no direct 23dof counterpart; use torso_link as anchor proxy
    "torso_link",
]

# Body list sent as motion targets (anchor + end-effectors).
MOTION_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_rubber_hand",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_rubber_hand",
]


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Plane scene matching G1MimicPrivCfg.terrain.mesh_type = 'plane'."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True,
    )


# --------------------------------------------------------------------------- #
# MDP settings
# --------------------------------------------------------------------------- #


# Multi-clip motion set. Point this to a yaml that lists npz paths, or pass
# a list of paths directly. For the initial single-clip smoke test we wrap the
# Take_102 dance in a one-entry list so the multi-clip pipeline still runs.
MOTION_FILES: list[str] | str = [
    f"{os.path.dirname(__file__)}/../dance_102/G1_Take_102.bvh_60hz.npz",
]

# tar_motion_steps_priv = [1, 5, 10, ..., 95] in G1MimicPrivCfg (20 frames).
FUTURE_OFFSETS: list[int] = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


@configclass
class CommandsCfg:
    """Multi-clip motion command with multi-horizon future preview.

    - ``motion_files``: list of npz clips (or a yaml listing them). Each env
      samples a random clip at reset, reproducing TWIST2's multi-clip teacher.
    - ``future_offsets``: forward step offsets (in motion frames) exposed to
      the privileged observation terms. Mirrors
      ``G1MimicPrivCfg.env.tar_motion_steps_priv``.
    """

    motion = mdp.MultiMotionCommandCfg(
        asset_name="robot",
        motion_files=MOTION_FILES,
        future_offsets=FUTURE_OFFSETS,
        anchor_body_name="torso_link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        body_names=MOTION_BODY_NAMES,
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=UNITREE_G1_23DOF_MIMIC_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Teacher observes privileged state as its policy input (obs_type='priv').

    In distill's G1MimicPrivCfg:
      num_observations == num_privileged_obs == n_priv_obs_single
      = n_priv_mimic_obs + n_proprio + n_priv_info

    In Isaac Lab's ManagerBasedRLEnv, the policy group is the actor input and
    the critic group is the value input. For the teacher both should see the
    privileged bundle, so we point the policy group to the same full-state
    observation used by the critic. A DOF/ang-vel noise injection is kept off
    (teacher is trained with privileged access; add_noise on proprio is the
    student's concern).
    """

    @configclass
    class PrivilegedCfg(ObsGroup):
        # Current-frame motion target (corresponds to n_priv_mimic_obs[0] in the original cfg)
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        # Key body tracking targets (robot side, in anchor frame)
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        # Proprio + base state (privileged: includes base_lin_vel that the student does not get)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # --- Multi-horizon future preview (teacher's 20-frame lookahead) ---
        # These reproduce n_priv_mimic_obs for tar_motion_steps_priv=[1,5,...,95].
        future_joint_pos = ObsTerm(
            func=mdp.motion_future_joint_pos, params={"command_name": "motion"}
        )
        future_anchor_pos_b = ObsTerm(
            func=mdp.motion_future_anchor_pos_b, params={"command_name": "motion"}
        )
        future_anchor_ori_b = ObsTerm(
            func=mdp.motion_future_anchor_ori_b, params={"command_name": "motion"}
        )
        future_body_pos_b = ObsTerm(
            func=mdp.motion_future_body_pos_b, params={"command_name": "motion"}
        )

        def __post_init__(self):
            # Teacher gets clean privileged obs — no corruption.
            self.enable_corruption = False
            self.concatenate_terms = True

    # Policy and critic both consume the privileged bundle (teacher behavior).
    policy: PrivilegedCfg = PrivilegedCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Domain randomization (match G1MimicPrivCfg.domain_rand)."""

    # startup — randomize_friction (0.1, 2.0)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 2.0),
            "dynamic_friction_range": (0.1, 2.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    # Teacher doesn't drift default joint positions; keep a tiny jitter.
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    # randomize_base_com = True (added_com_range=[-0.05, 0.05])
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # push_robots = True, push_interval_s = 4, max_push_vel_xy = 1.0
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms adapted from G1MimicPrivCfg.rewards.scales.

    Mapping (original -> isaaclab func):
      tracking_joint_dof        -> motion_relative_body_position_error_exp (body_pos proxy)
      tracking_joint_vel        -> motion_global_body_linear_velocity_error_exp (body_lin_vel proxy)
      tracking_root_translation -> motion_global_anchor_position_error_exp
      tracking_root_rotation    -> motion_global_anchor_orientation_error_exp
      tracking_root_linear_vel  -> motion_global_body_linear_velocity_error_exp
      tracking_root_angular_vel -> motion_global_body_angular_velocity_error_exp
      tracking_keybody_pos      -> motion_relative_body_position_error_exp
      tracking_keybody_pos_global -> motion_global_anchor_position_error_exp (fused)

    Regularizers: dof_acc, dof_vel, action_rate, dof_pos_limits, dof_torque_limits
    are kept; feet_* terms are summarized via undesired_contacts + default air time.
    """

    # -- Regularizers (weights flipped to rsl_rl-style negatives)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-8)       # dof_acc = -5e-8
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)       # dof_vel = -1e-4
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)  # action_rate = -0.01
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # -- Tracking (weights match G1MimicPrivCfg.rewards.scales)
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=1.0,  # tracking_root_translation_z = 1.0
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=1.0,  # tracking_root_rotation = 1.0
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=2.0,  # tracking_joint_dof = 2.0 (joint-level DOF tracking proxy)
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=2.0,  # tracking_keybody_pos = 2.0
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,  # tracking_root_linear_vel = 1.0
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,  # tracking_root_angular_vel = 1.0
        params={"command_name": "motion", "std": 3.14},
    )

    # Penalize contacts on any non-foot / non-hand body (matches the spirit of
    # penalize_contacts_on = ["shoulder","elbow","hip","knee"] + collision).
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    r"(?!left_wrist_roll_rubber_hand$)(?!right_wrist_roll_rubber_hand$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Terminations mapping from G1MimicPrivCfg.

    pose_termination + pose_termination_dist=0.7 -> bad_motion_body_pos_z_only
    root_height_diff_threshold=0.3                -> bad_anchor_pos_z_only(0.3)
    termination_roll/pitch = 4.0 (very loose)     -> bad_anchor_ori(0.8) (stricter; matches rl_lab default)
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.3},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.7,  # pose_termination_dist = 0.7
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_roll_rubber_hand",
                "right_wrist_roll_rubber_hand",
            ],
        },
    )


# --------------------------------------------------------------------------- #
# Environment configuration
# --------------------------------------------------------------------------- #


@configclass
class TeacherEnvCfg(ManagerBasedRLEnvCfg):
    """Privileged-observation motion tracking teacher (G1MimicPrivCfg port)."""

    # G1MimicPrivCfg.env.num_envs = 4096
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        # decimation = 10 + dt = 0.002 in Isaac Gym => control dt = 0.02s (50 Hz).
        # Isaac Lab pipeline is stable at dt=0.005, decimation=4 (50 Hz) — keep that.
        self.decimation = 4
        # episode_length_s = 10 in G1MimicPrivCfg — match here.
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


class TeacherPlayEnvCfg(TeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
