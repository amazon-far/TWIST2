"""G1 23-DoF Mimic Environment for Isaac Lab (Isaac Sim 5.1.0).

Ports legged_gym's HumanoidMimic + G1MimicDistill + G1MimicFuture into a
single Isaac Lab DirectRLEnv.  Teacher and Student modes are selected via
``cfg.obs_type``:

  "priv"           — Teacher: full privileged obs (multi-frame future + priv_info)
  "student"        — Student: current frame + proprio history only
  "student_future" — Student + future N frames (CMP, Curriculum Masked Privilege)

Design mirrors legged_gym exactly:
  • RSI (Reference State Initialization) at reset
  • Per-step motion library call to get reference pose
  • Same observation layout (priv_mimic_obs / mimic_obs / proprio / priv_info)
  • Same reward functions (tracking_joint_dof, tracking_keybody_pos, etc.)
  • Same termination conditions (roll/pitch limit, height diff, pose fail)
  • Motion difficulty curriculum (completion-rate based)
  • Error-aware motion sampling
  • Action delay + domain randomisation
"""

from __future__ import annotations

import math
import os
from dataclasses import MISSING
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
    subtract_frame_transforms,
)

from g1_23dof_mimic.assets.g1_23dof import G1_23DOF_CFG, JOINT_NAMES_ORDERED, NUM_DOFS
from g1_23dof_mimic.envs.mdp.motion_loader import MotionLib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euler_from_quat(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (roll, pitch, yaw) from (N,4) xyzw quaternion."""
    roll, pitch, yaw = euler_xyz_from_quat(q)
    return roll, pitch, yaw


def _convert_to_local_body_pos(root_quat: torch.Tensor, body_pos: torch.Tensor) -> torch.Tensor:
    """Rotate global-frame body offsets into root-local frame.
    body_pos: (N, K, 3)  already relative to root pos.
    """
    N, K, _ = body_pos.shape
    flat = body_pos.reshape(N * K, 3)
    q_rep = root_quat.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 4)
    local = quat_rotate_inverse(q_rep, flat)
    return local.reshape(N, K, 3)


def _convert_to_global_body_pos(
    root_pos: torch.Tensor, root_quat: torch.Tensor, body_pos_local: torch.Tensor
) -> torch.Tensor:
    """body_pos_local: (N, K, 3) local → world frame."""
    from isaaclab.utils.math import quat_apply
    N, K, _ = body_pos_local.shape
    flat = body_pos_local.reshape(N * K, 3)
    q_rep = root_quat.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 4)
    global_pos = quat_apply(q_rep, flat) + root_pos.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 3)
    return global_pos.reshape(N, K, 3)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@configclass
class G1MimicEnvCfg(DirectRLEnvCfg):
    """Configuration for G1 23-DoF Mimic environment.

    All observation / reward dimension constants are derived here so that
    policy network construction downstream has a single source of truth.
    """

    # ------------------------------------------------------------------ sim
    sim: SimulationCfg = SimulationCfg(dt=0.002, render_interval=10)
    decimation: int = 10  # policy runs at 50 Hz (0.002 * 10 = 0.02 s)

    # ------------------------------------------------------------------ scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # ------------------------------------------------------------------ robot
    robot_cfg: ArticulationCfg = G1_23DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ------------------------------------------------------------------ contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_ankle_roll_link",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )

    # ------------------------------------------------------------------ terrain
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
    )

    # ------------------------------------------------------------------ motion
    motion_file: str = MISSING  # path to twist2_dataset_23dof.yaml
    num_dofs_source: int = 29   # source pkl DoF (29); will be sliced to 23

    # Steps in teacher's future-frame window (priv obs)
    tar_motion_steps_priv: Tuple[int, ...] = (
        1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
        50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    )
    # Student observes only the current step (index 0 in priv window)
    tar_motion_steps: Tuple[int, ...] = (0,)
    # Additional future frames for "student_future" mode
    tar_motion_steps_future: Tuple[int, ...] = ()

    # ------------------------------------------------------------------ obs type
    obs_type: str = "priv"   # "priv" | "student" | "student_future"

    # ------------------------------------------------------------------ obs dims (auto-derived)
    # Per-step privileged mimic obs: root_pos(3)+dist(3)+roll+pitch+yaw(3)+vel(3)+ang_vel(3)
    #   +delta_pos(3)+delta_rot(3)+dof_pos(23)+key_body(9*3)
    NUM_KEY_BODIES: int = 9
    n_priv_mimic_obs_single: int = 21 + NUM_DOFS + NUM_KEY_BODIES * 3  # = 21+23+27 = 71
    # Per-step student mimic obs: root_vel_xy(2)+root_z(1)+roll+pitch(2)+yaw_angvel(1)+dof_pos(23)
    n_mimic_obs_single: int = 6 + NUM_DOFS   # = 29

    # Proprio: ang_vel(3)+imu(2)+dof_pos(23)+dof_vel(23)+last_action(23)
    n_proprio: int = 3 + 2 + NUM_DOFS * 3    # = 74

    # Priv info: lin_vel(3)+root_pos(3)+root_quat(4)+key_body(9*3)+contact(2)+mass(1)+friction(1)+motor(2*23)
    n_priv_info: int = 3 + 3 + 4 + NUM_KEY_BODIES * 3 + 2 + 1 + 1 + 2 * NUM_DOFS  # = 3+3+4+27+2+1+1+46=87

    history_len: int = 10   # how many frames of obs are stacked as history

    # Action dim (Isaac Lab 5.1: set action_space; num_actions kept for compatibility)
    num_actions: int = NUM_DOFS   # 23
    # observation_space / action_space are set dynamically in G1MimicEnv._setup_obs_dims()
    # before super().__init__() validates the cfg. We must give them placeholder int
    # values here so @configclass does not mark them as MISSING at class definition time.
    # They will be overwritten with correct values before validate() is called.
    observation_space: int = 1   # placeholder; overwritten by _setup_obs_dims
    action_space: int = NUM_DOFS  # 23 — correct value, won't change

    # ------------------------------------------------------------------ termination
    pose_termination_dist: float = 0.7   # max key-body distance before episode end
    root_height_diff_threshold: float = 0.3
    termination_roll: float = 4.0   # rad
    termination_pitch: float = 4.0  # rad

    # ------------------------------------------------------------------ motion curriculum
    motion_curriculum: bool = True
    motion_curriculum_gamma: float = 0.01
    use_adaptive_pose_termination: bool = False

    # ------------------------------------------------------------------ domain rand
    randomize_friction: bool = True
    friction_range: Tuple[float, float] = (0.1, 2.0)
    randomize_base_mass: bool = True
    added_mass_range: Tuple[float, float] = (-3.0, 3.0)
    randomize_motor: bool = True
    motor_strength_range: Tuple[float, float] = (0.8, 1.2)
    action_delay: bool = True
    action_buf_len: int = 8
    push_robots: bool = True
    push_interval_s: float = 4.0
    max_push_vel_xy: float = 1.0

    # ------------------------------------------------------------------ noise
    add_noise: bool = True
    noise_increasing_steps: int = 50_000
    ang_vel_noise: float = 0.2
    imu_noise: float = 0.2
    dof_pos_noise: float = 0.01
    dof_vel_noise: float = 0.15

    # ------------------------------------------------------------------ reward scales (mirrors legged_gym)
    tracking_joint_dof_scale: float = 2.0
    tracking_joint_vel_scale: float = 0.2
    tracking_root_z_scale: float = 1.0
    tracking_root_rot_scale: float = 1.0
    tracking_root_lin_vel_scale: float = 1.0
    tracking_root_ang_vel_scale: float = 1.0
    tracking_keybody_scale: float = 2.0
    alive_scale: float = 0.5
    feet_slip_scale: float = -0.1
    feet_contact_forces_scale: float = -5e-4
    dof_pos_limits_scale: float = -5.0
    action_rate_scale: float = -0.01
    dof_acc_scale: float = -5e-8
    ankle_dof_acc_scale: float = -1e-7

    # ------------------------------------------------------------------ reward sigma
    tracking_sigma: float = 0.2
    tracking_sigma_ang: float = 0.125

    # key bodies for robot tracking — names from local 23-DoF USD/XML.
    # (g1_23dof_rev_1_0.usd body names, confirmed via g1_23dof.xml)
    key_bodies: Tuple[str, ...] = (
        "left_ankle_roll_link",        "right_ankle_roll_link",
        "left_elbow_link",             "right_elbow_link",
        "left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand",
        "pelvis",
        "left_knee_link",              "right_knee_link",
    )

    # key bodies for motion pkl lookup — names from 29-DoF pkl link_body_list.
    # Must be the same joints in the same order as key_bodies above.
    key_bodies_motion: Tuple[str, ...] = (
        "left_ankle_roll_link",  "right_ankle_roll_link",
        "left_elbow_link",       "right_elbow_link",
        "left_rubber_hand",      "right_rubber_hand",
        "pelvis",
        "left_knee_link",        "right_knee_link",
    )

    # ------------------------------------------------------------------ episode
    episode_length_s: float = 20.0   # max; actual = motion length
    rand_reset: bool = True           # RSI: sample random time in motion
    randomize_start_pos: bool = True  # add random XY offset at spawn
    init_from_default_pos: bool = False  # True: always spawn in default standing pose
    global_obs: bool = False

    # ------------------------------------------------------------------ action scale
    action_scale: float = 0.5


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class G1MimicEnv(DirectRLEnv):
    """G1 23-DoF Motion Imitation environment for Isaac Lab.

    Observation spaces (set by subclass via _setup_obs_spaces):
      Teacher  (obs_type="priv"):
        obs_buf  = priv_mimic_obs * num_priv_steps
                   + proprio
                   + priv_info
                   (no history)
        privileged_obs_buf = same

      Student  (obs_type="student"):
        obs_buf  = mimic_obs_current + proprio + history*10
        privileged_obs_buf = teacher obs (for asymmetric AC)

      Student+Future (obs_type="student_future"):
        obs_buf  = (mimic_obs_current + proprio) + history*10
                   + future_mimic_flat
        privileged_obs_buf = teacher obs
    """

    cfg: G1MimicEnvCfg

    def __init__(self, cfg: G1MimicEnvCfg, render_mode: Optional[str] = None, **kwargs):
        # Compute observation spaces before super().__init__ which calls
        # _setup_scene and then queries num_observations / num_states.
        self._setup_obs_dims(cfg)
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._post_init()

    # ------------------------------------------------------------------
    # Dimension computation
    # ------------------------------------------------------------------

    def _setup_obs_dims(self, cfg: G1MimicEnvCfg):
        """Derive observation_space / state_space / action_space from cfg.

        Isaac Lab 5.1 uses observation_space / action_space (SpaceType = int for
        flat Box) instead of the deprecated num_observations / num_actions fields.
        """
        n_priv = cfg.n_priv_mimic_obs_single * len(cfg.tar_motion_steps_priv)
        n_stu  = cfg.n_mimic_obs_single  # current-frame student obs
        n_prop = cfg.n_proprio
        n_priv_info = cfg.n_priv_info
        n_future = cfg.n_mimic_obs_single * len(cfg.tar_motion_steps_future)
        n_obs_single = n_stu + n_prop  # one frame for history

        if cfg.obs_type == "priv":
            n_obs = n_priv + n_prop + n_priv_info
        elif cfg.obs_type == "student":
            n_obs = n_obs_single * (cfg.history_len + 1)
        elif cfg.obs_type == "student_future":
            n_obs = n_obs_single * (cfg.history_len + 1) + n_future
        else:
            raise ValueError(f"Unknown obs_type: {cfg.obs_type}")

        # Privileged obs = teacher obs (for asymmetric critic in student modes)
        n_states = n_priv + n_prop + n_priv_info

        # Isaac Lab 5.1 API: set observation_space / action_space directly
        cfg.observation_space = n_obs
        cfg.state_space = n_states
        cfg.action_space = cfg.num_actions  # 23

        # Store for use in forward methods
        self._n_priv_mimic   = n_priv
        self._n_stu_mimic    = n_stu
        self._n_proprio      = n_prop
        self._n_priv_info    = n_priv_info
        self._n_obs_single   = n_obs_single
        self._n_states       = n_states  # teacher obs dim (for priv_history buffer)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Post-init: motion lib, buffers, joint indices
    # ------------------------------------------------------------------

    def _post_init(self):
        """Called once after super().__init__() finishes."""
        device = self.device

        # ---- Motion library
        self._motion_lib = MotionLib(
            motion_file=self.cfg.motion_file,
            device=device,
            num_dofs_source=self.cfg.num_dofs_source,
        )

        # ---- Joint ordering index map (USD order → JOINT_NAMES_ORDERED order)
        usd_joint_names = self.robot.joint_names  # list from USD
        self._dof_reorder = torch.tensor(
            [usd_joint_names.index(n) for n in JOINT_NAMES_ORDERED],
            dtype=torch.long, device=device,
        )
        self._dof_reorder_inv = torch.argsort(self._dof_reorder)

        # ---- Key body indices
        all_body_names = self.robot.body_names
        self._key_body_ids = torch.tensor(
            [all_body_names.index(n) for n in self.cfg.key_bodies],
            dtype=torch.long, device=device,
        )
        self._key_body_ids_motion = self._motion_lib.get_key_body_idx(
            key_body_names=list(self.cfg.key_bodies_motion)
        )
        # Feet indices for contact / slip rewards
        foot_names = [n for n in all_body_names if "ankle_roll" in n]
        self._feet_ids = torch.tensor(
            [all_body_names.index(n) for n in foot_names],
            dtype=torch.long, device=device,
        )

        # ---- Target motion step tensors
        self._tar_steps_priv = torch.tensor(
            self.cfg.tar_motion_steps_priv, dtype=torch.long, device=device
        )
        self._tar_steps_stu = torch.tensor(
            self.cfg.tar_motion_steps, dtype=torch.long, device=device
        )
        # Index of each student step within the priv array
        self._tar_steps_stu_idx = torch.tensor(
            [list(self.cfg.tar_motion_steps_priv).index(s)
             if s in self.cfg.tar_motion_steps_priv else 0
             for s in self.cfg.tar_motion_steps],
            dtype=torch.long, device=device,
        )
        if self.cfg.tar_motion_steps_future:
            self._tar_steps_future = torch.tensor(
                self.cfg.tar_motion_steps_future, dtype=torch.long, device=device
            )
        else:
            self._tar_steps_future = None

        num_envs = self.num_envs

        # ---- Motion state buffers
        self._motion_ids         = torch.zeros(num_envs, dtype=torch.long,  device=device)
        self._motion_time_offset = torch.zeros(num_envs, dtype=torch.float, device=device)

        # Reference state
        self._ref_root_pos  = torch.zeros(num_envs, 3, device=device)
        self._ref_root_quat = torch.zeros(num_envs, 4, device=device)
        self._ref_root_lin_vel  = torch.zeros(num_envs, 3, device=device)
        self._ref_root_ang_vel  = torch.zeros(num_envs, 3, device=device)
        self._ref_dof_pos   = torch.zeros(num_envs, NUM_DOFS, device=device)
        self._ref_dof_vel   = torch.zeros(num_envs, NUM_DOFS, device=device)
        self._ref_body_pos  = torch.zeros(
            num_envs, len(self.robot.body_names), 3, device=device
        )
        # Per-frame step reference (used by reward / termination)
        self._ref_root_pos_delta_local = torch.zeros(num_envs, 3, device=device)
        self._ref_root_rot_delta_local = torch.zeros(num_envs, 3, device=device)

        # ---- Observation history buffers
        self._obs_history  = torch.zeros(num_envs, self.cfg.history_len, self._n_obs_single, device=device)
        self._priv_history = torch.zeros(num_envs, self.cfg.history_len, self._n_states, device=device)

        # ---- Action delay buffer
        self._action_history_buf = torch.zeros(num_envs, self.cfg.action_buf_len, NUM_DOFS, device=device)
        self._last_actions   = torch.zeros(num_envs, NUM_DOFS, device=device)
        self._last_dof_vel   = torch.zeros(num_envs, NUM_DOFS, device=device)

        # ---- Domain rand tensors
        self._mass_params   = torch.zeros(num_envs, 1, device=device)
        self._friction      = torch.ones(num_envs, 1, device=device)
        self._motor_strength = torch.ones(2, num_envs, NUM_DOFS, device=device)

        # ---- Episode tracking
        self._episode_length = torch.zeros(num_envs, device=device)
        # deviate tracking (consecutive pose fail frames)
        self._deviate_frames = torch.zeros(num_envs, device=device)

        # ---- Motion difficulty curriculum
        num_motions = self._motion_lib.num_motions()
        self._motion_difficulty = 10.0 * torch.ones(num_motions, device=device)
        self._motion_termination_dist = torch.ones(num_motions, device=device) * self.cfg.pose_termination_dist
        self._max_key_body_error = torch.zeros(num_motions, device=device)

        # ---- Default joint positions in reordered space
        default_pos_raw = self.robot.data.default_joint_pos[0]  # (num_dof_usd,)
        self._default_dof_pos = default_pos_raw[self._dof_reorder]  # (23,)

        # ---- Default standing pelvis height
        # Computed from USD InitialStateCfg: subtract the ankle contact point depth.
        # ankle_roll_link geom contact points are at z=-0.03 relative to ankle_roll_link,
        # so pelvis height = InitialStateCfg.pos.z - (drop from 1.0 to actual landing).
        # We use a fixed value derived from the MuJoCo XML (pelvis pos z=0.793) + small margin.
        self._default_root_height = 0.84  # tuned: MuJoCo XML=0.793, USD collision slightly higher

        # ---- Step counter (for noise curriculum, action delay)
        self._total_env_steps = 0

        # Apply initial domain randomization
        self._randomize_all(torch.arange(num_envs, device=device))

        # ---- Initial reset
        self.reset_idx(torch.arange(num_envs, device=device))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor):
        """Called by DirectRLEnv after episode termination."""
        if len(env_ids) == 0:
            return

        # Curriculum: update motion difficulty from episode completion rate
        if self.cfg.motion_curriculum:
            self._update_motion_difficulty(env_ids)

        # Sample motion + start time
        self._reset_ref_motion(env_ids)

        # RSI: reset robot to reference state
        self._apply_rsi(env_ids)

        # Domain rand
        self._randomize_all(env_ids)

        # Clear buffers
        self._action_history_buf[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._last_dof_vel[env_ids] = 0.0
        self._obs_history[env_ids]  = 0.0
        self._priv_history[env_ids] = 0.0
        self._deviate_frames[env_ids] = 0.0

    # Alias for external use (matches legged_gym API)
    def reset_idx(self, env_ids: torch.Tensor):
        self._reset_idx(env_ids)

    def _reset_ref_motion(self, env_ids: torch.Tensor):
        n = len(env_ids)
        motion_ids = self._motion_lib.sample_motions(
            n,
            motion_difficulty=self._motion_difficulty,
            max_key_body_error=self._max_key_body_error,
            use_error_aware_sampling=False,
        )
        if self.cfg.rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(n, device=self.device)

        self._motion_ids[env_ids]         = motion_ids
        self._motion_time_offset[env_ids] = motion_times

        # Compute reference pose at start time
        (root_pos, root_quat, root_vel, root_ang_vel,
         dof_pos, dof_vel, body_pos,
         delta_pos, delta_rot) = self._motion_lib.calc_motion_frame(motion_ids, motion_times)

        self._ref_root_pos[env_ids]  = root_pos
        self._ref_root_quat[env_ids] = root_quat  # already wxyz (converted in motion_loader)
        self._ref_root_lin_vel[env_ids]  = root_vel
        self._ref_root_ang_vel[env_ids]  = root_ang_vel
        self._ref_dof_pos[env_ids]   = dof_pos
        self._ref_dof_vel[env_ids]   = dof_vel
        # Store global body positions for pose-fail termination
        # body_pos from MotionLib is local; convert to global for comparison
        key_body_local = body_pos[:, self._key_body_ids_motion, :]  # (n, K, 3)
        key_body_global = _convert_to_global_body_pos(root_pos, root_quat, key_body_local)
        self._ref_body_pos[env_ids, :len(self._key_body_ids_motion)] = key_body_global

    def _apply_rsi(self, env_ids: torch.Tensor):
        """Reference State Initialization: set robot to reference pose."""
        n = len(env_ids)
        device = self.device

        if self.cfg.init_from_default_pos:
            # Spawn in default standing pose, zero velocity
            root_pos     = torch.zeros(n, 3, device=device)
            root_pos[:, 2] = self._default_root_height
            # Isaac Lab uses wxyz quaternion convention
            root_quat    = torch.tensor([1., 0., 0., 0.], device=device).expand(n, -1)
            root_vel     = torch.zeros(n, 3, device=device)
            root_ang_vel = torch.zeros(n, 3, device=device)
            dof_pos      = self._default_dof_pos.unsqueeze(0).expand(n, -1).clone()
            dof_vel      = torch.zeros(n, NUM_DOFS, device=device)
        else:
            root_pos     = self._ref_root_pos[env_ids].clone()
            root_quat    = self._ref_root_quat[env_ids].clone()  # already wxyz
            root_vel     = self._ref_root_lin_vel[env_ids].clone() * 0.8
            root_ang_vel = self._ref_root_ang_vel[env_ids].clone() * 0.8
            dof_pos      = self._ref_dof_pos[env_ids].clone()
            dof_vel      = self._ref_dof_vel[env_ids].clone() * 0.8

        # motion pkl root_pos z is already ground-relative; no offset needed

        # Add env origin offsets (world frame: XYZ all needed)
        env_origins = self._terrain.env_origins[env_ids]
        root_pos[:, 0] += env_origins[:, 0]
        root_pos[:, 1] += env_origins[:, 1]
        root_pos[:, 2] += env_origins[:, 2]  # flat terrain z=0 but add anyway
        if self.cfg.randomize_start_pos:
            rand_xy = (torch.rand(len(env_ids), 2, device=self.device) - 0.5) * 0.6
            root_pos[:, :2] += rand_xy

        # Reorder dof to USD order before writing
        dof_pos_usd = dof_pos[:, self._dof_reorder_inv]
        dof_vel_usd = dof_vel[:, self._dof_reorder_inv]

        # Write to articulation
        self.robot.write_root_pose_to_sim(
            torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids
        )
        self.robot.write_root_velocity_to_sim(
            torch.cat([root_vel, root_ang_vel], dim=-1), env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(dof_pos_usd, dof_vel_usd, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Pre-physics: apply action
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        """actions: (N, 23) in reordered space."""
        # Store in history
        self._action_history_buf = torch.cat(
            [self._action_history_buf[:, 1:], actions.unsqueeze(1)], dim=1
        )

        # Action delay curriculum
        if self.cfg.action_delay:
            start_step = 5000 * 24
            target_step = 20000 * 24
            if self._total_env_steps <= start_step:
                delay_prob = 0.0
            elif self._total_env_steps >= target_step:
                delay_prob = 0.5
            else:
                delay_prob = 0.5 * (self._total_env_steps - start_step) / (target_step - start_step)

            if torch.rand(1, device=self.device) < delay_prob:
                delayed = self._action_history_buf[:, -2]  # one step delay
            else:
                delayed = actions
        else:
            delayed = actions

        self._actions = torch.clamp(delayed, -5.0 / self.cfg.action_scale, 5.0 / self.cfg.action_scale)

    def _apply_action(self):
        """Convert action to joint position targets and write to sim."""
        # Action in reordered space → USD space
        target_dof = self._actions * self.cfg.action_scale + self._default_dof_pos
        target_usd = target_dof[:, self._dof_reorder_inv]
        self.robot.set_joint_position_target(target_usd)

    # ------------------------------------------------------------------
    # Post-physics: update motion reference
    # ------------------------------------------------------------------

    def _get_motion_times(self) -> torch.Tensor:
        return self.episode_length_buf * (self.cfg.decimation * self.cfg.sim.dt) + self._motion_time_offset

    def _update_ref_motion(self):
        """Recompute reference state for current time step."""
        motion_times = self._get_motion_times()
        (root_pos, root_quat, root_vel, root_ang_vel,
         dof_pos, dof_vel, body_pos,
         delta_pos, delta_rot) = self._motion_lib.calc_motion_frame(
            self._motion_ids, motion_times
        )
        # Offset for each env origin (xy only)
        env_origins = self._terrain.env_origins
        root_pos[:, :2] += env_origins[:, :2]

        self._ref_root_pos[:]  = root_pos
        self._ref_root_quat[:] = root_quat
        self._ref_root_lin_vel[:]  = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_dof_pos[:]   = dof_pos
        self._ref_dof_vel[:]   = dof_vel
        self._ref_root_pos_delta_local[:] = delta_pos
        self._ref_root_rot_delta_local[:] = delta_rot

        key_body_local = body_pos[:, self._key_body_ids_motion, :]
        key_body_global = _convert_to_global_body_pos(root_pos, root_quat, key_body_local)
        self._ref_body_pos[:, :len(self._key_body_ids_motion)] = key_body_global

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_dof_reordered(self, raw: torch.Tensor) -> torch.Tensor:
        """Convert USD-ordered dof tensor to JOINT_NAMES_ORDERED order."""
        return raw[:, self._dof_reorder]

    def _get_mimic_obs_batch(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute privileged mimic obs, student mimic obs, and optional future obs.

        Returns
        -------
        priv_mimic_obs : (N, n_priv_mimic_steps * n_priv_mimic_obs_single)
        mimic_obs      : (N, len(tar_steps_stu) * n_mimic_obs_single)
        future_obs     : (N, len(tar_steps_future) * n_mimic_obs_single) or None
        """
        dt = self.cfg.decimation * self.cfg.sim.dt
        motion_times = self._get_motion_times().unsqueeze(-1)  # (N, 1)

        # Gather all unique steps (priv + future)
        if self._tar_steps_future is not None and len(self._tar_steps_future) > 0:
            all_steps = torch.cat([self._tar_steps_priv, self._tar_steps_future])
        else:
            all_steps = self._tar_steps_priv
        n_total = all_steps.shape[0]
        n_priv  = self._tar_steps_priv.shape[0]

        obs_times = all_steps.float() * dt + motion_times  # (N, n_total)
        ids_tiled = self._motion_ids.unsqueeze(-1).expand(-1, n_total).reshape(-1)
        obs_times_flat = obs_times.reshape(-1)

        (root_pos, root_quat, root_vel, root_ang_vel,
         dof_pos, dof_vel, body_pos,
         delta_pos, delta_rot) = self._motion_lib.calc_motion_frame(ids_tiled, obs_times_flat)

        N = self.num_envs

        # Reshape to (N, n_total, dim)
        def rs(t): return t.reshape(N, n_total, t.shape[-1])

        root_pos_r    = rs(root_pos)
        root_quat_r   = rs(root_quat)
        root_vel_r    = rs(root_vel)
        root_ang_vel_r = rs(root_ang_vel)
        dof_pos_r     = rs(dof_pos)

        roll_r, pitch_r, yaw_r = _euler_from_quat(root_quat)
        roll_r  = roll_r.reshape(N, n_total, 1)
        pitch_r = pitch_r.reshape(N, n_total, 1)
        yaw_r   = yaw_r.reshape(N, n_total, 1)

        root_vel_local = quat_rotate_inverse(
            root_quat, root_vel
        ).reshape(N, n_total, 3)
        root_ang_vel_local = quat_rotate_inverse(
            root_quat, root_ang_vel
        ).reshape(N, n_total, 3)

        delta_pos_r = delta_pos.reshape(N, n_total, 3)
        delta_rot_r = delta_rot.reshape(N, n_total, 3)

        # Key body positions for priv obs
        key_body_local = body_pos[:, self._key_body_ids_motion, :]  # (N*n_total, K, 3)
        nk = key_body_local.shape[1]
        key_body_local_r = key_body_local.reshape(N, n_total, nk * 3)

        cur_root_pos = self.robot.data.root_pos_w  # (N, 3)
        dist_to_target = root_pos_r - cur_root_pos.unsqueeze(1)

        # --- Privileged mimic obs (teacher) ---
        priv_mimic = torch.cat([
            root_pos_r[:, :n_priv],             # 3
            dist_to_target[:, :n_priv],         # 3
            roll_r[:, :n_priv],                 # 1
            pitch_r[:, :n_priv],                # 1
            yaw_r[:, :n_priv],                  # 1
            root_vel_local[:, :n_priv],          # 3
            root_ang_vel_local[:, :n_priv],      # 3
            delta_pos_r[:, :n_priv],             # 3
            delta_rot_r[:, :n_priv],             # 3
            dof_pos_r[:, :n_priv],               # 23
            key_body_local_r[:, :n_priv],        # K*3
        ], dim=-1).reshape(N, -1)

        # --- Student mimic obs (nearest frame only, always 1 step = index 0) ---
        # legged_gym: student tar_motion_steps=[1], teacher tar_motion_steps=[1,5,...]
        # but stu_mimic is always 1 frame (the nearest priv step) regardless of
        # obs_type, because it feeds the obs_history buffer (n_mimic_obs_single wide).
        stu_mimic = torch.cat([
            root_vel_local[:, 0:1, :2].squeeze(1),       # 2
            root_pos_r[:, 0:1, 2:3].squeeze(1),          # 1
            roll_r[:, 0:1].squeeze(1),                   # 1
            pitch_r[:, 0:1].squeeze(1),                  # 1
            root_ang_vel_local[:, 0:1, 2:3].squeeze(1),  # 1
            dof_pos_r[:, 0:1].squeeze(1),                # 23
        ], dim=-1)   # (N, 29)

        # --- Future obs ---
        future_obs = None
        if self._tar_steps_future is not None and len(self._tar_steps_future) > 0:
            n_fut = self._tar_steps_future.shape[0]
            future_obs = torch.cat([
                root_vel_local[:, n_priv:, :2],      # 2
                root_pos_r[:, n_priv:, 2:3],         # 1
                roll_r[:, n_priv:],                  # 1
                pitch_r[:, n_priv:],                 # 1
                root_ang_vel_local[:, n_priv:, 2:3], # 1
                dof_pos_r[:, n_priv:],               # 23
            ], dim=-1).reshape(N, -1)

        return priv_mimic, stu_mimic, future_obs

    def _get_proprio_obs(self) -> torch.Tensor:
        """[ang_vel(3), imu(2), dof_pos(23), dof_vel(23), last_action(23)]"""
        dof_pos = self._get_dof_reordered(self.robot.data.joint_pos) - self._default_dof_pos
        dof_vel = self._get_dof_reordered(self.robot.data.joint_vel)

        root_quat = self.robot.data.root_quat_w
        ang_vel = quat_rotate_inverse(root_quat, self.robot.data.root_ang_vel_w)
        roll, pitch, _ = _euler_from_quat(root_quat)
        imu = torch.stack([roll, pitch], dim=-1)

        proprio = torch.cat([ang_vel, imu, dof_pos, dof_vel, self._last_actions], dim=-1)

        # Noise
        if self.cfg.add_noise:
            noise_scale = min(self._total_env_steps / (self.cfg.noise_increasing_steps * 24), 1.0)
            noise = torch.zeros_like(proprio)
            noise[:, :3] = self.cfg.ang_vel_noise
            noise[:, 3:5] = self.cfg.imu_noise
            noise[:, 5:5+NUM_DOFS] = self.cfg.dof_pos_noise
            noise[:, 5+NUM_DOFS:5+2*NUM_DOFS] = self.cfg.dof_vel_noise
            proprio += (2 * torch.rand_like(proprio) - 1) * noise * noise_scale

        # Zero ankle dof_vel (same as legged_gym)
        ankle_idx = [4, 5, 10, 11]
        dof_vel_start = 5 + NUM_DOFS
        for i in ankle_idx:
            proprio[:, dof_vel_start + i] = 0.0

        return proprio

    def _get_priv_info(self) -> torch.Tensor:
        """[lin_vel(3), root_pos(3), root_quat(4), key_body_local(K*3), contact(2), mass(1), friction(1), motor(2*23)]"""
        root_quat = self.robot.data.root_quat_w
        lin_vel = quat_rotate_inverse(root_quat, self.robot.data.root_lin_vel_w)
        root_pos = self.robot.data.root_pos_w

        # Key body positions in local frame
        body_pos_w = self.robot.data.body_pos_w  # (N, num_bodies, 3)
        key_pos = body_pos_w[:, self._key_body_ids, :]  # (N, K, 3)
        key_pos_rel = key_pos - root_pos.unsqueeze(1)
        key_pos_local = _convert_to_local_body_pos(root_quat, key_pos_rel)
        key_pos_flat = key_pos_local.reshape(self.num_envs, -1)

        # Foot contact (normal force > 5 N)
        contact_forces = self.contact_sensor.data.net_forces_w  # (N, num_sensors, 3)
        foot_contact = (contact_forces[:, :, 2].abs() > 5.0).float()
        if foot_contact.shape[1] > 2:
            foot_contact = foot_contact[:, :2]

        priv_info = torch.cat([
            lin_vel,
            root_pos,
            root_quat,
            key_pos_flat,
            foot_contact,
            self._mass_params,
            self._friction,
            self._motor_strength[0] - 1.0,
            self._motor_strength[1] - 1.0,
        ], dim=-1)
        return priv_info

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Build obs_buf and privileged_obs_buf."""
        priv_mimic, stu_mimic, future_obs = self._get_mimic_obs_batch()
        proprio = self._get_proprio_obs()
        priv_info = self._get_priv_info()

        # Teacher (privileged) obs
        priv_obs = torch.cat([priv_mimic, proprio, priv_info], dim=-1)

        # Current frame student obs (for history)
        obs_frame = torch.cat([stu_mimic, proprio], dim=-1)

        # Update history
        reset_mask = (self.episode_length_buf <= 1)
        if reset_mask.any():
            idx = reset_mask.nonzero(as_tuple=False).squeeze(-1)
            self._obs_history[idx] = obs_frame[idx].unsqueeze(1).expand(-1, self.cfg.history_len, -1)
            self._priv_history[idx] = priv_obs[idx].unsqueeze(1).expand(-1, self.cfg.history_len, -1)
        cont_mask = ~reset_mask
        if cont_mask.any():
            idx = cont_mask.nonzero(as_tuple=False).squeeze(-1)
            self._obs_history[idx, :-1] = self._obs_history[idx, 1:]
            self._obs_history[idx, -1]  = obs_frame[idx]
            self._priv_history[idx, :-1] = self._priv_history[idx, 1:]
            self._priv_history[idx, -1]  = priv_obs[idx]

        history_flat = self._obs_history.reshape(self.num_envs, -1)

        if self.cfg.obs_type == "priv":
            policy_obs = priv_obs
        elif self.cfg.obs_type == "student":
            policy_obs = torch.cat([obs_frame, history_flat], dim=-1)
        elif self.cfg.obs_type == "student_future":
            parts = [obs_frame, history_flat]
            if future_obs is not None:
                parts.append(future_obs)
            policy_obs = torch.cat(parts, dim=-1)
        else:
            policy_obs = priv_obs

        policy_obs = torch.clamp(policy_obs, -100.0, 100.0)
        priv_obs   = torch.clamp(priv_obs,   -100.0, 100.0)

        return {"policy": policy_obs, "critic": priv_obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        dof_pos = self._get_dof_reordered(self.robot.data.joint_pos)
        dof_vel = self._get_dof_reordered(self.robot.data.joint_vel)
        root_quat = self.robot.data.root_quat_w
        root_pos  = self.robot.data.root_pos_w
        roll, pitch, _ = _euler_from_quat(root_quat)

        sigma    = self.cfg.tracking_sigma
        sigma_ang = self.cfg.tracking_sigma_ang

        def gauss(err_sq, s): return torch.exp(-err_sq / (s * s))

        # --- joint dof tracking ---
        dof_diff = self._ref_dof_pos - dof_pos
        r_dof = gauss((dof_diff ** 2).mean(-1), sigma) * self.cfg.tracking_joint_dof_scale

        # --- joint vel tracking ---
        vel_diff = self._ref_dof_vel - dof_vel
        r_vel = gauss((vel_diff ** 2).mean(-1) / 100.0, sigma) * self.cfg.tracking_joint_vel_scale

        # --- root height tracking ---
        dz = (root_pos[:, 2] - self._ref_root_pos[:, 2]) ** 2
        r_z = gauss(dz, sigma) * self.cfg.tracking_root_z_scale

        # --- root rotation tracking (roll/pitch) ---
        ref_roll, ref_pitch, _ = _euler_from_quat(self._ref_root_quat)
        rot_err = (roll - ref_roll) ** 2 + (pitch - ref_pitch) ** 2
        r_rot = gauss(rot_err, sigma) * self.cfg.tracking_root_rot_scale

        # --- root lin vel tracking ---
        lin_vel_b = quat_rotate_inverse(root_quat, self.robot.data.root_lin_vel_w)
        ref_lin_vel_b = quat_rotate_inverse(self._ref_root_quat, self._ref_root_lin_vel)
        lin_err = ((lin_vel_b[:, :2] - ref_lin_vel_b[:, :2]) ** 2).sum(-1)
        r_lin = gauss(lin_err, sigma) * self.cfg.tracking_root_lin_vel_scale

        # --- root ang vel tracking (yaw) ---
        ang_vel_b = quat_rotate_inverse(root_quat, self.robot.data.root_ang_vel_w)
        ref_ang_vel_b = quat_rotate_inverse(self._ref_root_quat, self._ref_root_ang_vel)
        ang_err = (ang_vel_b[:, 2] - ref_ang_vel_b[:, 2]) ** 2
        r_ang = gauss(ang_err, sigma_ang) * self.cfg.tracking_root_ang_vel_scale

        # --- key body pos tracking ---
        body_pos_w = self.robot.data.body_pos_w
        key_pos = body_pos_w[:, self._key_body_ids, :]
        key_pos_rel = key_pos - root_pos.unsqueeze(1)
        key_pos_local = _convert_to_local_body_pos(root_quat, key_pos_rel)

        ref_key_pos = self._ref_body_pos[:, :len(self._key_body_ids), :]
        ref_key_rel = ref_key_pos - self._ref_root_pos.unsqueeze(1)
        ref_key_local = _convert_to_local_body_pos(self._ref_root_quat, ref_key_rel)

        key_err = ((key_pos_local - ref_key_local) ** 2).mean(-1).mean(-1)
        r_key = gauss(key_err, sigma) * self.cfg.tracking_keybody_scale

        # Update max key body error for error-aware sampling
        motion_ids = self._motion_ids
        key_dist = (key_pos_local - ref_key_local).norm(dim=-1).max(dim=-1).values  # (N,)
        self._max_key_body_error.scatter_reduce_(
            0, motion_ids, key_dist, reduce="amax", include_self=True
        )

        # --- alive ---
        r_alive = torch.ones(self.num_envs, device=self.device) * self.cfg.alive_scale

        # --- feet slip ---
        foot_vel_w = self.robot.data.body_lin_vel_w[:, self._feet_ids, :2]  # (N, 2, 2)
        foot_contact = (self.contact_sensor.data.net_forces_w[:, :, 2].abs() > 5.0)
        if foot_contact.shape[1] > 2:
            foot_contact = foot_contact[:, :2]
        slip = (foot_vel_w.norm(dim=-1) * foot_contact.float()).sum(-1)
        r_slip = slip * self.cfg.feet_slip_scale

        # --- feet contact forces ---
        foot_forces = self.contact_sensor.data.net_forces_w[:, :2].norm(dim=-1)
        r_contact_f = (foot_forces ** 2).sum(-1) * self.cfg.feet_contact_forces_scale

        # --- dof pos limits ---
        lo = self.robot.data.soft_joint_pos_limits[..., 0]
        hi = self.robot.data.soft_joint_pos_limits[..., 1]
        q  = self.robot.data.joint_pos
        over = (torch.clamp(q - hi, min=0.0) ** 2 + torch.clamp(lo - q, min=0.0) ** 2).sum(-1)
        r_limits = over * self.cfg.dof_pos_limits_scale

        # --- action rate ---
        act_rate = ((self._actions - self._last_actions) ** 2).mean(-1) * self.cfg.action_rate_scale

        # --- dof acc ---
        dof_acc = ((dof_vel - self._last_dof_vel) / (self.cfg.decimation * self.cfg.sim.dt)) ** 2
        r_dof_acc = dof_acc.mean(-1) * self.cfg.dof_acc_scale

        ankle_idx = [4, 5, 10, 11]
        ankle_acc = dof_acc[:, ankle_idx].mean(-1) * self.cfg.ankle_dof_acc_scale

        total = (r_dof + r_vel + r_z + r_rot + r_lin + r_ang + r_key
                 + r_alive + r_slip + r_contact_f + r_limits + act_rate
                 + r_dof_acc + ankle_acc)

        # Update last
        self._last_actions[:] = self._actions
        self._last_dof_vel[:] = dof_vel

        return total

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (terminated, time_out)."""
        root_quat = self.robot.data.root_quat_w
        root_pos  = self.robot.data.root_pos_w
        roll, pitch, _ = _euler_from_quat(root_quat)

        # Roll / pitch limit
        roll_cut  = roll.abs()  > self.cfg.termination_roll
        pitch_cut = pitch.abs() > self.cfg.termination_pitch

        # Root height diff
        height_diff = (root_pos[:, 2] - self._ref_root_pos[:, 2]).abs()
        height_cut  = height_diff > self.cfg.root_height_diff_threshold

        # Velocity too large
        vel_cut = self.robot.data.root_lin_vel_w.norm(dim=-1) > 5.0

        # Motion end
        motion_times = self._get_motion_times()
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        motion_end = motion_times >= motion_lengths

        # Pose tracking failure (key body distance)
        body_pos_w = self.robot.data.body_pos_w
        key_pos = body_pos_w[:, self._key_body_ids, :]
        key_pos_rel = key_pos - root_pos.unsqueeze(1)
        key_pos_local = _convert_to_local_body_pos(root_quat, key_pos_rel)

        ref_key_pos  = self._ref_body_pos[:, :len(self._key_body_ids), :]
        ref_key_rel  = ref_key_pos - self._ref_root_pos.unsqueeze(1)
        ref_key_local = _convert_to_local_body_pos(self._ref_root_quat, ref_key_rel)

        key_dist_sq = ((key_pos_local - ref_key_local) ** 2).sum(-1)  # (N, K)
        max_dist_sq = key_dist_sq.max(dim=-1).values                   # (N,)

        if self.cfg.use_adaptive_pose_termination:
            thresh_sq = self._motion_termination_dist[self._motion_ids] ** 2
        else:
            thresh_sq = self.cfg.pose_termination_dist ** 2

        pose_fail = max_dist_sq > thresh_sq

        terminated = (roll_cut | pitch_cut | height_cut | vel_cut | pose_fail | motion_end)
        time_out   = self.episode_length_buf >= self.max_episode_length

        # First-step protection
        first_step = (self.episode_length_buf == 0)
        terminated = terminated & ~first_step

        return terminated, time_out

    # ------------------------------------------------------------------
    # Motion difficulty curriculum (mirrors HumanoidMimic._update_motion_difficulty)
    # ------------------------------------------------------------------

    def _update_motion_difficulty(self, env_ids: torch.Tensor):
        reset_motion_ids = self._motion_ids[env_ids]
        dt = self.cfg.decimation * self.cfg.sim.dt
        completion = self.episode_length_buf[env_ids] * dt / \
                     self._motion_lib.get_motion_length(reset_motion_ids).clamp(min=1e-3)

        n_motions = self._motion_lib.num_motions()
        comp_sum   = torch.zeros(n_motions, device=self.device).scatter_add(0, reset_motion_ids, completion)
        comp_count = torch.zeros(n_motions, device=self.device).scatter_add(
            0, reset_motion_ids, torch.ones_like(completion)
        )
        comp_mean = comp_sum / comp_count.clamp(min=1)
        comp_mean[comp_count == 0] = 0.7

        add_idx   = comp_mean <= 0.50
        sub_idx   = (comp_mean >= 0.95) & (comp_mean < 0.99)
        super_sub = comp_mean >= 0.99

        gamma = self.cfg.motion_curriculum_gamma
        self._motion_difficulty[add_idx]   *= (1 + gamma)
        self._motion_difficulty[sub_idx]   *= (1 - gamma)
        self._motion_difficulty[super_sub] *= (1 - gamma * 20)
        self._motion_difficulty = self._motion_difficulty.clamp(1.0, 10.0)

        ratio = self._motion_difficulty / 10.0
        self._motion_termination_dist = (
            (self.cfg.pose_termination_dist - 0.2) * ratio + 0.2
        )

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------

    def _randomize_all(self, env_ids: torch.Tensor):
        n = len(env_ids)

        if self.cfg.randomize_base_mass:
            self._mass_params[env_ids] = torch.rand(n, 1, device=self.device) * (
                self.cfg.added_mass_range[1] - self.cfg.added_mass_range[0]
            ) + self.cfg.added_mass_range[0]

        if self.cfg.randomize_friction:
            self._friction[env_ids] = torch.rand(n, 1, device=self.device) * (
                self.cfg.friction_range[1] - self.cfg.friction_range[0]
            ) + self.cfg.friction_range[0]

        if self.cfg.randomize_motor:
            lo, hi = self.cfg.motor_strength_range
            self._motor_strength[0, env_ids] = torch.rand(n, NUM_DOFS, device=self.device) * (hi - lo) + lo
            self._motor_strength[1, env_ids] = torch.rand(n, NUM_DOFS, device=self.device) * (hi - lo) + lo

    # ------------------------------------------------------------------
    # Robot push (domain rand, mirrors legged_gym)
    # ------------------------------------------------------------------

    def _push_robots(self):
        """Apply random velocity impulses to robot base."""
        max_vel = self.cfg.max_push_vel_xy
        push_vel = (torch.rand(self.num_envs, 3, device=self.device) * 2 - 1)
        push_vel[:, :2] *= max_vel
        push_vel[:, 2] = 0.0
        cur_vel = self.robot.data.root_lin_vel_w
        self.robot.write_root_velocity_to_sim(
            torch.cat([cur_vel + push_vel, self.robot.data.root_ang_vel_w], dim=-1)
        )
