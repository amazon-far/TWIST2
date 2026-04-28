"""Teacher environment config for G1 23-DoF Mimic (Isaac Lab / Isaac Sim 5.1.0).

Mirrors legged_gym's G1MimicPrivCfg (obs_type='priv').

Teacher policy characteristics
-------------------------------
* Observes 20 future motion frames (multi-step priv mimic obs)
* Also receives privileged physics info: base lin_vel, key-body positions,
  contact mask, mass / friction / motor-strength params
* Trained with standard PPO (no distillation)
* Used as the "teacher" checkpoint for student distillation

Observation layout (single vector, no history)
-----------------------------------------------
  priv_mimic_obs = n_priv_steps × (21 + 23 + 27) = 20 × 71 = 1420
  proprio        = 3 + 2 + 23 + 23 + 23           =          74
  priv_info      = 3+3+4 + 27 + 2 + 1+1 + 46      =          87
  total          = 1420 + 74 + 87                  =        1581
"""

from __future__ import annotations

from g1_23dof_mimic.envs.mimic_env import G1MimicEnvCfg, G1MimicEnv
from isaaclab.utils import configclass


@configclass
class G1MimicTeacherCfg(G1MimicEnvCfg):
    # -----------------------------------------------------------------
    # Motion data
    # -----------------------------------------------------------------
    # NOTE: update this path to your 23-DoF dataset yaml.
    # If using 29-DoF pkl files set num_dofs_source=29 (default) and
    # the motion loader will slice automatically.
    motion_file: str = "/home/wego/TWIST2/isaacsim/motion_data_configs/example_dataset.yaml"
    num_dofs_source: int = 29   # source pkl DoF (auto-sliced to 23)

    # -----------------------------------------------------------------
    # Observation type
    # -----------------------------------------------------------------
    obs_type: str = "priv"   # full privileged obs (teacher mode)

    # Teacher sees 20 future frames + current (same as legged_gym G1MimicPrivCfg)
    tar_motion_steps_priv: tuple = (
        1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
        50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    )
    # tar_motion_steps: stu_mimic은 항상 priv[0] 1프레임 고정이므로 base 기본값 사용
    tar_motion_steps_future: tuple = ()  # teacher doesn't need extra future obs

    # -----------------------------------------------------------------
    # Scene
    # -----------------------------------------------------------------
    from isaaclab.scene import InteractiveSceneCfg
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # -----------------------------------------------------------------
    # Episode
    # -----------------------------------------------------------------
    episode_length_s: float = 20.0
    rand_reset: bool = False           # motion time=0 고정 (정자세 근처 스폰)
    randomize_start_pos: bool = False
    init_from_default_pos: bool = False  # RSI 사용 (motion 포즈로 스폰)

    # -----------------------------------------------------------------
    # Curriculum
    # -----------------------------------------------------------------
    motion_curriculum: bool = True
    motion_curriculum_gamma: float = 0.01
    use_adaptive_pose_termination: bool = False

    # -----------------------------------------------------------------
    # Termination thresholds (lenient for teacher — it has full info)
    # -----------------------------------------------------------------
    pose_termination_dist: float = 0.7
    root_height_diff_threshold: float = 0.3
    termination_roll: float = 4.0
    termination_pitch: float = 4.0

    # -----------------------------------------------------------------
    # Domain randomization (same as legged_gym G1MimicPrivCfg)
    # -----------------------------------------------------------------
    randomize_friction: bool = True
    friction_range: tuple = (0.1, 2.0)
    randomize_base_mass: bool = True
    added_mass_range: tuple = (-3.0, 3.0)
    randomize_motor: bool = True
    motor_strength_range: tuple = (0.8, 1.2)
    action_delay: bool = True
    action_buf_len: int = 8
    push_robots: bool = True
    push_interval_s: float = 4.0
    max_push_vel_xy: float = 1.0

    # -----------------------------------------------------------------
    # Noise (progressive)
    # -----------------------------------------------------------------
    add_noise: bool = True
    noise_increasing_steps: int = 50_000

    # -----------------------------------------------------------------
    # Reward scales (mirror legged_gym G1MimicPrivCfg.rewards.scales)
    # -----------------------------------------------------------------
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


# Convenience alias
TeacherEnv = G1MimicEnv
TeacherEnvCfg = G1MimicTeacherCfg
