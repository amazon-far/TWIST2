"""Student environment configs for G1 23-DoF Mimic (Isaac Lab / Isaac Sim 5.1.0).

Mirrors legged_gym's:
  G1MimicStuCfg       → G1MimicStudentCfg        (obs_type='student')
  G1MimicStuFutureCfg → G1MimicStudentFutureCfg  (obs_type='student_future')

Student policy characteristics
-------------------------------
* Policy ("actor") sees only current-frame motion obs + proprio + history
* Critic sees full teacher privileged obs (asymmetric Actor-Critic)
* Trained with RL + DAgger KL distillation from a pre-trained teacher

Observation layout — Student (obs_type='student')
--------------------------------------------------
  obs_frame   = mimic_obs_single(29) + proprio(74)  = 103
  history×10  = 103 × 10                            = 1030
  current     = 103
  total       = 1133

Observation layout — Student + Future (obs_type='student_future')
-----------------------------------------------------------------
  obs_frame + history = 1133
  future_obs = len(tar_motion_steps_future) × 29
  (default: 10 future steps → 290)
  total = 1133 + 290 = 1423
"""

from __future__ import annotations

from g1_23dof_mimic.envs.mimic_env import G1MimicEnvCfg, G1MimicEnv
from g1_23dof_mimic.envs.teacher_env_cfg import G1MimicTeacherCfg
from isaaclab.utils import configclass


@configclass
class G1MimicStudentCfg(G1MimicTeacherCfg):
    """Student policy — current frame + proprio history only.

    The critic still sees the full teacher privileged obs for asymmetric AC.
    """

    # -----------------------------------------------------------------
    # Observation type
    # -----------------------------------------------------------------
    obs_type: str = "student"

    # Student observes the current step only (index 0 in priv window = step=1)
    tar_motion_steps: tuple = (1,)
    tar_motion_steps_future: tuple = ()

    # Keep the same priv steps for teacher/critic
    tar_motion_steps_priv: tuple = (
        1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
        50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    )

    # -----------------------------------------------------------------
    # Student is slightly more lenient on termination (less info)
    # -----------------------------------------------------------------
    pose_termination_dist: float = 0.7
    root_height_diff_threshold: float = 0.3

    # -----------------------------------------------------------------
    # Larger scene for student (more diverse training episodes)
    # -----------------------------------------------------------------
    from isaaclab.scene import InteractiveSceneCfg
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)


@configclass
class G1MimicStudentFutureCfg(G1MimicStudentCfg):
    """Student policy with future N-frame motion conditioning (CMP).

    Mirrors legged_gym's G1MimicStuFutureCfg (obs_type='student_future').
    Future frames give the policy a look-ahead into upcoming motion phases,
    improving tracking quality while keeping the actor input smaller than
    the full teacher privileged obs.
    """

    obs_type: str = "student_future"

    # Future steps to include (subset of tar_motion_steps_priv extended range)
    # Set to empty tuple to disable future observations.
    tar_motion_steps_future: tuple = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)

    # Student still only sees current frame as base obs
    tar_motion_steps: tuple = (1,)


@configclass
class G1MimicStudentPlayCfg(G1MimicStudentCfg):
    """Smaller scene for visual evaluation / play."""

    from isaaclab.scene import InteractiveSceneCfg
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=2.5, replicate_physics=True)

    # Disable domain rand for clean playback
    randomize_friction: bool = False
    randomize_base_mass: bool = False
    randomize_motor: bool = False
    action_delay: bool = False
    push_robots: bool = False
    add_noise: bool = False


# Convenience aliases
StudentEnv    = G1MimicEnv
StudentEnvCfg = G1MimicStudentCfg
StudentFutureEnvCfg = G1MimicStudentFutureCfg
