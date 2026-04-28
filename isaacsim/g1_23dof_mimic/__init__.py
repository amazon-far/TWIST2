"""G1 23-DoF Motion Imitation package for Isaac Lab / Isaac Sim 5.1.0.

Register gym environments so Isaac Lab's gym registry can find them.

Tasks
-----
Isaac-G1-23DoF-Mimic-Teacher-v0
    Full privileged teacher policy (obs_type='priv').
    Mirrors legged_gym task: g1_priv_mimic

Isaac-G1-23DoF-Mimic-Student-v0
    Current-frame + history student policy (obs_type='student').
    Mirrors legged_gym task: g1_stu_mimic / g1_stu_rl

Isaac-G1-23DoF-Mimic-StudentFuture-v0
    Student + future N-frame conditioning (obs_type='student_future').
    Mirrors legged_gym task: g1_stu_future

Isaac-G1-23DoF-Mimic-Play-v0
    Evaluation / playback config (small env, no DR).
"""

import gymnasium as gym

from g1_23dof_mimic.envs.mimic_env import G1MimicEnv
from g1_23dof_mimic.envs.teacher_env_cfg import G1MimicTeacherCfg
from g1_23dof_mimic.envs.student_env_cfg import (
    G1MimicStudentCfg,
    G1MimicStudentFutureCfg,
    G1MimicStudentPlayCfg,
)
from g1_23dof_mimic.agents.ppo_teacher_cfg import G1MimicTeacherPPORunnerCfg
from g1_23dof_mimic.agents.ppo_student_cfg import (
    G1MimicStudentPPORunnerCfg,
    G1MimicStudentFuturePPORunnerCfg,
)

# ---- Teacher ---------------------------------------------------------------
gym.register(
    id="Isaac-G1-23DoF-Mimic-Teacher-v0",
    entry_point="g1_23dof_mimic.envs.mimic_env:G1MimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1MimicTeacherCfg,
        "rsl_rl_cfg_entry_point": G1MimicTeacherPPORunnerCfg,
    },
)

# ---- Student (current frame + history) ------------------------------------
gym.register(
    id="Isaac-G1-23DoF-Mimic-Student-v0",
    entry_point="g1_23dof_mimic.envs.mimic_env:G1MimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1MimicStudentCfg,
        "rsl_rl_cfg_entry_point": G1MimicStudentPPORunnerCfg,
    },
)

# ---- Student + Future frames (CMP) ----------------------------------------
gym.register(
    id="Isaac-G1-23DoF-Mimic-StudentFuture-v0",
    entry_point="g1_23dof_mimic.envs.mimic_env:G1MimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1MimicStudentFutureCfg,
        "rsl_rl_cfg_entry_point": G1MimicStudentFuturePPORunnerCfg,
    },
)

# ---- Play / evaluation (small scene, no DR) --------------------------------
gym.register(
    id="Isaac-G1-23DoF-Mimic-Play-v0",
    entry_point="g1_23dof_mimic.envs.mimic_env:G1MimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1MimicStudentPlayCfg,
        "rsl_rl_cfg_entry_point": G1MimicStudentPPORunnerCfg,
    },
)
