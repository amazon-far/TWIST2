import gymnasium as gym

gym.register(
    id="Unitree-G1-23dof-Mimic-Twist2-PrivTeacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:TeacherEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:TeacherPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)
