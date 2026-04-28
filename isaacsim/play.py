"""Playback / evaluation entry point for G1 23-DoF Mimic.

Loads a saved checkpoint and runs the policy in the simulator with
rendering enabled.

Usage
-----
# Play teacher
python play.py --task g1_23dof_teacher \\
               --ckpt logs/g1_23dof_mimic_teacher/run_xxx/model_30000.pt \\
               --num_envs 16

# Play student
python play.py --task g1_23dof_student \\
               --ckpt logs/g1_23dof_mimic_student/run_xxx/model_30000.pt \\
               --num_envs 4
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play G1 23-DoF Mimic policy")
parser.add_argument("--task",     type=str,  default="g1_23dof_student",
                    choices=["g1_23dof_teacher", "g1_23dof_student", "g1_23dof_student_future"])
parser.add_argument("--ckpt",     type=str,  required=True, help="Path to .pt checkpoint")
parser.add_argument("--num_envs", type=int,  default=16)
parser.add_argument("--device",   type=str,  default="cuda:0")
parser.add_argument("--use_jit",  action="store_true", help="Export and use TorchScript")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = False   # always render in play mode

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab_rl.rsl_rl.runner import OnPolicyRunner

TASK_MAP = {
    "g1_23dof_teacher": (
        "g1_23dof_mimic.envs.mimic_env:G1MimicEnv",
        "g1_23dof_mimic.envs.teacher_env_cfg:G1MimicTeacherCfg",
        "g1_23dof_mimic.agents.ppo_teacher_cfg:G1MimicTeacherPPORunnerCfg",
    ),
    "g1_23dof_student": (
        "g1_23dof_mimic.envs.mimic_env:G1MimicEnv",
        "g1_23dof_mimic.envs.student_env_cfg:G1MimicStudentPlayCfg",
        "g1_23dof_mimic.agents.ppo_student_cfg:G1MimicStudentPPORunnerCfg",
    ),
    "g1_23dof_student_future": (
        "g1_23dof_mimic.envs.mimic_env:G1MimicEnv",
        "g1_23dof_mimic.envs.student_env_cfg:G1MimicStudentPlayCfg",
        "g1_23dof_mimic.agents.ppo_student_cfg:G1MimicStudentFuturePPORunnerCfg",
    ),
}


def _import_cls(dotpath: str):
    module_path, cls_name = dotpath.rsplit(":", 1)
    import importlib
    return getattr(importlib.import_module(module_path), cls_name)


def main():
    device = args.device
    task   = args.task

    env_cls_path, env_cfg_path, runner_cfg_path = TASK_MAP[task]
    EnvCls       = _import_cls(env_cls_path)
    EnvCfgCls    = _import_cls(env_cfg_path)
    RunnerCfgCls = _import_cls(runner_cfg_path)

    env_cfg    = EnvCfgCls()
    runner_cfg = RunnerCfgCls()

    env_cfg.scene.num_envs = args.num_envs
    # Disable domain rand for clean visual evaluation
    env_cfg.randomize_friction   = False
    env_cfg.randomize_base_mass  = False
    env_cfg.randomize_motor      = False
    env_cfg.action_delay         = False
    env_cfg.push_robots          = False
    env_cfg.add_noise            = False

    env = EnvCls(cfg=env_cfg, render_mode="rgb_array")

    runner = OnPolicyRunner(env, runner_cfg, log_dir="/tmp/play", device=device)
    runner.load(args.ckpt)

    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs["policy"])
        obs, _, _, _, _ = env.step(actions)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
