"""Training entry point for G1 23-DoF Mimic (Isaac Lab / Isaac Sim 5.1.0).

Supports teacher and student (with optional DAgger distillation) training.

Usage examples
--------------
# Teacher
python train.py --task g1_23dof_teacher --num_envs 4096 --headless

# Student (obs_type='student', requires teacher checkpoint)
python train.py --task g1_23dof_student \\
                --teacher_ckpt logs/g1_23dof_mimic_teacher/.../model_30000.pt \\
                --num_envs 4096 --headless

# Student + Future frames
python train.py --task g1_23dof_student_future \\
                --teacher_ckpt logs/g1_23dof_mimic_teacher/.../model_30000.pt \\
                --num_envs 4096 --headless

# Resume
python train.py --task g1_23dof_teacher --resume \\
                --load_run logs/g1_23dof_mimic_teacher/run_xxx

Common CLI flags (passed through to RSL-RL runner)
---------------------------------------------------
--num_envs     : override number of parallel environments
--max_iter     : override maximum iterations
--device       : cuda:0 (default)
--headless     : disable rendering
--wandb_project: W&B project name (default: twist2)
--exptid       : experiment id suffix for log directory naming
--debug        : enable debug mode (4 envs, viewer enabled)
"""

import argparse
import os
import sys

# ---- Isaac Lab bootstrap (must happen before any omni/isaac imports) ----
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train G1 23-DoF Mimic policy")
parser.add_argument("--task",         type=str,   default="g1_23dof_teacher",
                    choices=["g1_23dof_teacher", "g1_23dof_student", "g1_23dof_student_future"],
                    help="Which task/policy to train")
parser.add_argument("--num_envs",     type=int,   default=None)
parser.add_argument("--max_iter",     type=int,   default=None)
# NOTE: --device and --headless are added by AppLauncher; do NOT duplicate them here
parser.add_argument("--debug",        action="store_true")
parser.add_argument("--exptid",       type=str,   default="run")
parser.add_argument("--wandb_project",type=str,   default="twist2")
parser.add_argument("--no_wandb",     action="store_true")
parser.add_argument("--teacher_ckpt", type=str,   default="",
                    help="Path to teacher checkpoint (.pt) for student DAgger training")
parser.add_argument("--resume",       action="store_true")
parser.add_argument("--load_run",     type=str,   default="")
parser.add_argument("--checkpoint",   type=int,   default=-1,
                    help="-1 = last saved checkpoint")
# AppLauncher adds its own args (--livestream, --enable_cameras, etc.)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

if args.debug:
    args.num_envs = 4
    args.headless = False   # AppLauncher reads this field

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- Now safe to import Isaac Lab / Omni modules -------------------------
import torch
import wandb
from datetime import datetime

from importlib import metadata
from packaging import version
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

_rsl_rl_version = metadata.version("rsl-rl-lib")

# ---- Task registry -------------------------------------------------------
TASK_MAP = {
    "g1_23dof_teacher": (
        "g1_23dof_mimic.envs.mimic_env:G1MimicEnv",
        "g1_23dof_mimic.envs.teacher_env_cfg:G1MimicTeacherCfg",
        "g1_23dof_mimic.agents.ppo_teacher_cfg:G1MimicTeacherPPORunnerCfg",
    ),
    "g1_23dof_student": (
        "g1_23dof_mimic.envs.mimic_env:G1MimicEnv",
        "g1_23dof_mimic.envs.student_env_cfg:G1MimicStudentCfg",
        "g1_23dof_mimic.agents.ppo_student_cfg:G1MimicStudentPPORunnerCfg",
    ),
    "g1_23dof_student_future": (
        "g1_23dof_mimic.envs.mimic_env:G1MimicEnv",
        "g1_23dof_mimic.envs.student_env_cfg:G1MimicStudentFutureCfg",
        "g1_23dof_mimic.agents.ppo_student_cfg:G1MimicStudentFuturePPORunnerCfg",
    ),
}


def _import_cls(dotpath: str):
    """Import 'module.path:ClassName'."""
    module_path, cls_name = dotpath.rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def main():
    device = args.device
    task   = args.task

    env_cls_path, env_cfg_path, runner_cfg_path = TASK_MAP[task]
    EnvCls       = _import_cls(env_cls_path)
    EnvCfgCls    = _import_cls(env_cfg_path)
    RunnerCfgCls = _import_cls(runner_cfg_path)

    # Build configs
    env_cfg    = EnvCfgCls()
    runner_cfg = RunnerCfgCls()

    # CLI overrides
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iter is not None:
        runner_cfg.max_iterations = args.max_iter
    if args.teacher_ckpt:
        runner_cfg.teacher_ckpt = args.teacher_ckpt
    if args.resume and args.load_run:
        runner_cfg.resume     = True
        runner_cfg.load_run   = args.load_run
        runner_cfg.checkpoint = args.checkpoint

    if args.debug:
        env_cfg.scene.num_envs = 4

    # Log directory
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name   = f"{args.exptid}_{timestamp}"
    log_root   = os.path.join(os.path.dirname(__file__), "logs", task)
    log_dir    = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # W&B
    if not args.no_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "task": task,
                    "obs_type": env_cfg.obs_type,
                    "num_envs": env_cfg.scene.num_envs,
                    "max_iter": runner_cfg.max_iterations,
                    "teacher_ckpt": getattr(runner_cfg, "teacher_ckpt", ""),
                },
                dir=log_dir,
            )
        except Exception as e:
            print(f"[Warning] W&B init failed: {e}. Continuing without logging.")

    # Build environment
    env = EnvCls(cfg=env_cfg, render_mode="rgb_array" if not args.headless else None)
    # rsl_rl requires a VecEnv wrapper
    env = RslRlVecEnvWrapper(env)

    # Strip deprecated fields (stochastic, init_noise_std, etc.) for rsl_rl >= 4.0
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, _rsl_rl_version)

    # Build runner — runner_cfg must be dict (configclass has .to_dict())
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=device)

    # Load teacher checkpoint for DAgger distillation
    if hasattr(runner_cfg, "teacher_ckpt") and runner_cfg.teacher_ckpt:
        _load_teacher(runner, runner_cfg.teacher_ckpt, device)

    # Train
    runner.learn(
        num_learning_iterations=runner_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()
    if not args.no_wandb:
        try:
            wandb.finish()
        except Exception:
            pass

    simulation_app.close()


def _load_teacher(runner, ckpt_path: str, device: str):
    """Load teacher ActorCritic into runner.alg.teacher_actor_critic."""
    import torch
    if not os.path.isfile(ckpt_path):
        print(f"[Warning] Teacher checkpoint not found: {ckpt_path}. Skipping.")
        return

    loaded = torch.load(ckpt_path, map_location=device)
    # RSL-RL saves {"model_state_dict": ..., ...}
    state_dict = loaded.get("model_state_dict", loaded)

    alg = runner.alg
    if hasattr(alg, "teacher_actor_critic"):
        alg.teacher_actor_critic.load_state_dict(state_dict, strict=False)
        alg.teacher_actor_critic.eval()
        alg.teacher_loaded = True
        print(f"[train] Teacher checkpoint loaded: {ckpt_path}")
    else:
        print("[Warning] Runner algorithm has no teacher_actor_critic attribute.")


if __name__ == "__main__":
    main()
