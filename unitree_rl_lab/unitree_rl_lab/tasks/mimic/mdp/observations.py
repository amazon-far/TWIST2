from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand, MultiMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


# =====================================================================
# Future-horizon observations (require MultiMotionCommand)
# =====================================================================


def _get_multi_cmd(env: "ManagerBasedEnv", command_name: str) -> MultiMotionCommand:
    cmd = env.command_manager.get_term(command_name)
    if not isinstance(cmd, MultiMotionCommand):
        raise TypeError(
            f"Observation expected MultiMotionCommand for '{command_name}', got {type(cmd).__name__}."
            " Switch CommandsCfg to MultiMotionCommandCfg to enable future-horizon previews."
        )
    return cmd


def motion_future_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Target joint positions at each future offset, flattened to (num_envs, K*num_dof)."""
    cmd = _get_multi_cmd(env, command_name)
    return cmd.future_joint_pos().reshape(env.num_envs, -1)


def motion_future_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future anchor positions in the robot's current anchor frame. (num_envs, K*3)."""
    cmd = _get_multi_cmd(env, command_name)
    fut_pos_w = cmd.future_body_pos_w()[:, :, cmd.motion_anchor_body_index]  # (N, K, 3)
    num_envs, K, _ = fut_pos_w.shape
    anchor_pos = cmd.robot_anchor_pos_w.unsqueeze(1).expand(-1, K, -1)
    anchor_quat = cmd.robot_anchor_quat_w.unsqueeze(1).expand(-1, K, -1)
    pos_b, _ = subtract_frame_transforms(
        anchor_pos.reshape(-1, 3),
        anchor_quat.reshape(-1, 4),
        (fut_pos_w + env.scene.env_origins.unsqueeze(1).expand(-1, K, -1)).reshape(-1, 3),
        cmd.future_body_quat_w()[:, :, cmd.motion_anchor_body_index].reshape(-1, 4),
    )
    return pos_b.reshape(num_envs, -1)


def motion_future_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future anchor orientations in robot anchor frame, 6D (first two columns of R). (num_envs, K*6)."""
    cmd = _get_multi_cmd(env, command_name)
    fut_pos_w = cmd.future_body_pos_w()[:, :, cmd.motion_anchor_body_index]
    fut_quat_w = cmd.future_body_quat_w()[:, :, cmd.motion_anchor_body_index]
    num_envs, K, _ = fut_pos_w.shape
    anchor_pos = cmd.robot_anchor_pos_w.unsqueeze(1).expand(-1, K, -1)
    anchor_quat = cmd.robot_anchor_quat_w.unsqueeze(1).expand(-1, K, -1)
    _, ori_b = subtract_frame_transforms(
        anchor_pos.reshape(-1, 3),
        anchor_quat.reshape(-1, 4),
        (fut_pos_w + env.scene.env_origins.unsqueeze(1).expand(-1, K, -1)).reshape(-1, 3),
        fut_quat_w.reshape(-1, 4),
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(num_envs, -1)


def motion_future_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future key-body positions in the robot anchor frame. (num_envs, K*num_bodies*3)."""
    cmd = _get_multi_cmd(env, command_name)
    fut_pos_w = cmd.future_body_pos_w()  # (N, K, B, 3)
    fut_quat_w = cmd.future_body_quat_w()
    num_envs, K, B, _ = fut_pos_w.shape
    anchor_pos = cmd.robot_anchor_pos_w[:, None, None, :].expand(-1, K, B, -1)
    anchor_quat = cmd.robot_anchor_quat_w[:, None, None, :].expand(-1, K, B, -1)
    env_origins = env.scene.env_origins[:, None, None, :].expand(-1, K, B, -1)
    pos_b, _ = subtract_frame_transforms(
        anchor_pos.reshape(-1, 3),
        anchor_quat.reshape(-1, 4),
        (fut_pos_w + env_origins).reshape(-1, 3),
        fut_quat_w.reshape(-1, 4),
    )
    return pos_b.reshape(num_envs, -1)
