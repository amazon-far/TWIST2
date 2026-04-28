"""Motion library wrapper for Isaac Sim 5.1 / Isaac Lab.

Wraps TWIST2's existing ``pose.utils.motion_lib_pkl.MotionLib`` to provide
a stable API identical in spirit to legged_gym's usage, but importable from
Isaac Lab environments.

Key responsibilities
--------------------
* Load pkl-based motion dataset from a YAML manifest.
* Sample motion IDs weighted by per-motion difficulty (curriculum).
* Sample random start times within motion length.
* Compute (root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
  body_pos, root_pos_delta_local, root_rot_delta_local) at arbitrary times.
* Slice 29-DoF pkl data to 23-DoF on the fly via KEEP_IDX_29_TO_23.
* Support error-aware sampling (weighted by max key-body tracking error).
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch

from g1_23dof_mimic.assets.g1_23dof import KEEP_IDX_29_TO_23


class MotionLib:
    """Thin adapter on top of TWIST2's MotionLib for Isaac Lab envs.

    Parameters
    ----------
    motion_file:
        Path to a YAML manifest listing pkl motion files (same format as
        ``legged_gym/motion_data_configs/twist2_dataset.yaml``).
    device:
        Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    sample_ratio:
        Fraction of motions to actually load (1.0 = all).
    motion_decompose:
        Whether to decompose motions (passed through to inner MotionLib).
    motion_smooth:
        Whether to smooth motions (passed through to inner MotionLib).
    num_dofs_source:
        DoF count in the pkl file (29 for TWIST2 default).  Frames are
        automatically sliced to 23 via KEEP_IDX_29_TO_23 before returning.
    """

    def __init__(
        self,
        motion_file: str,
        device: str = "cuda:0",
        sample_ratio: float = 1.0,
        motion_decompose: bool = False,
        motion_smooth: bool = True,
        num_dofs_source: int = 29,
    ):
        from pose.utils.motion_lib_pkl import MotionLib as _MotionLib  # heavy dep

        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"motion_file not found: {motion_file}")

        self._lib = _MotionLib(
            motion_file=motion_file,
            device=device,
            sample_ratio=sample_ratio,
            motion_decompose=motion_decompose,
            motion_smooth=motion_smooth,
        )
        self.device = device
        self._num_dofs_source = num_dofs_source
        self._slice_dofs = (num_dofs_source != 23)

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    def num_motions(self) -> int:
        return self._lib.num_motions()

    def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._lib.get_motion_length(motion_ids)

    def get_motion_names(self) -> List[str]:
        return self._lib.get_motion_names()

    def get_key_body_idx(self, key_body_names: List[str]) -> torch.Tensor:
        return self._lib.get_key_body_idx(key_body_names=key_body_names)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_motions(
        self,
        n: int,
        motion_difficulty: Optional[torch.Tensor] = None,
        max_key_body_error: Optional[torch.Tensor] = None,
        use_error_aware_sampling: bool = False,
        error_sampling_power: float = 5.0,
        error_sampling_threshold: float = 0.15,
    ) -> torch.Tensor:
        """Sample *n* motion IDs.

        When ``use_error_aware_sampling`` is True, combines difficulty weights
        with per-motion tracking error weights so harder / less-tracked motions
        are sampled more often (mirrors legged_gym's HumanoidMimic._reset_ref_motion).
        """
        return self._lib.sample_motions(
            n,
            motion_difficulty=motion_difficulty,
            max_key_body_error=max_key_body_error if use_error_aware_sampling else None,
            use_error_aware_sampling=use_error_aware_sampling,
            error_sampling_power=error_sampling_power,
            error_sampling_threshold=error_sampling_threshold,
        )

    def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Sample a random time within each motion's length."""
        return self._lib.sample_time(motion_ids)

    # ------------------------------------------------------------------
    # Frame computation
    # ------------------------------------------------------------------

    def calc_motion_frame(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Return motion state at given times.

        Returns
        -------
        root_pos              : (B, 3)
        root_rot              : (B, 4)  xyzw quaternion
        root_vel              : (B, 3)
        root_ang_vel          : (B, 3)
        dof_pos               : (B, 23)  — sliced from 29 if needed
        dof_vel               : (B, 23)
        body_pos              : (B, num_bodies, 3)  local body positions
        root_pos_delta_local  : (B, 3)
        root_rot_delta_local  : (B, 3)  euler angles
        """
        result = self._lib.calc_motion_frame(motion_ids, motion_times)
        (root_pos, root_rot, root_vel, root_ang_vel,
         dof_pos, dof_vel, body_pos,
         root_pos_delta_local, root_rot_delta_local) = result

        if self._slice_dofs:
            dof_pos = dof_pos[..., KEEP_IDX_29_TO_23]
            dof_vel = dof_vel[..., KEEP_IDX_29_TO_23]

        # pkl uses xyzw; convert to wxyz for Isaac Lab math convention
        root_rot = torch.cat([root_rot[..., 3:4], root_rot[..., :3]], dim=-1)

        return (root_pos, root_rot, root_vel, root_ang_vel,
                dof_pos, dof_vel, body_pos,
                root_pos_delta_local, root_rot_delta_local)
