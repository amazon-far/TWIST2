#!/usr/bin/env bash
# Train Student policy for G1 23-DoF Mimic (with DAgger distillation)
# Isaac Lab / Isaac Sim 5.1.0
#
# Usage:
#   bash train_student.sh <exptid> <teacher_ckpt> [device] [num_envs] [task]
#
# Example:
#   bash train_student.sh stu_run1 \
#       logs/g1_23dof_teacher/teacher_20240417/model_30000.pt \
#       cuda:0 4096 g1_23dof_student_future
#
# Prerequisites (run once in env_isaaclab):
#   cd /home/wego/TWIST2/pose   && pip install -e .
#   cd /home/wego/TWIST2/isaacsim && pip install -e .

set -e

ISAACLAB_SH="${ISAACLAB_SH:-/home/wego/isaacsim/IsaacLab/isaaclab.sh}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPTID="${1:-student_$(date +%Y%m%d_%H%M%S)}"
TEACHER_CKPT="${2:?'teacher_ckpt argument required. Usage: bash train_student.sh <exptid> <teacher_ckpt>'}"
DEVICE="${3:-cuda:0}"
NUM_ENVS="${4:-4096}"
TASK="${5:-g1_23dof_student_future}"   # or g1_23dof_student

if [[ ! -f "${ISAACLAB_SH}" ]]; then
    echo "ERROR: isaaclab.sh not found at ${ISAACLAB_SH}"
    echo "  Set ISAACLAB_SH=/path/to/isaaclab.sh and retry."
    exit 1
fi

"${ISAACLAB_SH}" -p "${SCRIPT_DIR}/train.py" \
    --task "${TASK}" \
    --exptid "${EXPTID}" \
    --device "${DEVICE}" \
    --num_envs "${NUM_ENVS}" \
    --teacher_ckpt "${TEACHER_CKPT}" \
    --headless

echo "Student training done. Checkpoint in: ${SCRIPT_DIR}/logs/${TASK}/${EXPTID}*"
