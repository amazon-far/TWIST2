#!/usr/bin/env bash
# Train Teacher policy for G1 23-DoF Mimic
# Isaac Lab / Isaac Sim 5.1.0
#
# Usage:
#   bash train_teacher.sh [exptid] [device] [num_envs] [max_iter] [load_run]
#
# Example (신규):
#   bash train_teacher.sh teacher_test cuda:0 256 500
#
# Example (기존 모델에서 추가학습):
#   bash train_teacher.sh teacher_resume cuda:0 256 "" logs/g1_23dof_teacher/teacher_test_20260417_131514
#
# Prerequisites (run once in env_isaaclab):
#   cd /home/wego/TWIST2/pose   && pip install -e .
#   cd /home/wego/TWIST2/isaacsim && pip install -e .

set -e

ISAACLAB_SH="${ISAACLAB_SH:-/home/wego/isaacsim/IsaacLab/isaaclab.sh}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPTID="${1:-teacher_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${2:-cuda:0}"
NUM_ENVS="${3:-4096}"
MAX_ITER="${4:-}"   # empty = use config default (30000)
LOAD_RUN="${5:-}"   # path to previous run dir for resume

if [[ ! -f "${ISAACLAB_SH}" ]]; then
    echo "ERROR: isaaclab.sh not found at ${ISAACLAB_SH}"
    echo "  Set ISAACLAB_SH=/path/to/isaaclab.sh and retry."
    exit 1
fi

MAX_ITER_ARG=""
[[ -n "${MAX_ITER}" ]] && MAX_ITER_ARG="--max_iter ${MAX_ITER}"

RESUME_ARG=""
[[ -n "${LOAD_RUN}" ]] && RESUME_ARG="--resume --load_run ${LOAD_RUN}"

"${ISAACLAB_SH}" -p "${SCRIPT_DIR}/train.py" \
    --task g1_23dof_teacher \
    --exptid "${EXPTID}" \
    --device "${DEVICE}" \
    --num_envs "${NUM_ENVS}" \
    ${MAX_ITER_ARG} \
    ${RESUME_ARG} \
    #--headless

echo "Teacher training done. Checkpoint in: ${SCRIPT_DIR}/logs/g1_23dof_teacher/${EXPTID}*"
