#!/usr/bin/env bash
# Resume off-policy GRPO training in background.
set -euo pipefail

RL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SECRETS_FILE=${OPENSEEKER_TOOL_SECRETS_FILE:-"${RL_ROOT}/.secrets/tool_api_keys.env"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
PROJECT_NAME=${PROJECT_NAME:-openseeker_agentic_rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_30b_a3b_grpo_lora_offpolicy}
OUTPUT_DIR=${OUTPUT_DIR:-"/root/autodl-tmp/output/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-"${OUTPUT_DIR}/global_step_4"}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-48}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-24}
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

if [[ "${RESUME_FROM_PATH}" != *"/global_step_"* ]]; then
  echo "RESUME_FROM_PATH must point to a global_step_* checkpoint: ${RESUME_FROM_PATH}" >&2
  exit 1
fi
if [[ ! -d "${RESUME_FROM_PATH}/actor" || ! -f "${RESUME_FROM_PATH}/data.pt" ]]; then
  echo "Incomplete resume checkpoint: expected actor/ and data.pt under ${RESUME_FROM_PATH}" >&2
  exit 1
fi

if [[ ! -r "${SECRETS_FILE}" ]]; then
  echo "Missing secrets file: ${SECRETS_FILE}" >&2
  exit 1
fi
if [[ "$(stat -c '%a' "${SECRETS_FILE}")" != "600" ]]; then
  echo "Refusing secrets file with permissions other than 600: ${SECRETS_FILE}" >&2
  exit 1
fi
set -a
# shellcheck source=/dev/null
source "${SECRETS_FILE}"
set +a
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is missing from ${SECRETS_FILE}" >&2
  exit 1
fi
if [[ -z "${OPENSEEKER_AUTODL_TOKEN:-}" ]]; then
  echo "OPENSEEKER_AUTODL_TOKEN is missing from ${SECRETS_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/trajectories" "${OUTPUT_DIR}/agent_logs"
TRAIN_PID_FILE="${OUTPUT_DIR}/train.pid"
TRAIN_STDOUT="${OUTPUT_DIR}/train.resume.launch.log"

if [[ "${DRY_RUN}" == true ]]; then
  sed -n 'p' <<EOF
Dry-run only; no process started.
output_dir=${OUTPUT_DIR}
resume_mode=resume_path
resume_from_path=${RESUME_FROM_PATH}
val_max_samples=${VAL_MAX_SAMPLES}
val_batch_size=${VAL_BATCH_SIZE}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
ngpus_per_node=${NGPUS_PER_NODE}
train_log=${TRAIN_STDOUT}
EOF
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; run this script on the GPU training host." >&2
  exit 1
fi
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | sed '/^$/d' | wc -l)
if (( AVAILABLE_GPUS < NGPUS_PER_NODE )); then
  echo "Need ${NGPUS_PER_NODE} visible GPUs but found ${AVAILABLE_GPUS}." >&2
  exit 1
fi

if [[ -f "${TRAIN_PID_FILE}" ]] && kill -0 "$(<"${TRAIN_PID_FILE}")" 2>/dev/null; then
  echo "Training is already running (pid $(<"${TRAIN_PID_FILE}"))." >&2
  exit 1
fi

nohup env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NGPUS_PER_NODE="${NGPUS_PER_NODE}" \
  PROJECT_NAME="${PROJECT_NAME}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  OPENSEEKER_TRAJECTORY_DIR="${OUTPUT_DIR}/trajectories" \
  OPENSEEKER_AGENT_LOG_DIR="${OUTPUT_DIR}/agent_logs" \
  bash "${RL_ROOT}/scripts/training/train_grpo_offpolicy.sh" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  > "${TRAIN_STDOUT}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${TRAIN_PID_FILE}"
echo "Resume training started: pid=${TRAIN_PID}, log=${TRAIN_STDOUT}"
echo "Resuming from: ${RESUME_FROM_PATH}"
echo "W&B project=${PROJECT_NAME}, experiment=${EXPERIMENT_NAME}"
