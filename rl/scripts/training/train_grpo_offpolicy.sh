#!/usr/bin/env bash
set -euo pipefail

RL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TOOL_SECRETS_FILE=${OPENSEEKER_TOOL_SECRETS_FILE:-"${RL_ROOT}/.secrets/tool_api_keys.env"}
if [[ -r "${TOOL_SECRETS_FILE}" ]]; then
  if [[ "$(stat -c '%a' "${TOOL_SECRETS_FILE}")" != "600" ]]; then
    echo "Refusing secrets file with permissions other than 600: ${TOOL_SECRETS_FILE}" >&2
    exit 1
  fi
  set -a
  # shellcheck source=/dev/null
  source "${TOOL_SECRETS_FILE}"
  set +a
fi

# The answer judge and visit summarizer share the DMX OpenAI-compatible API.
export JUDGE_BASE_URL=${JUDGE_BASE_URL:-${DMX_BASE_URL:-${TOOL_BASE_URL:-https://www.dmxapi.cn/v1}}}
export JUDGE_API_KEY=${JUDGE_API_KEY:-${DMX_API_KEY:-${TOOL_API_KEY:-}}}
export JUDGE_MODEL=${JUDGE_MODEL:-${DMX_MODEL:-${SUMMARY_MODEL:-qwen-plus}}}
export OPENSEEKER_JUDGE_ENABLED=${OPENSEEKER_JUDGE_ENABLED:-1}
export PYTHONPATH="${RL_ROOT}:${RL_ROOT}/vendor/verl${PYTHONPATH:+:${PYTHONPATH}}"

# CUDA 13 wheels place NVRTC and related libraries under site-packages. vLLM's
# cumem allocator is loaded by Ray workers and therefore needs this directory on
# the process library path before Ray starts.
PYTHON_SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_CUDA_WHEEL_LIB=${PYTHON_SITE_PACKAGES}/nvidia/cu13/lib
if [[ -d "${NVIDIA_CUDA_WHEEL_LIB}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_CUDA_WHEEL_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/modelscope_cache/Qwen/Qwen3-30B-A3B-Thinking-2507}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-/root/autodl-tmp/output/qwen3_30b_lora_2gpu_rtx6000_122k_exp6/v0-20260802-172409/checkpoint-380}
DATA_SOURCE=${DATA_SOURCE:-deepdive_qa}
TRAIN_FILE=${TRAIN_FILE:-${RL_ROOT}/data/deepdive_qa/train_rl.parquet}
VAL_FILE=${VAL_FILE:-${RL_ROOT}/data/deepdive_qa/eval_rl.parquet}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
if (( NGPUS_PER_NODE >= 4 )); then
  DEFAULT_ACTOR_SP_SIZE=2
else
  DEFAULT_ACTOR_SP_SIZE=1
fi
ACTOR_SP_SIZE=${ACTOR_SP_SIZE:-${DEFAULT_ACTOR_SP_SIZE}}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
# Four completions per prompt keep rollout cost bounded. The GRPO advantage
# below is mean-centered without per-group std scaling, which is less sensitive
# to the noisier group-std estimate at this group size.
ROLLOUT_N=${ROLLOUT_N:-4}
PPO_EPOCHS=${PPO_EPOCHS:-1}
AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-2}
AGENT_MAX_CONCURRENT_TRAJECTORIES=${AGENT_MAX_CONCURRENT_TRAJECTORIES:-8}
REWARD_WORKERS=${REWARD_WORKERS:-2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}
ROLLOUT_DISABLE_LOG_STATS=${ROLLOUT_DISABLE_LOG_STATS:-True}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.6}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:-20}
ROLLOUT_REPETITION_PENALTY=${ROLLOUT_REPETITION_PENALTY:-1.05}
USE_LIGER=${USE_LIGER:-False}
ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-bfloat16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-131072}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
# The actor's dynamic-batch limit is per sequence-parallel rank. Keep one full
# max-length sequence admissible across the SP group while avoiding packing two
# such sequences into a single actor microbatch.
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + ACTOR_SP_SIZE - 1) / ACTOR_SP_SIZE))}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
PROJECT_NAME=${PROJECT_NAME:-openseeker_agentic_rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_30b_a3b_grpo_lora_offpolicy}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/output/${PROJECT_NAME}/${EXPERIMENT_NAME}}

# Persist one complete Agent trajectory and one levelled runtime log per rollout.
# Set OPENSEEKER_SAVE_TRAJECTORIES=0 to disable only the JSON trajectory files.
export OPENSEEKER_TRAJECTORY_DIR=${OPENSEEKER_TRAJECTORY_DIR:-"${OUTPUT_DIR}/trajectories"}
export OPENSEEKER_AGENT_LOG_DIR=${OPENSEEKER_AGENT_LOG_DIR:-"${OUTPUT_DIR}/agent_logs"}
export OPENSEEKER_AGENT_LOG_LEVEL=${OPENSEEKER_AGENT_LOG_LEVEL:-INFO}
mkdir -p "${OPENSEEKER_TRAJECTORY_DIR}" "${OPENSEEKER_AGENT_LOG_DIR}"

python "${RL_ROOT}/src/preflight.py" \
  --model "${MODEL_PATH}" \
  --adapter "${LORA_ADAPTER_PATH}" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --data-source "${DATA_SOURCE}" \
  --require-search \
  --require-judge

if [[ "${OPENSEEKER_TELEMETRY:-0}" =~ ^(1|true|yes)$ ]]; then
  export OPENSEEKER_TELEMETRY_DIR=${OPENSEEKER_TELEMETRY_DIR:-"${OUTPUT_DIR}/telemetry"}
  mkdir -p "${OPENSEEKER_TELEMETRY_DIR}"
  python "${RL_ROOT}/scripts/diagnostics/runtime_sampler.py" \
    --output "${OPENSEEKER_TELEMETRY_DIR}/resources.jsonl" \
    --interval "${OPENSEEKER_TELEMETRY_INTERVAL_SECONDS:-1}" &
  TELEMETRY_SAMPLER_PID=$!
  stop_telemetry_sampler() {
    kill "${TELEMETRY_SAMPLER_PID}" 2>/dev/null || true
    wait "${TELEMETRY_SAMPLER_PID}" 2>/dev/null || true
  }
  trap stop_telemetry_sampler EXIT INT TERM
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.bypass_mode=True \
  algorithm.rollout_correction.loss_type=ppo_clip \
  algorithm.rollout_correction.rollout_is=null \
  algorithm.rollout_correction.rollout_rs=null \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_max_samples=48 \
  data.val_batch_size=24 \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.continuous_token.enable=True \
  data.continuous_token.model_family=qwen3 \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}" \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj]' \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_liger="${USE_LIGER}" \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
  +actor_rollout_ref.actor.use_rollout_log_probs=True \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ACTOR_SP_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
  +actor_rollout_ref.rollout.repetition_penalty="${ROLLOUT_REPETITION_PENALTY}" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.disable_log_stats="${ROLLOUT_DISABLE_LOG_STATS}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.max_model_len="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_WORKERS}" \
  actor_rollout_ref.rollout.agent.max_concurrent_trajectories="${AGENT_MAX_CONCURRENT_TRAJECTORIES}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=openseeker_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${RL_ROOT}/configs/agent_loops.yaml" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.num_workers="${REWARD_WORKERS}" \
  reward.custom_reward_function.path="${RL_ROOT}/src/reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.v1.trainer_mode=sync \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.val_before_train=False \
  trainer.test_freq=6 \
  trainer.save_freq=2 \
  +actor_rollout_ref.actor.checkpoint.save_lora_only=True \
  trainer.logger='[console,wandb]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  "$@"

