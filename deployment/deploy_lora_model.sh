## depoly lora model
#!/bin/bash

MODEL_PATH="/root/autodl-tmp/modelscope_cache/Qwen/Qwen3-8B"
LORA_PATH="/root/autodl-tmp/output/qqr/exp2-ck60"
LOG_FILE="/root/vllm_serve.log"
PORT=6006

# # LoRA 动态热切换配置
MAX_LORAS=4       # GPU 上同时常驻的 LoRA adapter 数量
MAX_CPU_LORAS=8   # CPU 侧缓存的 LoRA 数量(卸载后可快速热加载)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 允许运行时通过 /v1/load_lora_adapter 与 /v1/unload_lora_adapter 动态切换 LoRA。
# 启动时预加载 my_lora=<LORA_PATH>,之后可用同名 unload+load 热切换到其他 checkpoint,
# agent 端 AGENT_MODEL=my_lora 无需改动、无需重启。
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

echo "Starting vLLM server..."
echo "Model: ${MODEL_PATH}"
echo "LoRA: ${LORA_PATH} (name=my_lora, 支持运行时热切换)"
echo "max-loras: ${MAX_LORAS}, max-cpu-loras: ${MAX_CPU_LORAS}"
echo "Log: ${LOG_FILE}"
echo "Port: ${PORT}"

nohup vllm serve "${MODEL_PATH}" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --trust-remote-code \
    --enable-lora \
    --lora-modules "my_lora=${LORA_PATH}" \
    --max-lora-rank 16 \
    --max-loras ${MAX_LORAS} \
    --max-cpu-loras ${MAX_CPU_LORAS} \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes  \
    --host 0.0.0.0 \
    --port ${PORT} \
    > "${LOG_FILE}" 2>&1 &

echo "vLLM server started in background, PID: $!"
echo "Tail log: tail -f ${LOG_FILE}"
echo "切换 LoRA(同名热切换): unload my_lora 后再 load my_lora=<新ckpt路径>,agent 无需重启"


# ## depoly base model
# #!/bin/bash

# MODEL_PATH="/root/autodl-tmp/modelscope_cache/Qwen/Qwen3-8B"
# LOG_FILE="/root/vllm_serve.log"
# PORT=6006

# export CUDA_DEVICE_ORDER=PCI_BUS_ID

# echo "Starting vLLM server..."
# echo "Model: ${MODEL_PATH}"
# echo "Log: ${LOG_FILE}"
# echo "Port: ${PORT}"

# nohup vllm serve "${MODEL_PATH}" \
#   --tensor-parallel-size 2 \
#   --dtype bfloat16 \
#   --max-model-len 32768 \
#   --trust-remote-code \
#   --reasoning-parser qwen3 \
#   --enable-auto-tool-choice \
#   --tool-call-parser hermes  \
#   --host 0.0.0.0 \
#   --port "${PORT}" \
#   > "${LOG_FILE}" 2>&1 &

# echo "vLLM server started in background, PID: $!"
# echo "Tail log: tail -f ${LOG_FILE}"
