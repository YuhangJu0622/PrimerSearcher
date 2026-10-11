## depoly base model
#!/bin/bash

MODEL_PATH="/root/autodl-tmp/modelscope_cache/Qwen/Qwen3-8B"
LOG_FILE="/root/vllm_serve.log"
PORT=6006

export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Starting vLLM server..."
echo "Model: ${MODEL_PATH}"
echo "Log: ${LOG_FILE}"
echo "Port: ${PORT}"

nohup vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes  \
  --host 0.0.0.0 \
  --port "${PORT}" \
  > "${LOG_FILE}" 2>&1 &

echo "vLLM server started in background, PID: $!"
echo "Tail log: tail -f ${LOG_FILE}"
