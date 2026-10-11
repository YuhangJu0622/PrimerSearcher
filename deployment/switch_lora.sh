#!/bin/bash
# 热切换 LoRA 权重(同名 my_lora 策略)
# 用法: bash switch_lora.sh <新checkpoint目录路径>
# 例:   bash switch_lora.sh /root/autodl-tmp/output/qqr/exp2/exp2-ck140
#
# 机制: unload my_lora -> 用新路径 load 同名 my_lora
# agent 端 AGENT_MODEL=my_lora 不变,无需重启。
# 注意: unload 到 load 之间有几秒空窗,请在 agent 空闲时执行。

LORA_NAME="my_lora"
PORT=6006
BASE_URL="http://localhost:${PORT}"
NEW_PATH="${1:-}"

if [ -z "$NEW_PATH" ]; then
    echo "用法: bash $0 <新checkpoint目录路径>"
    exit 1
fi

if [ ! -d "$NEW_PATH" ]; then
    echo "错误: 路径不存在或不是目录: ${NEW_PATH}"
    exit 1
fi

echo "=== 切换前已加载 adapter ==="
curl -s "${BASE_URL}/v1/models" | jq -r '.data[].id' 2>/dev/null || curl -s "${BASE_URL}/v1/models"
echo ""

echo ">>> 卸载 ${LORA_NAME}"
curl -s -X POST "${BASE_URL}/v1/unload_lora_adapter" \
    -H "Content-Type: application/json" \
    -d "{\"lora_name\":\"${LORA_NAME}\"}"
echo ""

echo ">>> 加载 ${LORA_NAME}  <-  ${NEW_PATH}"
curl -s -X POST "${BASE_URL}/v1/load_lora_adapter" \
    -H "Content-Type: application/json" \
    -d "{\"lora_name\":\"${LORA_NAME}\",\"lora_path\":\"${NEW_PATH}\"}"
echo ""

echo "=== 切换后已加载 adapter ==="
curl -s "${BASE_URL}/v1/models" | jq -r '.data[].id' 2>/dev/null || curl -s "${BASE_URL}/v1/models"
echo ""
echo "完成。agent 继续 model=my_lora 即走新权重。"
