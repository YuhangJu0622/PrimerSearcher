<div align="center">
  <table>
    <tr>
      <td><img src="docs/assets/primersearcher-logo.svg" alt="PrimerSearcher logo" width="128" /></td>
      <td>&nbsp;&nbsp;&nbsp;</td>
      <td><h1>PrimerSearcher</h1></td>
    </tr>
  </table>

  <h3>Training Language Models for Autonomous Deep Search</h3>

  <hr />

  <p>
    <a href="https://www.modelscope.cn/datasets/YuhangJu/PrimerSearcher-SFT-Data"><img src="https://img.shields.io/badge/SFT%20Dataset-Long--trajectory%20data-5B8FF9?style=for-the-badge" alt="SFT Dataset" /></a>
    <a href="https://www.modelscope.cn/datasets/YuhangJu/PrimerSearcher-RL-Data"><img src="https://img.shields.io/badge/RL%20Dataset-Agentic%20RL%20data-00A67E?style=for-the-badge" alt="RL Dataset" /></a>
    <a href="https://www.modelscope.cn/models/YuhangJu/PrimerSearcher-SFT-30B-A3B"><img src="https://img.shields.io/badge/SFT%20Model-30B--A3B%20checkpoint-8E5BEF?style=for-the-badge" alt="SFT Model" /></a>
    <a href="https://www.modelscope.cn/models/YuhangJu/PrimerSearcher-RL-30B-A3B"><img src="https://img.shields.io/badge/RL%20Model-30B--A3B%20checkpoint-F97316?style=for-the-badge" alt="RL Model" /></a>
  </p>
</div>

## Introduction

复杂的多跳问题通常无法通过一次检索得到可靠答案。一个有效的深度检索 Agent 需要在多轮交互中理解问题、调用检索工具、阅读网页证据，并依据工具观察结果持续调整下一步行动。

PrimerSearcher 面向这类复杂问题构建自主深度检索 Agent。项目以高质量长轨迹数据为基础，对 Qwen3-30B-A3B-Thinking 进行后训练，将 ReAct 式的思考、搜索、网页访问与证据利用行为从外部工作流进一步内化到模型参数中，使模型能够完成更长程、更连贯的工具使用与证据驱动推理。

在 2026 天池 Data+AI 自建评测集上，基础 Qwen3-30B-A3B 模型的正确率为 28.0%；经过 SFT 微调后提升至 42.0%，再经 RL 微调后达到 53.0%。模型在复杂问题中可维持的自主检索轨迹也显著延长：平均工具调用轮次由 2.33 增至 52.4，最大轮次由 4 提升至 150。

## 🌟 Highlights

- **稳定的工具调用：面向长轨迹 ReAct**

  - **检索与网页证据的双层降级。** Search 对超时、非 200 响应或异常 JSON 会轮换全部 `SERPER_API_KEYS`；若请求成功但没有结果，则依次尝试原始 query、去除精确匹配引号、提取核心实体后的简化 query。这样既避免单个 Key 或短时服务异常终止轨迹，也减少过度限定检索词造成的零结果。

  - **网页访问从可读性到可用性的分级回退。** Visit 优先通过匿名 Jina Reader 获取干净 Markdown；失败后回退到带重定向与浏览器 User-Agent 的直接抓取，并用 BeautifulSoup 去除导航、脚本、页脚等噪声，优先提取文章正文。每个 URL 有独立的 120 秒总超时；多 URL 并发访问时单页失败会返回可行动的失败观察，而不会中断整轮研究。网页正文最多保留 50,000 字符；摘要模型输出为空时会缩短正文重试，并兼容 JSON、嵌入式 JSON 和非结构化输出，最终统一为 `Evidence` / `Summary` 证据包络。

  - **分层收敛：先纠偏，再停止工具，最后强制作答。** 前 3 轮且 120 秒内的过早答案会被拦截一次，要求逐项复核问题约束并补充确认来源；内容安全拦截时，前两次要求改用中性、学术化检索表达，第三次则压缩上下文、保留原问题和已发现结论后进入强制回答。达到轮次、时间或上下文上限时，工具被显式禁用，模型最多有 3 次机会按 `<answer>` 协议基于已有证据作答。

  - **跨轮次重复调用治理。** 每次工具调用都会生成规范化签名：忽略大小写与空白差异，并对 Search 的 query/engine、Visit 的 URL 等列表排序，因此参数重排不会绕过检测。同一逻辑调用最多实际执行 2 次；第 3、4 次会被拦截并要求模型盘点约束、候选答案、已有证据和缺失证据，转向新的实体、来源或验证方法；第 5 次相同调用，或连续 3 次重复调用被拦截后，系统直接进入强制回答阶段。这将循环式工具使用转化为显式的“重新规划或收敛”决策。

- **高效、可验证的后训练：长上下文 SFT 与 Off-Policy GRPO**

  - **122K 长轨迹 SFT：将工具使用协议内化为模型行为。** 在 Qwen3-30B-A3B-Thinking 上以 LoRA 微调 `q_proj`、`k_proj`、`v_proj`、`o_proj`，支持 122,880 token 上下文与 `ignore_empty_think` 损失处理，使模型学习完整的 ReAct 轨迹、工具调用格式和基于证据的最终回答，而非只拟合短答案。

  - **面向长上下文的参数与显存效率。** SFT 使用 BF16、LoRA、ZeRO-3、FlashAttention、sequence packing、padding-free、梯度检查点和 Liger kernel；在保留 122K 长轨迹训练能力的同时，减少激活、填充与优化器状态带来的显存浪费，并仅保存必要的 LoRA adapter。

  - **有限步、组内相对的 GRPO 更新。** RL 每个问题默认采样 4 条 Agent 工具交互轨迹，以 GRPO 计算同题轨迹的组内均值中心化优势，不使用组内标准差归一化；训练保留 rollout log-prob，并以 PPO-clip 形式进行有限轮更新，使长轨迹工具探索的更新幅度更可控。

  - **可验证奖励与可审计训练闭环。** Reward 按 `0.8 × 答案正确性 + 0.2 × 最终答案及轨迹格式正确性` 计算：先做归一化精确匹配，仅在失败时调用并持久缓存 LLM 等价性裁判；裁判不可用或输出不明确时 fail-closed，不给不确定样本正奖励。训练启动前会验证模型、LoRA、数据、Search 与 Judge，并保存逐条轨迹、运行日志和可选资源遥测，便于定位奖励、工具或性能问题。

## Quick Start

### 0. Prerequisites

Agent、SFT 与 RL 使用不同的运行环境。先准备 Python 环境；SFT 和 GRPO 必须在 Linux GPU 训练环境中执行。RL 环境请参考训练环境文档；`rl/requirements.base.lock.txt` 是当前机器的环境快照，并非可移植的依赖锁文件。

安装 Agent 依赖并从模板创建密钥配置：

```bash
pip install -r agent/requirements.txt
cp agent/.env.template agent/.env
```

编辑 `agent/.env`，填写以下变量：

| 变量 | 用途 |
| --- | --- |
| `SERPER_API_KEYS` | Serper 搜索 API Key；多个 Key 以逗号分隔。 |
| `AGENT_BASE_URL` | 主 Agent 模型的 OpenAI 兼容 API 地址。 |
| `AGENT_API_KEY` | 主 Agent 模型的 API Key。 |
| `AGENT_MODEL` | 主 Agent 模型名称。 |
| `DMX_API_KEY` | DMX 模型的 API Key，用于网页证据摘要，以及可选的答案裁判。 |

模板还提供了 DMX 地址和摘要/裁判模型的默认值，可按服务配置调整。

### 1. Run the Agent

输入 JSONL 的每一行必须包含 `id` 和 `question`。以下命令批量运行 Agent：

```bash
python agent/scripts/run_agent.py \
  --data agent/data/data_stage_1.jsonl \
  --output-dir agent/runs \
  --run-name baseline
```

结果、日志和每题轨迹将保存至 `agent/runs/logs/baseline/`。再次使用相同的 `--output-dir` 与 `--run-name` 运行时，脚本会读取已有结果并跳过已完成题目，从而恢复同一 run。

### 2. Evaluate Agent Results

离线评测会报告精确匹配结果；启用 `--use-llm` 后，会对精确匹配失败的答案进行语义判定，并同时报告两种准确率。

```bash
python agent/scripts/score.py \
  --results agent/runs/logs/baseline/result.jsonl \
  --ref agent/data/data_stage_1.jsonl \
  --use-llm
```

参考 JSONL 应包含 `id` 和 `answer`。启用语义判定时，需要在 `agent/.env` 中配置 `DMX_API_KEY`。

### 3. SFT Training

**仅限 Linux/GPU 环境。** 默认脚本使用 Qwen3-30B-A3B、LoRA 和 122K 上下文训练，需要安装 Swift、DeepSpeed，并提供 2 张 GPU。执行前，请修改脚本中的基础模型路径、训练/验证数据路径和输出目录。

```bash
bash sft/sft_qwen3_30b_lora_2gpu_rtx6000_122k.sh
```

### 4. Off-Policy GRPO Training

**仅限 Linux/GPU 环境。** 先配置私有工具密钥：

```bash
cp rl/configs/secrets.example.env rl/.secrets/tool_api_keys.env
chmod 600 rl/.secrets/tool_api_keys.env
```

在 `rl/.secrets/tool_api_keys.env` 中填入 Search、网页访问和 Judge 所需的密钥；随后设置或修改 `MODEL_PATH`、`LORA_ADAPTER_PATH`、`TRAIN_FILE`、`VAL_FILE` 和 GPU 数量，再启动训练：

```bash
bash rl/scripts/training/train_grpo_offpolicy.sh
```

默认配置已在 4 张 96 GiB GPU 上验证。脚本会先执行 preflight，确认基础模型、SFT adapter、训练/验证数据、Search 和 Judge 均可用，然后才启动训练。
## Open-source Release

PrimerSearcher 开源覆盖从 Agent 运行、SFT 到 Off-Policy GRPO 后训练的完整链路：

1. **Agent 完整代码** — 包含 ReAct 主循环、Search 与 Visit 工具、网页证据摘要、运行时容错、批量运行、结果保存与离线评测脚本。  
   [查看 Agent 代码](agent/)

2. **SFT 数据集** — 用于学习长轨迹 ReAct、工具调用协议和证据驱动回答。  
   [ModelScope 数据集](https://www.modelscope.cn/datasets/YuhangJu/PrimerSearcher-SFT-Data)

3. **SFT 训练配方** — 包含 Qwen3-30B-A3B 的 122K 长上下文 LoRA 训练脚本，以及 ZeRO-3、FlashAttention、packing、padding-free 等配置。  
   [查看 SFT 训练配方](sft/)

4. **SFT 模型** — 完成监督微调后的 PrimerSearcher-SFT-30B-A3B checkpoint。  
   [ModelScope 模型](https://www.modelscope.cn/models/YuhangJu/PrimerSearcher-SFT-30B-A3B)

5. **RL 数据集** — 用于 Agent 工具交互与结果验证的 GRPO 训练、验证数据。  
   [ModelScope 数据集](https://www.modelscope.cn/datasets/YuhangJu/PrimerSearcher-RL-Data)

6. **RL 运行环境与训练配置** — 包含 GRPO 启动脚本、预检、工具运行时、奖励函数、密钥模板、训练配置、轨迹记录和诊断工具。环境快照仅用于复现实验机器，不作为通用依赖锁。  
   [查看 RL 训练栈](rl/)

7. **RL 模型** — 经 Off-Policy GRPO 后训练的 PrimerSearcher-RL-30B-A3B checkpoint。  
   [ModelScope 模型](https://www.modelscope.cn/models/YuhangJu/PrimerSearcher-RL-30B-A3B)
