<div align="center">
  <table><tr><td><img src="docs/assets/primersearcher-logo.svg" alt="PrimerSearcher logo" width="128" /></td><td>&nbsp;&nbsp;&nbsp;</td><td><h1>PrimerSearcher</h1></td></tr></table>
  <h3>Training Language Models for Autonomous Deep Search</h3>
  <p><a href="README_zh.md">简体中文</a></p>
  <hr />
  <p>
    <a href="https://www.modelscope.cn/datasets/YuhangJu/PrimerSearcher-SFT-Data"><img src="https://img.shields.io/badge/SFT%20Dataset-Long--trajectory%20data-5B8FF9?style=for-the-badge" alt="SFT Dataset" /></a>
    <a href="https://www.modelscope.cn/datasets/YuhangJu/PrimerSearcher-RL-Data"><img src="https://img.shields.io/badge/RL%20Dataset-Agentic%20RL%20data-00A67E?style=for-the-badge" alt="RL Dataset" /></a>
    <a href="https://www.modelscope.cn/models/YuhangJu/PrimerSearcher-SFT-30B-A3B"><img src="https://img.shields.io/badge/SFT%20Model-30B--A3B%20checkpoint-8E5BEF?style=for-the-badge" alt="SFT Model" /></a>
    <a href="https://www.modelscope.cn/models/YuhangJu/PrimerSearcher-RL-30B-A3B"><img src="https://img.shields.io/badge/RL%20Model-30B--A3B%20checkpoint-F97316?style=for-the-badge" alt="RL Model" /></a>
  </p>
</div>

## Introduction

Complex multi-hop questions rarely yield reliable answers from a single search. An effective deep-search agent must understand the question, call retrieval tools, read web evidence, and continually adjust its next action based on tool observations over multiple turns.

PrimerSearcher builds autonomous deep-search agents for these problems. It post-trains Qwen3-30B-A3B-Thinking on high-quality long-trajectory data, further internalizing ReAct-style reasoning, search, web browsing, and evidence-use behavior from an external workflow into the model parameters. This enables longer, more coherent tool use and evidence-driven reasoning.

On the 2026 Tianchi Data+AI in-house evaluation set, the base Qwen3-30B-A3B model achieves 28.0% accuracy. SFT raises this to 42.0%, and RL further raises it to 53.0%. The agent's autonomous research trajectories on complex questions also become substantially longer: average tool-call rounds increase from 2.33 to 52.4, while the maximum rises from 4 to 150.

## 🌟 Highlights

- **Reliable tool use for long-horizon ReAct**

  - **Two-layer fallback for search and web evidence.** For timeouts, non-200 responses, or malformed JSON, Search rotates through all `SERPER_API_KEYS`. If a request succeeds but returns no results, it progressively tries the original query, a version without exact-match quotes, and a simplified query centered on the core entities. This prevents a single key or temporary service failure from terminating a trajectory and reduces zero-result searches caused by overly restrictive wording.

  - **Tiered web-access fallback from readability to availability.** Visit first uses the anonymous Jina Reader for clean Markdown. On failure, it falls back to direct fetching with redirects and a browser user agent, then uses BeautifulSoup to remove navigation, scripts, footers, and other noise while prioritizing article text. Each URL has its own 120-second total timeout. During concurrent multi-URL visits, a failed page produces an actionable failure observation instead of aborting the entire research turn. Web content is capped at 50,000 characters; when the summarizer returns an empty result, it retries with shortened content. JSON, embedded JSON, and unstructured outputs are all normalized into an `Evidence` / `Summary` envelope.

  - **Staged convergence: correct course, disable tools, then force an answer.** An answer produced too early—within the first three rounds and 120 seconds—is intercepted once, requiring the model to review every question constraint and add confirming sources. After a content-safety block, the first two retries require neutral, academic search phrasing; on the third, the context is compressed while preserving the original question and discoveries before a forced answer. When a round, time, or context limit is reached, tools are explicitly disabled and the model has up to three chances to answer from the available evidence using the `<answer>` protocol.

  - **Cross-turn duplicate-call control.** Each call receives a canonical signature that ignores capitalization and whitespace and sorts list-like arguments such as Search query/engine and Visit URLs, so parameter reordering cannot bypass detection. A logically identical call executes at most twice. The third and fourth attempts are intercepted and require the model to inventory constraints, candidate answers, available evidence, and missing evidence before pursuing new entities, sources, or validation methods. On the fifth identical call—or after three consecutively blocked duplicate calls—the system enters forced-answer mode. This turns looping tool use into an explicit decision to replan or converge.

- **Efficient, verifiable post-training: long-context SFT and off-policy GRPO**

  - **122K long-trajectory SFT that internalizes the tool-use protocol.** LoRA fine-tunes `q_proj`, `k_proj`, `v_proj`, and `o_proj` on Qwen3-30B-A3B-Thinking with a 122,880-token context window and `ignore_empty_think` loss handling. The model learns complete ReAct trajectories, tool-call formats, and evidence-grounded final answers rather than merely fitting short answers.

  - **Parameter and memory efficiency for long contexts.** SFT uses BF16, LoRA, ZeRO-3, FlashAttention, sequence packing, padding-free training, gradient checkpointing, and Liger Kernel. It retains the ability to train on 122K-long trajectories while reducing memory waste from activations, padding, and optimizer state, and saves only the required LoRA adapters.

  - **Finite-step, within-group relative GRPO updates.** RL samples four agent tool-interaction trajectories per question by default, then uses GRPO to compute a within-question mean-centered advantage without within-group standard-deviation normalization. Training retains rollout log probabilities and performs a limited number of PPO-clip-style updates, keeping the scale of exploration updates on long tool trajectories controlled.

  - **Verifiable rewards and an auditable training loop.** The reward is `0.8 × answer correctness + 0.2 × final-answer and trajectory-format correctness`. It first uses normalized exact matching and invokes a persistently cached LLM equivalence judge only on failures. If the judge is unavailable or ambiguous, it fails closed and does not award positive reward to uncertain samples. Before training, validation checks the model, LoRA, data, Search, and Judge; it then stores per-sample trajectories, run logs, and optional resource telemetry to make reward, tool, and performance issues diagnosable.

## Quick Start

### 0. Prerequisites

Agent, SFT, and RL use different environments. Set up the Python environment first; SFT and GRPO must run in a Linux GPU training environment. Refer to the training-environment documentation for RL. `rl/requirements.base.lock.txt` is a snapshot of the current machine's environment, not a portable dependency lock file.

Install the Agent dependencies and create the secret configuration from the template:

```bash
pip install -r agent/requirements.txt
cp agent/.env.template agent/.env
```

Edit `agent/.env` and set these variables:

| Variable | Purpose |
| --- | --- |
| `SERPER_API_KEYS` | Serper Search API key; separate multiple keys with commas. |
| `AGENT_BASE_URL` | OpenAI-compatible API endpoint for the primary Agent model. |
| `AGENT_API_KEY` | API key for the primary Agent model. |
| `AGENT_MODEL` | Primary Agent model name. |
| `DMX_API_KEY` | API key for the DMX model, used for web-evidence summarization and the optional answer judge. |

The template also supplies defaults for the DMX endpoint and summarizer/judge models, which can be adjusted for your service.

### 1. Run the Agent

Every input JSONL row must contain `id` and `question`. Run the Agent in batch with:

```bash
python agent/scripts/run_agent.py \
  --data agent/data/data_stage_1.jsonl \
  --output-dir agent/runs \
  --run-name baseline
```

Results, logs, and per-question trajectories are written to `agent/runs/logs/baseline/`. Re-running with the same `--output-dir` and `--run-name` reads existing results and skips completed questions, thereby resuming the same run.

### 2. Evaluate Agent Results

Offline evaluation reports exact-match results. With `--use-llm`, answers that fail exact matching are semantically judged as well, and both accuracies are reported.

```bash
python agent/scripts/score.py \
  --results agent/runs/logs/baseline/result.jsonl \
  --ref agent/data/data_stage_1.jsonl \
  --use-llm
```

The reference JSONL must contain `id` and `answer`. Semantic judging requires `DMX_API_KEY` in `agent/.env`.

### 3. SFT Training

**Linux/GPU environment only.** The default script trains Qwen3-30B-A3B with LoRA and a 122K context window. It requires Swift, DeepSpeed, and two GPUs. Before running it, update the base-model path, training/validation data paths, and output directory in the script.

```bash
bash sft/sft_qwen3_30b_lora_2gpu_rtx6000_122k.sh
```

### 4. Off-Policy GRPO Training

**Linux/GPU environment only.** First configure private tool credentials:

```bash
cp rl/configs/secrets.example.env rl/.secrets/tool_api_keys.env
chmod 600 rl/.secrets/tool_api_keys.env
```

Fill `rl/.secrets/tool_api_keys.env` with the credentials required by Search, web access, and Judge. Then set or update `MODEL_PATH`, `LORA_ADAPTER_PATH`, `TRAIN_FILE`, `VAL_FILE`, and the GPU count before starting training:

```bash
bash rl/scripts/training/train_grpo_offpolicy.sh
```

The default configuration has been validated on four 96 GiB GPUs. The script runs preflight checks for the base model, SFT adapter, training/validation data, Search, and Judge before starting training.

## Open-source Release

PrimerSearcher releases the complete path from Agent execution through SFT to off-policy GRPO post-training:

1. **Complete Agent source code** — ReAct control loop, Search and Visit tools, web-evidence summarization, runtime fault tolerance, batch execution, result persistence, and offline evaluation scripts.
   [Browse the Agent code](agent/)

2. **SFT dataset** — For learning long-horizon ReAct behavior, the tool-call protocol, and evidence-grounded answers.
   [ModelScope dataset](https://www.modelscope.cn/datasets/YuhangJu/PrimerSearcher-SFT-Data)

3. **SFT training recipe** — A 122K long-context LoRA training script for Qwen3-30B-A3B, plus ZeRO-3, FlashAttention, packing, padding-free, and related configuration.
   [Browse the SFT recipe](sft/)

4. **SFT model** — The supervised-fine-tuned PrimerSearcher-SFT-30B-A3B checkpoint.
   [ModelScope model](https://www.modelscope.cn/models/YuhangJu/PrimerSearcher-SFT-30B-A3B)

5. **RL dataset** — GRPO training and validation data for agent tool interaction and outcome verification.
   [ModelScope dataset](https://www.modelscope.cn/datasets/YuhangJu/PrimerSearcher-RL-Data)

6. **RL runtime and training configuration** — GRPO launch scripts, preflight checks, tool runtime, reward function, secret templates, training configuration, trajectory records, and diagnostic tools. The environment snapshot is only for reproducing the experimental machine and is not a general dependency lock.
   [Browse the RL training stack](rl/)

7. **RL model** — The PrimerSearcher-RL-30B-A3B checkpoint after off-policy GRPO post-training.
   [ModelScope model](https://www.modelscope.cn/models/YuhangJu/PrimerSearcher-RL-30B-A3B)
