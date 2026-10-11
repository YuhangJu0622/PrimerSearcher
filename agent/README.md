# Research Agent — 多跳推理问答智能体

> "寻找AI全能王"阿里云 Data+AI 工程师全球大奖赛（高校赛道）**第一名**作品

## 一、项目简介

本项目是面向 PAI-LangStudio 平台的 **Research Agent**，能够接收自然语言问题，通过自主规划子任务、调用搜索引擎和网页访问工具、整合多源证据，最终生成精确答案。该 Agent 在竞赛中需要回答 100 道涵盖事实验证、多步推理、单位换算等类型的复杂问题，仅依赖联网搜索完成作答。

核心设计思路参考了通义团队开源的 **DeepResearch** 项目（ReAct 范式），并在 Prompt 工程、搜索策略、答案归一化等方面做了针对性优化。

## 二、系统架构

```
用户问题 POST /
    |
agent.py (FastAPI 入口, dotenv 加载环境变量)
    |
agent_loop.py :: react_agent(question)
    |
+-- System Prompt (prompts.py) ----------------------------------+
|   ReAct 循环 (最多 40 轮, 超时与 token 上限保护)                 |
|   +-- 调用 AGENT_MODEL (OpenAI 兼容 API)                        |
|   |   统一设置 enable_thinking=True                             |
|   +-- 检测 <answer> -> 返回最终答案                              |
|   +-- 检测 <tool_call> -> 解析并执行工具                         |
|   |   +-- search -> tools_search.py :: batch_search              |
|   |   |   +-- Google/Serper API (英文/国际内容)                  |
|   |   |   +-- 阿里 IQS API (中文内容)                            |
|   |   +-- visit -> tools_visit.py :: visit_pages                 |
|   |       +-- Jina Reader API / httpx 直接抓取                   |
|   |       +-- BeautifulSoup 提取正文                              |
|   |       +-- qwen-plus LLM 摘要 (EXTRACTOR_PROMPT)             |
|   +-- token 上限 (122880) -> 强制生成答案                        |
|   +-- 内容安全过滤 -> 自动重试/换角度/强制回答                    |
+------------------------------------------------------------------+
    |
返回 {"answer": "纯文本答案"}
```

## 三、代码结构

```
src/
├── agent.py            # FastAPI 入口，提供 POST / 和 POST /stream 接口
├── agent_loop.py       # ReAct Agent 核心循环（推理、工具调用、答案提取）
├── prompts.py          # 主 Agent System Prompt、网页摘要 Prompt
├── tools_search.py     # 搜索工具（Google/Serper + 阿里 IQS 双引擎）
├── tools_visit.py      # 网页访问工具（Jina Reader + httpx + LLM 摘要）
├── eval.py             # 批量评测脚本（适配 question.jsonl）
├── test_eval.py        # 测试评测脚本（适配 test_data.jsonl）
├── .env                # API 密钥配置
├── question.jsonl      # 第一阶段 100 道验证题
├── submit_results.jsonl# 参考答案
├── test_data.jsonl     # 测试数据集（含答案）
├── agui.py             # AG-UI Protocol 事件流（平台原有）
├── skills.py           # Agent Skills 模块（平台原有）
├── skills/             # 技能目录（平台原有）
└── README.md           # 本文档
```

## 四、关键设计与优化

### 4.1 ReAct 推理循环

采用 **Think → Act → Observe** 的 ReAct 范式：

- **Think**：使用 `<think>` 标签进行推理，分解问题、规划搜索、验证约束
- **Act**：通过 `<tool_call>` 调用搜索或网页访问工具
- **Observe**：解析 `<tool_response>` 中的工具返回结果
- **Answer**：所有约束验证通过后，通过 `<answer>` 输出最终答案

模型开启了 `enable_thinking=True`（Qwen 深度思考模式），推理内容会被自动包裹在 `<think>` 标签中。

### 4.2 Prompt 设计

系统只保留两份静态提示词：

- `SYSTEM_PROMPT`：主 Agent 使用，包含 `search`、`visit` 工具定义以及 `<think>`、`<tool_call>`、`<answer>` 输出协议。用户问题不再拼接额外模板，而是直接作为 user message 传入。
- `EXTRACTOR_PROMPT`：由 `visit` 工具调用，用于从网页正文中提取证据并生成结构化摘要。

主 Agent 不再根据模型名称或环境变量选择提示词，也不再维护 normal、thinking、SFT 三套模板。

#### 4.2.1 Extractor Prompt（网页摘要提取）

Extractor Prompt 由 `visit` 工具内部调用，使用 **qwen-plus** 模型对网页内容进行结构化摘要。设计为 3 步提取流程：

1. **rational**（定位）：在网页内容中定位与用户目标直接相关的段落和数据
2. **evidence**（提取）：提取最相关的信息，保留完整原始上下文，可多段输出
3. **summary**（总结）：组织为简洁段落，判断信息对目标的贡献度

输出为 JSON 格式（含 `rational`、`evidence`、`summary` 三个字段），结构化输出的好处是：便于后续 Agent 循环中精确引用证据，同时控制注入到上下文中的信息量。

### 4.3 运行时动态 Prompt（ReAct 循环注入）

除了静态提示词，`agent_loop.py` 中还保留了必要的条件性动态 Prompt，在特定异常或终止条件下触发：

| 动态 Prompt | 触发条件 | 注入内容 | 源码位置 |
|------------|---------|---------|---------|
| **强制回答** | 超时或 Token 超限 | 要求模型根据已收集信息立即生成最终答案 | `_force_answer` |
| **早期回答拦截** | 前 3 轮且耗时 <120 秒时模型给出 `<answer>` | 要求重新检查所有约束并搜索额外确认来源 | `react_agent` |
| **内容安全过滤重定向** | API 返回 `data_inspection_failed` 错误 | 前 2 次更换中性搜索措辞，第 3 次清理上下文后强制回答 | `react_agent` |
| **无动作提示（Nudge）** | 输出中既无 `<tool_call>` 也无 `<answer>` | 提示模型调用工具或给出答案 | `react_agent` |

这些动态 Prompt 构成了一个"行为护栏"系统：既防止模型过早收敛（早期拦截），又确保在异常情况下（超时、风控、格式错误）仍能产出合理答案。与 4.6 节的鲁棒性机制互补，4.6 节侧重机制概述，本节提供 Prompt 级别的实现细节。

### 4.4 双引擎搜索策略

- **Google/Serper**：英文及国际内容搜索，支持多 API Key 自动轮换
- **阿里 IQS**：中文内容搜索，适合中国特定主题
- 模型可在单次搜索中混合使用两个引擎
- 查询自动简化：过长查询会被截断以提高命中率

### 4.5 网页访问与摘要

- **Jina Reader API**：优先使用，将网页转为干净的 Markdown
- **httpx + BeautifulSoup**：Jina 失败时的降级方案，直接抓取并提取正文
- **LLM 摘要**：使用 qwen-plus 对网页内容进行结构化摘要（rational/evidence/summary）

### 4.6 鲁棒性机制

- **超时保护**：超过 `TIMEOUT_SECONDS` 后强制生成答案
- **Token 上限保护**：上下文超过 122880 token 时强制生成答案
- **早期回答拦截**：前 3 轮且耗时不足 2 分钟的答案会被要求二次验证
- **内容安全过滤处理**：遇到风控拦截时自动换角度重试，最多 2 次后强制回答
- **答案归一化**：去除冗余前缀、引号包裹、多余标点

### 4.7 答案语言与格式

- 默认与问题语言一致（中文问题 → 中文答案，英文问题 → 英文答案）
- 外国专有名词使用最权威的标准译名（如"海尔-波普彗星"而非"Hale-Bopp"）
- 当问题上下文隐含特定语言要求（如"英文全称""[Name] Limited"格式）时，遵循该要求
- 回答前强制执行语言与格式检查，确保精确字符匹配

## 五、环境依赖

### 5.1 Python 依赖

```bash
pip install -r requirements.txt
```

### 5.2 环境变量（.env）

| 变量名 | 说明 |
|--------|------|
| `DASHSCOPE_API_KEY` | 阿里百炼 API Key，用于调用 Qwen 模型 |
| `IQS_API_KEY` | 阿里 IQS 搜索 API Key |
| `SERPER_API_KEYS` | Google Serper API Key（逗号分隔，支持多 Key 轮换） |
| `JINA_API_KEYS` | Jina Reader API Key（可选） |
| `AGENT_MODEL` | 主推理模型，默认 `qwen3.5-plus` |

### 5.3 模型使用

| 模型 | 用途 | 说明 |
|------|------|------|
| `qwen3.5-plus` | 主推理模型 | ReAct 循环中的推理与决策 |
| `qwen-plus` | 网页摘要模型 | 网页内容结构化摘要提取 |

所有模型均通过阿里云百炼 DashScope OpenAI 兼容接口调用，未进行任何微调。

## 六、关键参数

| 参数 | 值 | 位置 | 说明 |
|------|-----|------|------|
| AGENT_MODEL | `qwen3.5-plus` | agent_loop.py | 主推理模型 |
| MAX_ROUNDS | 40 | agent_loop.py | 最大 ReAct 轮数 |
| TIMEOUT_SECONDS | 648 | agent_loop.py | 超时时间（秒） |
| MAX_TOKENS_ESTIMATE | 122880 | agent_loop.py | token 上限 |
| MAX_TOOL_RESULT_CHARS | 15000 | agent_loop.py | 单个工具结果最大字符数 |
| temperature | 0.6 | agent_loop.py | LLM 生成温度 |

## 七、复现步骤

### 7.1 环境准备

1. 在 PAI-LangStudio 中创建应用（代码模式）
2. 配置 `.env` 文件，填入必要的 API Key
3. 安装依赖：`pip install -r requirements.txt`

### 7.2 本地调试

```bash
# 进入项目目录
cd /mnt/langstudio/flow/flow-5neot1jq8ikej6eew3/src

# 启动服务
python3 -m uvicorn agent:app --host 0.0.0.0 --port 8001

# 单题测试
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{"question": "法国的首都是哪里？"}'

# 预期返回
# {"answer":"巴黎"}
```

### 7.3 批量评测

```bash
# 评测验证集（question.jsonl），运行第 0-9 题
python3 eval.py 0 10 my_results.jsonl

# 评测测试集（test_data.jsonl），运行全部
python3 test_eval.py
```

### 7.4 部署为 EAS 服务

1. 在 LangStudio 中点击「服务部署」
2. 部署成功后获取 Endpoint 和 Token
3. 调用示例：

```bash
curl -X POST "https://<your-endpoint>/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{"question": "一位物理学领域的学者为一种经典棋盘游戏设计的评分系统..."}'
```

响应格式：

```json
{"answer": "魂武者"}
```

## 八、API 接口

### POST /（评测主接口）

- **输入**：`{"question": "自然语言问题"}`
- **输出**：`{"answer": "纯文本答案"}`
- **超时**：单请求最长 10 分钟

### POST /stream（流式接口）

- **输入**：同上
- **输出**：SSE 格式，`data: {"answer": "..."}`

## 九、参考与致谢

- [DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)（通义团队开源项目）：本项目的 ReAct 架构和工具协议设计参考了该项目
- 搜索引擎：Google Serper API、阿里云 IQS
- 网页解析：Jina Reader API
- 模型服务：阿里云百炼 DashScope（Qwen 系列模型）

## 十、License

本项目基于 [MIT License](LICENSE) 开源。
