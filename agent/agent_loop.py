import json
import os
import re
import time
from typing import Callable, Optional

import json5
import tiktoken
from openai import AsyncOpenAI

from agent.logger_config import get_logger
from agent.prompts import SYSTEM_PROMPT
from agent.tools.tools_search import batch_search
from agent.tools.tools_visit import visit_pages

logger = get_logger()

TrajectoryCallback = Callable[[list], None]


AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen3.5-plus")

_IS_NODESK = "nodesk" in AGENT_BASE_URL.lower()

MAX_ROUNDS = 150
TIMEOUT_SECONDS = 540*12  # 9 minutes (leave 1 min buffer for 10 min limit)
MAX_TOKENS_ESTIMATE = 122880  # “qwen2.5-7b”的上下文长度32k，"qwen3.5-plus"的上下文长度1M
MAX_TOOL_RESULT_CHARS = 15000  # Max chars per tool result to keep context manageable
MAX_IDENTICAL_TOOL_EXECUTIONS = 2
MAX_IDENTICAL_TOOL_ATTEMPTS = 5
MAX_CONSECUTIVE_BLOCKED_DUPLICATES = 3
TOOL_SIGNATURE_LOG_CHARS = 300

_client = AsyncOpenAI(
    base_url=AGENT_BASE_URL,
    api_key=AGENT_API_KEY,
)


class _TrackedMessages(list):
    """Conversation list that checkpoints itself after every append."""

    def __init__(
        self,
        initial_messages: list,
        callback: Optional[TrajectoryCallback] = None,
    ):
        super().__init__(initial_messages)
        self._callback = callback
        self._checkpoint()

    def _checkpoint(self) -> None:
        if self._callback is None:
            return
        try:
            self._callback(list(self))
        except Exception as e:
            logger.warning("Failed to save trajectory checkpoint: %s", e)

    def append(self, message: dict) -> None:
        super().append(message)
        self._checkpoint()


# Initialize tiktoken encoder for accurate token counting
try:
    _tokenizer = tiktoken.get_encoding("cl100k_base")
    logger.debug("Using tiktoken for accurate token counting")
except Exception as e:
    logger.warning("Failed to load tiktoken encoder: %s", e)
    _tokenizer = None


def _extract_between(text: str, start_tag: str, end_tag: str) -> str:
    """Extract content between two tags."""
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return ""
    start_idx += len(start_tag)
    end_idx = text.find(end_tag, start_idx)
    if end_idx == -1:
        return text[start_idx:].strip()
    return text[start_idx:end_idx].strip()


def _post_think(content: str) -> str:
    """Return the portion of an assistant turn after the final </think> tag.

    The model may mention or quote <answer>/<tool_call> inside its <think>
    reasoning (e.g. restating the format instruction), which must not be
    parsed as an actual answer or tool call. Only tags emitted after the
    reasoning block closes are authoritative. Falls back to the whole content
    when there is no </think> tag.
    """
    if "</think>" in content:
        return content[content.rfind("</think>") + len("</think>"):]
    return content


def _estimate_tokens(messages: list) -> int:
    """Accurate token counting using tiktoken (fallback to rough estimation)."""
    if _tokenizer is not None:
        try:
            total_tokens = 0
            for msg in messages:
                content = msg.get("content", "")
                total_tokens += len(_tokenizer.encode(content))
            return total_tokens
        except Exception as e:
            logger.warning("Tokenizer error: %s, falling back to char estimation", e)

    # Fallback: rough estimation
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return int(total_chars / 1.5)


def _truncate_tool_result(result: str) -> str:
    """Truncate tool results to keep context manageable."""
    if len(result) > MAX_TOOL_RESULT_CHARS:
        return result[:MAX_TOOL_RESULT_CHARS] + "\n\n[... result truncated due to length ...]"
    return result


def _normalize_assistant_output(content: str) -> str:
    """Match the assistant turn endings used by the SFT conversations."""
    normalized = content.strip()
    if normalized.endswith("</tool_call>"):
        return normalized + "\n\n"
    return normalized


def _truncate_after_first_tool_call(content: str) -> tuple[str, bool]:
    """Hard-stop locally after the first authoritative tool-call block.

    Providers are expected to honor ``stop=["</tool_call>"]``, but keeping a
    client-side guard prevents any trailing hallucinated turns from entering
    the agent conversation if that server-side stop is absent or ineffective.
    When a provider returns reasoning and content in one string, tool-call
    examples inside the leading <think> block are ignored.
    """
    search_start = 0
    stripped_start = len(content) - len(content.lstrip())
    if content.startswith("<think>", stripped_start):
        think_end = content.find("</think>", stripped_start)
        if think_end != -1:
            search_start = think_end + len("</think>")

    tool_start = content.find("<tool_call>", search_start)
    if tool_start == -1:
        return content, False

    tool_end = content.find("</tool_call>", tool_start + len("<tool_call>"))
    if tool_end == -1:
        return content, False

    cutoff = tool_end + len("</tool_call>")
    if not content[cutoff:].strip():
        return content, False
    return content[:cutoff], True


def _restore_stripped_stop(content: str, finish_reason: Optional[str]) -> str:
    """Restore closing tags removed by an OpenAI-compatible API."""
    if finish_reason == "length":
        return content
    if "<tool_call>" in content and "</tool_call>" not in content:
        return content + "</tool_call>"
    if "<answer>" in content and "</answer>" not in content:
        return content + "</answer>"
    return content


def _normalize_answer(answer: str) -> str:
    """Clean up answer format: remove common prefixes, trailing punctuation, and quote wrapping."""
    text = answer.strip()
    # Remove common prefixes
    for prefix in [
        "The answer is ", "the answer is ",
        "Answer: ", "answer: ",
        "答案是", "答案：",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Remove wrapping quotes (Chinese and English)
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            text = text[1:-1].strip()
        if (text[0] == '\u201c' and text[-1] == '\u201d') or (text[0] == '\u2018' and text[-1] == '\u2019'):
            text = text[1:-1].strip()
        if (text[0] == '\u300c' and text[-1] == '\u300d'):
            text = text[1:-1].strip()
    return text


def _extract_tool_call(content: str) -> Optional[str]:
    """Extract tool call JSON from content, supporting multiple formats."""
    # Format 1: <tool_call>...</tool_call>
    if "<tool_call>" in content and "</tool_call>" in content:
        return _extract_between(content, "<tool_call>", "</tool_call>")
    if "<tool_call>" in content:
        return content[content.find("<tool_call>") + len("<tool_call>"):].strip()

    # Format 2: <function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>
    func_match = re.search(r'<function=(search|visit)>', content)
    if func_match:
        tool_name = func_match.group(1)
        func_start = func_match.start()
        func_end_tag = content.find("</function>", func_start)
        if func_end_tag == -1:
            func_block = content[func_start:]
        else:
            func_block = content[func_start:func_end_tag]
        args = {}
        for param_match in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', func_block, re.DOTALL):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()
            try:
                args[param_name] = json5.loads(param_value)
            except Exception:
                args[param_name] = param_value
        result = json.dumps({"name": tool_name, "arguments": args})
        logger.debug("Converted <function=%s> to JSON: %s", tool_name, result[:200])
        return result

    # Format 3: bare JSON tool call (no tags)
    bare_match = re.search(r'\{"name":\s*"(search|visit)"', content)
    if bare_match:
        json_start = bare_match.start()
        brace_depth = 0
        json_end = json_start
        for i in range(json_start, len(content)):
            if content[i] == '{':
                brace_depth += 1
            elif content[i] == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    json_end = i + 1
                    break
        return content[json_start:json_end]
    return None


def _normalize_signature_text(value: str, *, lowercase: bool = False) -> str:
    """Normalize insignificant text differences for tool-call comparison."""
    normalized = " ".join(value.split())
    return normalized.lower() if lowercase else normalized


def _tool_call_signature(tool_name: str, tool_args) -> str:
    """Build a stable signature for detecting repeated logical tool calls."""
    normalized_name = _normalize_signature_text(str(tool_name), lowercase=True)

    if not isinstance(tool_args, dict):
        normalized_args = tool_args
    else:
        normalized_args = dict(tool_args)

        if normalized_name == "search":
            queries = normalized_args.get("query", [])
            if isinstance(queries, str):
                queries = [queries]
            if isinstance(queries, list):
                normalized_args["query"] = sorted(
                    _normalize_signature_text(str(query), lowercase=True)
                    for query in queries
                )

            engines = normalized_args.get("engine")
            if isinstance(engines, str):
                engines = [engines]
            if isinstance(engines, list):
                normalized_args["engine"] = sorted(
                    _normalize_signature_text(str(engine), lowercase=True)
                    for engine in engines
                )

        elif normalized_name == "visit":
            urls = normalized_args.get("url", [])
            if isinstance(urls, str):
                urls = [urls]
            if isinstance(urls, list):
                normalized_args["url"] = sorted(
                    _normalize_signature_text(str(url)) for url in urls
                )

            goal = normalized_args.get("goal")
            if isinstance(goal, str):
                normalized_args["goal"] = _normalize_signature_text(
                    goal,
                    lowercase=True,
                )

    return json.dumps(
        {"name": normalized_name, "arguments": normalized_args},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _signature_for_log(signature: str) -> str:
    """Keep repeat-detection logs useful without emitting huge arguments."""
    if len(signature) <= TOOL_SIGNATURE_LOG_CHARS:
        return signature
    return signature[:TOOL_SIGNATURE_LOG_CHARS] + "..."


CONTENT_FILTER_MARKER = "__CONTENT_FILTERED__"


async def _call_llm(
    messages: list,
    stop: Optional[list] = None,
    temperature: float = 0.6,
    max_tokens: int = 8192,
) -> str:
    """Call LLM via OpenAI-compatible API with retries."""
    extra_body = {}
    if _IS_NODESK:
        extra_body["channel"] = "DMX"
        extra_body["channel_url"] = "https://www.dmxapi.cn/v1/chat/completions"
    extra_body["enable_thinking"] = True
    # vLLM-only: keep the </tool_call> stop token in the output so it need not
    # be re-appended client-side. Gateways that drop unknown fields ignore it.
    extra_body["include_stop_str_in_output"] = True

    for attempt in range(3):
        try:
            extra_body["repetition_penalty"] = 1.05
            resp = await _client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                # Stop on </tool_call> so the model is cut off right after its
                # tool call and cannot hallucinate a <tool_response> of its own.
                # If the API strips the stop token, _restore_stripped_stop
                # re-appends </tool_call>, and _normalize_assistant_output
                # matches the SFT turn ending.
                stop=stop or ["</tool_call>"],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
                frequency_penalty=0.1,
                extra_body=extra_body,
            )
            choice = resp.choices[0]
            msg = choice.message
            finish_reason = getattr(choice, "finish_reason", None)
            usage = getattr(resp, "usage", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            logger.info(
                "LLM response: request_id=%s attempt=%d finish_reason=%r "
                "completion_tokens=%s",
                getattr(resp, "id", None),
                attempt + 1,
                finish_reason,
                completion_tokens,
            )
            # OpenAI-compatible providers may return reasoning separately.
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            content = msg.content or ""
            # Clean up garbled tag fragments at start of content (e.g. "ynchroneg>")
            content = re.sub(r'^[a-z]*>', '', content.lstrip())
            content = _restore_stripped_stop(
                content,
                finish_reason,
            )
            untruncated_content = content
            content, was_truncated = _truncate_after_first_tool_call(content)
            # Reassemble: wrap reasoning in <think> tags + content
            if reasoning:
                full = f"<think>\n{reasoning}\n</think>\n\n{content}"
                untruncated_full = (
                    f"<think>\n{reasoning}\n</think>\n\n{untruncated_content}"
                )
            else:
                full = content
                untruncated_full = untruncated_content
            if was_truncated:
                logger.warning(
                    "Client-side tool-call truncation applied "
                    "(request_id=%s, finish_reason=%r).\n"
                    "----- BEFORE TRUNCATION -----\n%s\n"
                    "----- AFTER TRUNCATION -----\n%s\n"
                    "----- END TRUNCATION LOG -----",
                    getattr(resp, "id", None),
                    finish_reason,
                    untruncated_full,
                    full,
                )
            if full and full.strip():
                return _normalize_assistant_output(full)
            logger.warning("Empty response on attempt %d", attempt + 1)
        except Exception as e:
            err_str = str(e)
            logger.warning("LLM call attempt %d failed: %s", attempt + 1, e)
            # Content safety filter - no point retrying the same prompt
            if "data_inspection_failed" in err_str:
                logger.warning("Content filter triggered, signaling redirect")
                return CONTENT_FILTER_MARKER
            if attempt < 2:
                await _async_sleep(2 ** attempt)

    return "I was unable to process this request due to an API error."


async def _async_sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)


async def _force_answer(messages: list) -> str:
    """Force the LLM to generate a final answer based on gathered info.

    Uses a final-answer instruction and retries with stronger prompts if the
    model still outputs tool calls due to training inertia.
    """
    force_msg = (
        "The current round limit has been reached or the remaining time is "
        "insufficient, so you can no longer call any further tools. Based only "
        "on the conversation in this round, the information already obtained, "
        "and the tool results, provide your final response immediately.\n\n"
        "Do not request further tool calls or additional information, and do "
        "not merely describe follow-up investigation steps. Regardless of "
        "whether the information is complete, you must provide a clear final "
        "answer; if you cannot determine it with certainty, give the most "
        "reasonable guess.\n\n"
        "Strictly output only in the following format:\n\n"
        "<think>\n"
        "Briefly state the key information and reasoning underlying your conclusion.\n"
        "</think>\n\n"
        "<answer>\n"
        "A direct and complete final answer for the user.\n"
        "</answer>"
    )
    messages.append({"role": "user", "content": force_msg})

    for attempt in range(3):
        content = await _call_llm(messages, temperature=0.6)
        messages.append({"role": "assistant", "content": content})
        post_think = _post_think(content)

        # Model still output a tool call — retry with stronger nudge
        if "<tool_call>" in post_think and "<answer>" not in post_think:
            if attempt < 2:
                logger.warning(
                    "_force_answer attempt %d/3: model output tool_call, retrying",
                    attempt + 1,
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "<tool_response>\n"
                        "Tool execution is unavailable because the round or time limit "
                        "has been reached. The system has entered the forced-answer "
                        "phase. No tool calls will be executed or return results.\n"
                        "</tool_response>"
                    ),
                })
                retry_msg = (
                    "Tool use is no longer available. Do NOT output <tool_call>, "
                    "request more information, or describe further investigation.\n\n"
                    "You must now provide the best possible final answer using only "
                    "the information already present in the conversation. If the "
                    "evidence is incomplete, provide your most reasonable guess "
                    "rather than refusing to answer or deferring the task.\n\n"
                    "Output only the following format, with no text outside these tags:\n\n"
                    "<think>\n"
                    "Briefly state the key information and reasoning underlying your "
                    "conclusion.\n"
                    "</think>\n\n"
                    "<answer>\n"
                    "A direct and complete final answer for the user.\n"
                    "</answer>"
                )
                messages.append({"role": "user", "content": retry_msg})
                continue
            # All retries exhausted
            logger.error(
                "_force_answer attempt %d/3: still tool_call after all retries, giving up",
                attempt + 1,
            )
            return "Unable to determine answer."

        # Got a valid answer
        if "<answer>" in post_think:
            answer = _extract_between(post_think, "<answer>", "</answer>")
            if answer:
                logger.info("_force_answer attempt %d/3: got valid answer", attempt + 1)
                return _normalize_answer(answer)
            # </answer> was the stop token, content after <answer> is the answer
            idx = post_think.find("<answer>")
            answer = post_think[idx + len("<answer>"):].strip()
            if answer:
                logger.info("_force_answer attempt %d/3: got answer (stop-token truncated)", attempt + 1)
                return _normalize_answer(answer)

        # Neither tool_call nor answer — ask again
        logger.warning(
            "_force_answer attempt %d/3: no tool_call and no <answer>, retrying",
            attempt + 1,
        )
        if attempt < 2:
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response is invalid because it did not provide "
                    "a final answer in the required format.\n\n"
                    "This is the forced-answer stage. Tool use is unavailable. Do "
                    "not output a tool call, request additional information, or "
                    "describe further investigation.\n\n"
                    "Using only the information already present in the conversation, "
                    "provide the best possible final answer now. If the evidence is "
                    "incomplete, give your most reasonable guess.\n\n"
                    "Output only the following format, with no text outside these tags:\n\n"
                    "<think>\n"
                    "Briefly state the key information and reasoning underlying your "
                    "conclusion.\n"
                    "</think>\n\n"
                    "<answer>\n"
                    "A direct and complete final answer for the user.\n"
                    "</answer>"
                ),
            })
            continue

    logger.error("_force_answer: all 3 attempts exhausted, no valid answer")
    return "Unable to determine answer."


async def _execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a tool call and return the result string."""
    try:
        if tool_name == "search":
            queries = tool_args.get("query", [])
            if isinstance(queries, str):
                queries = [queries]
            engines = tool_args.get("engine", None)
            if isinstance(engines, str):
                engines = [engines]
            result = batch_search(queries, engines=engines)
            return _truncate_tool_result(result)
        elif tool_name == "visit":
            urls = tool_args.get("url", [])
            goal = tool_args.get("goal", "Extract relevant information")
            result = await visit_pages(urls, goal)
            return _truncate_tool_result(result)
        else:
            return f"Error: Unknown tool '{tool_name}'"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


async def react_agent(
    question: str,
    *,
    trajectory_callback: Optional[TrajectoryCallback] = None,
) -> str:
    """Run the ReAct agent with the training-aligned prompt."""
    messages = _TrackedMessages(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        trajectory_callback,
    )

    start_time = time.time()
    early_answer_blocked = False  # only block early answer once
    content_filter_count = 0  # track consecutive content filter hits
    tool_call_attempt_counts = {}
    tool_call_execution_counts = {}
    consecutive_blocked_duplicate_calls = 0

    for round_idx in range(MAX_ROUNDS):
        # Timeout check
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT_SECONDS:
            logger.warning("Timeout after %.0fs at round %d", elapsed, round_idx + 1)
            return await _force_answer(messages)

        # Call LLM
        current_tokens = _estimate_tokens(messages)
        logger.info("Round %d, elapsed %.0fs, tokens %d", round_idx + 1, elapsed, current_tokens)
        content = await _call_llm(messages)

        # Handle content filter: redirect or force a neutral search
        if content == CONTENT_FILTER_MARKER:
            content_filter_count += 1
            logger.warning("Content filter hit #%d at round %d", content_filter_count, round_idx + 1)
            if content_filter_count <= 2:
                redirect_msg = (
                    "The previous response was blocked by content filters. "
                    "Please try a different approach: rephrase your search using more neutral, "
                    "academic, or indirect terms. Focus on factual and historical aspects. "
                    "Use <think> to plan a new search angle, then make a tool call."
                )
                messages.append({"role": "assistant", "content": "<think>My previous response was filtered. I need to rephrase my approach using more neutral terms.</think>"})
                messages.append({"role": "user", "content": redirect_msg})
            else:
                # After 2 redirects, context itself is toxic — force answer with sanitized context
                logger.warning("Too many content filters, forcing answer with sanitized context")
                # Keep only system prompt, original question, and a summary of findings
                sanitized_messages = _TrackedMessages(
                    [messages[0], messages[1]],  # system + user question
                    trajectory_callback,
                )
                # Check if model already found an answer in earlier rounds
                for msg in messages:
                    if msg["role"] == "assistant":
                        c = _post_think(msg["content"])
                        if "<answer>" in c:
                            answer_text = _extract_between(c, "<answer>", "</answer>")
                            if answer_text:
                                return _normalize_answer(answer_text)
                # Extract key findings from assistant messages (avoid sensitive details)
                findings = []
                for msg in messages:
                    if msg["role"] == "assistant":
                        for line in msg["content"].split("\n"):
                            if any(kw in line for kw in ["答案", "answer", "结论", "确认", "因此", "全名", "全称"]):
                                findings.append(line[:200])
                if findings:
                    sanitized_messages.append({
                        "role": "user",
                        "content": "Based on your earlier research, you found:\n" + "\n".join(findings[-5:]) + "\nNow give your final answer."
                    })
                return await _force_answer(sanitized_messages)
            continue

        # Clean up: remove any hallucinated <tool_response> tags
        if "<tool_response>" in content:
            content = content[: content.find("<tool_response>")]

        # Detect stalled output (very short meaningless content like "<" or empty)
        if len(content.strip()) < 5 and "<answer>" not in content:
            logger.warning("Stalled output detected: '%s', retrying", content.strip())
            continue

        messages.append({"role": "assistant", "content": content})

        logger.debug("Output content: %s", content)
        # Only tags emitted after the reasoning block are authoritative; the
        # model may quote <answer>/<tool_call> inside <think> (e.g. restating
        # the format instruction), which must not count as an answer/tool call.
        post_think = _post_think(content)
        # Check for final answer (with or without closing tag)
        if "<answer>" in post_think:
            if "</answer>" in post_think:
                answer_text = _extract_between(post_think, "<answer>", "</answer>")
            else:
                # No closing tag — extract everything after <answer>
                answer_text = post_think[post_think.find("<answer>") + len("<answer>"):].strip()
                # Take only the first line to avoid grabbing trailing content
                if "\n" in answer_text:
                    answer_text = answer_text.split("\n")[0].strip()
                logger.debug("Detected <answer> without </answer>, extracted: %s", answer_text[:100])

            if answer_text:
                # Early answer gating: if answered too quickly, inject verification prompt
                elapsed = time.time() - start_time
                if round_idx < 3 and elapsed < 120 and not early_answer_blocked:
                    early_answer_blocked = True
                    verify_msg = (
                        "You answered very quickly. For complex multi-hop questions, "
                        "this often leads to errors. Please verify:\n"
                        "1. Re-read the original question\n"
                        "2. Check EVERY constraint\n"
                        "3. Search for one more confirming source\n"
                        "If correct, repeat your answer. If not, search more."
                    )
                    messages.append({"role": "user", "content": verify_msg})
                    logger.info("Early answer blocked at round %d, requesting verification", round_idx + 1)
                    continue

                logger.info("Got answer at round %d: %s", round_idx + 1, answer_text[:100])
                return _normalize_answer(answer_text)

        # Check for tool call — support both <tool_call> tags and bare JSON
        tool_call_str = _extract_tool_call(post_think)

        if tool_call_str:
            try:
                tool_data = json5.loads(tool_call_str)
                tool_name = tool_data.get("name", "")
                tool_args = tool_data.get("arguments", {})
                signature = _tool_call_signature(tool_name, tool_args)
                signature_log = _signature_for_log(signature)
                attempt_count = tool_call_attempt_counts.get(signature, 0) + 1
                tool_call_attempt_counts[signature] = attempt_count

                if attempt_count > MAX_IDENTICAL_TOOL_EXECUTIONS:
                    if attempt_count >= MAX_IDENTICAL_TOOL_ATTEMPTS:
                        logger.warning(
                            "Repeated tool call reached attempt limit at round %d; "
                            "forcing answer: tool=%s attempt_count=%d signature=%s",
                            round_idx + 1,
                            tool_name,
                            attempt_count,
                            signature_log,
                        )
                        messages.append({
                            "role": "user",
                            "content": (
                                "<tool_response>\n"
                                "This tool call was not executed because this research "
                                "direction has reached its repeated-attempt limit. Tool "
                                "execution is no longer available; the system has entered "
                                "the forced-answer phase.\n"
                                "</tool_response>"
                            ),
                        })
                        return await _force_answer(messages)

                    consecutive_blocked_duplicate_calls += 1
                    if (
                        consecutive_blocked_duplicate_calls
                        >= MAX_CONSECUTIVE_BLOCKED_DUPLICATES
                    ):
                        logger.warning(
                            "Consecutive blocked duplicate calls reached limit at round %d; "
                            "forcing answer: tool=%s blocked_count=%d signature=%s",
                            round_idx + 1,
                            tool_name,
                            consecutive_blocked_duplicate_calls,
                            signature_log,
                        )
                        messages.append({
                            "role": "user",
                            "content": (
                                "<tool_response>\n"
                                "This tool call was not executed because repeated research "
                                "directions have been blocked multiple times. Tool execution "
                                "is no longer available; the system has entered the "
                                "forced-answer phase.\n"
                                "</tool_response>"
                            ),
                        })
                        return await _force_answer(messages)

                    logger.warning(
                        "Blocking repeated tool call at round %d and requesting a new "
                        "direction: tool=%s attempt_count=%d signature=%s",
                        round_idx + 1,
                        tool_name,
                        attempt_count,
                        signature_log,
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "<tool_response>\n"
                            "This tool call was not executed because this exact research "
                            "direction has already been attempted repeatedly. Choose an "
                            "untried direction before making another tool call.\n"
                            "</tool_response>"
                        ),
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            "The previous tool call was rejected because this research "
                            "direction has already been attempted repeatedly.\n\n"
                            "Before taking the next action, reassess the task carefully:\n\n"
                            "1. What constraints does the original question contain?\n"
                            "2. What candidate answers have been identified so far?\n"
                            "3. What evidence supports or contradicts each candidate?\n"
                            "4. Which research directions have already been ruled out or exhausted?\n"
                            "5. What specific missing evidence is most important?\n"
                            "6. Is the best next step to open an existing source, investigate "
                            "a genuinely different direction, or provide the final answer?\n\n"
                            "Do not repeat any previously attempted search or use trivial "
                            "rewording of the same query. If you make another tool call, it "
                            "must pursue a substantially different entity, source, constraint, "
                            "or verification method. If the existing evidence is sufficient, "
                            "provide the final answer instead."
                        ),
                    })
                    continue

                logger.info("Calling tool: %s", tool_name)

                result = await _execute_tool(tool_name, tool_args)
                consecutive_blocked_duplicate_calls = 0
                execution_count = tool_call_execution_counts.get(signature, 0) + 1
                tool_call_execution_counts[signature] = execution_count
                logger.info(
                    "Recorded tool call execution at round %d: tool=%s "
                    "execution_count=%d signature=%s",
                    round_idx + 1,
                    tool_name,
                    execution_count,
                    signature_log,
                )

                messages.append({
                    "role": "user",
                    "content": f"<tool_response>{result}</tool_response>",
                })
            except Exception as e:
                error_msg = (
                    f'Error: Tool call is not valid JSON. '
                    f'Tool call must contain a valid "name" and "arguments" field. '
                    f'Error: {str(e)}'
                )
                messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{error_msg}\n</tool_response>",
                })
        else:
            # No tool call and no answer — nudge the LLM
            nudge_msg = (
                "You did not make a tool call or provide an answer. "
                "Please use <think> to reason, then either make a tool call or provide your answer."
            )
            messages.append({"role": "user", "content": nudge_msg})

        # Token limit check
        if _estimate_tokens(messages) > MAX_TOKENS_ESTIMATE:
            logger.warning("Token limit reached at round %d", round_idx + 1)
            return await _force_answer(messages)

    # Exhausted all rounds
    logger.warning("Max rounds (%d) exhausted", MAX_ROUNDS)
    return await _force_answer(messages)
