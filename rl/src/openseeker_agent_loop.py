"""Token-native VERL loop mirroring the production OpenSeeker agent guards.

The control flow and recovery prompts are copied from ``Project/agent`` but the
policy call is replaced by VERL's token-in/token-out API. Model tokens are never
re-encoded; every tool result or recovery instruction is an environment token
with response mask zero.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import json5

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.workers.rollout.replica import TokenOutput

# The rollout configuration supplies temperature/top-p/top-k/repetition penalty.
# VERL does not expose frequency_penalty in its rollout config, so enforce the
# production Agent's value on every token-native generation request here.
ORIGINAL_AGENT_FREQUENCY_PENALTY = 0.1
from src.rollout_artifacts import TraceLogger, TrajectoryWriter
from src.telemetry import TelemetryWriter
from src.tool_runtime import ToolResult, get_tool_environment


logger = logging.getLogger(__name__)

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
FUNCTION_RE = re.compile(r"<function=(search|visit)>", re.IGNORECASE)
PARAMETER_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)
CONTENT_FILTER_MARKER = "data_inspection_failed"

PROTOCOL_TAGS = (
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<answer>",
    "</answer>",
)
STRICT_TURN_RE = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*"
    r"(?:(?P<tool_call><tool_call>(?P<tool_payload>.*?)</tool_call>)"
    r"|(?P<answer><answer>(?P<answer_content>.*?)</answer>))\s*\Z",
    re.DOTALL,
)


def validate_assistant_turn_format(content: str) -> tuple[bool, str | None, str | None]:
    """Validate one logical assistant turn against the strict XML protocol.

    The active Qwen3 template opens ``<think>`` in the generation prompt, so
    ``content`` contains only the generated suffix. Reconstruct that opening
    tag here for validation without changing the generated token sequence.
    """
    logical_turn = "<think>\n" + content.replace("\r\n", "\n").replace("\r", "\n")
    match = STRICT_TURN_RE.fullmatch(logical_turn)
    if match is None:
        return False, None, "invalid_structure"

    action = "tool_call" if match.group("tool_call") is not None else "answer"
    expected_counts = {tag: 0 for tag in PROTOCOL_TAGS}
    expected_counts["<think>"] = 1
    expected_counts["</think>"] = 1
    expected_counts[f"<{action}>"] = 1
    expected_counts[f"</{action}>"] = 1
    if any(logical_turn.count(tag) != expected for tag, expected in expected_counts.items()):
        return False, action, "duplicate_or_mixed_protocol_tags"

    if not match.group("think").strip():
        return False, action, "empty_think"
    action_content = (
        match.group("tool_payload") if action == "tool_call" else match.group("answer_content")
    )
    if not action_content or not action_content.strip():
        return False, action, f"empty_{action}"
    return True, action, None

EARLY_ANSWER_VERIFY = (
    "You answered very quickly. For complex multi-hop questions, this often leads to errors. Please verify:\n"
    "1. Re-read the original question\n"
    "2. Check EVERY constraint\n"
    "3. Search for one more confirming source\n"
    "If correct, repeat your answer. If not, search more."
)
CONTENT_FILTER_REDIRECT = (
    "The previous response was blocked by content filters. Please try a different approach: rephrase your "
    "search using more neutral, academic, or indirect terms. Focus on factual and historical aspects. "
    "Use <think> to plan a new search angle, then make a tool call."
)
NO_ACTION_NUDGE = (
    "You did not make a tool call or provide an answer. Please use <think> to reason, then either make a "
    "tool call or provide your answer."
)
FORCE_ANSWER_INITIAL = (
    "The current round limit has been reached or the remaining time is insufficient, so you can no longer "
    "call any further tools. Based only on the conversation in this round, the information already obtained, "
    "and the tool results, provide your final response immediately.\n\n"
    "Do not request further tool calls or additional information, and do not merely describe follow-up "
    "investigation steps. Regardless of whether the information is complete, you must provide a clear final "
    "answer; if you cannot determine it with certainty, give the most reasonable guess.\n\n"
    "Strictly output only in the following format:\n\n"
    "<think>\nBriefly state the key information and reasoning underlying your conclusion.\n</think>\n\n"
    "<answer>\nA direct and complete final answer for the user.\n</answer>"
)
FORCE_TOOL_UNAVAILABLE = (
    "<tool_response>\nTool execution is unavailable because the round or time limit has been reached. "
    "The system has entered the forced-answer phase. No tool calls will be executed or return results.\n"
    "</tool_response>"
)
FORCE_TOOL_RETRY = (
    "Tool use is no longer available. Do NOT output <tool_call>, request more information, or describe "
    "further investigation.\n\nYou must now provide the best possible final answer using only the information "
    "already present in the conversation. If the evidence is incomplete, provide your most reasonable guess "
    "rather than refusing to answer or deferring the task.\n\nOutput only the following format, with no text "
    "outside these tags:\n\n<think>\nBriefly state the key information and reasoning underlying your conclusion.\n"
    "</think>\n\n<answer>\nA direct and complete final answer for the user.\n</answer>"
)
FORCE_INVALID_RETRY = (
    "Your previous response is invalid because it did not provide a final answer in the required format.\n\n"
    "This is the forced-answer stage. Tool use is unavailable. Do not output a tool call, request additional "
    "information, or describe further investigation.\n\nUsing only the information already present in the "
    "conversation, provide the best possible final answer now. If the evidence is incomplete, give your most "
    "reasonable guess.\n\nOutput only the following format, with no text outside these tags:\n\n"
    "<think>\nBriefly state the key information and reasoning underlying your conclusion.\n</think>\n\n"
    "<answer>\nA direct and complete final answer for the user.\n</answer>"
)
DUPLICATE_BLOCK_RESPONSE = (
    "<tool_response>\nThis tool call was not executed because this exact research direction has already "
    "been attempted repeatedly. Choose an untried direction before making another tool call.\n</tool_response>"
)
DUPLICATE_REASSESS = (
    "The previous tool call was rejected because this research direction has already been attempted "
    "repeatedly.\n\nBefore taking the next action, reassess the task carefully:\n\n"
    "1. What constraints does the original question contain?\n"
    "2. What candidate answers have been identified so far?\n"
    "3. What evidence supports or contradicts each candidate?\n"
    "4. Which research directions have already been ruled out or exhausted?\n"
    "5. What specific missing evidence is most important?\n"
    "6. Is the best next step to open an existing source, investigate a genuinely different direction, or "
    "provide the final answer?\n\nDo not repeat any previously attempted search or use trivial rewording of the "
    "same query. If you make another tool call, it must pursue a substantially different entity, source, "
    "constraint, or verification method. If the existing evidence is sufficient, provide the final answer instead."
)


class GenerationExhausted(RuntimeError):
    pass


class GenerationTimedOut(GenerationExhausted):
    pass


class ContentFiltered(RuntimeError):
    pass


def _extract_tool_payload(content: str) -> str | None:
    if "<tool_call>" in content:
        start = content.find("<tool_call>") + len("<tool_call>")
        end = content.find("</tool_call>", start)
        return content[start : end if end >= 0 else None].strip()

    function_match = FUNCTION_RE.search(content)
    if function_match:
        function_start = function_match.start()
        function_end = content.find("</function>", function_start)
        block = content[function_start : function_end if function_end >= 0 else None]
        arguments: dict[str, Any] = {}
        for parameter in PARAMETER_RE.finditer(block):
            value = parameter.group(2).strip()
            try:
                arguments[parameter.group(1)] = json5.loads(value)
            except Exception:
                arguments[parameter.group(1)] = value
        return json.dumps({"name": function_match.group(1), "arguments": arguments})

    bare_match = re.search(r'\{"name":\s*"(search|visit)"', content)
    if bare_match:
        json_start = bare_match.start()
        brace_depth = 0
        json_end = json_start
        for index in range(json_start, len(content)):
            if content[index] == "{":
                brace_depth += 1
            elif content[index] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    json_end = index + 1
                    break
        return content[json_start:json_end]
    return None


def extract_answer_after_think(content: str) -> str | None:
    """Return a complete answer block only when it follows the final thought."""
    think_end = content.rfind("</think>")
    if think_end < 0:
        return None
    match = ANSWER_RE.search(content, think_end + len("</think>"))
    if match is None:
        return None
    answer = match.group(1).strip()
    return answer or None


def extract_tool_call(content: str) -> tuple[str, Any] | None:
    payload_text = _extract_tool_payload(content)
    if payload_text is None:
        return None
    payload = json5.loads(payload_text)
    return payload.get("name", ""), payload.get("arguments", {})


def tool_signature(name: str, arguments: Any) -> str:
    normalized_name = name.strip().lower()
    if not isinstance(arguments, dict):
        normalized_arguments = arguments
    else:
        normalized_arguments = dict(arguments)
    if isinstance(normalized_arguments, dict) and normalized_name == "search":
        queries = normalized_arguments.get("query", [])
        if isinstance(queries, str):
            queries = [queries]
        normalized_arguments["query"] = [
            " ".join(str(query).split()) for query in queries
        ]
    elif isinstance(normalized_arguments, dict) and normalized_name == "visit":
        urls = normalized_arguments.get("url", [])
        if isinstance(urls, str):
            urls = [urls]
        normalized_arguments["url"] = [str(url).strip() for url in urls]
        if "goal" in normalized_arguments:
            normalized_arguments["goal"] = " ".join(
                str(normalized_arguments["goal"]).split()
            )
    return json.dumps(
        {"name": normalized_name, "arguments": normalized_arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


class OpenSeekerAgentLoop(AgentLoopBase):
    """OpenSeeker ReAct loop with production-equivalent stability guards."""

    def __init__(
        self,
        *args,
        max_assistant_turns: int = 150,
        max_tokens_per_turn: int = 8192,
        force_answer_reserve_tokens: int = 12288,
        force_answer_max_attempts: int = 3,
        force_answer_timeout_attempts: int = 5,
        generation_max_retries: int = 3,
        generation_retry_base_seconds: float = 1.0,
        generation_timeout_seconds: float = 120.0,
        tool_timeout_seconds: float = 120.0,
        max_identical_tool_executions: int = 2,
        max_identical_tool_attempts: int = 5,
        max_consecutive_blocked_duplicates: int = 3,
        early_answer_turns: int = 3,
        early_answer_seconds: float = 120.0,
        content_filter_max_redirects: int = 2,
        stalled_output_min_chars: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not self.enable_continuous_token:
            raise ValueError(
                "OpenSeekerAgentLoop requires data.continuous_token.enable=true; "
                "re-encoding assistant turns is unsafe for on-policy RL"
            )
        self.max_assistant_turns = max_assistant_turns
        self.max_tokens_per_turn = max_tokens_per_turn
        self.force_answer_reserve_tokens = force_answer_reserve_tokens
        self.force_answer_max_attempts = force_answer_max_attempts
        if force_answer_timeout_attempts < 1:
            raise ValueError("force_answer_timeout_attempts must be at least one")
        self.force_answer_timeout_attempts = force_answer_timeout_attempts
        self.generation_max_retries = generation_max_retries
        self.generation_retry_base_seconds = generation_retry_base_seconds
        self.generation_timeout_seconds = generation_timeout_seconds
        self.tool_timeout_seconds = tool_timeout_seconds
        self.max_identical_tool_executions = max_identical_tool_executions
        self.max_identical_tool_attempts = max_identical_tool_attempts
        self.max_consecutive_blocked_duplicates = max_consecutive_blocked_duplicates
        self.early_answer_turns = early_answer_turns
        self.early_answer_seconds = early_answer_seconds
        self.content_filter_max_redirects = content_filter_max_redirects
        self.stalled_output_min_chars = stalled_output_min_chars
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.environment = get_tool_environment()
        self.telemetry = TelemetryWriter("agent")
        self.trajectory_writer = TrajectoryWriter()

    def _emit_telemetry(self, trace_id: str, event: str, **fields: Any) -> None:
        # Unit tests construct a minimal loop via __new__; telemetry must remain optional.
        telemetry = getattr(self, "telemetry", None)
        if telemetry is not None:
            telemetry.emit(trace_id, event, **fields)

    def _write_trajectory(
        self,
        trace_id: str,
        *,
        started_at: datetime,
        status: str,
        messages: list[dict[str, str]],
        extra_fields: dict[str, Any],
        trace_logger: TraceLogger,
    ) -> None:
        writer = getattr(self, "trajectory_writer", None)
        if writer is None:
            writer = TrajectoryWriter()
        try:
            path = writer.write(
                trace_id,
                started_at=started_at,
                status=status,
                messages=messages,
                extra_fields=extra_fields,
            )
        except Exception as error:
            trace_logger.error("Failed to save trajectory: %s", error)
            return
        if path is not None:
            trace_logger.info("Trajectory saved: %s", path)

    def _close_stopped_tool_turn(
        self,
        messages: list[dict[str, str]],
        runtime_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
    ) -> tuple[list[int], list[int], list[float] | None]:
        """Close a tool-call turn whose generation stopped before ChatML EOS.

        Rollout generation deliberately stops at ``</tool_call>`` and retains
        that stop token. Qwen's canonical chat template nevertheless closes
        every assistant message with ``<|im_end|>``. The missing close token is
        supplied by the environment, so it must carry neither policy loss nor
        an old-policy log probability.
        """
        if not messages or messages[-1].get("role") != "assistant":
            return runtime_ids, response_mask, response_logprobs
        if not messages[-1].get("content", "").rstrip().endswith("</tool_call>"):
            return runtime_ids, response_mask, response_logprobs

        im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if not im_end_ids:
            raise ValueError("Qwen <|im_end|> encoded to no tokens")
        im_end_ids = [int(token_id) for token_id in im_end_ids]
        if runtime_ids[-len(im_end_ids) :] == im_end_ids:
            return runtime_ids, response_mask, response_logprobs

        runtime_ids = [*runtime_ids, *im_end_ids]
        response_mask = [*response_mask, *([0] * len(im_end_ids))]
        if response_logprobs is not None:
            response_logprobs = [*response_logprobs, *([0.0] * len(im_end_ids))]
        return runtime_ids, response_mask, response_logprobs

    async def _append_user_messages(
        self,
        messages: list[dict[str, str]],
        runtime_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
        contents: list[str],
    ) -> tuple[list[int], list[int], list[float] | None]:
        """Append an environment-message group with one generation prompt.

        Keeping related recovery messages in one CT merge prevents an
        assistant generation header from being inserted between two user
        messages (the previous "ghost assistant" boundary).
        """
        if not contents:
            return runtime_ids, response_mask, response_logprobs
        previous_messages = list(messages)
        runtime_ids, response_mask, response_logprobs = self._close_stopped_tool_turn(
            previous_messages, runtime_ids, response_mask, response_logprobs
        )
        messages.extend({"role": "user", "content": content} for content in contents)
        (
            merge_result,
            response_mask,
            response_logprobs,
        ) = await self.ct_merge_non_assistant_msg(
            previous_messages, messages, runtime_ids, response_mask, response_logprobs
        )
        return merge_result.token_ids, response_mask, response_logprobs

    async def _append_user_message(
        self,
        messages: list[dict[str, str]],
        runtime_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
        content: str,
    ) -> tuple[list[int], list[int], list[float] | None]:
        return await self._append_user_messages(
            messages, runtime_ids, response_mask, response_logprobs, [content]
        )

    async def _generate_once(
        self,
        *,
        request_id: str,
        runtime_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
        sampling_params: dict[str, Any],
        trace_logger: TraceLogger,
        turn: int = 0,
        attempt: int = 0,
    ) -> tuple[list[int], list[int], list[float] | None, TokenOutput]:
        remaining = self.response_length - len(response_mask)
        if remaining <= 0:
            raise GenerationExhausted("response token budget exhausted")
        params = dict(sampling_params)
        params["max_tokens"] = min(self.max_tokens_per_turn, remaining)
        params["stop"] = ["</tool_call>"]
        params["frequency_penalty"] = ORIGINAL_AGENT_FREQUENCY_PENALTY
        if str(self.rollout_config.name).lower() == "vllm":
            params["include_stop_str_in_output"] = True
        generation_timeout = self.generation_timeout_seconds
        started = time.perf_counter()
        self._emit_telemetry(
            request_id, "vllm_submit", turn=turn, attempt=attempt,
            prompt_tokens=len(runtime_ids), max_tokens=params["max_tokens"],
        )
        trace_logger.info(
            "VLLM submit: turn=%d attempt=%d prompt_tokens=%d max_tokens=%d timeout_s=%.1f",
            turn, attempt, len(runtime_ids), params["max_tokens"], generation_timeout,
        )
        try:
            output: TokenOutput = await asyncio.wait_for(
                self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=runtime_ids,
                    sampling_params=params,
                ),
                timeout=generation_timeout,
            )
        except Exception as error:
            elapsed_s = time.perf_counter() - started
            self._emit_telemetry(
                request_id, "vllm_error", turn=turn, attempt=attempt,
                elapsed_s=elapsed_s, error_type=type(error).__name__,
            )
            trace_logger.warning(
                "VLLM error: turn=%d attempt=%d elapsed_s=%.3f error=%s: %s",
                turn, attempt, elapsed_s, type(error).__name__, error,
            )
            raise
        elapsed_s = time.perf_counter() - started
        self._emit_telemetry(
            request_id, "vllm_complete", turn=turn, attempt=attempt,
            elapsed_s=elapsed_s, generated_tokens=len(output.token_ids),
            replica_rank=output.extra_fields.get("vllm_replica_rank"),
            server_elapsed_s=output.extra_fields.get("vllm_server_elapsed_s"),
            server_first_output_s=output.extra_fields.get("vllm_server_first_output_s"),
        )
        trace_logger.info(
            "VLLM complete: turn=%d attempt=%d elapsed_s=%.3f generated_tokens=%d",
            turn, attempt, elapsed_s, len(output.token_ids),
        )
        (
            merge_result,
            response_mask,
            response_logprobs,
        ) = await self.ct_merge_assistant_token(
            runtime_ids,
            output.token_ids,
            response_mask,
            response_logprobs,
            assistant_logprobs=output.log_probs,
        )
        return merge_result.token_ids, response_mask, response_logprobs, output

    async def _generate_with_retries(
        self,
        *,
        request_id: str,
        runtime_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
        sampling_params: dict[str, Any],
        metrics: dict[str, Any],
        stability: Counter[str],
        trace_logger: TraceLogger,
        turn: int = 0,
        max_attempts: int | None = None,
        retry_timeouts_only: bool = False,
    ) -> tuple[list[int], list[int], list[float] | None, TokenOutput]:
        attempts = max_attempts or self.generation_max_retries
        if attempts < 1:
            raise ValueError("generation attempts must be at least one")
        last_error: Exception | None = None
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                result = await self._generate_once(
                    request_id=request_id,
                    runtime_ids=runtime_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    sampling_params=sampling_params,
                    trace_logger=trace_logger,
                    turn=turn,
                    attempt=attempt + 1,
                )
                metrics["generate_sequences"] += time.perf_counter() - started
                return result
            except GenerationExhausted:
                metrics["generate_sequences"] += time.perf_counter() - started
                raise
            except asyncio.TimeoutError as error:
                metrics["generate_sequences"] += time.perf_counter() - started
                last_error = error
                stability["generation_errors"] += 1
                stability["generation_timeouts"] += 1
                if attempt + 1 < attempts:
                    stability["generation_retries"] += 1
                    retry_delay = self.generation_retry_base_seconds * (2**attempt)
                    trace_logger.warning(
                        "Retrying timed-out VLLM generation: turn=%d next_attempt=%d delay_s=%.3f",
                        turn,
                        attempt + 2,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                if retry_timeouts_only:
                    raise GenerationTimedOut(
                        f"generation timed out after {attempts} attempts: {last_error}"
                    ) from error
            except Exception as error:
                metrics["generate_sequences"] += time.perf_counter() - started
                last_error = error
                stability["generation_errors"] += 1
                if CONTENT_FILTER_MARKER in str(error):
                    raise ContentFiltered(str(error)) from error
                if retry_timeouts_only:
                    raise GenerationExhausted(
                        f"generation failed without a timeout: {error}"
                    ) from error
                if attempt + 1 < attempts:
                    stability["generation_retries"] += 1
                    retry_delay = self.generation_retry_base_seconds * (2**attempt)
                    trace_logger.warning(
                        "Retrying VLLM generation: turn=%d next_attempt=%d delay_s=%.3f",
                        turn,
                        attempt + 2,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
        raise GenerationExhausted(
            f"generation failed after {attempts} attempts: {last_error}"
        ) from last_error

    async def _begin_force_answer(
        self,
        messages: list[dict[str, str]],
        runtime_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
        observation: str | None = None,
    ) -> tuple[list[int], list[int], list[float] | None]:
        contents = [observation] if observation else []
        contents.append(FORCE_ANSWER_INITIAL)
        return await self._append_user_messages(
            messages,
            runtime_ids,
            response_mask,
            response_logprobs,
            contents,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = [dict(message) for message in kwargs["raw_prompt"]]
        runtime_ids = await self.ct_build_initial_tokens(messages)
        response_mask: list[int] = []
        response_logprobs: list[float] | None = (
            [] if sampling_params.get("logprobs") else None
        )
        request_id = uuid4().hex
        started_wallclock_at = datetime.now(UTC)
        started_at = time.monotonic()
        trace_logger = TraceLogger(request_id)
        trace_logger.info(
            "Trajectory start: initial_messages=%d max_response_tokens=%d",
            len(messages), self.response_length,
        )
        self._emit_telemetry(request_id, "trajectory_start", initial_messages=len(messages), max_response_tokens=self.response_length)
        metrics: dict[str, Any] = {
            "generate_sequences": 0.0,
            "tool_calls": 0.0,
            "compute_score": 0.0,
            "num_preempted": 0,
        }
        stability: Counter[str] = Counter()
        attempt_counts: Counter[str] = Counter()
        execution_counts: Counter[str] = Counter()
        regular_turns = 0
        total_assistant_turns = 0
        tool_calls = 0
        tool_errors = 0
        cached_tool_calls = 0
        consecutive_blocked_duplicates = 0
        forced_answer = False
        force_attempts = 0
        answered = False
        early_answer_blocked = False
        content_filter_count = 0
        termination_reason = "unknown"
        latest_routed_experts = None
        assistant_turn_formats: list[dict[str, Any]] = []

        while True:
            elapsed = time.monotonic() - started_at
            if len(response_mask) >= self.response_length:
                trace_logger.warning("Response token limit reached: tokens=%d", len(response_mask))
                termination_reason = "response_token_limit"
                break

            if not forced_answer:
                force_reason: str | None = None
                if regular_turns >= self.max_assistant_turns:
                    force_reason = "assistant_turn_limit"
                elif (
                    self.response_length - len(response_mask)
                    <= self.force_answer_reserve_tokens
                ):
                    force_reason = "response_token_reserve"
                if force_reason:
                    (
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                    ) = await self._begin_force_answer(
                        messages, runtime_ids, response_mask, response_logprobs
                    )
                    forced_answer = True
                    termination_reason = force_reason
                    trace_logger.warning("Entering forced-answer mode: reason=%s", force_reason)
                    continue

            if forced_answer and force_attempts >= self.force_answer_max_attempts:
                trace_logger.warning(
                    "Forced-answer attempt limit reached: attempts=%d", force_attempts
                )
                termination_reason = "force_answer_exhausted"
                break

            try:
                (
                    runtime_ids,
                    response_mask,
                    response_logprobs,
                    output,
                ) = await self._generate_with_retries(
                    request_id=request_id,
                    runtime_ids=runtime_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    sampling_params=sampling_params,
                    metrics=metrics,
                    stability=stability,
                    trace_logger=trace_logger,
                    turn=total_assistant_turns + 1,
                    max_attempts=(
                        self.force_answer_timeout_attempts if forced_answer else None
                    ),
                    retry_timeouts_only=forced_answer,
                )
            except GenerationTimedOut:
                force_attempts += 1
                stability["force_answer_attempts"] += 1
                stability["force_answer_timeout_count"] += self.force_answer_timeout_attempts
                trace_logger.warning(
                    "Forced-answer API timed out %d times in one round; terminating rollout",
                    self.force_answer_timeout_attempts,
                )
                termination_reason = "force_answer_model_call_failed"
                break
            except ContentFiltered:
                content_filter_count += 1
                trace_logger.warning("Content filter hit: count=%d", content_filter_count)
                stability["content_filter_hits"] += 1
                if (
                    content_filter_count <= self.content_filter_max_redirects
                    and not forced_answer
                ):
                    (
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                    ) = await self._append_user_message(
                        messages,
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                        CONTENT_FILTER_REDIRECT,
                    )
                    continue
                if not forced_answer:
                    (
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                    ) = await self._begin_force_answer(
                        messages,
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                        "<tool_response>Repeated content filtering disabled further tool use.</tool_response>",
                    )
                    forced_answer = True
                    termination_reason = "content_filter_limit"
                    trace_logger.warning("Entering forced-answer mode: repeated content filters")
                    continue
                force_attempts += 1
                stability["force_answer_attempts"] += 1
                if force_attempts < self.force_answer_max_attempts:
                    (
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                    ) = await self._append_user_message(
                        messages,
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                        FORCE_INVALID_RETRY,
                    )
                    continue
                termination_reason = "content_filter_during_force"
                break
            except GenerationExhausted as error:
                trace_logger.warning("OpenSeeker generation stopped: %s", error)
                if "response token budget exhausted" in str(error):
                    termination_reason = "response_token_limit"
                    break
                if not forced_answer:
                    (
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                    ) = await self._begin_force_answer(
                        messages,
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                        "<tool_response>The model API failed repeatedly; entering forced-answer recovery."
                        "</tool_response>",
                    )
                    forced_answer = True
                    termination_reason = "generation_failure_recovery"
                    trace_logger.warning("Entering forced-answer mode: repeated generation failure")
                    continue
                force_attempts += 1
                stability["force_answer_attempts"] += 1
                if force_attempts < self.force_answer_max_attempts:
                    await asyncio.sleep(self.generation_retry_base_seconds)
                    continue
                termination_reason = "generation_exhausted"
                break

            metrics["num_preempted"] += output.num_preempted or 0
            if output.routed_experts is not None:
                latest_routed_experts = output.routed_experts
            total_assistant_turns += 1
            content = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            turn_format_ok, turn_kind, turn_format_reason = validate_assistant_turn_format(content)
            assistant_turn_formats.append(
                {
                    "turn": total_assistant_turns,
                    "valid": turn_format_ok,
                    "kind": turn_kind,
                    "reason": turn_format_reason,
                }
            )
            if not turn_format_ok:
                stability["invalid_assistant_formats"] += 1
                trace_logger.warning(
                    "Invalid assistant format: turn=%d reason=%s",
                    total_assistant_turns,
                    turn_format_reason,
                )
            messages.append({"role": "assistant", "content": content})
            trace_logger.debug(
                "Assistant output: turn=%d chars=%d content=%s",
                total_assistant_turns, len(content), content,
            )
            if forced_answer:
                force_attempts += 1
                stability["force_answer_attempts"] += 1
            else:
                regular_turns += 1

            parsed_content = content.split("<tool_response>", 1)[0]
            answer = extract_answer_after_think(parsed_content)
            if answer is not None:
                if (
                    not forced_answer
                    and self.early_answer_turns > 0
                    and regular_turns <= self.early_answer_turns
                    and elapsed < self.early_answer_seconds
                    and not early_answer_blocked
                ):
                    early_answer_blocked = True
                    stability["early_answers_blocked"] += 1
                    trace_logger.info("Early answer blocked for verification: turn=%d", total_assistant_turns)
                    (
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                    ) = await self._append_user_message(
                        messages,
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                        EARLY_ANSWER_VERIFY,
                    )
                    continue
                answered = True
                termination_reason = "answered"
                trace_logger.info("Final answer detected: turn=%d", total_assistant_turns)
                break

            if forced_answer:
                if force_attempts >= self.force_answer_max_attempts:
                    termination_reason = "force_answer_exhausted"
                    break
                if "<tool_call>" in parsed_content:
                    trace_logger.warning(
                        "Tool call rejected during forced-answer mode: turn=%d", total_assistant_turns
                    )
                    (
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                    ) = await self._append_user_messages(
                        messages,
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                        [FORCE_TOOL_UNAVAILABLE, FORCE_TOOL_RETRY],
                    )
                else:
                    (
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                    ) = await self._append_user_message(
                        messages,
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                        FORCE_INVALID_RETRY,
                    )
                continue

            if len(parsed_content.strip()) < self.stalled_output_min_chars:
                stability["stalled_outputs"] += 1
                trace_logger.warning("Stalled assistant output: turn=%d", total_assistant_turns)
                (
                    runtime_ids,
                    response_mask,
                    response_logprobs,
                ) = await self._append_user_message(
                    messages,
                    runtime_ids,
                    response_mask,
                    response_logprobs,
                    NO_ACTION_NUDGE,
                )
                continue

            try:
                parsed_call = extract_tool_call(parsed_content)
            except Exception as error:
                tool_errors += 1
                stability["invalid_tool_calls"] += 1
                trace_logger.warning("Invalid tool call: turn=%d error=%s", total_assistant_turns, error)
                error_message = (
                    'Error: Tool call is not valid JSON. Tool call must contain a valid "name" and '
                    f'"arguments" field. Error: {error}'
                )
                (
                    runtime_ids,
                    response_mask,
                    response_logprobs,
                ) = await self._append_user_message(
                    messages,
                    runtime_ids,
                    response_mask,
                    response_logprobs,
                    f"<tool_response>\n{error_message}\n</tool_response>",
                )
                continue

            if parsed_call is None:
                stability["no_action_outputs"] += 1
                trace_logger.warning("Assistant output had neither tool call nor answer: turn=%d", total_assistant_turns)
                (
                    runtime_ids,
                    response_mask,
                    response_logprobs,
                ) = await self._append_user_message(
                    messages,
                    runtime_ids,
                    response_mask,
                    response_logprobs,
                    NO_ACTION_NUDGE,
                )
                continue

            name, arguments = parsed_call
            signature = tool_signature(name, arguments)
            attempt_counts[signature] += 1
            attempt_count = attempt_counts[signature]
            if attempt_count > self.max_identical_tool_executions:
                stability["duplicate_blocks"] += 1
                if attempt_count >= self.max_identical_tool_attempts:
                    trace_logger.warning(
                        "Repeated tool call reached attempt limit: turn=%d tool=%s attempts=%d",
                        total_assistant_turns, name, attempt_count,
                    )
                    (
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                    ) = await self._begin_force_answer(
                        messages,
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                        "<tool_response>\nThis tool call was not executed because this research direction has reached "
                        "its repeated-attempt limit. Tool execution is no longer available; the system has "
                        "entered the forced-answer phase.\n</tool_response>",
                    )
                    forced_answer = True
                    termination_reason = "duplicate_attempt_limit"
                    continue

                consecutive_blocked_duplicates += 1
                if (
                    consecutive_blocked_duplicates
                    >= self.max_consecutive_blocked_duplicates
                ):
                    trace_logger.warning(
                        "Consecutive duplicate blocks reached limit: turn=%d tool=%s blocked=%d",
                        total_assistant_turns, name, consecutive_blocked_duplicates,
                    )
                    (
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                    ) = await self._begin_force_answer(
                        messages,
                        runtime_ids,
                        response_mask,
                        response_logprobs,
                        "<tool_response>\nThis tool call was not executed because repeated research directions have "
                        "been blocked multiple times. Tool execution is no longer available; the system has "
                        "entered the forced-answer phase.\n</tool_response>",
                    )
                    forced_answer = True
                    termination_reason = "consecutive_duplicate_limit"
                    continue

                trace_logger.warning(
                    "Blocking repeated tool call: turn=%d tool=%s attempts=%d",
                    total_assistant_turns, name, attempt_count,
                )
                (
                    runtime_ids,
                    response_mask,
                    response_logprobs,
                ) = await self._append_user_messages(
                    messages,
                    runtime_ids,
                    response_mask,
                    response_logprobs,
                    [DUPLICATE_BLOCK_RESPONSE, DUPLICATE_REASSESS],
                )
                continue

            tool_started = time.perf_counter()
            trace_logger.info(
                "Tool start: turn=%d tool=%s arguments=%s",
                total_assistant_turns, name, json.dumps(arguments, ensure_ascii=False, default=str),
            )
            self._emit_telemetry(request_id, "tool_start", turn=total_assistant_turns, tool=name)
            try:
                result: ToolResult = await asyncio.wait_for(
                    self.environment.execute(
                        name, arguments, sample_id=request_id
                    ),
                    timeout=self.tool_timeout_seconds,
                )
            except TimeoutError:
                result = ToolResult(
                    content=f"Error executing {name}: tool timed out after {self.tool_timeout_seconds:.0f}s",
                    success=False,
                    error="tool_timeout",
                )
                stability["tool_timeouts"] += 1
                trace_logger.error(
                    "Tool timeout: turn=%d tool=%s timeout_s=%.1f",
                    total_assistant_turns, name, self.tool_timeout_seconds,
                )
            tool_elapsed = time.perf_counter() - tool_started
            metrics["tool_calls"] += tool_elapsed
            self._emit_telemetry(
                request_id, "tool_complete", turn=total_assistant_turns, tool=name,
                elapsed_s=tool_elapsed, success=result.success, cached=result.cached, has_error=bool(result.error),
            )
            if result.success:
                trace_logger.info(
                    "Tool complete: turn=%d tool=%s elapsed_s=%.3f cached=%s",
                    total_assistant_turns, name, tool_elapsed, result.cached,
                )
            else:
                trace_logger.error(
                    "Tool failed: turn=%d tool=%s elapsed_s=%.3f cached=%s error=%s",
                    total_assistant_turns, name, tool_elapsed, result.cached, result.error,
                )
            tool_calls += 1
            tool_errors += int(not result.success)
            cached_tool_calls += int(result.cached)
            execution_counts[signature] += 1
            consecutive_blocked_duplicates = 0
            (
                runtime_ids,
                response_mask,
                response_logprobs,
            ) = await self._append_user_message(
                messages,
                runtime_ids,
                response_mask,
                response_logprobs,
                f"<tool_response>\n{result.content}\n</tool_response>",
            )

        response_ids = runtime_ids[-len(response_mask) :] if response_mask else []
        prompt_ids = (
            runtime_ids[: len(runtime_ids) - len(response_mask)]
            if response_mask
            else runtime_ids
        )
        extra_fields: dict[str, Any] = {
            # Used by post-rollout reward and training log writers.
            "trace_id": request_id,
            "tool_calls": tool_calls,
            "tool_errors": tool_errors,
            "cached_tool_calls": cached_tool_calls,
            "tool_execution_counts": sum(execution_counts.values()),
            "assistant_turn_format_ok": bool(assistant_turn_formats)
            and all(turn["valid"] for turn in assistant_turn_formats),
            "format_valid_assistant_turns": sum(
                int(turn["valid"]) for turn in assistant_turn_formats
            ),
            "format_invalid_assistant_turns": sum(
                int(not turn["valid"]) for turn in assistant_turn_formats
            ),
            "assistant_turn_formats": assistant_turn_formats,
            "answered": answered,
            "forced_answer": forced_answer,
            "termination_reason": termination_reason,
            "elapsed_seconds": time.monotonic() - started_at,
            "regular_assistant_turns": regular_turns,
            "total_assistant_turns": total_assistant_turns,
            "turn_scores": [],
            "tool_rewards": [],
        }
        extra_fields.update(stability)
        self._emit_telemetry(
            request_id, "trajectory_complete", elapsed_s=extra_fields["elapsed_seconds"],
            response_tokens=len(response_ids), messages=len(messages), tool_calls=tool_calls,
            tool_errors=tool_errors, cached_tool_calls=cached_tool_calls,
            regular_assistant_turns=regular_turns, total_assistant_turns=total_assistant_turns,
            termination_reason=termination_reason, answered=answered, forced_answer=forced_answer,
        )
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length]
            if response_logprobs
            else None,
            routed_experts=latest_routed_experts,
            num_turns=len(messages),
            metrics=metrics,
            extra_fields=extra_fields,
        )
        self._write_trajectory(
            request_id,
            started_at=started_wallclock_at,
            status="completed",
            messages=messages,
            extra_fields=extra_fields,
            trace_logger=trace_logger,
        )
        trace_logger.info(
            "Trajectory complete: reason=%s elapsed_s=%.3f messages=%d tool_calls=%d",
            termination_reason, extra_fields["elapsed_seconds"], len(messages), tool_calls,
        )
        trace_logger.close()
        return output
