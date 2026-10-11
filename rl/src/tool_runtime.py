"""Asynchronous, cached deep-search environment for OpenSeeker rollouts.

This module intentionally does not import ``Project/agent``.  RL rollouts need a
process-safe, async environment with reproducible caching and without a second
policy-model call hidden inside the agent loop.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import ipaddress
import json
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.rollout_artifacts import TraceLogger

try:
    from diskcache import Cache
except ImportError:  # pragma: no cover - deployment dependency, not core logic
    Cache = None


TOOL_ENV_VERSION = "openseeker-tools-v4"
SEARCH_CACHE_VERSION = f"{TOOL_ENV_VERSION}-result-15k"
VISIT_CACHE_VERSION = f"{TOOL_ENV_VERSION}-result-15k"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DMX_CONFIG_PATH = PROJECT_ROOT / ".secrets" / "tool_api_keys.env"
DEFAULT_DMX_BASE_URL = "https://www.dmxapi.cn/v1"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs" / "tool_cache"
DEFAULT_SERPER_BLACKLIST_PATH = (
    PROJECT_ROOT / "outputs" / "tool_state" / "serper_invalid_keys.json"
)
DEFAULT_SERPER_LOG_DIR = PROJECT_ROOT / "outputs" / "tool_logs" / "search"
VISIT_STAGE_TIMEOUT_SECONDS = 40.0
VISIT_MAX_ATTEMPTS = 3
SUMMARY_CALL_TIMEOUT_SECONDS = 20.0
SUMMARY_MAX_ATTEMPTS = 3
VISIT_RETRY_DELAYS = (0.5, 1.5)

# Kept textually aligned with Project/agent/prompts.py::EXTRACTOR_PROMPT so the
# qwen-plus visit summarizer sees the same task and JSON schema in both agents.
EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**
"""


class SearchArgs(BaseModel):
    query: list[str] = Field(min_length=1, max_length=30)
    engine: list[str] | str | None = None

    @field_validator("query")
    @classmethod
    def validate_queries(cls, values: list[str]) -> list[str]:
        cleaned = [
            " ".join(value.split()) for value in values if value and value.strip()
        ]
        if not cleaned:
            raise ValueError("at least one non-empty query is required")
        return [value[:500] for value in cleaned]


class VisitArgs(BaseModel):
    url: list[str] | str
    goal: str = Field(min_length=1, max_length=2000)

    @field_validator("url")
    @classmethod
    def validate_urls(cls, value: list[str] | str) -> list[str]:
        values = [value] if isinstance(value, str) else value
        cleaned = [item.strip() for item in values if item and item.strip()]
        if not cleaned:
            raise ValueError("at least one URL is required")
        if len(cleaned) > 4:
            raise ValueError("at most four URLs may be visited in one call")
        return cleaned


@dataclass(slots=True)
class ToolResult:
    content: str
    success: bool
    cached: bool = False
    latency_seconds: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class _SearchQueryOutcome:
    content: str
    kind: str


@dataclass(slots=True)
class _VisitFetchResult:
    success: bool
    source: str
    content: str = ""
    http_status: int | None = None
    error_code: str | None = None
    action: str | None = None
    detail: str = ""
    final_url: str | None = None
    attempts: int = 1


@dataclass(slots=True)
class _VisitBatchOutcome:
    content: str
    success: bool
    cacheable: bool
    error: str | None = None


@dataclass(slots=True)
class _SearchBatchOutcome:
    content: str
    success: bool
    cacheable: bool
    error: str | None = None


class _PersistentSerperBlacklist:
    """Process-safe, manually reversible blacklist for invalid Serper keys."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @staticmethod
    def fingerprint(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "keys": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("keys"), dict):
            raise ValueError("invalid Serper blacklist structure")
        return data

    def fingerprints(self) -> set[str]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                return set(self._read_unlocked()["keys"])
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def add(
        self,
        key: str,
        *,
        http_status: int,
        reason: str,
        key_index: int,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = self.fingerprint(key)
        now = datetime.now(UTC).isoformat()
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            temporary_path: Path | None = None
            try:
                data = self._read_unlocked()
                record = data["keys"].get(fingerprint, {})
                data["keys"][fingerprint] = {
                    "failure_class": "invalid_key",
                    "http_status": http_status,
                    "key_index": key_index,
                    "first_seen_at": record.get("first_seen_at", now),
                    "last_seen_at": now,
                    "occurrences": int(record.get("occurrences", 0)) + 1,
                    "reason": reason[:300],
                }
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    delete=False,
                ) as temporary:
                    json.dump(data, temporary, ensure_ascii=False, indent=2, sort_keys=True)
                    temporary.write("\n")
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    temporary_path = Path(temporary.name)
                os.replace(temporary_path, self.path)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class _SerperEventWriter:
    """Append key-safe JSONL events to one file per rollout sample."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    @staticmethod
    def _safe_name(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:128]
        return cleaned or "unknown"

    def write(self, sample_id: str, event: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self._safe_name(sample_id)}.jsonl"
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "sample_id": sample_id,
            **event,
        }
        with path.open("a", encoding="utf-8") as output:
            fcntl.flock(output.fileno(), fcntl.LOCK_EX)
            try:
                output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()
            finally:
                fcntl.flock(output.fileno(), fcntl.LOCK_UN)


class _ToolCache:
    def __init__(self, directory: str | Path, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._cache = Cache(str(directory)) if Cache is not None else None

    def get(self, key: str) -> str | None:
        if self._cache is None:
            return None
        value = self._cache.get(key)
        return value if isinstance(value, str) else None

    def set(self, key: str, value: str) -> None:
        if self._cache is not None:
            self._cache.set(key, value, expire=self.ttl_seconds)


def _read_assignment_file(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE assignments without executing shell code."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name] = value
    return values


class DeepSearchEnvironment:
    """Search/visit tools used as the external environment during rollout."""

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        cache_ttl_seconds: int = 7 * 24 * 3600,
        max_concurrency: int = 16,
        max_search_result_chars: int = 15_000,
        max_visit_result_chars: int = 15_000,
        request_timeout_seconds: float = 10.0,
        serper_blacklist_path: str | Path | None = None,
        serper_log_dir: str | Path | None = None,
    ):
        resolved_cache_dir = Path(
            cache_dir or os.getenv("OPENSEEKER_TOOL_CACHE", DEFAULT_CACHE_DIR)
        )
        self.cache = _ToolCache(resolved_cache_dir, cache_ttl_seconds)
        # Search is aligned with the original agent's 15k observation cap.
        # Visit uses the same 15k observation cap as the original agent.
        self.max_search_result_chars = max_search_result_chars
        self.max_visit_result_chars = max_visit_result_chars
        self.request_timeout_seconds = max(
            0.1,
            float(os.getenv("SERPER_HTTP_TIMEOUT_SECONDS", request_timeout_seconds)),
        )
        self.serper_same_key_max_retries = max(
            0, int(os.getenv("SERPER_SAME_KEY_MAX_RETRIES", "3"))
        )
        self.serper_max_key_switches = max(
            0, int(os.getenv("SERPER_MAX_KEY_SWITCHES", "5"))
        )
        self.serper_retry_base_seconds = max(
            0.0, float(os.getenv("SERPER_RETRY_BASE_SECONDS", "0.25"))
        )
        isolated_state_root = resolved_cache_dir if cache_dir is not None else None
        blacklist_path = serper_blacklist_path or os.getenv(
            "SERPER_BLACKLIST_PATH",
            str(
                isolated_state_root / "serper_invalid_keys.json"
                if isolated_state_root is not None
                else DEFAULT_SERPER_BLACKLIST_PATH
            ),
        )
        log_dir = serper_log_dir or os.getenv(
            "OPENSEEKER_SEARCH_LOG_DIR",
            str(
                isolated_state_root / "serper_logs"
                if isolated_state_root is not None
                else DEFAULT_SERPER_LOG_DIR
            ),
        )
        self._serper_blacklist = _PersistentSerperBlacklist(blacklist_path)
        self._serper_event_writer = _SerperEventWriter(log_dir)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._serper_keys = self._split_keys("SERPER_API_KEYS")
        configured_jina_keys = self._split_keys("JINA_API_KEYS")
        self._jina_key = (
            os.getenv("JINA_API_KEY", "").strip()
            or (configured_jina_keys[0] if configured_jina_keys else "")
        )
        dmx_path = Path(
            os.getenv("OPENSEEKER_DMX_CONFIG", DEFAULT_DMX_CONFIG_PATH)
        )
        dmx = _read_assignment_file(dmx_path)
        self._summary_base_url = (
            os.getenv("DMX_BASE_URL", "").strip()
            or dmx.get("DMX_BASE_URL", "").strip()
            or os.getenv("TOOL_BASE_URL", "").strip()
            or dmx.get("TOOL_BASE_URL", "").strip()
            or DEFAULT_DMX_BASE_URL
        )
        self._summary_api_key = (
            os.getenv("DMX_API_KEY", "").strip()
            or dmx.get("DMX_API_KEY", "").strip()
            or os.getenv("TOOL_API_KEY", "").strip()
            or dmx.get("TOOL_API_KEY", "").strip()
        )
        self._summary_model = (
            os.getenv("DMX_MODEL", "").strip()
            or dmx.get("DMX_MODEL", "").strip()
            or os.getenv("SUMMARY_MODEL", "").strip()
            or dmx.get("SUMMARY_MODEL", "").strip()
            or "qwen-plus"
        )
        self._summary_enabled = os.getenv(
            "OPENSEEKER_VISIT_SUMMARIZE", "1"
        ).lower() in {"1", "true", "yes"}
        self._jina_enabled = os.getenv(
            "JINA_ENABLED", "1"
        ).lower() in {"1", "true", "yes"}

    @staticmethod
    def _split_keys(name: str) -> list[str]:
        return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]

    @staticmethod
    def _cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
        payload = json.dumps(
            {"version": SEARCH_CACHE_VERSION if tool_name == "search" else VISIT_CACHE_VERSION, "tool": tool_name, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _truncate(self, value: str, *, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "\n\n[tool result truncated]"

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        sample_id: str | None = None,
    ) -> ToolResult:
        tool_call_id = uuid4().hex
        resolved_sample_id = sample_id or f"standalone-{tool_call_id}"
        trace_logger = TraceLogger(resolved_sample_id)
        trace_logger.info(
            "Tool runtime start: tool=%s tool_call_id=%s", tool_name, tool_call_id
        )
        try:
            return await self._execute_with_trace(
                tool_name,
                arguments,
                sample_id=resolved_sample_id,
                tool_call_id=tool_call_id,
                trace_logger=trace_logger,
            )
        finally:
            trace_logger.close()

    @staticmethod
    def _terminal_tool_result(
        trace_logger: TraceLogger,
        tool_name: str,
        tool_call_id: str,
        result: ToolResult,
    ) -> ToolResult:
        if result.success:
            trace_logger.info(
                "TOOL_TERMINAL outcome=success tool=%s tool_call_id=%s cached=%s latency_s=%.3f",
                tool_name, tool_call_id, result.cached, result.latency_seconds,
            )
        else:
            # A tool failure is recoverable by the AgentLoop, which can try a
            # different direction or produce a forced answer. It is still a
            # monitor-worthy terminal outcome for this invocation.
            trace_logger.warning(
                "TOOL_TERMINAL outcome=failure tool=%s tool_call_id=%s cached=%s latency_s=%.3f error=%s",
                tool_name, tool_call_id, result.cached, result.latency_seconds,
                result.error or "unknown",
            )
        return result

    async def _execute_with_trace(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        sample_id: str,
        tool_call_id: str,
        trace_logger: TraceLogger,
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            if tool_name == "search":
                parsed = SearchArgs.model_validate(arguments)
                normalized = parsed.model_dump(mode="json")
                trace_logger.info(
                    "Search arguments accepted: tool_call_id=%s query_count=%d",
                    tool_call_id, len(parsed.query),
                )
            elif tool_name == "visit":
                parsed = VisitArgs.model_validate(arguments)
                normalized = parsed.model_dump(mode="json")
                trace_logger.info(
                    "Visit arguments accepted: tool_call_id=%s url_count=%d",
                    tool_call_id, len(parsed.url),
                )
            else:
                return self._terminal_tool_result(
                    trace_logger, tool_name, tool_call_id,
                    ToolResult(
                        content=f"Error: unknown tool {tool_name!r}",
                        success=False,
                        latency_seconds=time.perf_counter() - started,
                        error="unknown_tool",
                    ),
                )
        except ValidationError as error:
            trace_logger.warning(
                "Tool argument validation failed: tool=%s tool_call_id=%s error=%s",
                tool_name, tool_call_id, error.errors()[0].get("type", "validation_error"),
            )
            return self._terminal_tool_result(
                trace_logger, tool_name, tool_call_id,
                ToolResult(
                    content=f"Error: invalid arguments for {tool_name}: {error}",
                    success=False,
                    latency_seconds=time.perf_counter() - started,
                    error="invalid_arguments",
                ),
            )

        cache_key = self._cache_key(tool_name, normalized)
        cached = await asyncio.to_thread(self.cache.get, cache_key)
        if cached is not None:
            trace_logger.info(
                "Tool cache hit: tool=%s tool_call_id=%s", tool_name, tool_call_id
            )
            if tool_name == "search":
                await self._emit_serper_event(
                    sample_id,
                    {"event": "search_cache_hit", "tool_call_id": tool_call_id},
                )
            return self._terminal_tool_result(
                trace_logger, tool_name, tool_call_id,
                ToolResult(
                    content=cached,
                    success=True,
                    cached=True,
                    latency_seconds=time.perf_counter() - started,
                ),
            )

        try:
            async with self._semaphore:
                if tool_name == "search":
                    search_outcome = await self._search(
                        parsed,
                        sample_id=sample_id,
                        tool_call_id=tool_call_id,
                        trace_logger=trace_logger,
                    )
                    content = search_outcome.content
                    success = search_outcome.success
                    cacheable = search_outcome.cacheable
                    execution_error = search_outcome.error
                else:
                    visit_outcome = await self._visit(parsed, trace_logger=trace_logger)
                    content = visit_outcome.content
                    success = visit_outcome.success
                    cacheable = visit_outcome.cacheable
                    execution_error = visit_outcome.error
            max_chars = (
                self.max_search_result_chars
                if tool_name == "search"
                else self.max_visit_result_chars
            )
            content = self._truncate(content, max_chars=max_chars)
            if cacheable:
                await asyncio.to_thread(self.cache.set, cache_key, content)
            return self._terminal_tool_result(
                trace_logger, tool_name, tool_call_id,
                ToolResult(
                    content=content,
                    success=success,
                    latency_seconds=time.perf_counter() - started,
                    error=execution_error,
                ),
            )
        except Exception as error:  # tools return observations; rollouts must survive
            trace_logger.warning(
                "Tool runtime exception after fallbacks: tool=%s tool_call_id=%s error=%s",
                tool_name, tool_call_id, type(error).__name__,
            )
            message = f"Error executing {tool_name}: {type(error).__name__}: {error}"
            return self._terminal_tool_result(
                trace_logger, tool_name, tool_call_id,
                ToolResult(
                    content=message,
                    success=False,
                    latency_seconds=time.perf_counter() - started,
                    error=type(error).__name__,
                ),
            )

    async def _emit_serper_event(
        self,
        sample_id: str,
        event: dict[str, Any],
    ) -> None:
        try:
            await asyncio.to_thread(self._serper_event_writer.write, sample_id, event)
        except Exception:
            # Observability must never make the search tool unavailable.
            return

    async def _search(
        self,
        args: SearchArgs,
        *,
        sample_id: str,
        tool_call_id: str,
        trace_logger: TraceLogger,
    ) -> _SearchBatchOutcome:
        engines = args.engine
        if isinstance(engines, str):
            engines = [engines]
        engines = list(engines or [])
        engines.extend([""] * (len(args.query) - len(engines)))
        outcomes = await asyncio.gather(
            *(
                self._search_one(
                    query,
                    engine,
                    sample_id=sample_id,
                    tool_call_id=tool_call_id,
                    trace_logger=trace_logger,
                )
                for query, engine in zip(args.query, engines, strict=False)
            )
        )
        failed = [outcome for outcome in outcomes if outcome.kind not in {"success", "empty"}]
        if failed:
            trace_logger.warning(
                "Search fallbacks exhausted: tool_call_id=%s query_count=%d failed_queries=%d error_codes=%s",
                tool_call_id, len(args.query), len(failed),
                ",".join(sorted({outcome.kind for outcome in failed})),
            )
        elif all(outcome.kind == "empty" for outcome in outcomes):
            trace_logger.info(
                "Search completed with zero results: tool_call_id=%s query_count=%d",
                tool_call_id, len(args.query),
            )
        else:
            trace_logger.info(
                "Search completed: tool_call_id=%s query_count=%d", tool_call_id, len(args.query)
            )
        return _SearchBatchOutcome(
            content="\n\n---\n\n".join(outcome.content for outcome in outcomes),
            success=not failed,
            cacheable=bool(outcomes) and all(outcome.kind == "success" for outcome in outcomes),
            error=failed[0].kind if failed else None,
        )

    async def _search_one(
        self,
        query: str,
        engine: str,
        *,
        sample_id: str = "standalone",
        tool_call_id: str = "standalone",
        trace_logger: TraceLogger | None = None,
    ) -> _SearchQueryOutcome:
        # Keep accepting engine for trajectory compatibility; Serper is exclusive.
        del engine
        attempted_keys: set[str] = set()
        result = await self._serper_search(
            query,
            sample_id=sample_id,
            tool_call_id=tool_call_id,
            attempted_keys=attempted_keys,
        )
        if result.kind != "empty":
            return result

        if trace_logger is not None:
            trace_logger.debug(
                "Search returned zero results; starting query recovery: tool_call_id=%s",
                tool_call_id,
            )
        cleaned = self._clean_query(query)
        if cleaned != query:
            retry = await self._serper_search(
                cleaned,
                sample_id=sample_id,
                tool_call_id=tool_call_id,
                attempted_keys=attempted_keys,
            )
            if retry.kind == "success":
                return retry
            if retry.kind != "empty":
                return result
            result = retry

        simplified = self._simplify_query(cleaned)
        if simplified != cleaned:
            retry = await self._serper_search(
                simplified,
                sample_id=sample_id,
                tool_call_id=tool_call_id,
                attempted_keys=attempted_keys,
            )
            if retry.kind == "success":
                return retry
            if retry.kind == "empty":
                result = retry
        return result

    @staticmethod
    def _clean_query(query: str) -> str:
        """Widen over-constrained exact-match queries without changing terms."""
        cleaned = query.strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        return " ".join(cleaned.replace('" "', " ").replace('"', "").split())

    @staticmethod
    def _simplify_query(query: str) -> str:
        """Remove common connective words while retaining the core entities."""
        if len(query) < 10:
            return query
        simplified = re.sub(
            r"的|which|that|whose|who|when|where",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        simplified = re.sub(r"\s+", " ", simplified).strip()
        if len(simplified) < len(query) * 0.3:
            return query
        return simplified

    @staticmethod
    def _empty_search_outcome(query: str) -> _SearchQueryOutcome:
        return _SearchQueryOutcome(
            content=f"### A Google search for {query!r} found 0 results:\n",
            kind="empty",
        )

    @staticmethod
    def _unavailable_search_outcome(query: str) -> _SearchQueryOutcome:
        return _SearchQueryOutcome(
            content=(
                f"Error: Serper search could not be completed for {query!r}.\n"
                "All allowed key attempts failed. No search result was obtained."
            ),
            kind="serper_unavailable",
        )

    @staticmethod
    def _key_fingerprint(key: str) -> str:
        return _PersistentSerperBlacklist.fingerprint(key)

    def _safe_serper_reason(self, value: str) -> str:
        reason = " ".join((value or "").split())
        for key in self._serper_keys:
            reason = reason.replace(key, "[REDACTED]")
        return reason[:300] or "no response body"

    @staticmethod
    def _is_permanent_key_failure(status_code: int, reason: str) -> bool:
        if status_code == 401:
            return True
        if status_code != 403:
            return False
        lowered = reason.lower()
        return any(
            marker in lowered
            for marker in (
                "invalid api key",
                "api key is invalid",
                "key has been revoked",
                "api key revoked",
                "account disabled",
                "key disabled",
                "api key unauthorized",
            )
        )

    async def _record_serper_attempt(
        self,
        *,
        sample_id: str,
        tool_call_id: str,
        query: str,
        key: str,
        key_index: int,
        key_attempt: int,
        switch_count: int,
        failure_class: str | None,
        http_status: int | None,
        latency_seconds: float,
        reason: str | None = None,
        blacklisted_permanently: bool = False,
    ) -> None:
        await self._emit_serper_event(
            sample_id,
            {
                "event": "serper_attempt",
                "tool_call_id": tool_call_id,
                "query": query,
                "key_index": key_index + 1,
                "key_fingerprint": self._key_fingerprint(key)[:12],
                "key_attempt": key_attempt,
                "switch_count": switch_count,
                "failure_class": failure_class,
                "http_status": http_status,
                "latency_ms": round(latency_seconds * 1000, 3),
                "reason": reason,
                "blacklisted_permanently": blacklisted_permanently,
            },
        )

    async def _serper_search(
        self,
        query: str,
        *,
        sample_id: str = "standalone",
        tool_call_id: str = "standalone",
        attempted_keys: set[str] | None = None,
    ) -> _SearchQueryOutcome:
        if not self._serper_keys:
            return _SearchQueryOutcome(
                content="Error: SERPER_API_KEYS is not configured",
                kind="serper_not_configured",
            )

        attempted = attempted_keys if attempted_keys is not None else set()
        try:
            permanently_blocked = await asyncio.to_thread(
                self._serper_blacklist.fingerprints
            )
        except Exception as error:
            permanently_blocked = set()
            await self._emit_serper_event(
                sample_id,
                {
                    "event": "serper_blacklist_read_failed",
                    "tool_call_id": tool_call_id,
                    "failure_class": "blacklist_io_error",
                    "reason": self._safe_serper_reason(str(error)),
                },
            )

        start = int(hashlib.sha256(query.encode()).hexdigest(), 16) % len(
            self._serper_keys
        )
        candidates: list[tuple[int, str]] = []
        selected_fingerprints: set[str] = set()
        for offset in range(len(self._serper_keys)):
            key_index = (start + offset) % len(self._serper_keys)
            key = self._serper_keys[key_index]
            fingerprint = self._key_fingerprint(key)
            if (
                fingerprint in permanently_blocked
                or fingerprint in attempted
                or fingerprint in selected_fingerprints
            ):
                continue
            candidates.append((key_index, key))
            selected_fingerprints.add(fingerprint)
            if len(candidates) >= self.serper_max_key_switches + 1:
                break

        if not candidates:
            outcome = self._unavailable_search_outcome(query)
            await self._emit_serper_event(
                sample_id,
                {
                    "event": "serper_search_finished",
                    "tool_call_id": tool_call_id,
                    "query": query,
                    "outcome": outcome.kind,
                    "reason": "no eligible Serper keys",
                },
            )
            return outcome

        transient_statuses = {408, 425, 500, 502, 503, 504}
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            for switch_count, (key_index, key) in enumerate(candidates):
                fingerprint = self._key_fingerprint(key)
                for key_attempt in range(1, self.serper_same_key_max_retries + 2):
                    request_started = time.perf_counter()
                    try:
                        response = await client.post(
                            "https://google.serper.dev/search",
                            headers={
                                "X-API-Key": key,
                                "Content-Type": "application/json",
                            },
                            json={"q": query, "num": 10},
                        )
                    except httpx.HTTPError as error:
                        reason = self._safe_serper_reason(str(error))
                        await self._record_serper_attempt(
                            sample_id=sample_id,
                            tool_call_id=tool_call_id,
                            query=query,
                            key=key,
                            key_index=key_index,
                            key_attempt=key_attempt,
                            switch_count=switch_count,
                            failure_class="transient",
                            http_status=None,
                            latency_seconds=time.perf_counter() - request_started,
                            reason=reason,
                        )
                        if key_attempt <= self.serper_same_key_max_retries:
                            delay = self.serper_retry_base_seconds * (2 ** (key_attempt - 1))
                            if delay:
                                await asyncio.sleep(delay + random.uniform(0, delay * 0.25))
                            continue
                        attempted.add(fingerprint)
                        break

                    status_code = response.status_code
                    reason = self._safe_serper_reason(response.text)
                    latency = time.perf_counter() - request_started
                    if status_code == 200:
                        try:
                            data = response.json()
                            if not isinstance(data, dict):
                                raise ValueError("response JSON must be an object")
                            items = data.get("organic", [])
                            if not isinstance(items, list):
                                raise ValueError("organic results must be a list")
                        except (ValueError, TypeError) as error:
                            await self._record_serper_attempt(
                                sample_id=sample_id,
                                tool_call_id=tool_call_id,
                                query=query,
                                key=key,
                                key_index=key_index,
                                key_attempt=key_attempt,
                                switch_count=switch_count,
                                failure_class="transient",
                                http_status=status_code,
                                latency_seconds=latency,
                                reason=self._safe_serper_reason(str(error)),
                            )
                            if key_attempt <= self.serper_same_key_max_retries:
                                continue
                            attempted.add(fingerprint)
                            break

                        await self._record_serper_attempt(
                            sample_id=sample_id,
                            tool_call_id=tool_call_id,
                            query=query,
                            key=key,
                            key_index=key_index,
                            key_attempt=key_attempt,
                            switch_count=switch_count,
                            failure_class=None if items else "empty_result",
                            http_status=status_code,
                            latency_seconds=latency,
                        )
                        if items:
                            return _SearchQueryOutcome(
                                content=self._format_search_results(query, items, "Google"),
                                kind="success",
                            )
                        return self._empty_search_outcome(query)

                    if status_code in {400, 404, 422}:
                        await self._record_serper_attempt(
                            sample_id=sample_id,
                            tool_call_id=tool_call_id,
                            query=query,
                            key=key,
                            key_index=key_index,
                            key_attempt=key_attempt,
                            switch_count=switch_count,
                            failure_class="invalid_request",
                            http_status=status_code,
                            latency_seconds=latency,
                            reason=reason,
                        )
                        return _SearchQueryOutcome(
                            content=(
                                f"Error: Serper rejected the search request for {query!r} "
                                f"with HTTP {status_code}."
                            ),
                            kind="invalid_request",
                        )

                    permanent = self._is_permanent_key_failure(status_code, reason)
                    if permanent:
                        blacklist_written = True
                        try:
                            await asyncio.to_thread(
                                self._serper_blacklist.add,
                                key,
                                http_status=status_code,
                                reason=reason,
                                key_index=key_index + 1,
                            )
                            permanently_blocked.add(fingerprint)
                        except Exception as error:
                            blacklist_written = False
                            await self._emit_serper_event(
                                sample_id,
                                {
                                    "event": "serper_blacklist_write_failed",
                                    "tool_call_id": tool_call_id,
                                    "key_fingerprint": fingerprint[:12],
                                    "failure_class": "blacklist_io_error",
                                    "reason": self._safe_serper_reason(str(error)),
                                },
                            )
                        await self._record_serper_attempt(
                            sample_id=sample_id,
                            tool_call_id=tool_call_id,
                            query=query,
                            key=key,
                            key_index=key_index,
                            key_attempt=key_attempt,
                            switch_count=switch_count,
                            failure_class="invalid_key",
                            http_status=status_code,
                            latency_seconds=latency,
                            reason=reason,
                            blacklisted_permanently=blacklist_written,
                        )
                        attempted.add(fingerprint)
                        break

                    if status_code == 429:
                        await self._record_serper_attempt(
                            sample_id=sample_id,
                            tool_call_id=tool_call_id,
                            query=query,
                            key=key,
                            key_index=key_index,
                            key_attempt=key_attempt,
                            switch_count=switch_count,
                            failure_class="rate_limited",
                            http_status=status_code,
                            latency_seconds=latency,
                            reason=reason,
                        )
                        attempted.add(fingerprint)
                        break

                    if status_code in transient_statuses or status_code >= 500:
                        await self._record_serper_attempt(
                            sample_id=sample_id,
                            tool_call_id=tool_call_id,
                            query=query,
                            key=key,
                            key_index=key_index,
                            key_attempt=key_attempt,
                            switch_count=switch_count,
                            failure_class="transient",
                            http_status=status_code,
                            latency_seconds=latency,
                            reason=reason,
                        )
                        if key_attempt <= self.serper_same_key_max_retries:
                            delay = self.serper_retry_base_seconds * (2 ** (key_attempt - 1))
                            if delay:
                                await asyncio.sleep(delay + random.uniform(0, delay * 0.25))
                            continue
                        attempted.add(fingerprint)
                        break

                    await self._record_serper_attempt(
                        sample_id=sample_id,
                        tool_call_id=tool_call_id,
                        query=query,
                        key=key,
                        key_index=key_index,
                        key_attempt=key_attempt,
                        switch_count=switch_count,
                        failure_class="access_failure",
                        http_status=status_code,
                        latency_seconds=latency,
                        reason=reason,
                    )
                    attempted.add(fingerprint)
                    break

        outcome = self._unavailable_search_outcome(query)
        await self._emit_serper_event(
            sample_id,
            {
                "event": "serper_search_finished",
                "tool_call_id": tool_call_id,
                "query": query,
                "outcome": outcome.kind,
                "attempted_key_count": len(attempted),
            },
        )
        return outcome

    @staticmethod
    def _format_search_results(
        query: str, items: list[dict[str, Any]], engine: str
    ) -> str:
        lines = [f"### A {engine} search for {query!r} found {len(items)} results:"]
        for index, item in enumerate(items[:10], 1):
            title = item.get("title") or item.get("name") or "Untitled"
            url = item.get("link") or item.get("url") or ""
            snippet = (
                item.get("snippet") or item.get("summary") or item.get("content") or ""
            )
            published = item.get("date") or item.get("publishedTime") or ""
            entry = f"{index}. [{title}]({url})"
            if published:
                entry += f"\nDate published: {published}"
            if snippet:
                entry += f"\n{snippet}"
            lines.append(entry)
        return "\n\n".join(lines)

    async def _visit(
        self, args: VisitArgs, *, trace_logger: TraceLogger
    ) -> _VisitBatchOutcome:
        results = await asyncio.gather(
            *(self._visit_one(url, args.goal, trace_logger=trace_logger) for url in args.url)
        )
        failed_count = sum(result.startswith("[visit_failed]") for result in results)
        success = failed_count == 0
        if failed_count:
            trace_logger.warning(
                "Visit fallbacks exhausted: url_count=%d failed_urls=%d",
                len(args.url), failed_count,
            )
        else:
            trace_logger.info("Visit completed: url_count=%d", len(args.url))
        return _VisitBatchOutcome(
            content="\n\n---\n\n".join(results),
            success=success,
            cacheable=success,
            error=None if success else "visit_failed",
        )

    async def _visit_one(
        self, url: str, goal: str, *, trace_logger: TraceLogger | None = None
    ) -> str:
        trace_logger = trace_logger or TraceLogger("standalone-visit")
        await self._validate_visit_url(url)
        stages = []
        if self._jina_enabled:
            stages.append("anonymous_jina")
        stages.append("direct_http")
        if self._jina_enabled and self._jina_key:
            stages.append("authenticated_jina")

        last_result: _VisitFetchResult | None = None
        for index, source in enumerate(stages):
            result = await self._fetch_visit_stage(url, source)
            last_result = result
            if not result.success:
                if result.action == "stop":
                    trace_logger.warning(
                        "Visit stopped: source=%s code=%s attempts=%d",
                        source, result.error_code or "UNKNOWN", result.attempts,
                    )
                    return self._format_visit_failure(url, result)
                trace_logger.warning(
                    "Visit fallback: source=%s code=%s attempts=%d next_stage=%s",
                    source, result.error_code or "UNKNOWN", result.attempts,
                    stages[index + 1] if index + 1 < len(stages) else "none",
                )
                continue

            text = result.content
            if source == "direct_http":
                text = self._extract_main_text(text)
            if self._looks_like_blocked_page(text):
                blocked = _VisitFetchResult(
                    success=False,
                    source=source,
                    http_status=result.http_status,
                    error_code="TARGET_BLOCKED",
                    action="stop",
                    detail="response is an access-denied or challenge page",
                )
                return self._format_visit_failure(url, blocked)
            if len(text.strip()) < 20:
                last_result = _VisitFetchResult(
                    success=False,
                    source=source,
                    http_status=result.http_status,
                    error_code="CONTENT_UNUSABLE",
                    action="fallback",
                    detail="extracted content is empty or too short",
                )
                if index + 1 < len(stages):
                    trace_logger.warning(
                        "Visit fallback: source=%s code=CONTENT_UNUSABLE next_stage=%s",
                        source, stages[index + 1],
                    )
                    continue
                trace_logger.warning(
                    "Visit fallbacks exhausted: source=%s code=CONTENT_UNUSABLE", source
                )
                return self._format_visit_failure(url, last_result)

            if self._summary_enabled and self._summary_base_url and self._summary_api_key:
                summary = await self._summarize(text, goal)
                if "[Summary API failed after" in summary:
                    trace_logger.warning("Visit summary fallback used: source=%s", source)
            else:
                summary = text[: self.max_visit_result_chars]
            trace_logger.info(
                "Visit source succeeded: source=%s attempts=%d", source, result.attempts
            )
            return (
                f"The useful information in {url} for user goal {goal} as follows: \n\n"
                f"{summary}"
            )

        if last_result is None:
            last_result = _VisitFetchResult(
                success=False,
                source="visit",
                error_code="NO_FETCH_STAGE_AVAILABLE",
                action="stop",
            )
        trace_logger.warning(
            "Visit fallbacks exhausted: source=%s code=%s",
            last_result.source if last_result is not None else "none",
            last_result.error_code if last_result is not None and last_result.error_code else "UNKNOWN",
        )
        return self._format_visit_failure(url, last_result)

    async def _validate_visit_url(self, url: str) -> None:
        """Validate URL syntax without depending on local DNS answers.

        Outbound requests use the configured HTTP(S) proxy, whose resolver can
        legitimately differ from this process's resolver (for example with
        Mihomo fake-IP DNS). Block clearly unsafe literal IP targets, but do
        not pre-resolve normal hostnames here.
        """
        parsed = urlparse(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError("visit only accepts absolute http(s) URLs")
        hostname = parsed.hostname.lower().rstrip(".")
        if hostname in {"localhost", "localhost.localdomain"}:
            raise ValueError("local URLs are not allowed")
        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            return
        if not ip.is_global:
            raise ValueError(
                "private, loopback, link-local, and reserved literal IP URLs are not allowed"
            )

    @staticmethod
    def _fetch_failure(
        source: str,
        code: str,
        action: str,
        *,
        status: int | None = None,
        detail: str = "",
        final_url: str | None = None,
    ) -> _VisitFetchResult:
        return _VisitFetchResult(
            success=False,
            source=source,
            http_status=status,
            error_code=code,
            action=action,
            detail=detail,
            final_url=final_url,
        )

    def _classify_jina_error(
        self,
        source: str,
        status: int,
        body: str,
        final_url: str,
    ) -> _VisitFetchResult:
        normalized = body.lower()[:4000]
        common = {
            "status": status,
            "detail": body[:500],
            "final_url": final_url,
        }
        target_markers = (
            "robots.txt",
            "robots denied",
            "blocked by robots",
            "upstream forbidden",
            "forbidden by target",
            "blocked by the site",
            "captcha",
        )
        provider_markers = (
            "api key",
            "invalid token",
            "authorization",
            "unauthorized",
            "quota",
            "rate limit",
        )
        if any(marker in normalized for marker in target_markers):
            return self._fetch_failure(source, "TARGET_BLOCKED", "stop", **common)
        if status in (408, 425, 429, 500, 502, 503, 504):
            return self._fetch_failure(source, "JINA_TRANSIENT", "retry", **common)
        if status in (400, 422):
            return self._fetch_failure(source, "INVALID_URL", "stop", **common)
        if status in (404, 410):
            return self._fetch_failure(source, "TARGET_NOT_FOUND", "stop", **common)
        if status in (401, 402) or any(
            marker in normalized for marker in provider_markers
        ):
            action = "fallback" if source == "anonymous_jina" else "stop"
            return self._fetch_failure(
                source, "JINA_AUTH_OR_QUOTA", action, **common
            )
        if status == 403:
            action = "fallback" if source == "anonymous_jina" else "stop"
            return self._fetch_failure(
                source, "JINA_FORBIDDEN_UNKNOWN", action, **common
            )
        if 400 <= status < 500:
            return self._fetch_failure(source, "JINA_CLIENT_ERROR", "stop", **common)
        return self._fetch_failure(source, "JINA_TRANSIENT", "retry", **common)

    def _classify_direct_error(
        self,
        status: int,
        body: str,
        final_url: str,
    ) -> _VisitFetchResult:
        common = {
            "status": status,
            "detail": body[:500],
            "final_url": final_url,
        }
        if status in (408, 425, 500, 502, 503, 504):
            return self._fetch_failure(
                "direct_http", "TRANSIENT_HTTP", "retry", **common
            )
        terminal_codes = {
            400: "TARGET_BAD_REQUEST",
            401: "TARGET_UNAUTHORIZED",
            403: "TARGET_FORBIDDEN",
            404: "TARGET_NOT_FOUND",
            405: "TARGET_METHOD_NOT_ALLOWED",
            410: "TARGET_GONE",
            429: "TARGET_RATE_LIMITED",
            451: "TARGET_LEGAL_RESTRICTION",
        }
        if status in terminal_codes:
            return self._fetch_failure(
                "direct_http", terminal_codes[status], "stop", **common
            )
        if 400 <= status < 500:
            return self._fetch_failure(
                "direct_http", "TARGET_CLIENT_ERROR", "stop", **common
            )
        return self._fetch_failure(
            "direct_http", "TRANSIENT_HTTP", "retry", **common
        )

    def _classify_fetch_exception(
        self,
        source: str,
        error: Exception,
    ) -> _VisitFetchResult:
        if isinstance(error, httpx.InvalidURL):
            return self._fetch_failure(source, "INVALID_URL", "stop", detail=str(error))
        if isinstance(error, httpx.TimeoutException):
            code = "NETWORK_TIMEOUT"
        elif isinstance(error, (httpx.NetworkError, httpx.RemoteProtocolError)):
            code = "NETWORK_TRANSIENT"
        else:
            code = "FETCH_TRANSIENT"
        return self._fetch_failure(source, code, "retry", detail=str(error))

    async def _fetch_jina_once(
        self,
        url: str,
        source: str,
        timeout: float,
    ) -> _VisitFetchResult:
        headers = {"Accept": "text/markdown", "X-No-Cache": "true"}
        if source == "authenticated_jina":
            headers["Authorization"] = f"Bearer {self._jina_key}"
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                verify=True,
            ) as client:
                response = await client.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                )
            final_url = str(response.url)
            if response.status_code != 200:
                return self._classify_jina_error(
                    source,
                    response.status_code,
                    response.text,
                    final_url,
                )
            text = response.text.strip()
            if text:
                return _VisitFetchResult(
                    success=True,
                    source=source,
                    content=text,
                    http_status=200,
                    final_url=final_url,
                )
            return self._fetch_failure(
                source,
                "CONTENT_UNUSABLE",
                "fallback",
                status=200,
                detail="Jina returned empty content",
                final_url=final_url,
            )
        except Exception as error:
            return self._classify_fetch_exception(source, error)

    async def _fetch_direct_once(
        self,
        url: str,
        timeout: float,
    ) -> _VisitFetchResult:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        }
        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=timeout,
                verify=True,
            ) as client:
                current_url = url
                for _ in range(6):
                    response = await client.get(current_url, headers=headers)
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            return self._fetch_failure(
                                "direct_http",
                                "REDIRECT_MISSING_LOCATION",
                                "fallback",
                            )
                        current_url = urljoin(current_url, location)
                        await self._validate_visit_url(current_url)
                        continue
                    final_url = str(response.url)
                    if response.status_code == 200:
                        return _VisitFetchResult(
                            success=True,
                            source="direct_http",
                            content=response.text,
                            http_status=200,
                            final_url=final_url,
                        )
                    return self._classify_direct_error(
                        response.status_code,
                        response.text,
                        final_url,
                    )
            return self._fetch_failure(
                "direct_http",
                "TOO_MANY_REDIRECTS",
                "fallback",
            )
        except Exception as error:
            return self._classify_fetch_exception("direct_http", error)

    async def _fetch_visit_stage(
        self,
        url: str,
        source: str,
    ) -> _VisitFetchResult:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + VISIT_STAGE_TIMEOUT_SECONDS
        last_result: _VisitFetchResult | None = None

        for attempt in range(1, VISIT_MAX_ATTEMPTS + 1):
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                if source == "direct_http":
                    request = self._fetch_direct_once(url, remaining)
                else:
                    request = self._fetch_jina_once(url, source, remaining)
                result = await asyncio.wait_for(request, timeout=remaining)
            except asyncio.TimeoutError:
                result = self._fetch_failure(
                    source,
                    "NETWORK_TIMEOUT",
                    "retry",
                    detail=f"attempt exceeded remaining {remaining:.1f}s",
                )
            result.attempts = attempt
            last_result = result
            if result.success or result.action != "retry":
                return result
            if attempt >= VISIT_MAX_ATTEMPTS:
                break
            delay = VISIT_RETRY_DELAYS[
                min(attempt - 1, len(VISIT_RETRY_DELAYS) - 1)
            ]
            if deadline - loop.time() <= delay:
                break
            await asyncio.sleep(delay)

        if last_result is None:
            return self._fetch_failure(
                source,
                "STAGE_TIMEOUT",
                "fallback",
                detail=f"stage exceeded {VISIT_STAGE_TIMEOUT_SECONDS:.0f}s",
            )
        last_result.action = "fallback"
        return last_result

    @staticmethod
    def _looks_like_blocked_page(text: str) -> bool:
        normalized = text.lower()
        if len(normalized) > 5000:
            return False
        markers = (
            "verify you are human",
            "checking your browser",
            "attention required! | cloudflare",
            "enable javascript and cookies to continue",
            "access denied",
            "captcha",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _format_visit_failure(url: str, result: _VisitFetchResult) -> str:
        status = (
            f" status={result.http_status}"
            if result.http_status is not None
            else ""
        )
        return (
            f"[visit_failed] url={url} source={result.source} "
            f"code={result.error_code or 'UNKNOWN'}{status} "
            f"attempts={result.attempts}"
        )

    @staticmethod
    def _extract_main_text(html: str) -> str:
        """Production Agent BeautifulSoup main-text extraction."""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside",
                "iframe",
                "noscript",
                "svg",
                "form",
            ]
        ):
            tag.decompose()

        article = soup.find("article")
        if article:
            text = article.get_text(separator="\n", strip=True)
        else:
            body = soup.find("body")
            text = (
                body.get_text(separator="\n", strip=True)
                if body
                else soup.get_text(separator="\n", strip=True)
            )
        return "\n".join(line.strip() for line in text.split("\n") if line.strip())

    async def _summarize(self, content: str, goal: str) -> str:
        """Call the configured summary API up to three times, 20 seconds each."""
        from openai import AsyncOpenAI

        content = content[:50_000]
        fallback_content = content[:8_000]
        client = AsyncOpenAI(
            base_url=self._summary_base_url,
            api_key=self._summary_api_key,
            timeout=SUMMARY_CALL_TIMEOUT_SECONDS,
            max_retries=0,
        )
        for _ in range(SUMMARY_MAX_ATTEMPTS):
            prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self._summary_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=4096,
                    ),
                    timeout=SUMMARY_CALL_TIMEOUT_SECONDS,
                )
                raw = response.choices[0].message.content or ""
                if not raw or len(raw.strip()) < 10:
                    content = content[: int(len(content) * 0.7)]
                    continue

                raw_clean = raw.strip()
                fence = chr(96) * 3
                if raw_clean.startswith(fence):
                    raw_clean = raw_clean.split(fence, 2)[1]
                    if raw_clean.startswith("json"):
                        raw_clean = raw_clean[4:]
                    raw_clean = raw_clean.strip()
                parsed = None
                try:
                    parsed = json.loads(raw_clean)
                except json.JSONDecodeError:
                    left = raw_clean.find("{")
                    right = raw_clean.rfind("}")
                    if left != -1 and right != -1 and left < right:
                        try:
                            parsed = json.loads(raw_clean[left : right + 1])
                        except json.JSONDecodeError:
                            pass
                if isinstance(parsed, dict):
                    return (
                        f"Evidence in page: \n{parsed.get('evidence', '')}\n\n"
                        f"Summary: \n{parsed.get('summary', '')}"
                    )
                return f"Evidence in page: \n{raw}\n\nSummary: \n{raw}"
            except Exception:
                continue

        return (
            "Evidence in page: \n"
            "[Summary API failed after 3 attempts; extracted webpage content follows.]\n"
            f"{fallback_content}\n\n"
            "Summary: \nThe webpage was fetched successfully, but the summary API failed."
        )


_ENVIRONMENT: DeepSearchEnvironment | None = None


def get_tool_environment() -> DeepSearchEnvironment:
    """Return one environment per Ray worker process."""
    global _ENVIRONMENT
    if _ENVIRONMENT is None:
        _ENVIRONMENT = DeepSearchEnvironment()
    return _ENVIRONMENT
