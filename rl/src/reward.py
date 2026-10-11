"""Hybrid deterministic + LLM-judge reward for OpenSeeker QA."""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from diskcache import Cache

from src.rollout_artifacts import append_training_result


ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
JUDGE_PROMPT_VERSION = "openseeker-yes-no-v2"
DEFAULT_JUDGE_CACHE = Path(__file__).resolve().parents[1] / "outputs" / "judge_cache"
DEFAULT_DMX_BASE_URL = "https://www.dmxapi.cn/v1"
YES_TOKEN_RE = re.compile(r"(?<![A-Z])YES(?![A-Z])")
NO_TOKEN_RE = re.compile(r"(?<![A-Z])NO(?![A-Z])")


def normalize_answer(value: str) -> str:
    """Normalize surface variation while retaining alphanumeric semantics."""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.category(char).startswith("M")
    )
    value = "".join(
        " "
        if (char in string.punctuation or unicodedata.category(char).startswith("P"))
        else char
        for char in value
    )
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def extract_final_answer(solution_str: str) -> tuple[str, bool]:
    """Return the last answer block and strict final-answer format status.

    An absent answer block is represented by an empty candidate so callers do
    not accidentally score the complete rollout transcript as an answer.
    """
    matches = list(ANSWER_BLOCK_RE.finditer(solution_str))
    if not matches:
        return "", False
    final = matches[-1]
    format_ok = bool(final.group(1).strip()) and not solution_str[final.end() :].strip()
    return final.group(1).strip(), format_ok


def _candidate_references(ground_truth: Any, extra_info: dict[str, Any]) -> list[str]:
    values: list[Any]
    if isinstance(ground_truth, (list, tuple)):
        values = list(ground_truth)
    else:
        values = [ground_truth]
    aliases = extra_info.get("acceptable_answers", [])
    if isinstance(aliases, str):
        aliases = [aliases]
    values.extend(aliases or [])
    return list(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


def exact_match(candidate: str, references: Iterable[str]) -> bool:
    normalized_candidate = normalize_answer(candidate)
    return bool(normalized_candidate) and any(
        normalized_candidate == normalize_answer(reference) for reference in references
    )


def parse_yes_no_judgment(raw: str) -> tuple[bool, str]:
    """Accept tolerant but unambiguous YES/NO judge output."""
    normalized = raw.strip().upper()
    has_yes = bool(YES_TOKEN_RE.search(normalized))
    has_no = bool(NO_TOKEN_RE.search(normalized))
    if has_yes and not has_no:
        return True, "yes"
    if has_no:
        return False, "no" if not has_yes else "ambiguous yes/no"
    return False, "no clear yes/no"


class JudgeDecision:
    """Small immutable-style result object safe for VERL's file module loader."""

    __slots__ = ("equivalent", "confidence", "reason", "available", "cached")

    def __init__(
        self,
        equivalent: bool,
        confidence: float,
        reason: str,
        available: bool = True,
        cached: bool = False,
    ) -> None:
        self.equivalent = equivalent
        self.confidence = confidence
        self.reason = reason
        self.available = available
        self.cached = cached


class LLMAnswerJudge:
    """YES/NO equivalence judge with a persistent cache and fail-closed behavior."""

    def __init__(self) -> None:
        self.enabled = os.getenv("OPENSEEKER_JUDGE_ENABLED", "1").lower() in {
            "1",
            "true",
            "yes",
        }
        self.base_url = (
            os.getenv("JUDGE_BASE_URL", "").strip()
            or os.getenv("DMX_BASE_URL", "").strip()
            or os.getenv("TOOL_BASE_URL", "").strip()
            or DEFAULT_DMX_BASE_URL
        )
        self.api_key = (
            os.getenv("JUDGE_API_KEY", "").strip()
            or os.getenv("DMX_API_KEY", "").strip()
            or os.getenv("TOOL_API_KEY", "").strip()
        )
        self.model = (
            os.getenv("JUDGE_MODEL", "").strip()
            or os.getenv("DMX_MODEL", "").strip()
            or os.getenv("SUMMARY_MODEL", "").strip()
            or "qwen-plus"
        )
        self.timeout = float(os.getenv("JUDGE_TIMEOUT_SECONDS", "45"))
        self.max_retries = int(os.getenv("JUDGE_MAX_RETRIES", "2"))
        self.cache = Cache(
            os.getenv("OPENSEEKER_JUDGE_CACHE", str(DEFAULT_JUDGE_CACHE))
        )
        self._client = None

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.base_url and self.api_key and self.model)

    def _key(self, question: str, references: list[str], candidate: str) -> str:
        payload = json.dumps(
            {
                "version": JUDGE_PROMPT_VERSION,
                "model": self.model,
                "question": question,
                "references": references,
                "candidate": candidate,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def judge(
        self, question: str, references: list[str], candidate: str
    ) -> JudgeDecision:
        if not self.available or not question.strip() or not candidate.strip():
            return JudgeDecision(False, 0.0, "judge unavailable", available=False)
        key = self._key(question, references, candidate)
        cached = self.cache.get(key)
        if isinstance(cached, dict):
            return JudgeDecision(
                bool(cached.get("equivalent")),
                float(cached.get("confidence", 0.0)),
                str(cached.get("reason", "")),
                cached=True,
            )

        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )

        expected = "\n".join(references)
        user = f"""你需要作为一个公正的裁判，评估模型针对问题给出的预测答案是否与标准参考答案含义相同。

[预测答案]: {candidate[:3000]}
[标准参考答案]: {expected[:6000]}

判断规则：
1. 如果预测答案与任一标准参考答案表达了相同的核心含义、人物、地点或事实，视为正确。
2. 如果预测答案包含了多余的废话，只要核心答案存在且正确，依然视为正确。
3. 语言不同但含义相同（例如："美国"和"United States"）视为正确。
4. 缩写和全称对应视为正确。
5. 如果预测答案缺少核心信息，或者表达了不同的含义，视为错误。

请仔细思考，并仅回复 "YES" 或 "NO"。"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user}],
                temperature=0,
                max_tokens=10,
            )
            raw = response.choices[0].message.content or ""
            equivalent, reason = parse_yes_no_judgment(raw)
            decision = {
                "equivalent": equivalent,
                "confidence": 1.0 if equivalent else 0.0,
                "reason": reason,
            }
            self.cache.set(key, decision)
            return JudgeDecision(**decision)
        except Exception as error:
            # Reward infrastructure must never turn an uncertain sample into a positive.
            return JudgeDecision(
                False, 0.0, f"judge error: {type(error).__name__}", available=False
            )


_JUDGE: LLMAnswerJudge | None = None


def get_judge() -> LLMAnswerJudge:
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = LLMAnswerJudge()
    return _JUDGE


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    format_weight: float = 0.2,
) -> dict[str, float | str | bool]:
    """Compute trajectory reward, invoking the judge only after exact-match failure."""
    if data_source not in {"openseeker_qa", "deepdive_qa"}:
        raise ValueError(f"unsupported data_source: {data_source!r}")
    if not 0.0 <= format_weight <= 1.0:
        raise ValueError("format_weight must be between zero and one")

    extra_info = dict(extra_info or {})
    candidate, format_ok = extract_final_answer(solution_str)
    trajectory_format_ok = bool(extra_info.get("assistant_turn_format_ok", False))
    if not candidate:
        # A rewardable answer must be explicitly provided in a non-empty
        # <answer> block. Do not exact-match or LLM-judge raw rollout text.
        result: dict[str, float | str | bool] = {
            "score": 0.0,
            "answer_score": 0.0,
            "format_score": 0.0,
            "exact_match": False,
            "judge_match": False,
            "judge_available": False,
            "judge_cached": False,
            "judge_confidence": 0.0,
            "predicted_answer": "",
            "trajectory_format_ok": trajectory_format_ok,
        }
        _log_reward_result(extra_info, result)
        return result
    references = _candidate_references(ground_truth, extra_info)
    matched_exactly = exact_match(candidate, references)
    judge_decision = JudgeDecision(
        False, 0.0, "exact match; judge skipped", available=False
    )
    if not matched_exactly:
        judge_decision = get_judge().judge(
            str(extra_info.get("question", "")), references, candidate
        )

    answer_ok = matched_exactly or judge_decision.equivalent
    answer_score = float(answer_ok)
    format_score = float(format_ok and trajectory_format_ok)
    total = (1.0 - format_weight) * answer_score + format_weight * format_score
    result = {
        "score": total,
        "answer_score": answer_score,
        "format_score": format_score,
        "exact_match": matched_exactly,
        "judge_match": judge_decision.equivalent,
        "judge_available": judge_decision.available,
        "judge_cached": judge_decision.cached,
        "judge_confidence": judge_decision.confidence,
        "predicted_answer": candidate[:500],
        "trajectory_format_ok": trajectory_format_ok,
    }
    _log_reward_result(extra_info, result)
    return result


def _log_reward_result(
    extra_info: dict[str, Any], result: dict[str, float | str | bool]
) -> None:
    """Persist scalar reward components beside the trajectory, when enabled."""
    trace_id = str(extra_info.get("trace_id", "")).strip()
    if not trace_id:
        return
    append_training_result(
        trace_id,
        "reward",
        score=result["score"],
        answer_score=result["answer_score"],
        format_score=result["format_score"],
        exact_match=result["exact_match"],
        judge_match=result["judge_match"],
        trajectory_format_ok=result["trajectory_format_ok"],
    )
