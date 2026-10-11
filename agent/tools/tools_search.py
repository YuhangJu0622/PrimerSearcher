import json
import os
import random
import re
from typing import List

import httpx

from agent.logger_config import get_logger

logger = get_logger()

# Serper (Google) search API — the only search backend.
SERPER_API_KEYS = [k.strip() for k in os.getenv(
    "SERPER_API_KEYS",
    ""
).split(",") if k.strip()]
SERPER_BASE = "https://google.serper.dev/search"
SERPER_TIMEOUT = 15


def _clean_query(query: str) -> str:
    """Remove excessive exact-match double-quotes from search queries.

    Some models (e.g., DeepSeek-v4-pro) wrap every search term in double
    quotes (Google exact-match syntax).  This is far too restrictive for
    web-search APIs and yields poor / empty result sets.  Strip them.
    """
    if not query:
        return query
    q = query.strip()
    # Strip wrapping paired quotes
    if q.startswith('"') and q.endswith('"'):
        q = q[1:-1].strip()
    # Remove internal quoted-phrase boundaries while keeping the words
    q = q.replace('" "', ' ')
    # Remove any remaining double-quote characters
    q = q.replace('"', '')
    # Collapse multiple spaces
    q = ' '.join(q.split())
    return q


def _simplify_query(query: str) -> str:
    """Simplify complex query by removing modifiers and keeping core entities."""
    if not query or len(query) < 10:
        return query  # Already simple enough

    # Remove common Chinese/English modifiers and conjunctions
    simplified = re.sub(r'的|which|that|whose|who|when|where', ' ', query, flags=re.IGNORECASE)
    # Remove multiple spaces
    simplified = re.sub(r'\s+', ' ', simplified).strip()

    # If simplified is too different (too short), return original
    if len(simplified) < len(query) * 0.3:
        return query

    return simplified if simplified != query else query


def _serper_error_summary(response_text: str) -> str:
    """Return a single-line, key-safe excerpt of a Serper error response."""
    summary = " ".join((response_text or "").split())
    for configured_key in SERPER_API_KEYS:
        summary = summary.replace(configured_key, "[REDACTED]")
    return summary[:300] or "no response body"


def _is_empty_result(result: str) -> bool:
    """Check whether a formatted Serper result contains no results."""
    return not result or "found 0 results" in result


def _format_serper_results(query: str, data: dict) -> str:
    """Format Serper JSON response."""
    organic = data.get("organic", [])
    if not organic:
        return f"### A Google search for '{query}' found 0 results:\n\n"

    web_snippets = []
    for idx, item in enumerate(organic, 1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        date = item.get("date", "")

        date_str = ""
        if date:
            date_str = f"\nDate published: {date}"

        snippet_text = f"\n\n{snippet}" if snippet else ""
        entry = f"{idx}. [{title}]({link}){date_str}{snippet_text}"
        web_snippets.append(entry)

    content = (
        f"### A Google search for '{query}' found {len(web_snippets)} results:\n\n"
        + "\n\n".join(web_snippets)
    )
    return content


def serper_search(query: str) -> str:
    """Search Serper, rotating to the next key after every failed attempt.

    Returns the formatted result string, or the "found 0 results" marker when
    no key could complete the request.
    """
    if not SERPER_API_KEYS:
        logger.warning("No Serper API keys configured")
        return f"### A Google search for '{query}' found 0 results:\n\n"

    start = random.randint(0, len(SERPER_API_KEYS) - 1)
    for i in range(len(SERPER_API_KEYS)):
        key_index = (start + i) % len(SERPER_API_KEYS)
        key = SERPER_API_KEYS[key_index]
        headers = {"X-API-Key": key, "Content-Type": "application/json"}
        payload = {"q": query, "num": 10}
        try:
            resp = httpx.post(SERPER_BASE, headers=headers, json=payload, timeout=SERPER_TIMEOUT)
        except Exception as e:
            logger.warning(
                "Serper key %d/%d request failed; rotating to the next key. error=%s",
                key_index + 1,
                len(SERPER_API_KEYS),
                _serper_error_summary(str(e)),
            )
            continue

        if resp.status_code != 200:
            logger.warning(
                "Serper key %d/%d returned HTTP %d; rotating to the next key. reason=%s",
                key_index + 1,
                len(SERPER_API_KEYS),
                resp.status_code,
                _serper_error_summary(resp.text),
            )
            continue

        try:
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("response JSON must be an object")
            return _format_serper_results(query, data)
        except Exception as e:
            logger.warning(
                "Serper key %d/%d returned an unusable response (HTTP 200); rotating to the next key. error=%s",
                key_index + 1,
                len(SERPER_API_KEYS),
                _serper_error_summary(str(e)),
            )
            continue

    logger.warning("All %d Serper keys failed for this query", len(SERPER_API_KEYS))
    return f"### A Google search for '{query}' found 0 results:\n\n"


def _search_one(query: str) -> str:
    """Search a single query on Serper, recovering poor results by widening it.

    Falls back in order: raw query -> de-quoted query -> simplified query.
    """
    result = serper_search(query)
    if not _is_empty_result(result):
        return result

    cleaned = _clean_query(query)
    if cleaned != query:
        logger.info("No results for '%s', retrying de-quoted: '%s'", query[:60], cleaned[:60])
        retry = serper_search(cleaned)
        if not _is_empty_result(retry):
            return retry
        result = retry

    simplified = _simplify_query(cleaned)
    if simplified != cleaned:
        logger.info("No results for '%s', retrying simplified: '%s'", cleaned[:60], simplified[:60])
        retry = serper_search(simplified)
        if not _is_empty_result(retry):
            return retry
        result = retry

    return result


def batch_search(queries: List[str], engines: List[str] = None) -> str:
    """Batch search on Serper (the sole search backend).

    ``engines`` is accepted for call-site compatibility but ignored: IQS has
    been removed and every query goes through Serper.
    """
    if isinstance(queries, str):
        queries = [queries]

    results = []
    for q in queries:
        logger.info('Query: "%s" -> Google/Serper', q[:60])
        results.append(_search_one(q))

    return "\n---\n".join(results)
