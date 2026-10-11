import asyncio
import json
import os
from typing import List, Union

import httpx
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from agent.logger_config import get_logger
from agent.prompts import EXTRACTOR_PROMPT

logger = get_logger()

# Summary/judge third-party model goes directly through DMX (no NoDesk gateway).
DMX_BASE_URL = os.getenv("DMX_BASE_URL", "https://www.dmxapi.cn/v1")
DMX_API_KEY = os.getenv("DMX_API_KEY", "")
VISIT_TIMEOUT = 20
WEBCONTENT_MAXLENGTH = 50000
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "qwen-plus")

# Jina Reader API configuration (anonymous access only — no API key).
JINA_ENABLED = os.getenv("JINA_ENABLED", "1").strip() in ("1", "true", "yes")
JINA_BASE = "https://r.jina.ai/"
JINA_TIMEOUT = 30

_summary_client = AsyncOpenAI(
    base_url=DMX_BASE_URL,
    api_key=DMX_API_KEY,
)


async def fetch_page_jina(url: str) -> str:
    """Fetch webpage as clean markdown via Jina Reader API (anonymous, no key)."""
    headers = {"Accept": "text/markdown", "X-No-Cache": "true"}
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=JINA_TIMEOUT,
            verify=False,
        ) as client:
            resp = await client.get(f"{JINA_BASE}{url}", headers=headers)
            if resp.status_code == 200:
                text = resp.text.strip()
                if text:
                    return text
                return "[visit] Jina returned empty content"
            return f"[visit] Jina HTTP {resp.status_code} for {url}"
    except Exception as e:
        return f"[visit] Jina fetch failed: {str(e)}"


async def fetch_page(url: str) -> str:
    """Fetch webpage content using httpx."""
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
            follow_redirects=True,
            timeout=VISIT_TIMEOUT,
            verify=False,
        ) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.text
            else:
                return f"[visit] HTTP {resp.status_code} error for {url}"
    except Exception as e:
        return f"[visit] Failed to fetch {url}: {str(e)}"


def extract_main_text(html: str) -> str:
    """Extract main text content from HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "iframe", "noscript", "svg", "form"]):
        tag.decompose()

    # Try to find article content first
    article = soup.find("article")
    if article:
        text = article.get_text(separator="\n", strip=True)
    else:
        # Fall back to body
        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

    # Clean up: remove excessive blank lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    return text


async def summarize_content(content: str, goal: str, max_retries: int = 2) -> str:
    """Use LLM to extract relevant information from webpage content."""
    content = content[:WEBCONTENT_MAXLENGTH]

    messages = [
        {
            "role": "user",
            "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal),
        }
    ]

    for attempt in range(max_retries):
        try:
            resp = await _summary_client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content
            if not raw or len(raw) < 10:
                # Truncate content and retry
                content = content[: int(len(content) * 0.7)]
                messages[0]["content"] = EXTRACTOR_PROMPT.format(
                    webpage_content=content, goal=goal
                )
                continue

            # Try to parse as JSON
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                raw_clean = raw_clean.split("```")[1]
                if raw_clean.startswith("json"):
                    raw_clean = raw_clean[4:]
                raw_clean = raw_clean.strip()

            try:
                data = json.loads(raw_clean)
                evidence = data.get("evidence", "")
                summary = data.get("summary", "")
                return f"Evidence in page: \n{evidence}\n\nSummary: \n{summary}"
            except json.JSONDecodeError:
                # Extract JSON from string
                left = raw_clean.find("{")
                right = raw_clean.rfind("}")
                if left != -1 and right != -1 and left < right:
                    try:
                        data = json.loads(raw_clean[left : right + 1])
                        evidence = data.get("evidence", "")
                        summary = data.get("summary", "")
                        return f"Evidence in page: \n{evidence}\n\nSummary: \n{summary}"
                    except json.JSONDecodeError:
                        pass
                # Keep the same evidence/summary envelope used by training data.
                return f"Evidence in page: \n{raw}\n\nSummary: \n{raw}"

        except Exception as e:
            logger.warning("Summary attempt %d failed: %s", attempt + 1, e)

    return (
        "Evidence in page: \nThe webpage content could not be processed.\n\n"
        "Summary: \nThe webpage content could not be processed."
    )


async def visit_page(url: str, goal: str) -> str:
    """Visit a single webpage and return extracted information.
    Tries Jina Reader API first, falls back to httpx + BeautifulSoup."""
    text = await fetch_page_jina(url) if JINA_ENABLED else "[visit] Jina disabled"

    if text.startswith("[visit]"):
        logger.info("Visit: \"%s\" -> httpx (Jina failed: %s)", url, text)
        html = await fetch_page(url)
        if html.startswith("[visit]"):
            return f"[visit] Failed to visit the url: {url}"
        text = extract_main_text(html)
    else:
        logger.info("Visit: \"%s\" -> Jina Reader", url)

    if not text or len(text) < 20:
        return (
            f"The useful information in {url} for user goal {goal} as follows: \n\n"
            f"Evidence in page: \nEmpty or very short content.\n\n"
            f"Summary: \nThe webpage had no extractable content."
        )

    summary = await summarize_content(text, goal)
    return (
        f"The useful information in {url} for user goal {goal} as follows: \n\n"
        f"{summary}"
    )


async def visit_pages(urls: Union[str, List[str]], goal: str) -> str:
    """Visit one or more webpages concurrently and return combined results."""
    if isinstance(urls, str):
        urls = [urls]

    async def _safe_visit(url: str) -> str:
        try:
            return await asyncio.wait_for(visit_page(url, goal), timeout=120)
        except asyncio.TimeoutError:
            return f"[visit] Failed to visit the url: {url}"
        except Exception as e:
            logger.warning("Visit failed for %s: %s", url, e)
            return f"[visit] Failed to visit the url: {url}"

    results = await asyncio.gather(*[_safe_visit(url) for url in urls])
    return "\n---\n".join(results)
