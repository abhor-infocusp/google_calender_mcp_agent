"""Async HTTP client for the judge service.

RL training imports `verdict(...)` and uses it as a drop-in for the old
Gemini-based evaluate_trajectory(). On persistent failure (connection refused,
non-200, malformed body) the call raises — callers must NOT silently fall
back to Gemini, per project policy. We do retry transient transport errors
(timeouts, connection resets) a small number of times before giving up.

Environment:
  JUDGE_URL          base URL of the FastAPI service (default http://127.0.0.1:8765)
  JUDGE_TIMEOUT      total per-call HTTP timeout in seconds (default 180)
  JUDGE_RETRIES      retries on transport errors before raising (default 3)
  JUDGE_RETRY_BACKOFF base seconds for exponential backoff (default 2.0)
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import httpx

JUDGE_URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:8765")
JUDGE_TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "180"))
JUDGE_RETRIES = int(os.environ.get("JUDGE_RETRIES", "3"))
JUDGE_RETRY_BACKOFF = float(os.environ.get("JUDGE_RETRY_BACKOFF", "2.0"))


class JudgeUnavailable(RuntimeError):
    """Judge service did not return a usable verdict."""


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=JUDGE_URL, timeout=httpx.Timeout(JUDGE_TIMEOUT))
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def verdict(
    *,
    cat: str,
    query: str,
    before: str,
    after: str,
    final: str = "",
    expected: str = "",
    scenario_id: str | int | None = None,
) -> dict[str, Any]:
    """POST /verdict and return the parsed body. Raises JudgeUnavailable on failure."""
    body = {
        "cat": cat,
        "query": query,
        "final": final,
        "expected": expected,
        "before": before,
        "after": after,
        "scenario_id": scenario_id,
    }
    last_err: Exception | None = None
    for attempt in range(JUDGE_RETRIES + 1):
        try:
            r = await _get_client().post("/verdict", json=body)
            break
        except httpx.HTTPError as e:
            last_err = e
            if attempt >= JUDGE_RETRIES:
                raise JudgeUnavailable(
                    f"transport error after {attempt + 1} attempts: {e!r}"
                ) from e
            delay = JUDGE_RETRY_BACKOFF * (2 ** attempt)
            print(
                f"[JUDGE RETRY] {type(e).__name__} on attempt {attempt + 1}/"
                f"{JUDGE_RETRIES + 1}; sleeping {delay:.1f}s",
                file=sys.stderr, flush=True,
            )
            await asyncio.sleep(delay)
    else:
        raise JudgeUnavailable(f"transport error: {last_err!r}") from last_err
    if r.status_code != 200:
        raise JudgeUnavailable(f"HTTP {r.status_code}: {r.text[:300]}")
    try:
        out = r.json()
    except ValueError as e:
        raise JudgeUnavailable(f"non-JSON body: {r.text[:200]}") from e
    if "verdict" not in out:
        raise JudgeUnavailable(f"missing verdict field: {out}")
    return out


async def health() -> dict[str, Any]:
    r = await _get_client().get("/health")
    r.raise_for_status()
    return r.json()
