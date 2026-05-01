"""Async HTTP client for the judge service.

RL training imports `verdict(...)` and uses it as a drop-in for the old
Gemini-based evaluate_trajectory(). On any failure (connection refused,
non-200, malformed body) the call raises — callers must NOT silently fall
back to Gemini, per project policy.

Environment:
  JUDGE_URL     base URL of the FastAPI service (default http://127.0.0.1:8765)
  JUDGE_TIMEOUT total per-call timeout in seconds (default 120)
"""
from __future__ import annotations

import os
from typing import Any

import httpx

JUDGE_URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:8765")
JUDGE_TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "120"))


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
    try:
        r = await _get_client().post("/verdict", json=body)
    except httpx.HTTPError as e:
        raise JudgeUnavailable(f"transport error: {e!r}") from e
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
