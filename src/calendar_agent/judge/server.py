"""FastAPI judge service.

Wraps a local vLLM OpenAI-compatible server (Qwen3-14B fp8) with the
canonical router prompt. RL training POSTs `{cat, query, final, expected,
before, after}`; we build the prompt server-side, generate full reasoning
to keep accuracy at 95.44%, and return only the verdict.

Every call is appended to JSONL so the corpus doubles as Phase 1.5
distillation training data.

Environment:
  VLLM_BASE       upstream vLLM /v1 base (default http://127.0.0.1:8000/v1)
  VLLM_MODEL      served-model-name registered with vLLM (default 'judge')
  JUDGE_LOG_DIR   directory for calls.jsonl (default runs/judge_service_<date>/)
  JUDGE_PORT      bind port (default 8765)

Runs as: PYTHONPATH=src python -m calendar_agent.judge.server
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from calendar_agent.judge.prompts import (
    PROMPT_VERSION,
    build_router,
    extract_verdict,
)

VLLM_BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "judge")
JUDGE_PORT = int(os.environ.get("JUDGE_PORT", "8765"))

_default_log_dir = (
    Path(__file__).resolve().parents[3]
    / "runs"
    / f"judge_service_{datetime.now().strftime('%Y%m%d')}"
)
LOG_DIR = Path(os.environ.get("JUDGE_LOG_DIR", str(_default_log_dir)))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "calls.jsonl"

START_TIME = time.monotonic()
_log_lock = asyncio.Lock()
# Long-lived client; vLLM handles continuous batching, we just keep sockets warm.
_client: httpx.AsyncClient | None = None


class VerdictRequest(BaseModel):
    cat: str
    query: str
    final: str = ""
    expected: str = ""
    before: str
    after: str
    scenario_id: str | int | None = None
    # Optional sampling overrides; default is greedy.
    temperature: float | None = None


class VerdictResponse(BaseModel):
    verdict: str
    prompt_version: str
    latency_ms: int


app = FastAPI(title="calendar-judge", version=PROMPT_VERSION)


@app.on_event("startup")
async def _startup() -> None:
    global _client
    _client = httpx.AsyncClient(base_url=VLLM_BASE, timeout=httpx.Timeout(120.0))


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _client is not None:
        await _client.aclose()


@app.get("/health")
async def health() -> dict[str, Any]:
    ok = False
    detail = ""
    try:
        assert _client is not None
        r = await _client.get("/models", timeout=5.0)
        ok = r.status_code == 200
        detail = r.text[:200] if not ok else ""
    except Exception as e:  # noqa: BLE001
        detail = repr(e)
    return {
        "ok": ok,
        "model": VLLM_MODEL,
        "prompt_version": PROMPT_VERSION,
        "uptime_s": round(time.monotonic() - START_TIME, 1),
        "vllm_base": VLLM_BASE,
        "log_path": str(LOG_PATH),
        "detail": detail,
    }


async def _append_log(record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    async with _log_lock:
        with LOG_PATH.open("a") as f:
            f.write(line + "\n")


@app.post("/verdict", response_model=VerdictResponse)
async def verdict(req: VerdictRequest) -> VerdictResponse:
    rec = {
        "cat": req.cat,
        "query": req.query,
        "final": req.final,
        "expected": req.expected,
        "before": req.before,
        "after": req.after,
    }
    sys_prompt, user_prompt, opts = build_router(rec)

    payload: dict[str, Any] = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": opts.get("max_tokens", 1024),
        "temperature": req.temperature if req.temperature is not None else 0.0,
    }

    t0 = time.monotonic()
    assert _client is not None
    try:
        r = await _client.post("/chat/completions", json=payload)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"vLLM unreachable: {e!r}") from e
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"vLLM {r.status_code}: {r.text[:300]}")

    body = r.json()
    raw = body["choices"][0]["message"]["content"]
    v = extract_verdict(raw)
    latency_ms = int((time.monotonic() - t0) * 1000)

    await _append_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "scenario_id": req.scenario_id,
        "cat": req.cat,
        "query": req.query,
        "final": req.final,
        "expected": req.expected,
        "before": req.before,
        "after": req.after,
        "system_prompt": sys_prompt,
        "user_prompt": user_prompt,
        "raw_response": raw,
        "verdict": v,
        "prompt_version": PROMPT_VERSION,
        "model": VLLM_MODEL,
        "latency_ms": latency_ms,
        "usage": body.get("usage"),
    })

    return VerdictResponse(verdict=v, prompt_version=PROMPT_VERSION, latency_ms=latency_ms)


def main() -> None:
    import uvicorn
    uvicorn.run(
        "calendar_agent.judge.server:app",
        host="127.0.0.1",
        port=JUDGE_PORT,
        log_level=os.environ.get("JUDGE_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
