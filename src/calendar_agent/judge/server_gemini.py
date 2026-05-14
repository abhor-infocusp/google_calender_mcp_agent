"""Gemini-backed FastAPI judge service.

Drop-in replacement for `calendar_agent.judge.server` (the local-vLLM judge).
Same `/verdict` and `/health` API, same log format, same VerdictResponse shape.
RL training only needs to point at a different host/port.

Routing:
  Default JUDGE_ROUTER=structured (structured_prompts.build_router_structured),
  i.e. per-cat structured prompts (Modifier, Chaos, Complex, IR, RelTime use
  structured fields; Schedule + Vague fall back to plain `router`). On the
  285-traj manual oracle this hits 92.98% — beats plain router (91.93%) and
  the previous production rl-sft-4952 (~91.5%).

Env:
  JUDGE_ROUTER     structured | router | qwen_v2 | gemini_v2  (default structured)
  JUDGE_PORT       bind port (default 8765 — same as local service)
  JUDGE_LOG_DIR    where to append calls.jsonl
  GEMINI_MODEL     vertex model name (default gemini-2.0-flash-001)
  GEMINI_PROJECT   GCP project (default internal-ml-exp)
  GEMINI_LOCATION  region (default us-central1)
  GEMINI_MAX_TOKENS max output tokens (default 2048)

Run:
  PYTHONPATH=src python -m calendar_agent.judge.server_gemini
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import google.auth.transport.requests
import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import GenerationConfig, GenerativeModel

from calendar_agent.judge.prompts import (
    PROMPT_VERSION,
    build_router,
    build_router_qwen_v2,
    build_router_gemini_v2,
    extract_verdict,
)
from calendar_agent.judge.structured_prompts import build_router_structured
from calendar_agent.paths import CREDENTIALS_PATH

_ROUTERS = {
    "structured": build_router_structured,   # default — current best on 285 oracle
    "router":     build_router,
    "qwen_v2":    build_router_qwen_v2,
    "gemini_v2":  build_router_gemini_v2,
}
JUDGE_ROUTER = os.environ.get("JUDGE_ROUTER", "structured")
if JUDGE_ROUTER not in _ROUTERS:
    raise ValueError(f"unknown JUDGE_ROUTER={JUDGE_ROUTER!r}; choose {list(_ROUTERS)}")
_BUILD_ROUTER = _ROUTERS[JUDGE_ROUTER]

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")
GEMINI_PROJECT = os.environ.get("GEMINI_PROJECT", "internal-ml-exp")
GEMINI_LOCATION = os.environ.get("GEMINI_LOCATION", "us-central1")
GEMINI_MAX_TOKENS = int(os.environ.get("GEMINI_MAX_TOKENS", "2048"))
JUDGE_PORT = int(os.environ.get("JUDGE_PORT", "8765"))

_default_log_dir = (
    Path(__file__).resolve().parents[3]
    / "runs"
    / f"judge_service_gemini_{datetime.now().strftime('%Y%m%d')}"
)
LOG_DIR = Path(os.environ.get("JUDGE_LOG_DIR", str(_default_log_dir)))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "calls.jsonl"

START_TIME = time.monotonic()
_log_lock = asyncio.Lock()
_model: GenerativeModel | None = None


class VerdictRequest(BaseModel):
    cat: str
    query: str
    final: str = ""
    expected: str = ""
    before: str
    after: str
    scenario_id: str | int | None = None
    temperature: float | None = None


class VerdictResponse(BaseModel):
    verdict: str
    prompt_version: str
    latency_ms: int


app = FastAPI(title="calendar-judge-gemini",
              version=f"{PROMPT_VERSION}/{JUDGE_ROUTER}/{GEMINI_MODEL}")


def _init_vertex() -> None:
    cd = json.load(open(CREDENTIALS_PATH))
    creds = OAuth2Credentials(
        token=None, refresh_token=cd["refresh_token"],
        client_id=cd["client_id"], client_secret=cd["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(google.auth.transport.requests.Request())
    vertexai.init(project=GEMINI_PROJECT, location=GEMINI_LOCATION, credentials=creds)


@app.on_event("startup")
async def _startup() -> None:
    global _model
    _init_vertex()
    # System instruction is set per-call (varies by category), so build a
    # shareable model handle once.
    _model = None  # built lazily per call


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "backend": "vertex-ai",
        "model": GEMINI_MODEL,
        "router": JUDGE_ROUTER,
        "prompt_version": PROMPT_VERSION,
        "uptime_s": round(time.monotonic() - START_TIME, 1),
        "log_path": str(LOG_PATH),
    }


async def _append_log(record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False, default=str)
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
    try:
        sys_prompt, user_prompt, opts = _BUILD_ROUTER(rec)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"router error: {e!r}") from e

    gen_cfg = GenerationConfig(
        temperature=req.temperature if req.temperature is not None else 0.0,
        top_p=1.0,
        max_output_tokens=opts.get("max_tokens", GEMINI_MAX_TOKENS),
    )

    t0 = time.monotonic()
    try:
        # GenerativeModel is cheap to construct — instantiate per-call so the
        # system instruction can vary by category.
        loop = asyncio.get_running_loop()
        def _call() -> str:
            model = GenerativeModel(GEMINI_MODEL, system_instruction=[sys_prompt])
            resp = model.generate_content(user_prompt, generation_config=gen_cfg)
            return resp.text
        raw = await loop.run_in_executor(None, _call)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"vertex error: {e!r}") from e

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
        "router": JUDGE_ROUTER,
        "model": GEMINI_MODEL,
        "latency_ms": latency_ms,
    })

    return VerdictResponse(verdict=v, prompt_version=PROMPT_VERSION, latency_ms=latency_ms)


def main() -> None:
    import uvicorn
    uvicorn.run(
        "calendar_agent.judge.server_gemini:app",
        host="0.0.0.0",
        port=JUDGE_PORT,
        log_level=os.environ.get("JUDGE_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
