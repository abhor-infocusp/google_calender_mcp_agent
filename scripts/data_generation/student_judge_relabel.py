#!/usr/bin/env python3
"""Re-judge candidate rows with v2 routers and capture full CoT.

Two backends:
  --backend qwen     hits local vLLM at QWEN_BASE; uses build_router_qwen_v2.
                     Captures verdict + raw CoT (SFT target).
  --backend gemini   hits Vertex gemini-2.0-flash-001; uses build_router_gemini_v2.
                     Captures verdict only (used for agreement filter).

Reads:  data/judge/v2_20260502/student_candidates.jsonl
Writes: data/judge/v2_20260502/relabel_qwen.jsonl   (or relabel_gemini.jsonl)

Idempotent: if output exists, skips rows whose rollout_hash already has a
verdict line.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from calendar_agent.judge.prompts import (  # noqa: E402
    build_router_qwen_v2, build_router_gemini_v2, extract_verdict,
)

DEFAULT_IN = REPO / "data/judge/v2_20260502/student_candidates.jsonl"
QWEN_OUT = REPO / "data/judge/v2_20260502/relabel_qwen.jsonl"
GEMINI_OUT = REPO / "data/judge/v2_20260502/relabel_gemini.jsonl"

QWEN_BASE = os.environ.get("QWEN_BASE", "http://localhost:8000/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "judge")

# Cats blocked from silver (per Phase-3 P(both wrong | agree) > 5%)
SILVER_BLOCKED = {
    "Complex Logic & Conflict (Advanced)",
    "Vague & Contextual (Reasoning Required)",
}


def call_qwen(client: httpx.Client, sys_p: str, user_p: str, no_think: bool = True,
              max_tokens: int = 512) -> tuple[str, int]:
    if no_think:
        sys_p = sys_p + "\n\n/no_think"
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }
    t0 = time.monotonic()
    try:
        r = client.post(f"{QWEN_BASE}/chat/completions", json=payload, timeout=120.0)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raw = f"<<<ERR:{type(e).__name__}>>>"
    return raw, int((time.monotonic() - t0) * 1000)


def call_gemini(model, sys_p: str, user_p: str) -> tuple[str, int]:
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_output_tokens=2048)
    t0 = time.monotonic()
    try:
        m = GenerativeModel("gemini-2.0-flash-001", system_instruction=[sys_p])
        r = m.generate_content(user_p, generation_config=cfg)
        raw = r.text.strip()
    except Exception as e:
        raw = f"<<<ERR:{type(e).__name__}>>>"
    return raw, int((time.monotonic() - t0) * 1000)


def init_gemini():
    import vertexai, json as _j
    import google.auth.transport.requests
    from google.oauth2.credentials import Credentials as OAuth2Credentials
    from calendar_agent.paths import CREDENTIALS_PATH
    cd = _j.load(open(CREDENTIALS_PATH))
    creds = OAuth2Credentials(
        token=None, refresh_token=cd["refresh_token"], client_id=cd["client_id"],
        client_secret=cd["client_secret"], token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(google.auth.transport.requests.Request())
    vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)


def already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    seen = set()
    for line in out_path.open():
        try:
            d = json.loads(line)
            # Skip error rows so they get retried on resume
            if d.get(f"qwen_v2_err") or d.get(f"gemini_v2_err"):
                continue
            seen.add(d["rollout_hash"])
        except Exception:
            pass
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["qwen", "gemini"], required=True)
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--shard", default="0/1", help="X/N: take rows where hash(rollout_hash) %% N == X")
    ap.add_argument("--skip-blocked", action="store_true",
                    help="skip silver-blocked cats (saves Gemini quota; Qwen still wants them for CoT)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (QWEN_OUT if args.backend == "qwen" else GEMINI_OUT)
    rows = [json.loads(l) for l in open(args.in_path)]
    if args.skip_blocked:
        rows = [r for r in rows if r["cat"] not in SILVER_BLOCKED]
    shard_x, shard_n = (int(s) for s in args.shard.split("/"))
    if shard_n > 1:
        rows = [r for r in rows if (int(r["rollout_hash"], 16) % shard_n) == shard_x]
        print(f"shard {shard_x}/{shard_n}: {len(rows)} rows in this shard")
    done = already_done(out_path)
    todo = [r for r in rows if r["rollout_hash"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"backend={args.backend} input={len(rows)} done={len(done)} todo={len(todo)} → {out_path}")

    if args.backend == "qwen":
        client = httpx.Client(timeout=httpx.Timeout(120.0))
        builder = build_router_qwen_v2
        def call(r):
            s, u, _ = builder(r); return call_qwen(client, s, u)
    else:
        init_gemini()
        builder = build_router_gemini_v2
        def call(r):
            s, u, _ = builder(r); return call_gemini(None, s, u)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("a", buffering=1)
    n_ok = n_err = 0
    t0 = time.monotonic()

    def one(r):
        raw, ms = call(r)
        verdict = extract_verdict(raw if not raw.startswith("<<<ERR") else "Incorrect")
        is_err = raw.startswith("<<<ERR")
        return r, raw, ms, verdict, is_err

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, r): r for r in todo}
        for i, fut in enumerate(as_completed(futs)):
            r, raw, ms, verdict, is_err = fut.result()
            rec = {
                "sid": r["sid"], "rollout_hash": r["rollout_hash"], "cat": r["cat"],
                f"{args.backend}_v2_verdict": verdict,
                f"{args.backend}_v2_raw": raw,
                f"{args.backend}_v2_latency_ms": ms,
                f"{args.backend}_v2_err": is_err,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += (0 if is_err else 1); n_err += (1 if is_err else 0)
            if (i + 1) % 100 == 0 or i + 1 == len(todo):
                el = time.monotonic() - t0
                rate = (i + 1) / max(el, 1e-3)
                print(f"  {i+1}/{len(todo)}  ok={n_ok} err={n_err}  {rate:.2f}/s  eta={int((len(todo)-i-1)/max(rate,1e-3))}s")

    fout.close()
    print(f"done: ok={n_ok} err={n_err}  → {out_path}")


if __name__ == "__main__":
    main()
