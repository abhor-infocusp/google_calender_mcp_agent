#!/usr/bin/env python3
"""Run base-14B + sft-14B + gemini-v2 on a jsonl in parallel.

Used for validation experiments around the silver-label filter switch.

Outputs one line per input row:
  {sid, rollout_hash, cat, label?, base, sft, gem,
   base_raw, sft_raw, gem_raw, base_ms, sft_ms, gem_ms}

Usage:
  PYTHONPATH=src python scripts/eval/judge_3way_run.py \
      --in data/judge/v2_20260502/eval.jsonl \
      --out runs/judge_filter_validation_20260505/tier1_run1.jsonl \
      --judges base,sft,gem --workers 12

  # CoT mode (full reasoning, no /no_think):
  PYTHONPATH=src python scripts/eval/judge_3way_run.py \
      --in <sample.jsonl> --out <out.jsonl> --judges base,sft \
      --think --max-tokens 1024 --workers 8
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from calendar_agent.judge.prompts import (  # noqa: E402
    build_router_qwen_v2, build_router_gemini_v2, extract_verdict,
)

import os
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8011/v1")
SFT_URL = os.environ.get("SFT_URL", "http://localhost:8012/v1")
BASE_MODEL = os.environ.get("BASE_MODEL", "judge")
SFT_MODEL = os.environ.get("SFT_MODEL", "sft14b-judge")


def call_vllm(client, base_url, model, sys_p, user_p, max_tokens, temperature, no_think):
    if no_think:
        sys_p = sys_p + "\n\n/no_think"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        "max_tokens": max_tokens, "temperature": temperature,
    }
    t0 = time.monotonic()
    try:
        r = client.post(f"{base_url}/chat/completions", json=payload, timeout=180.0)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raw = f"<<<ERR:{type(e).__name__}>>>"
    return raw, int((time.monotonic() - t0) * 1000)


_GEMINI_INIT = False


def init_gemini():
    global _GEMINI_INIT
    if _GEMINI_INIT:
        return
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
    _GEMINI_INIT = True


def call_gemini(sys_p, user_p):
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


def normalize_rec(rec):
    """Some pool rows lack `expected`/`final` etc. Fill with empty strings."""
    return {
        "cat": rec.get("cat", ""),
        "query": rec.get("query", "") or "",
        "final": rec.get("final", "") or "",
        "expected": rec.get("expected", "") or "",
        "before": rec.get("before", "") or "",
        "after": rec.get("after", "") or "",
    }


def predict_one(client, rec, judges, max_tokens, no_think):
    r = normalize_rec(rec)
    out = {
        "sid": rec.get("sid"), "rollout_hash": rec.get("rollout_hash"),
        "cat": rec.get("cat", ""), "label": rec.get("label"),
    }
    if "base" in judges or "sft" in judges:
        sys_p, user_p, _ = build_router_qwen_v2(r)
        if "base" in judges:
            raw, ms = call_vllm(client, BASE_URL, BASE_MODEL, sys_p, user_p, max_tokens, 0.0, no_think)
            out["base"] = extract_verdict(raw if not raw.startswith("<<<ERR") else "Incorrect")
            out["base_raw"] = raw
            out["base_ms"] = ms
        if "sft" in judges:
            raw, ms = call_vllm(client, SFT_URL, SFT_MODEL, sys_p, user_p, max_tokens, 0.0, no_think)
            out["sft"] = extract_verdict(raw if not raw.startswith("<<<ERR") else "Incorrect")
            out["sft_raw"] = raw
            out["sft_ms"] = ms
    if "gem" in judges:
        sys_p, user_p, _ = build_router_gemini_v2(r)
        raw, ms = call_gemini(sys_p, user_p)
        out["gem"] = extract_verdict(raw if not raw.startswith("<<<ERR") else "Incorrect")
        out["gem_raw"] = raw
        out["gem_ms"] = ms
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judges", default="base,sft,gem", help="comma list of base,sft,gem")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="random subsample size (seed=42)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--think", action="store_true", help="enable thinking (omit /no_think)")
    args = ap.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    if "gem" in judges:
        init_gemini()

    rows = [json.loads(l) for l in open(args.in_path)]
    if args.sample and args.sample < len(rows):
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.sample]
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", buffering=1)

    client = httpx.Client(timeout=httpx.Timeout(180.0))
    no_think = not args.think

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(predict_one, client, r, judges, args.max_tokens, no_think): r for r in rows}
        n = 0
        for fut in as_completed(futs):
            res = fut.result()
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            n += 1
            if n % 25 == 0 or n == len(rows):
                el = time.monotonic() - t0
                print(f"  {n}/{len(rows)}  ({el:.0f}s)", flush=True)
    fout.close()
    print(f"done → {out_path}  n={len(rows)}")


if __name__ == "__main__":
    main()
