#!/usr/bin/env python3
"""Latency benchmark: time per-call wall time for each judge configuration,
calls SERIAL so we measure individual call cost (not concurrent throughput).

Configs:
  gemini       - Gemini 2.0 Flash via Vertex AI, rl_train.py prompt
  14b_large    - vLLM Qwen3-14B fp8, router fewshot_v3 prompt (~4k tokens)
  14b_small    - vLLM Qwen3-14B fp8, short prompt (no examples)
  8b_small     - vLLM Qwen3-8B fp8, short prompt

Usage:
  PYTHONPATH=src python scripts/eval/judge_latency_bench.py --n 30
"""
import argparse, asyncio, json, os, statistics, sys, time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eval"))

from judge_prompt_tune import (
    RECS, build_router, diff_states, CHECKLIST_V2_SYS,
)


def short_prompt(rec):
    """Short judge prompt: system rules only, no examples."""
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    user = (
        f"User query: {q}\n\n"
        f"Assistant response: {final or '(no response)'}\n\n"
        f"Expected behavior: {exp or '(not specified)'}\n\n"
        f"Calendar diff:\n{diff}\n\n"
        f"BEFORE:\n{rec['before']}\n\nAFTER:\n{rec['after']}\n\n"
        "End with Correct or Incorrect on the last line."
    )
    return CHECKLIST_V2_SYS, user


def large_prompt(rec):
    sys_, user, _ = build_router(rec)
    return sys_, user


# ── Gemini ──
GEMINI_AVAILABLE = False
try:
    from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT
    from calendar_agent.paths import CREDENTIALS_PATH
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(CREDENTIALS_PATH))
    import vertexai
    from vertexai.generative_models import GenerativeModel
    vertexai.init(project=os.environ.get("GCP_PROJECT", "nextsense-research"),
                  location="us-central1")
    _gemini_model = GenerativeModel("gemini-2.0-flash-001",
                                    system_instruction=[EVAL_SYSTEM_PROMPT])
    GEMINI_AVAILABLE = True
except Exception as e:
    print(f"[gemini setup failed: {type(e).__name__}: {e}]")


def call_gemini_sync(rec):
    diff = diff_states(rec["before"], rec["after"])
    prompt = (
        f"Query: {rec['query']}\n\n"
        f"Response: {rec['final'] or '(no response)'}\n\n"
        f"Expected: {rec['expected'] or '(not specified)'}\n\n"
        f"Before:\n{rec['before']}\n\nAfter:\n{rec['after']}\n\n"
        "Was the task completed correctly? End with one word: Correct or Incorrect."
    )
    t0 = time.time()
    r = _gemini_model.generate_content(prompt)
    dt = time.time() - t0
    return dt, r.text


# ── vLLM serial caller ──
async def call_vllm(client, model, system, user, n_tokens=512):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": n_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = await client.post("/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    dt = time.time() - t0
    txt = r.json()["choices"][0]["message"]["content"]
    return dt, txt


async def bench_vllm(name, base, model, builder, recs, n_tokens=512):
    times = []
    print(f"\n[{name}] base={base} model={model} n={len(recs)} (serial)")
    async with httpx.AsyncClient(base_url=base, timeout=120) as client:
        # 1 warmup
        sysp, userp = builder(recs[0])
        try:
            await call_vllm(client, model, sysp, userp, n_tokens)
        except Exception as e:
            print(f"  warmup failed: {e}")
            return name, []
        for i, rec in enumerate(recs):
            sysp, userp = builder(rec)
            try:
                dt, _ = await call_vllm(client, model, sysp, userp, n_tokens)
                times.append(dt)
                if i % 5 == 0: print(f"  [{i+1}/{len(recs)}] {dt:.2f}s")
            except Exception as e:
                print(f"  [{i+1}] FAILED: {e}")
    return name, times


def bench_gemini(recs):
    name = "gemini"
    times = []
    print(f"\n[{name}] gemini-2.0-flash via Vertex AI n={len(recs)} (serial)")
    if not GEMINI_AVAILABLE:
        print("  Gemini not available, skipping")
        return name, []
    # 1 warmup
    try:
        call_gemini_sync(recs[0])
    except Exception as e:
        print(f"  warmup failed: {e}")
        return name, []
    for i, rec in enumerate(recs):
        try:
            dt, _ = call_gemini_sync(rec)
            times.append(dt)
            if i % 5 == 0: print(f"  [{i+1}/{len(recs)}] {dt:.2f}s")
        except Exception as e:
            print(f"  [{i+1}] FAILED: {e}")
    return name, times


def stats(times):
    if not times: return {}
    s = sorted(times)
    n = len(s)
    return {
        "n": n,
        "min": round(s[0], 2),
        "p50": round(s[n//2], 2),
        "mean": round(sum(s)/n, 2),
        "p90": round(s[int(0.9*n)], 2),
        "p99": round(s[min(n-1, int(0.99*n))], 2),
        "max": round(s[-1], 2),
        "total": round(sum(s), 1),
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--skip-gemini", action="store_true")
    p.add_argument("--14b-base", dest="b14", default="http://localhost:8013/v1")
    p.add_argument("--8b-base",  dest="b8",  default="http://localhost:8014/v1")
    args = p.parse_args()

    # Pick the same recs for every config (deterministic)
    import random
    rng = random.Random(42)
    sample = rng.sample(RECS, args.n)

    results = {}

    # 14B large (router prompt)
    name, t = await bench_vllm("14b_large", args.b14, "judge", large_prompt, sample, n_tokens=1024)
    results[name] = stats(t)

    # 14B small (short prompt)
    name, t = await bench_vllm("14b_small", args.b14, "judge", short_prompt, sample, n_tokens=512)
    results[name] = stats(t)

    # 8B small
    name, t = await bench_vllm("8b_small", args.b8, "judge-8b", short_prompt, sample, n_tokens=512)
    results[name] = stats(t)

    # Gemini (synchronous, but small-batch — call in thread)
    if not args.skip_gemini:
        loop = asyncio.get_event_loop()
        name, t = await loop.run_in_executor(None, bench_gemini, sample)
        results[name] = stats(t)

    # Print summary table
    print("\n\n=" * 5 + " LATENCY SUMMARY " + "=" * 5)
    print(f"{'config':<14} {'n':>4} {'min':>6} {'p50':>6} {'mean':>6} {'p90':>6} {'p99':>6} {'max':>6} {'total':>8}")
    for name, s in results.items():
        if not s:
            print(f"{name:<14} (no data)")
            continue
        print(f"{name:<14} {s['n']:>4} {s['min']:>6.2f} {s['p50']:>6.2f} {s['mean']:>6.2f} "
              f"{s['p90']:>6.2f} {s['p99']:>6.2f} {s['max']:>6.2f} {s['total']:>8.1f}")

    out = REPO / "runs/judge_prompt_tune_20260430/results/latency_bench.json"
    json.dump({"results": results, "n": args.n}, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
