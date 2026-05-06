#!/usr/bin/env python3
"""Serial per-call latency benchmark for gpt-oss-20B judge, mirrors the
Qwen rows in judge_latency_bench.py so the README table is apples-to-apples.

Usage:
    PYTHONPATH=src python scripts/eval/judge_latency_bench_gptoss.py \\
        --n 30 --reasoning-effort low

Saves:
    runs/judge_prompt_tune_20260430/results/latency_bench_gptoss.json
"""
import argparse, asyncio, json, random, sys, time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eval"))
sys.path.insert(0, str(REPO / "src"))

from judge_prompt_tune import RECS, build_router, diff_states, CHECKLIST_V2_SYS  # noqa: E402

API_BASE_DEFAULT = "http://azkaban:8020/v1"
MODEL_DEFAULT = "judge-gptoss"


def short_prompt(rec):
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
    s, u, _ = build_router(rec)
    return s, u


async def call_one(client, model, system, user, n_tokens, reasoning):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": n_tokens,
        "chat_template_kwargs": {"reasoning_effort": reasoning},
    }
    t0 = time.time()
    r = await client.post("/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    return time.time() - t0


async def bench(name, base, model, builder, recs, n_tokens, reasoning):
    times = []
    print(f"\n[{name}] base={base} model={model} reasoning={reasoning} n={len(recs)} (serial)")
    async with httpx.AsyncClient(base_url=base, timeout=300) as client:
        s, u = builder(recs[0])
        try:
            await call_one(client, model, s, u, n_tokens, reasoning)
        except Exception as e:
            print(f"  warmup failed: {e}")
            return name, []
        for i, rec in enumerate(recs):
            s, u = builder(rec)
            try:
                dt = await call_one(client, model, s, u, n_tokens, reasoning)
                times.append(dt)
                if i % 5 == 0:
                    print(f"  [{i+1}/{len(recs)}] {dt:.2f}s")
            except Exception as e:
                print(f"  [{i+1}] FAILED: {e}")
    return name, times


def stats(times):
    if not times: return {}
    s = sorted(times); n = len(s)
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
    p.add_argument("--base", default=API_BASE_DEFAULT)
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    args = p.parse_args()

    sample = random.Random(42).sample(RECS, args.n)

    results = {}
    name, t = await bench("gptoss20b_router", args.base, args.model, large_prompt, sample,
                          n_tokens=1024, reasoning=args.reasoning_effort)
    results[f"{name}_{args.reasoning_effort}"] = stats(t)

    name, t = await bench("gptoss20b_short", args.base, args.model, short_prompt, sample,
                          n_tokens=512, reasoning=args.reasoning_effort)
    results[f"{name}_{args.reasoning_effort}"] = stats(t)

    print("\n\n=== LATENCY SUMMARY (serial, gpt-oss-20b) ===")
    print(f"{'config':<36} {'n':>4} {'min':>6} {'p50':>6} {'mean':>6} {'p90':>6} {'p99':>6} {'max':>6} {'total':>8}")
    for name, s in results.items():
        if not s:
            print(f"{name:<36} (no data)")
            continue
        print(f"{name:<36} {s['n']:>4} {s['min']:>6.2f} {s['p50']:>6.2f} {s['mean']:>6.2f} "
              f"{s['p90']:>6.2f} {s['p99']:>6.2f} {s['max']:>6.2f} {s['total']:>8.1f}")

    out = REPO / "runs/judge_prompt_tune_20260430/results/latency_bench_gptoss.json"
    json.dump({"results": results, "n": args.n,
               "reasoning_effort": args.reasoning_effort}, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
