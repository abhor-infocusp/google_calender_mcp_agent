#!/usr/bin/env python3
"""Evaluate gpt-oss-20B as the judge against the 285 manual-labeled ART hold-out.

Reuses prompt builders + scoring from judge_prompt_tune.py. Differs only in the
HTTP payload: gpt-oss uses harmony format with `reasoning_effort` instead of
Qwen's `enable_thinking` chat-template kwarg.

The vLLM server (judge_serve_gpt_oss_20b.sbatch) listens on :8020 with
served-model-name=judge-gptoss.

Usage:
    PYTHONPATH=src python scripts/eval/judge_eval_gpt_oss.py \\
        --variant router --concurrency 16 --reasoning-effort low

    # Quick sanity (10 samples, raw output dumped):
    PYTHONPATH=src python scripts/eval/judge_eval_gpt_oss.py \\
        --variant router --limit 10 --reasoning-effort low --dump-raw
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eval"))

from judge_prompt_tune import (  # noqa: E402
    RECS, TRUTH, VARIANTS, OUT_DIR, extract_verdict, write_summary_row,
)

API_BASE = os.environ.get("JUDGE_API_BASE", "http://azkaban:8020/v1")
MODEL = os.environ.get("JUDGE_MODEL", "judge-gptoss")


async def query_one(client: httpx.AsyncClient, idx: int, system: str, user: str,
                    opts: dict, reasoning_effort: str, dump_raw: bool) -> dict:
    """gpt-oss flavored chat completion. Sets reasoning_effort and folds any
    server-emitted reasoning_content into the inspectable raw text so
    extract_verdict can search both reasoning and answer for the verdict word."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": opts.get("temperature", 0.0),
        "max_tokens": opts.get("max_tokens", 1024),
        # gpt-oss/harmony reasoning level — supported via chat_template_kwargs in vLLM
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
    }
    t0 = time.time()
    for attempt in range(3):
        try:
            r = await client.post("/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            # Combined view — verdict word may appear in answer or in reasoning
            combined = (content or "").strip()
            if not combined and reasoning:
                # Some fronts return verdict only in reasoning. Use it as fallback.
                combined = reasoning
            verdict = extract_verdict(combined)
            out = {
                "idx": idx,
                "raw": content,
                "verdict": verdict,
                "latency_s": round(time.time() - t0, 2),
            }
            if dump_raw:
                out["reasoning"] = reasoning
            return out
        except Exception as e:
            if attempt == 2:
                return {"idx": idx, "raw": f"[ERROR {e}]", "verdict": "Incorrect",
                        "latency_s": round(time.time() - t0, 2)}
            await asyncio.sleep(2)


async def run(variant: str, concurrency: int, limit: int | None,
              reasoning_effort: str, max_tokens: int | None, dump_raw: bool,
              tag: str) -> dict:
    builder = VARIANTS[variant]
    items = list(enumerate(RECS))
    if limit:
        items = items[:limit]
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = [None] * len(items)
    per_call_latencies: list[float] = []

    async with httpx.AsyncClient(base_url=API_BASE, timeout=300) as client:
        async def go(i, rec):
            async with sem:
                system, user, opts = builder(rec)
                if max_tokens is not None:
                    opts = dict(opts); opts["max_tokens"] = max_tokens
                res = await query_one(client, i, system, user, opts,
                                       reasoning_effort, dump_raw)
                results[i] = res
                per_call_latencies.append(res["latency_s"])

        t0 = time.time()
        await asyncio.gather(*(go(i, rec) for i, rec in items))
        wall = time.time() - t0

    correct = total = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion = defaultdict(int)
    for r in results:
        if r is None: continue
        i = r["idx"]; truth = TRUTH[i]; cat = RECS[i]["cat"]
        total += 1
        by_cat[cat][1] += 1
        if r["verdict"] == truth:
            correct += 1
            by_cat[cat][0] += 1
        confusion[(truth, r["verdict"])] += 1

    p50 = sorted(per_call_latencies)[len(per_call_latencies)//2] if per_call_latencies else 0
    p90 = sorted(per_call_latencies)[int(0.9*len(per_call_latencies))] if per_call_latencies else 0
    name = f"gptoss20b_{variant}_{reasoning_effort}" + (f"_{tag}" if tag else "")
    summary = {
        "variant": name,
        "n": total,
        "correct": correct,
        "acc_pct": round(100 * correct / total, 2) if total else 0.0,
        "wall_s": round(wall, 1),
        "p50_s": round(p50, 2),
        "p90_s": round(p90, 2),
        "concurrency": concurrency,
        "per_category": {c: round(100 * a / t, 2) for c, (a, t) in by_cat.items()},
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
    }
    out_path = OUT_DIR / f"{name}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            if r is None: continue
            f.write(json.dumps({**r, "truth": TRUTH[r["idx"]],
                                "cat": RECS[r["idx"]]["cat"]}) + "\n")
    write_summary_row(summary)
    print(f"\nWrote {out_path}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="router")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--dump-raw", action="store_true",
                   help="also persist reasoning_content for inspection")
    p.add_argument("--tag", default="", help="suffix for output name")
    args = p.parse_args()

    if args.variant not in VARIANTS:
        p.error(f"unknown variant {args.variant}; see judge_prompt_tune.py --list")

    print(f"Variant: {args.variant} | reasoning={args.reasoning_effort} | "
          f"n={args.limit or len(RECS)} | concurrency={args.concurrency}")
    print(f"API: {API_BASE}  model={MODEL}")
    s = asyncio.run(run(args.variant, args.concurrency, args.limit,
                         args.reasoning_effort, args.max_tokens,
                         args.dump_raw, args.tag))
    print(f"\nResult: {s['acc_pct']}%  ({s['correct']}/{s['n']})  in {s['wall_s']}s")
    print(f"Per-call: p50={s['p50_s']}s  p90={s['p90_s']}s")
    print(f"Per-category: {s['per_category']}")
    print(f"Confusion: {s['confusion']}")


if __name__ == "__main__":
    main()
