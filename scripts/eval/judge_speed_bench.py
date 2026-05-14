#!/usr/bin/env python3
"""Latency bench for the deployed local judge service at $JUDGE_URL.

Posts the curated v2 eval set to /verdict in two regimes:
  - serial         (concurrency=1) — per-call cost
  - concurrent     (configurable) — throughput / queueing

Reports p50/p90/p99/mean wall ms per request, plus aggregate QPS.

Usage:
  PYTHONPATH=src /home/abhor/miniconda3/envs/agentic/bin/python \
      scripts/eval/judge_speed_bench.py --n 60 --concurrency 8
"""
import argparse, asyncio, json, os, statistics, time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
EVAL_JSONL = REPO / "data/judge/v2_20260502/eval.jsonl"


def load_samples(n):
    rows = []
    with open(EVAL_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            rows.append({
                "cat": r.get("cat", ""),
                "query": r.get("query", ""),
                "final": r.get("final", ""),
                "expected": r.get("expected", ""),
                "before": r.get("before", ""),
                "after": r.get("after", ""),
                "scenario_id": r.get("sid", ""),
            })
            if len(rows) >= n: break
    return rows


async def one_call(client, url, payload):
    t0 = time.perf_counter()
    r = await client.post(f"{url}/verdict", json=payload, timeout=120.0)
    r.raise_for_status()
    dt = (time.perf_counter() - t0) * 1000.0
    return dt, r.json().get("verdict", "?")


async def run_serial(url, samples):
    out = []
    async with httpx.AsyncClient() as client:
        for s in samples:
            dt, v = await one_call(client, url, s)
            out.append(dt)
    return out


async def run_concurrent(url, samples, concurrency):
    sem = asyncio.Semaphore(concurrency)
    out = []
    async with httpx.AsyncClient() as client:
        async def worker(s):
            async with sem:
                dt, _ = await one_call(client, url, s)
                out.append(dt)
        await asyncio.gather(*(worker(s) for s in samples))
    return out


def stats(latencies):
    if not latencies:
        return {}
    s = sorted(latencies)
    def pct(p):
        i = int(round(p / 100.0 * (len(s) - 1)))
        return s[i]
    return {
        "n": len(s),
        "mean_ms": statistics.fmean(s),
        "p50_ms": pct(50),
        "p90_ms": pct(90),
        "p99_ms": pct(99),
        "min_ms": s[0],
        "max_ms": s[-1],
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--url", default=os.environ.get("JUDGE_URL", "http://127.0.0.1:8765"))
    p.add_argument("--out", default="runs/judge_speed_20260506/results.json")
    args = p.parse_args()

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Health check
    async with httpx.AsyncClient() as c:
        h = await c.get(f"{args.url}/health", timeout=5.0)
        h.raise_for_status()
        info = h.json()
    print(f"Judge: model={info.get('model')} prompt={info.get('prompt_version')}")

    samples = load_samples(args.n)
    print(f"Loaded {len(samples)} samples from {EVAL_JSONL.name}")

    print(f"\n[1/3] Warmup (concurrency=1, n=3)...")
    await run_serial(args.url, samples[:3])

    print(f"\n[2/3] Serial (n={len(samples)})...")
    t0 = time.perf_counter()
    serial_lat = await run_serial(args.url, samples)
    serial_wall = time.perf_counter() - t0
    s_stats = stats(serial_lat)
    print(json.dumps(s_stats, indent=2))
    print(f"  wall={serial_wall:.1f}s  qps={len(serial_lat)/serial_wall:.2f}")

    print(f"\n[3/3] Concurrent (concurrency={args.concurrency}, n={len(samples)})...")
    t0 = time.perf_counter()
    conc_lat = await run_concurrent(args.url, samples, args.concurrency)
    conc_wall = time.perf_counter() - t0
    c_stats = stats(conc_lat)
    print(json.dumps(c_stats, indent=2))
    print(f"  wall={conc_wall:.1f}s  qps={len(conc_lat)/conc_wall:.2f}")

    result = {
        "judge_url": args.url,
        "judge_info": info,
        "n_samples": len(samples),
        "concurrency": args.concurrency,
        "serial": {**s_stats, "wall_s": serial_wall, "qps": len(serial_lat) / serial_wall},
        "concurrent": {**c_stats, "wall_s": conc_wall, "qps": len(conc_lat) / conc_wall},
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
