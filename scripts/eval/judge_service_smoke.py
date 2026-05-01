"""Smoke-test the judge service against the manual oracle.

Reads N records from runs/judge_baseline_20260430/eval/manual_review_input.jsonl
+ manual_verdicts.jsonl, POSTs each to the running judge service, and prints
agreement vs the manual labels. Confirms the router prompt + verdict
extraction round-trip works through the FastAPI sidecar.

Run with the service up (sbatch scripts/serving/judge_service.sbatch):

    PYTHONPATH=src python scripts/eval/judge_service_smoke.py --n 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from calendar_agent.judge.client import verdict, health, aclose

REPO = Path(__file__).resolve().parents[2]
INPUT_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_review_input.jsonl"
TRUTH_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_verdicts.jsonl"


async def main(n: int, concurrency: int) -> None:
    print("health:", await health())
    recs = [json.loads(l) for l in INPUT_JSONL.open()]
    truth = {int(json.loads(l)["idx"]): json.loads(l)["verdict"] for l in TRUTH_JSONL.open()}
    sample = recs[:n]
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int, r: dict) -> tuple[int, str, str, int]:
        async with sem:
            t0 = time.monotonic()
            resp = await verdict(
                cat=r["cat"], query=r["query"], final=r.get("final", ""),
                expected=r.get("expected", ""), before=r["before"], after=r["after"],
                scenario_id=r.get("sid"),
            )
            return i, resp["verdict"], truth.get(i, "?"), int((time.monotonic() - t0) * 1000)

    results = await asyncio.gather(*(one(i, r) for i, r in enumerate(sample)))
    agree = sum(1 for _, v, t, _l in results if v == t)
    print(f"\nagreement: {agree}/{len(results)}")
    for i, v, t, lat in results:
        mark = "✓" if v == t else "✗"
        print(f"  [{i:3d}] {mark} judge={v:<9} manual={t:<9} {lat}ms  {sample[i]['cat']}")
    await aclose()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()
    asyncio.run(main(args.n, args.concurrency))
