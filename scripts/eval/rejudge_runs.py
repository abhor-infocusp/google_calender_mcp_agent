"""Re-judge stored eval JSONs against the Gemini-flash structured judge.

Reads eval result JSONs (baseline.json or checkpoint-*.json) that already
contain `trajectory`, `before`, `after`, `expected`, `final_output`, `query`,
`category` — POSTs each row to the judge service `/verdict` endpoint and
writes a `*_rejudged.json` next to the original. Then prints overall accuracy.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

JUDGE_URL = "http://127.0.0.1:8765"
CONCURRENCY = 20
TIMEOUT = 60.0


async def judge_one(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, idx: int, row: dict
) -> tuple[int, str]:
    payload = {
        "cat": row["category"],
        "query": row["query"],
        "final": row.get("final_output", "") or "",
        "expected": row.get("expected", "") or "",
        "before": row.get("before", "") or "",
        "after": row.get("after", "") or "",
    }
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post("/verdict", json=payload, timeout=TIMEOUT)
                r.raise_for_status()
                return idx, r.json()["verdict"]
            except Exception as e:
                if attempt == 2:
                    print(f"  [{idx}] FAILED after retries: {e}", file=sys.stderr)
                    return idx, "Incorrect"
                await asyncio.sleep(1.0 * (attempt + 1))
        return idx, "Incorrect"


async def rejudge_file(path: Path) -> dict:
    raw = json.load(open(path))
    # Two formats: baseline.json -> {"test": {"results": [...]}}, checkpoint json -> {"test": {"results":[...]}} too
    bucket = raw.get("test") or raw.get("rl") or raw
    results = bucket["results"]

    print(f"\n=== {path}  ({len(results)} rows) ===", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    async with httpx.AsyncClient(base_url=JUDGE_URL, timeout=TIMEOUT) as client:
        tasks = [judge_one(client, sem, i, r) for i, r in enumerate(results)]
        done = 0
        for coro in asyncio.as_completed(tasks):
            i, v = await coro
            results[i]["verdict"] = v
            done += 1
            if done % 100 == 0 or done == len(results):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                print(f"  {done}/{len(results)}  ({rate:.1f}/s)", flush=True)

    correct = sum(1 for r in results if r.get("verdict") == "Correct")
    pct = 100.0 * correct / len(results)
    cats: dict[str, list[int]] = {}
    for r in results:
        c = r["category"]
        cats.setdefault(c, [0, 0])
        cats[c][1] += 1
        if r.get("verdict") == "Correct":
            cats[c][0] += 1

    out = path.with_name(path.stem + "_rejudged.json")
    json.dump(raw, open(out, "w"))
    print(f"  Overall: {correct}/{len(results)} = {pct:.1f}%")
    for c, (k, n) in sorted(cats.items()):
        print(f"    {c}: {k}/{n} = {100*k/n:.0f}%")
    print(f"  Saved -> {out}")
    return {"path": str(path), "correct": correct, "total": len(results), "pct": pct, "cats": cats}


async def main(files: list[Path]) -> None:
    summary = []
    for f in files:
        if not f.exists():
            print(f"SKIP missing: {f}", file=sys.stderr)
            continue
        summary.append(await rejudge_file(f))
    print("\n\n=== SUMMARY ===")
    for s in summary:
        print(f"{s['pct']:5.1f}%  {s['correct']:>4}/{s['total']}  {s['path']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    asyncio.run(main(args.files))
