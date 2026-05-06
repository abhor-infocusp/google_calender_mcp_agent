#!/usr/bin/env python3
"""Hit the LIVE production judge service on the 285-traj manual oracle and
report accuracy + per-category + latency. Used for Phase-0 verification and
any subsequent regression checks.

Usage:
    PYTHONPATH=src /home/abhor/miniconda3/envs/agentic/bin/python \
        scripts/eval/judge_local_live_eval.py \
        [--out runs/judge_local_live_eval_<tag>/]
"""
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
INPUT_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_review_input.jsonl"
TRUTH_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_verdicts.jsonl"
URL = "http://localhost:8765/verdict"

CAT_SHORT = {
    "Complex Logic & Conflict (Advanced)": "Complex",
    "Human Chaos (Edge Cases/Fragments)": "Chaos",
    "Information Retrieval (Querying)": "IR",
    "Modifier & Correction (Rescheduling/Updates)": "Modifier",
    "Relative Time References (today, tomorrow, yesterday, this week)": "RelTime",
    "Schedule a Single Event": "Schedule",
    "Vague & Contextual (Reasoning Required)": "Vague",
}


def load_oracle() -> list[dict]:
    inputs = [json.loads(l) for l in open(INPUT_JSONL)]
    truth = [json.loads(l) for l in open(TRUTH_JSONL)]
    for i, rec in enumerate(inputs):
        if i < len(truth):
            rec["gt"] = truth[i].get("verdict") or rec.get("gt")
    return [r for r in inputs if r.get("gt") in ("Correct", "Incorrect")]


def call(client: httpx.Client, rec: dict) -> dict:
    body = {
        "cat": rec["cat"], "query": rec["query"], "final": rec.get("final", ""),
        "expected": rec.get("expected", ""), "before": rec["before"], "after": rec["after"],
        "scenario_id": rec.get("sid"),
    }
    t0 = time.time()
    try:
        r = client.post(URL, json=body)
        r.raise_for_status()
        d = r.json()
        return {"sid": rec["sid"], "cat": rec["cat"], "gt": rec["gt"],
                "pred": d["verdict"], "latency_s": time.time() - t0, "err": ""}
    except Exception as e:
        return {"sid": rec["sid"], "cat": rec["cat"], "gt": rec["gt"],
                "pred": "ERR", "latency_s": time.time() - t0, "err": str(e)[:120]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else REPO / f"runs/judge_local_live_eval_{datetime.now():%Y%m%d_%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = load_oracle()
    print(f"oracle size: {len(recs)} → {out_dir}")
    client = httpx.Client(timeout=httpx.Timeout(180.0))

    t0 = time.time()
    results: list[dict] = [None] * len(recs)  # type: ignore
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(call, client, r): i for i, r in enumerate(recs)}
        for k, fut in enumerate(as_completed(futs), 1):
            results[futs[fut]] = fut.result()
            if k % 50 == 0 or k == len(recs):
                print(f"  {k}/{len(recs)} in {time.time()-t0:.0f}s")

    with open(out_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n = len(results)
    errs = sum(1 for r in results if r["pred"] == "ERR")
    right = sum(1 for r in results if r["pred"] == r["gt"])
    lats = sorted(r["latency_s"] for r in results if r["pred"] != "ERR")
    p50 = lats[len(lats) // 2] if lats else 0
    p90 = lats[int(0.9 * len(lats))] if lats else 0
    p99 = lats[int(0.99 * len(lats))] if lats else 0
    fp = sum(1 for r in results if r["pred"] == "Correct" and r["gt"] == "Incorrect")
    fn = sum(1 for r in results if r["pred"] == "Incorrect" and r["gt"] == "Correct")

    print(f"\n=== LIVE judge service on 285-traj oracle ===")
    print(f"overall:  {right}/{n} = {100*right/n:.2f}%   errors={errs}")
    print(f"latency:  p50={p50:.2f}s  p90={p90:.2f}s  p99={p99:.2f}s")
    print(f"confusion: FP={fp}  FN={fn}\n")

    by = defaultdict(lambda: {"n": 0, "r": 0})
    for r in results:
        by[r["cat"]]["n"] += 1
        if r["pred"] == r["gt"]: by[r["cat"]]["r"] += 1
    print(f"{'cat':<10} {'right/total':>14} {'acc':>8}")
    for c, d in sorted(by.items(), key=lambda kv: -kv[1]["r"] / max(kv[1]["n"], 1)):
        print(f"{CAT_SHORT.get(c, c[:18]):<10} {d['r']}/{d['n']:<5} {100*d['r']/max(d['n'],1):>7.2f}%")

    summary = {
        "total": n, "correct": right, "accuracy": right / n,
        "errors": errs, "fp": fp, "fn": fn,
        "p50_s": p50, "p90_s": p90, "p99_s": p99,
        "by_cat": {CAT_SHORT.get(c, c): {"n": d["n"], "right": d["r"], "acc": d["r"] / max(d["n"], 1)}
                   for c, d in by.items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
