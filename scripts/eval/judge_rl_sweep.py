#!/usr/bin/env python3
"""Sweep tier-1 eval over many RL checkpoints loaded as LoRA modules.

Hits a single vLLM server with one base model + many LoRA names.
Iterates over LoRA names, evaluating tier-1 (or any input jsonl) for each.

Usage:
  PYTHONPATH=src python scripts/eval/judge_rl_sweep.py \
      --base http://localhost:8011/v1 \
      --in data/judge/v2_20260502/eval.jsonl \
      --out runs/judge_filter_validation_20260505/sweep_base.jsonl \
      --names rl-base-0500,rl-base-1000,...,rl-base-5077 --workers 8
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from calendar_agent.judge.prompts import build_router_qwen_v2, extract_verdict  # noqa


def call(client, base_url, model, sys_p, user_p, max_tokens=512):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }
    t0 = time.monotonic()
    try:
        r = client.post(f"{base_url}/chat/completions", json=payload, timeout=180.0)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raw = f"<<<ERR:{type(e).__name__}>>>"
    return raw, int((time.monotonic() - t0) * 1000)


def predict_one(client, base_url, model, rec):
    r = {
        "cat": rec.get("cat", ""), "query": rec.get("query", "") or "",
        "final": rec.get("final", "") or "", "expected": rec.get("expected", "") or "",
        "before": rec.get("before", "") or "", "after": rec.get("after", "") or "",
    }
    sys_p, user_p, _ = build_router_qwen_v2(r)
    sys_p = sys_p + "\n\n/no_think"
    raw, ms = call(client, base_url, model, sys_p, user_p)
    return {
        "sid": rec.get("sid"), "rollout_hash": rec.get("rollout_hash"),
        "cat": rec.get("cat", ""), "label": rec.get("label"),
        "model": model, "verdict": extract_verdict(raw if not raw.startswith("<<<ERR") else "Incorrect"),
        "raw_len": len(raw), "ms": ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", required=True, help="comma-separated LoRA model names")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_path)]
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    fout = open(args.out, "w", buffering=1)
    client = httpx.Client(timeout=httpx.Timeout(180.0))

    for name in names:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(predict_one, client, args.base, name, r): r for r in rows}
            done = 0
            for fut in as_completed(futs):
                res = fut.result()
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                done += 1
        ok = sum(1 for line in open(args.out) for r in [json.loads(line)] if r["model"] == name and r["verdict"] == r["label"])
        n = sum(1 for line in open(args.out) for r in [json.loads(line)] if r["model"] == name)
        print(f"  done in {time.monotonic()-t0:.0f}s  acc={ok/n*100:.1f}% ({ok}/{n})", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
