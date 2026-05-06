#!/usr/bin/env python3
"""Run rl-sft-4952 on tier-1/tier-2 using BOTH prompts and compare.

  - "router"   : build_router_qwen_v2 + /no_think  (current deployment)
  - "eval"     : EVAL_SYSTEM_PROMPT + the original eval user-prompt
                 (the non-router one Gemini used to use)

Hits vLLM directly at $JUDGE_VLLM_BASE (default :8000), model="judge".
Outputs per-row predictions and prints a side-by-side accuracy summary.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from calendar_agent.judge.prompts import build_router_qwen_v2, extract_verdict
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT

VLLM_BASE = os.environ.get("JUDGE_VLLM_BASE", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("JUDGE_VLLM_MODEL", "judge")


def call(client, sys_p, user_p, max_tokens=512):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }
    t0 = time.monotonic()
    try:
        r = client.post(f"{VLLM_BASE}/chat/completions", json=payload, timeout=180.0)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raw = f"<<<ERR:{type(e).__name__}>>>"
    return raw, int((time.monotonic() - t0) * 1000)


def predict(client, rec, mode):
    r = {
        "cat": rec.get("cat", ""), "query": rec.get("query", "") or "",
        "final": rec.get("final", "") or "", "expected": rec.get("expected", "") or "",
        "before": rec.get("before", "") or "", "after": rec.get("after", "") or "",
    }
    if mode == "router":
        sys_p, user_p, _ = build_router_qwen_v2(r)
        sys_p = sys_p + "\n\n/no_think"
    elif mode == "eval":
        sys_p = EVAL_SYSTEM_PROMPT + "\n\n/no_think"
        user_p = (
            f"Query: {r['query']}\n\n"
            f"Response: {r['final'] or '(no response)'}\n\n"
            f"Expected: {r['expected'] or '(not specified)'}\n\n"
            f"Before:\n{r['before']}\n\n"
            f"After:\n{r['after']}\n\n"
            "Was the task completed correctly? End with one word: Correct or Incorrect."
        )
    else:
        raise ValueError(mode)
    raw, ms = call(client, sys_p, user_p)
    v = extract_verdict(raw if not raw.startswith("<<<ERR") else "Incorrect")
    return {"sid": rec.get("sid"), "rollout_hash": rec.get("rollout_hash"),
            "cat": rec.get("cat", ""), "label": rec.get("label"),
            "mode": mode, "verdict": v, "raw_len": len(raw), "ms": ms, "raw": raw}


def wilson(k, n, z=1.96):
    if n == 0: return (0,0,0)
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (p, max(0,c-h), min(1,c+h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_path)]
    print(f"Loaded {len(rows)} rows from {args.in_path}")
    fout = open(args.out, "w", buffering=1)
    client = httpx.Client(timeout=httpx.Timeout(180.0))

    summary = {}
    for mode in ("router", "eval"):
        print(f"\n=== mode={mode} ===")
        t0 = time.monotonic()
        preds = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(predict, client, r, mode): r for r in rows}
            for fut in as_completed(futs):
                p = fut.result()
                preds.append(p)
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")
        # Aggregate
        ok = sum(1 for p in preds if p["verdict"] == p["label"])
        acc, lo, hi = wilson(ok, len(preds))
        med_ms = sorted(p["ms"] for p in preds)[len(preds)//2]
        med_len = sorted(p["raw_len"] for p in preds)[len(preds)//2]
        print(f"  acc={acc*100:5.1f}% [{lo*100:.1f}, {hi*100:.1f}]  median_ms={med_ms}  median_len={med_len}  elapsed={time.monotonic()-t0:.0f}s")
        # Per-cat
        cats = sorted(set(p["cat"] for p in preds))
        for c in cats:
            cp = [p for p in preds if p["cat"] == c]
            cok = sum(1 for p in cp if p["verdict"] == p["label"])
            print(f"    {c[:50]:50s}  {cok}/{len(cp)} = {cok/len(cp)*100:.1f}%")
        summary[mode] = {"acc": acc, "ci": (lo, hi), "med_ms": med_ms, "med_len": med_len, "n": len(preds)}

    print("\n=== summary ===")
    for mode, s in summary.items():
        print(f"  {mode:8s}  acc={s['acc']*100:5.1f}%  med_ms={s['med_ms']}  med_len={s['med_len']}  n={s['n']}")


if __name__ == "__main__":
    main()
