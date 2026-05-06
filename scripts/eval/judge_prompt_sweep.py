#!/usr/bin/env python3
"""Sweep multiple prompt builders against rl-sft-4952 on tier-1 + tier-2.

Tests every builder and reports tier-1 + tier-2 accuracy with Wilson CIs
plus median latency and output length. Outputs raw predictions per
(builder × tier) row to allow downstream analysis.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from calendar_agent.judge.prompts import (
    build_router, build_router_qwen_v2,
    build_fewshot, build_fewshot_v3, build_fewshot_v4_dayfocus,
    build_fewshot_v3_lenient, build_fewshot_v4_dayfocus_lenient,
    build_cot_checklist_v2,
    extract_verdict, LENIENT_ADDENDUM,
)
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT


VLLM_BASE = os.environ.get("JUDGE_VLLM_BASE", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("JUDGE_VLLM_MODEL", "judge")


# ── New experimental prompts targeted at terse-correct models ──

def build_minimal(rec):
    """Bare-bones: just the data + a verdict request. No reasoning structure."""
    sys_p = "You are an expert evaluator of calendar-assistant behavior. End your response with the single word Correct or Incorrect on the last line."
    user = (
        f"User asked: {rec['query']}\n\n"
        f"Assistant said: {rec['final'] or '(empty)'}\n\n"
        f"Expected: {rec.get('expected') or '(unspecified)'}\n\n"
        f"Calendar BEFORE:\n{rec['before']}\n\n"
        f"Calendar AFTER:\n{rec['after']}\n\n"
        "Was the assistant's behavior correct? Verdict:"
    )
    return sys_p, user, {"max_tokens": 512}


def build_eval_classic(rec):
    """The original Gemini evaluation prompt."""
    user = (
        f"Query: {rec['query']}\n\n"
        f"Response: {rec['final'] or '(no response)'}\n\n"
        f"Expected: {rec.get('expected') or '(not specified)'}\n\n"
        f"Before:\n{rec['before']}\n\n"
        f"After:\n{rec['after']}\n\n"
        "Was the task completed correctly? End with one word: Correct or Incorrect."
    )
    return EVAL_SYSTEM_PROMPT, user, {"max_tokens": 512}


def build_router_qwen_v2_lenient(rec):
    """router_qwen_v2 with the LENIENT_ADDENDUM appended."""
    s, u, o = build_router_qwen_v2(rec)
    return s + LENIENT_ADDENDUM, u, o


def build_minimal_lenient(rec):
    s, u, o = build_minimal(rec)
    return s + LENIENT_ADDENDUM, u, o


BUILDERS = {
    # Currently deployed
    "router_qwen_v2 (DEPLOYED)": build_router_qwen_v2,
    # Single-strategy (no per-cat dispatch)
    "fewshot_v3":                build_fewshot_v3,
    "fewshot_v4_dayfocus":       build_fewshot_v4_dayfocus,
    "fewshot":                   build_fewshot,
    "cot_checklist_v2":          build_cot_checklist_v2,
    # Original v1 per-cat
    "router_v1":                 build_router,
    # Lenient variants
    "fewshot_v3_lenient":        build_fewshot_v3_lenient,
    "fewshot_v4_dayfocus_lenient": build_fewshot_v4_dayfocus_lenient,
    "router_qwen_v2_lenient":    build_router_qwen_v2_lenient,
    # Stripped down
    "eval_classic":              build_eval_classic,
    "minimal":                   build_minimal,
    "minimal_lenient":           build_minimal_lenient,
}


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


def predict(client, rec, builder, no_think=True):
    r = {
        "cat": rec.get("cat", ""), "query": rec.get("query", "") or "",
        "final": rec.get("final", "") or "", "expected": rec.get("expected", "") or "",
        "before": rec.get("before", "") or "", "after": rec.get("after", "") or "",
    }
    sys_p, user_p, opts = builder(r)
    if no_think:
        sys_p = sys_p + "\n\n/no_think"
    raw, ms = call(client, sys_p, user_p, max_tokens=opts.get("max_tokens", 512))
    v = extract_verdict(raw if not raw.startswith("<<<ERR") else "Incorrect")
    return {"sid": rec.get("sid"), "cat": rec.get("cat", ""), "label": rec.get("label"),
            "verdict": v, "raw_len": len(raw), "ms": ms, "raw": raw}


def wilson(k, n, z=1.96):
    if n == 0: return (0,0,0)
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (p, max(0, c-h), min(1, c+h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1", default="data/judge/v2_20260502/eval.jsonl")
    ap.add_argument("--tier2", default="data/judge/v2_20260502/disagreements.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--builders", default="", help="comma list, empty=all")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=httpx.Timeout(180.0))

    selected = (args.builders.split(",") if args.builders
                else list(BUILDERS.keys()))
    selected = [s for s in selected if s in BUILDERS]
    print(f"Sweeping {len(selected)} builders.")

    summary = []  # list of {builder, tier, n, ok, acc, lo, hi, med_ms, med_len}

    for tier_name, path in [("tier1", args.tier1), ("tier2", args.tier2)]:
        rows = [json.loads(l) for l in open(path)]
        print(f"\n>>>>> {tier_name} (n={len(rows)}) <<<<<")
        for builder_name in selected:
            builder = BUILDERS[builder_name]
            t0 = time.monotonic()
            preds = []
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(predict, client, r, builder): r for r in rows}
                for fut in as_completed(futs):
                    preds.append(fut.result())
            ok = sum(1 for p in preds if p["verdict"] == p["label"])
            p, lo, hi = wilson(ok, len(preds))
            ms = sorted(x["ms"] for x in preds)
            lens = sorted(x["raw_len"] for x in preds)
            row = {
                "builder": builder_name, "tier": tier_name,
                "n": len(preds), "ok": ok,
                "acc": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                "med_ms": ms[len(ms)//2], "med_len": lens[len(lens)//2],
                "elapsed_s": round(time.monotonic()-t0, 1),
            }
            summary.append(row)
            print(f"  {builder_name:28s}  {p*100:5.1f}% [{lo*100:.1f},{hi*100:.1f}]  med_ms={row['med_ms']:>6}  med_len={row['med_len']:>5}  ({row['elapsed_s']}s)")
            # Save predictions per builder per tier
            with (out_dir / f"{builder_name.replace(' ', '_').replace('(','').replace(')','')}_{tier_name}.jsonl").open("w") as f:
                for pp in preds:
                    f.write(json.dumps({**pp, "builder": builder_name}, ensure_ascii=False) + "\n")

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"{'builder':28s}  {'tier1 acc':>17s}  {'tier2 acc':>17s}  {'p50 ms':>7s}  {'p50 len':>8s}")
    by_b = {}
    for r in summary:
        by_b.setdefault(r["builder"], {})[r["tier"]] = r
    for b, t in by_b.items():
        t1 = t.get("tier1"); t2 = t.get("tier2")
        t1s = f"{t1['acc']*100:5.1f}% [{t1['lo']*100:.1f},{t1['hi']*100:.1f}]" if t1 else "—"
        t2s = f"{t2['acc']*100:5.1f}% [{t2['lo']*100:.1f},{t2['hi']*100:.1f}]" if t2 else "—"
        ms = t1.get("med_ms") if t1 else "—"
        ln = t1.get("med_len") if t1 else "—"
        print(f"{b:28s}  {t1s:>17}  {t2s:>17}  {str(ms):>7}  {str(ln):>8}")


if __name__ == "__main__":
    main()
