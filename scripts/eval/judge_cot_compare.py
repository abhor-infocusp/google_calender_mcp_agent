#!/usr/bin/env python3
"""Compare base-14B vs SFT-14B CoT outputs on the same prompts.

Uses runs/judge_filter_validation_20260505/cot_compare_50.jsonl which has
base_raw and sft_raw from /no_think production prompts.

Reports:
  - length distribution (mean, median, p10, p90)
  - rows where both judges agreed on verdict — same length range?
  - rows where they disagreed — which judge's CoT looks more grounded?
  - structural signals: bullet count, "Step", "Action", citations to before/after
"""
from __future__ import annotations
import json, re, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P = REPO / "runs/judge_filter_validation_20260505/cot_compare_50.jsonl"


def stats(xs):
    if not xs:
        return None
    s = sorted(xs)
    return {
        "n": len(xs),
        "mean": int(statistics.mean(xs)),
        "median": int(statistics.median(xs)),
        "p10": s[len(s) // 10],
        "p90": s[-len(s) // 10] if len(s) >= 10 else s[-1],
        "min": s[0], "max": s[-1],
    }


def structural(raw):
    s = raw or ""
    return {
        "len": len(s),
        "lines": s.count("\n") + 1,
        "bullets": len(re.findall(r"(?m)^\s*[-*•]\s", s)),
        "step_kw": len(re.findall(r"(?i)\b(step\s*\d|first|second|then|next)\b", s)),
        "before_after_ref": len(re.findall(r"(?i)\b(before|after)\b", s)),
        "trailing_verdict": s.strip().split()[-1] if s.strip() else "",
    }


def main():
    rows = [json.loads(l) for l in P.open()]
    print(f"Loaded {len(rows)} rows")
    base_lens, sft_lens = [], []
    base_struct, sft_struct = [], []
    for r in rows:
        bs = structural(r.get("base_raw", ""))
        ss = structural(r.get("sft_raw", ""))
        base_lens.append(bs["len"]); sft_lens.append(ss["len"])
        base_struct.append(bs); sft_struct.append(ss)

    print("\nLength (chars):")
    print(f"  base  {stats(base_lens)}")
    print(f"  sft   {stats(sft_lens)}")

    # Avg structural signals
    def avg(items, k): return sum(x[k] for x in items) / len(items)
    print("\nAverage structural signals:")
    print(f"  {'metric':22s} {'base':>8s} {'sft':>8s}")
    for k in ("lines", "bullets", "step_kw", "before_after_ref"):
        print(f"  {k:22s} {avg(base_struct,k):8.2f} {avg(sft_struct,k):8.2f}")

    # Verdict agreement
    same = sum(1 for r in rows if r["base"] == r["sft"])
    print(f"\nVerdict agreement: {same}/{len(rows)} = {same/len(rows)*100:.1f}%")

    # Sample 3 of each: agree, disagree
    agree = [r for r in rows if r["base"] == r["sft"]]
    disagree = [r for r in rows if r["base"] != r["sft"]]
    print(f"\nDisagreements: {len(disagree)}/{len(rows)}")
    for r in disagree[:5]:
        print(f"\n  --- sid={r['sid']} cat={r['cat'][:40]} label={r.get('label')} base={r['base']} sft={r['sft']}")
        print(f"  BASE ({len(r['base_raw'])}c): {r['base_raw'][:300]!r}")
        print(f"  SFT  ({len(r['sft_raw'])}c): {r['sft_raw'][:300]!r}")

    # Random 2 agree samples
    print("\n--- agree samples ---")
    for r in agree[:2]:
        print(f"\n  sid={r['sid']} verdict={r['base']} (label={r.get('label')})")
        print(f"  BASE ({len(r['base_raw'])}c): {r['base_raw'][:300]!r}")
        print(f"  SFT  ({len(r['sft_raw'])}c): {r['sft_raw'][:300]!r}")


if __name__ == "__main__":
    main()
