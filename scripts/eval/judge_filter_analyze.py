#!/usr/bin/env python3
"""Analyze judge_3way_run.py outputs to decide silver-label filter switch.

Reads:
  runs/judge_filter_validation_20260505/tier1_run1.jsonl
  runs/judge_filter_validation_20260505/tier1_run2.jsonl
  runs/judge_filter_validation_20260505/tier2_disagreements.jsonl
  runs/judge_filter_validation_20260505/pool500_basesft.jsonl

For each tier:
  - per-judge accuracy + Wilson 95% CI
  - per-judge agreement-rate with each other
  - for each candidate filter F = {pair} ∈ {base∧gem, sft∧gem, base∧sft, all3}:
      coverage   = P(filter triggers, i.e. judges agree)
      noise      = P(label wrong | filter triggers)  (Wilson CI)

Run-1 vs run-2 jitter:
  - per-judge accuracy delta
  - per-row flip rate (different verdict between runs)

Pool 500:
  - distribution-shift check: Base vs SFT agreement rate matches tier-1?
  - join with relabel_qwen.jsonl + relabel_gemini.jsonl for base/qwen-v2 vs gem
"""
from __future__ import annotations
import json, math
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/judge_filter_validation_20260505"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def load(p): return [json.loads(l) for l in Path(p).open()]


def per_judge_acc(rows, judge_keys=("base", "sft", "gem")):
    out = {}
    for jk in judge_keys:
        if jk not in rows[0]:
            continue
        ok = sum(1 for r in rows if r[jk] == r.get("label"))
        n = sum(1 for r in rows if r.get("label") is not None and jk in r)
        p, lo, hi = wilson(ok, n)
        out[jk] = {"n": n, "acc": p, "ci": (lo, hi)}
    return out


def filter_stats(rows, j1, j2):
    """Stats for silver filter j1 ∧ j2."""
    n = len(rows)
    triggers = [r for r in rows if r.get(j1) == r.get(j2)]
    coverage = len(triggers) / n if n else 0
    if not triggers:
        return {"coverage": 0, "noise_n": 0}
    # noise = P(label wrong | both agreed); label is the agreed verdict
    wrong = sum(1 for r in triggers if r[j1] != r.get("label"))
    p, lo, hi = wilson(wrong, len(triggers))
    return {
        "coverage": coverage,
        "n_agree": len(triggers),
        "noise": p,
        "noise_ci": (lo, hi),
    }


def jitter(run1, run2, judge_keys=("base", "sft")):
    by_sid_r1 = {(r["sid"], r.get("rollout_hash")): r for r in run1}
    out = {}
    for jk in judge_keys:
        if jk not in run1[0]:
            continue
        same, diff, n = 0, 0, 0
        acc1, acc2 = 0, 0
        for r2 in run2:
            key = (r2["sid"], r2.get("rollout_hash"))
            r1 = by_sid_r1.get(key)
            if not r1:
                continue
            n += 1
            if r1[jk] == r2[jk]:
                same += 1
            else:
                diff += 1
            if r1[jk] == r1.get("label"):
                acc1 += 1
            if r2[jk] == r2.get("label"):
                acc2 += 1
        out[jk] = {
            "n_compared": n,
            "flip_rate": diff / n if n else 0,
            "acc_run1": acc1 / n if n else 0,
            "acc_run2": acc2 / n if n else 0,
            "delta": (acc2 - acc1) / n if n else 0,
        }
    return out


def main():
    print("=" * 60)
    print("TIER 1 — eval.jsonl (n=110, gold)")
    print("=" * 60)
    r1 = load(RUN / "tier1_run1.jsonl")
    r2 = load(RUN / "tier1_run2.jsonl")
    for label, rows in [("RUN1", r1), ("RUN2", r2)]:
        print(f"\n  {label} per-judge accuracy:")
        for jk, s in per_judge_acc(rows).items():
            lo, hi = s["ci"]
            print(f"    {jk:5s}  {s['acc']*100:5.1f}% [{lo*100:5.1f}, {hi*100:5.1f}]  n={s['n']}")

    print("\n  RUN1 vs RUN2 jitter:")
    for jk, s in jitter(r1, r2).items():
        print(f"    {jk:5s}  flip={s['flip_rate']*100:.1f}%  acc1={s['acc_run1']*100:.1f}%  acc2={s['acc_run2']*100:.1f}%  Δ={s['delta']*100:+.1f}pp")

    print("\n  RUN1 silver-filter candidates (P(label wrong | agree)):")
    for j1, j2 in [("base", "gem"), ("sft", "gem"), ("base", "sft")]:
        s = filter_stats(r1, j1, j2)
        lo, hi = s["noise_ci"]
        print(f"    {j1}∧{j2:5s}  coverage={s['coverage']*100:5.1f}%  noise={s['noise']*100:5.2f}% [{lo*100:5.2f}, {hi*100:5.2f}]  n_agree={s['n_agree']}")

    print("\n  RUN2 silver-filter candidates:")
    for j1, j2 in [("base", "gem"), ("sft", "gem"), ("base", "sft")]:
        s = filter_stats(r2, j1, j2)
        lo, hi = s["noise_ci"]
        print(f"    {j1}∧{j2:5s}  coverage={s['coverage']*100:5.1f}%  noise={s['noise']*100:5.2f}% [{lo*100:5.2f}, {hi*100:5.2f}]  n_agree={s['n_agree']}")

    # 3-way intersection
    print("\n  RUN1 3-way agreement (base ∧ sft ∧ gem):")
    n3 = sum(1 for r in r1 if r["base"] == r["sft"] == r["gem"])
    w3 = sum(1 for r in r1 if r["base"] == r["sft"] == r["gem"] != r.get("label"))
    p, lo, hi = wilson(w3, n3)
    print(f"    coverage={n3/len(r1)*100:.1f}%  noise={p*100:.2f}% [{lo*100:.2f}, {hi*100:.2f}]  n_agree={n3}")

    print("\n" + "=" * 60)
    print("TIER 2 — disagreements.jsonl (n=85, hard cases, gold)")
    print("=" * 60)
    t2 = load(RUN / "tier2_disagreements.jsonl")
    print("\n  per-judge accuracy:")
    for jk, s in per_judge_acc(t2).items():
        lo, hi = s["ci"]
        print(f"    {jk:5s}  {s['acc']*100:5.1f}% [{lo*100:5.1f}, {hi*100:5.1f}]  n={s['n']}")
    print("\n  silver-filter candidates:")
    for j1, j2 in [("base", "gem"), ("sft", "gem"), ("base", "sft")]:
        s = filter_stats(t2, j1, j2)
        lo, hi = s["noise_ci"]
        print(f"    {j1}∧{j2:5s}  coverage={s['coverage']*100:5.1f}%  noise={s['noise']*100:5.2f}% [{lo*100:5.2f}, {hi*100:5.2f}]  n_agree={s['n_agree']}")

    print("\n" + "=" * 60)
    print("POOL 500 — distribution-shift check (no gold labels)")
    print("=" * 60)
    p500_path = RUN / "pool500_basesft.jsonl"
    if p500_path.exists():
        p500 = load(p500_path)
        # join with relabel files
        rq = {f"{r['rollout_hash']}": r for r in load(REPO / "data/judge/v2_20260502/relabel_qwen.jsonl")}
        rg = {f"{r['rollout_hash']}": r for r in load(REPO / "data/judge/v2_20260502/relabel_gemini.jsonl")}
        joined = []
        for r in p500:
            h = r.get("rollout_hash")
            qv = rq.get(h, {}).get("qwen_v2_verdict")
            gv = rg.get(h, {}).get("gemini_v2_verdict")
            if qv and gv:
                joined.append({**r, "qwen_v2": qv, "gem": gv})
        print(f"\n  joined rows (have base+sft+qwen-v2+gem): {len(joined)}/{len(p500)}")
        # pairwise agreement rates
        print("\n  pairwise agreement rates (no gold, just consistency):")
        for j1, j2 in [("base", "qwen_v2"), ("base", "sft"), ("base", "gem"),
                       ("sft", "gem"), ("sft", "qwen_v2"), ("qwen_v2", "gem")]:
            agree = sum(1 for r in joined if r[j1] == r[j2])
            print(f"    {j1:8s} vs {j2:8s}  {agree/len(joined)*100:5.1f}%")
        # noise can't be computed without gold; instead, "filter F triggered AND
        # third judge disagrees" is a noise-proxy
        print("\n  filter triggered AND remaining judge disagrees (noise-proxy %):")
        for j1, j2, jk in [("base", "gem", "sft"),
                           ("sft", "gem", "base"),
                           ("base", "sft", "gem"),
                           ("qwen_v2", "gem", "sft")]:
            triggers = [r for r in joined if r[j1] == r[j2]]
            if not triggers:
                continue
            disagree = sum(1 for r in triggers if r[jk] != r[j1])
            p, lo, hi = wilson(disagree, len(triggers))
            print(f"    {j1}∧{j2:8s} (3rd={jk}) coverage={len(triggers)/len(joined)*100:5.1f}%  3rd-disagrees={p*100:5.2f}% [{lo*100:5.2f}, {hi*100:5.2f}]  n={len(triggers)}")


if __name__ == "__main__":
    main()
