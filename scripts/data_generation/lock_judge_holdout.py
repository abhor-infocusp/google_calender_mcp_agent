#!/usr/bin/env python3
"""Lock the 30% stratified holdout from the 285-traj manual oracle.

The holdout sids are NEVER touched during prompt tuning. They're scored only
at the final ship gate. This file is the single source of truth for what's
held out.

Usage:
    PYTHONPATH=src /home/abhor/miniconda3/envs/agentic/bin/python \
        scripts/data_generation/lock_judge_holdout.py
"""
from __future__ import annotations
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INPUT_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_review_input.jsonl"
TRUTH_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_verdicts.jsonl"
OUT = REPO / "data/judge/v2_20260502/holdout_sids.json"

SEED = 20260502  # date-based, never change once committed
HOLDOUT_FRAC = 0.30


def main() -> int:
    inputs = [json.loads(l) for l in open(INPUT_JSONL)]
    truth = [json.loads(l) for l in open(TRUTH_JSONL)]
    # Stratified by category. Sort sids within each cat for determinism.
    by_cat: dict[str, list[str]] = defaultdict(list)
    cat_of: dict[str, str] = {}
    gt_of: dict[str, str] = {}
    for i, rec in enumerate(inputs):
        sid = rec["sid"]
        cat = rec["cat"]
        verdict = (truth[i].get("verdict") if i < len(truth) else None) or rec.get("gt")
        if verdict not in ("Correct", "Incorrect"):
            continue
        by_cat[cat].append(sid)
        cat_of[sid] = cat
        gt_of[sid] = verdict

    rng = random.Random(SEED)
    holdout: list[str] = []
    train: list[str] = []
    per_cat = []
    for cat, sids in sorted(by_cat.items()):
        sids_sorted = sorted(sids)  # determinism — input file order may vary
        rng.shuffle(sids_sorted)
        n_holdout = round(len(sids_sorted) * HOLDOUT_FRAC)
        h = sids_sorted[:n_holdout]
        t = sids_sorted[n_holdout:]
        # also keep verdict balance roughly representative — quick check
        h_correct = sum(1 for s in h if gt_of[s] == "Correct")
        t_correct = sum(1 for s in t if gt_of[s] == "Correct")
        holdout.extend(h)
        train.extend(t)
        per_cat.append({
            "cat": cat,
            "n_total": len(sids_sorted),
            "n_holdout": len(h),
            "n_train": len(t),
            "holdout_correct_frac": round(h_correct / max(len(h), 1), 3),
            "train_correct_frac": round(t_correct / max(len(t), 1), 3),
        })

    payload = {
        "_doc": "30% stratified holdout sids from the 285-traj manual oracle. "
                "NEVER tune against these. Used only at final ship gate.",
        "seed": SEED,
        "holdout_frac": HOLDOUT_FRAC,
        "created_at": datetime.now().isoformat(),
        "n_holdout": len(holdout),
        "n_train": len(train),
        "per_category": per_cat,
        "holdout_sids": sorted(holdout),
        "train_sids":   sorted(train),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"holdout: {len(holdout)} sids  →  train: {len(train)} sids  "
          f"(total {len(holdout)+len(train)})")
    print(f"\n{'cat':<55} {'n':>4} {'hold':>5} {'train':>6} {'h.C%':>6} {'t.C%':>6}")
    for r in per_cat:
        print(f"{r['cat'][:55]:<55} {r['n_total']:>4} {r['n_holdout']:>5} "
              f"{r['n_train']:>6} {100*r['holdout_correct_frac']:>5.1f}% "
              f"{100*r['train_correct_frac']:>5.1f}%")
    print(f"\nwritten → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
