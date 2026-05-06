#!/usr/bin/env python3
"""Assemble data/judge/v2_20260502/{train,eval,disagreements,metadata}.jsonl
from gold + (filtered) silver sources.

Sources, in trust order:
  - oracle           : 285 manual_verdicts.jsonl  (gold)
  - manual_v2_agent  : 139 Claude-agent labels from Phase 1   (gold)
  - adjudicated      : 85 Claude-agent labels from Phase 3 disagreements (gold)
  - two_way_agree    : agreements from Phase-3 700-call live sweep (silver)

Silver filters (per plan):
  - Drop holdout sids from train.
  - Drop silver records in cats failing P(both-wrong) ≤ 5% gate.
    From Phase 3: Complex (14.3%) and Vague (14.3%) FAIL → silver dropped.
  - Cap silver per cat at 2× gold count.
  - Match natural class ratio (target 73/27 Correct, ±5%).

Outputs:
  data/judge/v2_20260502/train.jsonl
  data/judge/v2_20260502/eval.jsonl
  data/judge/v2_20260502/disagreements.jsonl
  data/judge/v2_20260502/metadata.json
"""
from __future__ import annotations
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORACLE_INPUT = REPO / "runs/judge_baseline_20260430/eval/manual_review_input.jsonl"
ORACLE_TRUTH = REPO / "runs/judge_baseline_20260430/eval/manual_verdicts.jsonl"
NEW_LABELS = REPO / "data/judge/v2_20260502/manual_v2_labels.jsonl"
ADJUDICATED = REPO / "data/judge/v2_20260502/disagreement_adjudicated.jsonl"
LABELS_TO_COLLECT = REPO / "data/judge/v2_20260502/labels_to_collect.jsonl"
ADJUDICATE_POOL = REPO / "data/judge/v2_20260502/disagreement_adjudicate_pool.jsonl"
LIVE_DETAIL = REPO / "data/judge/v2_20260502/live_agreement_detail.jsonl"
HOLDOUT_JSON = REPO / "data/judge/v2_20260502/holdout_sids.json"

OUT = REPO / "data/judge/v2_20260502"

# Phase-3 P(both wrong | agree) — fail gate: ≤ 5%.
SILVER_BLOCKED_CATS = {
    "Complex Logic & Conflict (Advanced)",          # 14.29% (1/7)
    "Vague & Contextual (Reasoning Required)",      # 14.29% (1/7)
}

SEED = 20260502
TARGET_CORRECT_RATIO = 0.73   # natural live rate
RATIO_TOLERANCE = 0.05
SILVER_CAP_MULT = 2.0


def rollout_hash(rec: dict) -> str:
    blob = (rec.get("final", "") or "") + (rec.get("before", "") or "") + (rec.get("after", "") or "")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def make_record(*, sid: str, rh: str, cat: str, query: str, final: str, expected: str,
                before: str, after: str, label: str, label_source: str,
                label_confidence: str = "high",
                qwen_v2: str | None = None, gemini_v2: str | None = None,
                claude_agent: str | None = None) -> dict:
    return {
        "sid": sid, "rollout_hash": rh, "cat": cat,
        "query": query, "final": final, "expected": expected,
        "before": before, "after": after,
        "label": label,
        "label_source": label_source,
        "label_confidence": label_confidence,
        "judge_qwen_v2": qwen_v2,
        "judge_gemini_v2": gemini_v2,
        "judge_claude_agent": claude_agent,
        "prompt_version": {"qwen_v2": "router-qwen-v2-20260502",
                           "gemini_v2": "router-gemini-v2-20260502"},
    }


def load_oracle_gold() -> list[dict]:
    """285 records with manual gold verdicts."""
    inputs = [json.loads(l) for l in open(ORACLE_INPUT)]
    truth = [json.loads(l) for l in open(ORACLE_TRUTH)]
    out = []
    for i, rec in enumerate(inputs):
        gt = (truth[i].get("verdict") if i < len(truth) else None) or rec.get("gt")
        if gt not in ("Correct", "Incorrect"):
            continue
        rh = rollout_hash(rec)
        out.append(make_record(
            sid=rec["sid"], rh=rh, cat=rec["cat"],
            query=rec["query"], final=rec.get("final", ""),
            expected=rec.get("expected", ""),
            before=rec["before"], after=rec["after"],
            label=gt, label_source="oracle", label_confidence="high",
        ))
    return out


def load_manual_v2_gold() -> list[dict]:
    """139 records from Phase 1 agent labeling."""
    if not NEW_LABELS.exists() or not LABELS_TO_COLLECT.exists():
        return []
    pool = {json.loads(l)["label_id"]: json.loads(l) for l in open(LABELS_TO_COLLECT)}
    out = []
    for line in open(NEW_LABELS):
        lab = json.loads(line)
        lid = lab["label_id"]
        if lid not in pool:
            continue
        r = pool[lid]
        out.append(make_record(
            sid=r["sid"], rh=r["rollout_hash"], cat=r["cat"],
            query=r["query"], final=r.get("final", ""),
            expected=r.get("expected", ""),
            before=r["before"], after=r["after"],
            label=lab["verdict"], label_source="manual_v2_agent",
            label_confidence=lab.get("confidence", "medium"),
            claude_agent=lab["verdict"],
        ))
    return out


def load_adjudicated_gold() -> list[dict]:
    """85 records from Phase 3 disagreement adjudication."""
    if not ADJUDICATED.exists() or not ADJUDICATE_POOL.exists():
        return []
    pool = {json.loads(l)["label_id"]: json.loads(l) for l in open(ADJUDICATE_POOL)}
    out = []
    for line in open(ADJUDICATED):
        lab = json.loads(line)
        lid = lab["label_id"]
        if lid not in pool:
            continue
        r = pool[lid]
        out.append(make_record(
            sid=r["sid"], rh=r["rollout_hash"], cat=r["cat"],
            query=r["query"], final=r["final"], expected=r["expected"],
            before=r["before"], after=r["after"],
            label=lab["verdict"], label_source="adjudicated",
            label_confidence=lab.get("confidence", "medium"),
            qwen_v2=r.get("qwen_v2"), gemini_v2=r.get("gemini_v2"),
            claude_agent=lab["verdict"],
        ))
    return out


def load_silver() -> list[dict]:
    """Live agreements from Phase 3 700-call sweep."""
    if not LIVE_DETAIL.exists():
        return []
    out = []
    for line in open(LIVE_DETAIL):
        r = json.loads(line)
        if r.get("qwen") != r.get("gemini"):
            continue  # disagreements live elsewhere
        if r["cat"] in SILVER_BLOCKED_CATS:
            continue  # gate-failed cats lose silver
        out.append(make_record(
            sid=r["sid"], rh=r["rollout_hash"], cat=r["cat"],
            query=r["query"], final=r["final"], expected=r["expected"],
            before=r["before"], after=r["after"],
            label=r["qwen"], label_source="two_way_agree",
            label_confidence="medium",
            qwen_v2=r["qwen"], gemini_v2=r["gemini"],
        ))
    return out


def dedup_by_key(rows: list[dict]) -> list[dict]:
    seen = set(); out = []
    for r in rows:
        k = (r["sid"], r["rollout_hash"])
        if k in seen: continue
        seen.add(k); out.append(r)
    return out


def split_train_eval(gold: list[dict], holdout_sids: set[str]) -> tuple[list[dict], list[dict]]:
    train = [r for r in gold if r["sid"] not in holdout_sids]
    eval_ = [r for r in gold if r["sid"] in holdout_sids]
    return train, eval_


def cap_silver_per_cat(silver: list[dict], gold_train: list[dict],
                       cap_mult: float, rng: random.Random) -> list[dict]:
    """At most cap_mult × (gold_train count for that cat)."""
    gold_per_cat: dict[str, int] = Counter(r["cat"] for r in gold_train)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in silver:
        by_cat[r["cat"]].append(r)
    out = []
    for cat, rows in by_cat.items():
        cap = int(gold_per_cat.get(cat, 0) * cap_mult)
        if cap <= 0:
            continue  # no gold in this cat → don't trust silver alone
        rng.shuffle(rows)
        out.extend(rows[:cap])
    return out


def class_balance_per_cat(silver: list[dict], target: float, tolerance: float,
                          rng: random.Random) -> list[dict]:
    """Drop excess Correct or Incorrect to bring per-cat ratio into target ± tolerance."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in silver:
        by_cat[r["cat"]].append(r)
    out = []
    for cat, rows in by_cat.items():
        c = [r for r in rows if r["label"] == "Correct"]
        i = [r for r in rows if r["label"] == "Incorrect"]
        n = len(rows)
        if n == 0: continue
        # Want correct_count / n ∈ [target - tol, target + tol]
        target_c = round(target * n)
        if len(c) > target_c + tolerance * n:
            rng.shuffle(c); c = c[:int(target_c + tolerance * n)]
        if len(i) > (1 - target) * n + tolerance * n:
            rng.shuffle(i); i = i[: int((1 - target) * n + tolerance * n)]
        out.extend(c); out.extend(i)
    return out


def main():
    rng = random.Random(SEED)
    holdout_sids = set(json.load(open(HOLDOUT_JSON))["holdout_sids"])

    print("Loading sources...")
    oracle = load_oracle_gold()
    manual_v2 = load_manual_v2_gold()
    adjudicated = load_adjudicated_gold()
    silver = load_silver()
    print(f"  oracle:          {len(oracle)}")
    print(f"  manual_v2_agent: {len(manual_v2)}")
    print(f"  adjudicated:     {len(adjudicated)}")
    print(f"  silver (raw):    {len(silver)}  (after blocked cats)")

    # Combine gold; dedup by (sid, rollout_hash). Priority: oracle > adjudicated > manual_v2.
    # If a record appears in oracle AND elsewhere, oracle wins.
    by_key: dict[tuple[str, str], dict] = {}
    for r in oracle:                  by_key[(r["sid"], r["rollout_hash"])] = r
    for r in adjudicated:
        key = (r["sid"], r["rollout_hash"])
        if key not in by_key: by_key[key] = r
    for r in manual_v2:
        key = (r["sid"], r["rollout_hash"])
        if key not in by_key: by_key[key] = r
    gold_all = list(by_key.values())
    print(f"  gold (deduped):  {len(gold_all)}")

    # Train/eval split by holdout sids
    gold_train, gold_eval = split_train_eval(gold_all, holdout_sids)
    print(f"\nGold split:")
    print(f"  gold_train: {len(gold_train)}")
    print(f"  gold_eval:  {len(gold_eval)}")

    # Silver: dedup, drop holdout sids, drop any sid that already appears in gold_train
    silver = dedup_by_key(silver)
    train_keys = {(r["sid"], r["rollout_hash"]) for r in gold_train}
    eval_sids = {r["sid"] for r in gold_eval}
    silver = [r for r in silver if r["sid"] not in holdout_sids
              and (r["sid"], r["rollout_hash"]) not in train_keys
              and r["sid"] not in eval_sids]
    print(f"  silver after dedup/holdout-drop: {len(silver)}")

    silver = cap_silver_per_cat(silver, gold_train, SILVER_CAP_MULT, rng)
    print(f"  silver after 2× gold cap:       {len(silver)}")

    silver = class_balance_per_cat(silver, TARGET_CORRECT_RATIO, RATIO_TOLERANCE, rng)
    print(f"  silver after class balancing:   {len(silver)}")

    train = gold_train + silver
    eval_ = gold_eval

    # Final integrity checks
    eval_sids2 = {r["sid"] for r in eval_}
    leak = [r for r in train if r["sid"] in eval_sids2]
    assert not leak, f"leakage: {len(leak)} train records share sids with eval"

    # Disagreements file: every adjudicated record gets stored separately too
    disagreements = adjudicated  # already loaded

    # Metadata
    meta = {
        "version": "v2_20260502",
        "created_at": datetime.now().isoformat(),
        "seed": SEED,
        "sources": {
            "oracle":          len([r for r in gold_all if r["label_source"] == "oracle"]),
            "adjudicated":     len([r for r in gold_all if r["label_source"] == "adjudicated"]),
            "manual_v2_agent": len([r for r in gold_all if r["label_source"] == "manual_v2_agent"]),
            "two_way_agree":   len(silver),
        },
        "gold_train": len(gold_train),
        "gold_eval": len(eval_),
        "silver_train": len(silver),
        "train_total": len(train),
        "eval_total": len(eval_),
        "silver_blocked_cats": sorted(SILVER_BLOCKED_CATS),
        "silver_cap_mult": SILVER_CAP_MULT,
        "target_correct_ratio": TARGET_CORRECT_RATIO,
        "ratio_tolerance": RATIO_TOLERANCE,
        "p_both_wrong_per_cat": {  # from Phase 3 spotcheck (n=7 per cat)
            "Complex Logic & Conflict (Advanced)": 0.1429,
            "Human Chaos (Edge Cases/Fragments)": 0.0,
            "Information Retrieval (Querying)": 0.0,
            "Modifier & Correction (Rescheduling/Updates)": 0.0,
            "Relative Time References (today, tomorrow, yesterday, this week)": 0.0,
            "Schedule a Single Event": 0.0,
            "Vague & Contextual (Reasoning Required)": 0.1429,
        },
        "p_both_wrong_overall": 0.0408,
        "phase3_holdout_acc": {
            "qwen_v2_overall_n": 111,
            "gemini_v2_overall_n": 111,
        },
    }
    # Per-cat / per-source breakdown
    breakdown: dict = defaultdict(lambda: defaultdict(int))
    for r in train:
        breakdown[r["cat"]][f"train_{r['label_source']}"] += 1
        breakdown[r["cat"]][f"train_{r['label']}"] += 1
    for r in eval_:
        breakdown[r["cat"]][f"eval_{r['label_source']}"] += 1
        breakdown[r["cat"]][f"eval_{r['label']}"] += 1
    meta["per_cat_breakdown"] = {k: dict(v) for k, v in breakdown.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    rng.shuffle(train)
    with open(OUT / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "eval.jsonl", "w") as f:
        for r in eval_:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "disagreements.jsonl", "w") as f:
        for r in disagreements:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Summary print
    print(f"\n=== FINAL DATASET v2_20260502 ===")
    print(f"  train.jsonl:         {len(train)} records")
    print(f"    gold:              {len(gold_train)}")
    print(f"    silver:            {len(silver)}")
    print(f"  eval.jsonl:          {len(eval_)} records (gold only)")
    print(f"  disagreements.jsonl: {len(disagreements)} records")

    # Per-cat split
    print(f"\nTrain per category:")
    train_cat = Counter(r["cat"] for r in train)
    train_C = Counter((r["cat"], r["label"]) for r in train)
    for c, n in sorted(train_cat.items(), key=lambda kv: -kv[1]):
        cc = train_C[(c, "Correct")]; ii = train_C[(c, "Incorrect")]
        print(f"  {c[:50]:<52} n={n:<4} C={cc:<3}({100*cc/n:.0f}%) I={ii:<3}({100*ii/n:.0f}%)")
    print(f"\nEval per category:")
    eval_cat = Counter(r["cat"] for r in eval_)
    eval_C = Counter((r["cat"], r["label"]) for r in eval_)
    for c, n in sorted(eval_cat.items(), key=lambda kv: -kv[1]):
        cc = eval_C[(c, "Correct")]; ii = eval_C[(c, "Incorrect")]
        print(f"  {c[:50]:<52} n={n:<4} C={cc:<3} I={ii:<3}")

    print(f"\nartifacts → {OUT}")


if __name__ == "__main__":
    main()
