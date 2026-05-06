#!/usr/bin/env python3
"""Prepare a JSONL pool of records to label.

Used by:
  --mode calibration   50 sids from the 285 oracle, stratified by category,
                       drawn from the train half (NOT holdout). Used to
                       validate the labeling agent against manual gold.
  --mode phase1        150 fresh records: 50 inter-judge disagreements +
                       30 capped + 40 uncovered + 30 diversity. (See plan.)

Output schema (per line, both modes):
{
  "label_id": "<sid>__<rollout_hash>",        # primary key
  "sid": "...", "rollout_hash": "...",
  "cat": "...",
  "query": "...", "final": "...", "expected": "...",
  "before": "...", "after": "...",
  "selection_bucket": "calibration"|"disagreement"|"capped"|"uncovered"|"diversity",
  "manual_gt": "Correct"|"Incorrect"|null    # only filled for calibration mode
}
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INPUT_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_review_input.jsonl"
TRUTH_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_verdicts.jsonl"
HOLDOUT_JSON = REPO / "data/judge/v2_20260502/holdout_sids.json"
LIVE_CALLS = REPO / "runs/judge_service_20260501/calls.jsonl"
DISAGREE_JSONL = REPO / "runs/judge_inter_agreement_20260502/results.jsonl"

CALIBRATION_OUT = REPO / "data/judge/v2_20260502/calibration_pool.jsonl"
PHASE1_OUT      = REPO / "data/judge/v2_20260502/labels_to_collect.jsonl"

SEED = 20260502


def rollout_hash(rec: dict) -> str:
    blob = (rec.get("final", "") or "") + (rec.get("before", "") or "") + (rec.get("after", "") or "")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def load_oracle_with_truth() -> list[dict]:
    inputs = [json.loads(l) for l in open(INPUT_JSONL)]
    truth = [json.loads(l) for l in open(TRUTH_JSONL)]
    out = []
    for i, rec in enumerate(inputs):
        rec["manual_gt"] = (truth[i].get("verdict") if i < len(truth) else None) or rec.get("gt")
        if rec["manual_gt"] in ("Correct", "Incorrect"):
            out.append(rec)
    return out


def cmd_calibration(out_path: Path, n: int = 50) -> None:
    holdout_sids = set(json.load(open(HOLDOUT_JSON))["holdout_sids"])
    recs = [r for r in load_oracle_with_truth() if r["sid"] not in holdout_sids]

    # Stratified — same count per category where possible
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_cat[r["cat"]].append(r)

    rng = random.Random(SEED + 1)
    cats = sorted(by_cat)
    per_cat = max(1, n // len(cats))
    picked: list[dict] = []
    for c in cats:
        sids = sorted(by_cat[c], key=lambda r: r["sid"])
        rng.shuffle(sids)
        picked.extend(sids[:per_cat])
    # if under quota, fill with random extras
    if len(picked) < n:
        remainder = [r for c in cats for r in sorted(by_cat[c], key=lambda r: r["sid"]) if r not in picked]
        rng.shuffle(remainder)
        picked.extend(remainder[: n - len(picked)])
    picked = picked[:n]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in picked:
            row = {
                "label_id": f"{r['sid']}__{rollout_hash(r)}",
                "sid": r["sid"], "rollout_hash": rollout_hash(r),
                "cat": r["cat"],
                "query": r["query"], "final": r.get("final", ""),
                "expected": r.get("expected", ""),
                "before": r["before"], "after": r["after"],
                "selection_bucket": "calibration",
                "manual_gt": r["manual_gt"],
            }
            f.write(json.dumps(row) + "\n")
    print(f"calibration pool: {len(picked)} records → {out_path}")
    cnt = defaultdict(int)
    gt_cnt = defaultdict(int)
    for r in picked:
        cnt[r["cat"]] += 1
        gt_cnt[r["manual_gt"]] += 1
    for c, k in sorted(cnt.items()):
        print(f"  {c[:50]:<52} {k}")
    print(f"  gt: {dict(gt_cnt)}")


def cmd_phase1(out_path: Path) -> None:
    holdout_sids = set(json.load(open(HOLDOUT_JSON))["holdout_sids"])
    rng = random.Random(SEED + 2)
    picked: list[dict] = []

    # Bucket A: 50 inter-judge disagreement cases.
    # 26 actual disagreements from runs/judge_inter_agreement_20260502/. For
    # another 24 we use high-latency live cases as a proxy for "uncertain"
    # (no gemini verdicts on the broader pool to identify true disagreements).
    a_all = [json.loads(l) for l in open(DISAGREE_JSONL)]
    a_existing = [r for r in a_all if r.get("local") != r.get("gemini")]
    a_seen: set[str] = {r["sid"] for r in a_existing}

    # Build sid -> latest live call mapping
    live_by_sid: dict[str, dict] = {}
    capped_by_sid: dict[str, list[dict]] = defaultdict(list)
    for ln in open(LIVE_CALLS):
        c = json.loads(ln)
        sid = c.get("scenario_id")
        if not sid: continue
        live_by_sid[sid] = c
        if c.get("usage", {}).get("completion_tokens", 0) >= 1024:
            capped_by_sid[sid].append(c)

    def to_label_row(c: dict, bucket: str) -> dict:
        return {
            "label_id": f"{c.get('scenario_id')}__{rollout_hash(c)}",
            "sid": c.get("scenario_id"),
            "rollout_hash": rollout_hash(c),
            "cat": c.get("cat"),
            "query": c.get("query", ""),
            "final": c.get("final", ""),
            "expected": c.get("expected", ""),
            "before": c.get("before", ""),
            "after": c.get("after", ""),
            "selection_bucket": bucket,
            "manual_gt": None,
        }

    # Bucket A: 26 known disagreements + 24 high-latency live cases as "uncertain"
    # placeholders. (Fresh disagreements after Phase 2 will be re-collected.)
    for sid in a_seen:
        if sid in live_by_sid and sid not in holdout_sids:
            picked.append(to_label_row(live_by_sid[sid], "disagreement"))
    # Top up to 50
    extra_a = sorted(
        [c for s, c in live_by_sid.items() if s not in a_seen and s not in holdout_sids],
        key=lambda c: -c.get("latency_ms", 0),
    )[: 50 - len(picked)]
    for c in extra_a:
        picked.append(to_label_row(c, "disagreement"))

    # Bucket B: 30 capped-thinking cases, weak cats first
    weak_cats = {
        "Vague & Contextual (Reasoning Required)",
        "Complex Logic & Conflict (Advanced)",
        "Human Chaos (Edge Cases/Fragments)",
        "Relative Time References (today, tomorrow, yesterday, this week)",
    }
    capped_pool = [c for sid, lst in capped_by_sid.items() for c in lst
                   if sid not in holdout_sids
                   and f"{sid}__{rollout_hash(c)}" not in {p['label_id'] for p in picked}]
    weak_capped = [c for c in capped_pool if c.get("cat") in weak_cats]
    other_capped = [c for c in capped_pool if c.get("cat") not in weak_cats]
    rng.shuffle(weak_capped); rng.shuffle(other_capped)
    bucket_b = weak_capped[:24] + other_capped[:6]
    for c in bucket_b:
        picked.append(to_label_row(c, "capped"))

    # Bucket C: 40 oracle-uncovered sids, weak cats only
    oracle_sids = {json.loads(l)["sid"] for l in open(INPUT_JSONL)}
    uncovered_sids = [s for s in live_by_sid if s not in oracle_sids and s not in holdout_sids]
    uncovered_weak = [s for s in uncovered_sids if live_by_sid[s].get("cat") in weak_cats]
    by_cat_pool: dict[str, list[str]] = defaultdict(list)
    for s in uncovered_weak:
        by_cat_pool[live_by_sid[s]["cat"]].append(s)
    bucket_c: list[str] = []
    for c, sids in by_cat_pool.items():
        rng.shuffle(sids)
        bucket_c.extend(sids[:10])
    for s in bucket_c[:40]:
        c = live_by_sid[s]
        lid = f"{s}__{rollout_hash(c)}"
        if lid not in {p["label_id"] for p in picked}:
            picked.append(to_label_row(c, "uncovered"))

    # Bucket D: 30 random fresh cases, ≥3 per cat
    diversity_pool = [s for s in live_by_sid if s not in holdout_sids
                      and f"{s}__{rollout_hash(live_by_sid[s])}" not in {p['label_id'] for p in picked}]
    by_cat_div: dict[str, list[str]] = defaultdict(list)
    for s in diversity_pool:
        by_cat_div[live_by_sid[s]["cat"]].append(s)
    bucket_d_sids: list[str] = []
    for c, sids in by_cat_div.items():
        rng.shuffle(sids)
        bucket_d_sids.extend(sids[:5])
    for s in bucket_d_sids[:30]:
        picked.append(to_label_row(live_by_sid[s], "diversity"))

    # Final dedup just in case
    seen_lid = set(); final = []
    for p in picked:
        if p["label_id"] not in seen_lid:
            seen_lid.add(p["label_id"]); final.append(p)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in final:
            f.write(json.dumps(p) + "\n")

    print(f"phase 1 pool: {len(final)} records → {out_path}")
    by_b = defaultdict(int); by_c = defaultdict(int)
    for p in final:
        by_b[p["selection_bucket"]] += 1
        by_c[p["cat"]] += 1
    print(f"  buckets: {dict(by_b)}")
    print(f"  by cat:  {dict(by_c)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calibration", "phase1"], required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.mode == "calibration":
        cmd_calibration(args.out or CALIBRATION_OUT, n=50)
    else:
        cmd_phase1(args.out or PHASE1_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
