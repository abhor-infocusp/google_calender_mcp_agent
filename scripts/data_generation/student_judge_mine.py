#!/usr/bin/env python3
"""Mine candidate judge-training rows from existing run logs.

Sources (both have before/after already materialized):
  1. runs/judge_service_20260501/calls.jsonl  — live RL judge calls (5,396 rows, 537 sids)
  2. runs/**/eval/checkpoint-*.json:rl.results — SFT eval on rl_data (3,774 rows, 280 sids)

Pipeline:
  - Normalize to a common schema
  - Filter holdout sids (data/judge/v2_20260502/holdout_sids.json)
  - De-dup by sha256(final|before|after)[:12]
  - Per-sid cap (default 20)
  - Stratify by source step bucket where available

Output:  data/judge/v2_20260502/student_candidates.jsonl
Schema:
  {sid, rollout_hash, cat, query, final, expected, before, after,
   src, src_step, prior_verdict, prior_judge}
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, random
from collections import defaultdict, Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOLDOUT_PATH = os.path.join(REPO, "data/judge/v2_20260502/holdout_sids.json")
DEFAULT_OUT = os.path.join(REPO, "data/judge/v2_20260502/student_candidates.jsonl")

CALLS = os.path.join(REPO, "runs/judge_service_20260501/calls.jsonl")
EVAL_GLOB = os.path.join(REPO, "runs/**/eval/checkpoint-*.json")


def rhash(final: str, before: str, after: str) -> str:
    s = f"{final or ''}\x1f{before or ''}\x1f{after or ''}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def load_holdout() -> set[str]:
    with open(HOLDOUT_PATH) as f:
        d = json.load(f)
    return set(d["holdout_sids"])


def iter_calls_jsonl() -> list[dict]:
    rows = []
    with open(CALLS) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "sid": r["scenario_id"],
                "cat": r.get("cat", ""),
                "query": r.get("query", ""),
                "final": r.get("final", ""),
                "expected": r.get("expected", ""),
                "before": r.get("before", ""),
                "after": r.get("after", ""),
                "src": "calls",
                "src_step": None,
                "prior_verdict": r.get("verdict"),
                "prior_judge": r.get("prompt_version"),
            })
    return rows


def iter_eval_jsons() -> list[dict]:
    rows = []
    for fp in glob.glob(EVAL_GLOB, recursive=True):
        ckpt = os.path.basename(fp).replace("checkpoint-", "").replace(".json", "")
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        for r in d.get("rl", {}).get("results", []):
            sid = f"cal_{r.get('cal')}_q_{r.get('qi')}"
            rows.append({
                "sid": sid,
                "cat": r.get("category", ""),
                "query": r.get("query", ""),
                "final": r.get("final_output", "") or "",
                "expected": r.get("expected", "") or "",
                "before": r.get("before", "") or "",
                "after": r.get("after", "") or "",
                "src": "eval",
                "src_step": int(ckpt) if ckpt.isdigit() else None,
                "prior_verdict": r.get("verdict"),
                "prior_judge": "gemini",
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--per-sid-cap", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260504)
    args = ap.parse_args()

    random.seed(args.seed)
    holdout = load_holdout()
    print(f"holdout sids: {len(holdout)}")

    raw = iter_calls_jsonl() + iter_eval_jsons()
    print(f"raw rows: {len(raw)}")

    # Filter holdout + drop empties
    after_holdout = [r for r in raw if r["sid"] not in holdout and r["final"] and r["query"]]
    print(f"after holdout+empty filter: {len(after_holdout)}")

    # Hash + de-dup
    seen = set()
    uniq = []
    for r in after_holdout:
        h = rhash(r["final"], r["before"], r["after"])
        if h in seen:
            continue
        seen.add(h)
        r["rollout_hash"] = h
        uniq.append(r)
    print(f"after de-dup: {len(uniq)}")

    # Per-sid cap (random sample, prefer diversity across step buckets)
    by_sid = defaultdict(list)
    for r in uniq:
        by_sid[r["sid"]].append(r)
    capped = []
    for sid, rs in by_sid.items():
        if len(rs) <= args.per_sid_cap:
            capped.extend(rs)
        else:
            random.shuffle(rs)
            capped.extend(rs[:args.per_sid_cap])
    print(f"after per-sid cap (max {args.per_sid_cap}): {len(capped)}")

    # Stats
    by_cat = Counter(r["cat"] for r in capped)
    by_src = Counter(r["src"] for r in capped)
    by_prior = Counter(r["prior_verdict"] for r in capped)
    print("by cat:", dict(by_cat))
    print("by src:", dict(by_src))
    print("prior verdicts (untrusted):", dict(by_prior))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in capped:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
