#!/usr/bin/env python3
"""Build judge SFT v3 dataset from existing Gemini-labelled corpora.

Combines two reasoning-bearing sources we already have on disk:

  A. runs/**/eval/checkpoint-*.json  (per-ckpt eval results, Gemini-flash labels)
     Schema per row: query, final_output, expected, before, after,
                     verdict, judge_reasoning, cal, qi.

  B. data/judge/v2_20260502/relabel_gemini.jsonl  (Gemini-v2 relabel pass on
     adaptive-RL parquet rollouts).
     Joined by rollout_hash to data/judge/v2_20260502/student_candidates_parquet.jsonl
     for query/final/before/after.

Pipeline:
  1. Collect rows from A and B, normalise to a common schema.
  2. Drop rows whose sid is in v2 holdout (no leak with manual eval set).
  3. Dedupe by sha1(query|final|before|after)[:16].
  4. Split 95/5 by sha1(sid) so all instances of a scenario stay together.
  5. Emit data/judge/v3_20260507/{train,val}.jsonl with messages
     = [system, user, assistant] using the byte-identical user prompt template
     from rl_train.py / evaluation.py / judge_data_prep.py.

No new Gemini calls. Output rows have label_source = "v1_eval_json"
or "v2_relabel" so downstream ablations can split by provenance.
"""
from __future__ import annotations
import glob
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT  # noqa: E402

# ── Paths ──────────────────────────────────────────────────
EVAL_GLOB = str(REPO / "runs" / "**" / "eval" / "checkpoint-*.json")
V2_DIR = REPO / "data/judge/v2_20260502"
V2_RELABEL = V2_DIR / "relabel_gemini.jsonl"
V2_CANDS = V2_DIR / "student_candidates_parquet.jsonl"
HOLDOUT_JSON = V2_DIR / "holdout_sids.json"

OUT_DIR = REPO / "data/judge/v3_20260507"
TRAIN_PATH = OUT_DIR / "train.jsonl"
VAL_PATH = OUT_DIR / "val.jsonl"
META_PATH = OUT_DIR / "metadata.json"

VAL_PCT = 5  # 5% val, 95% train (split by sid)
SEED = 20260507


# ── Helpers ────────────────────────────────────────────────
def build_user_prompt(query: str, final_output: str, expected: str,
                      before_text: str, after_text: str) -> str:
    """Byte-identical to rl_train.py / evaluation.py / judge_data_prep.py."""
    return f"""\
Query: {query}

Response: {final_output if final_output else '(no response)'}

Expected: {expected if expected else '(not specified)'}

Before:
{before_text}

After:
{after_text}

Was the task completed correctly? End with one word: Correct or Incorrect."""


def dedupe_key(query: str, final: str, before: str, after: str) -> str:
    s = f"{query}|{final}|{before}|{after}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def is_val(sid: str) -> bool:
    h = int(hashlib.sha1(sid.encode("utf-8")).hexdigest(), 16)
    return (h % 100) < VAL_PCT


def to_messages(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(
                row["query"], row["final"], row.get("expected", "") or "",
                row["before"], row["after"],
            )},
            {"role": "assistant", "content": row["reasoning"]},
        ]
    }


# ── Sources ────────────────────────────────────────────────
def load_source_a() -> list[dict]:
    """v1 eval-JSON corpus."""
    files = sorted(glob.glob(EVAL_GLOB, recursive=True))
    rows: list[dict] = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for _, section in d.items():
            if not isinstance(section, dict) or "results" not in section:
                continue
            for r in section["results"]:
                if not r.get("query"):
                    continue
                if r.get("verdict") not in ("Correct", "Incorrect"):
                    continue
                if not r.get("judge_reasoning"):
                    continue
                cal, qi = r.get("cal"), r.get("qi")
                if cal is None or qi is None:
                    continue
                rows.append({
                    "sid": f"cal_{cal}_q_{qi}",
                    "cat": r.get("category", "Unknown"),
                    "query": r["query"],
                    "final": r.get("final_output", "") or "",
                    "expected": r.get("expected", "") or "",
                    "before": r.get("before", "") or "",
                    "after": r.get("after", "") or "",
                    "verdict": r["verdict"],
                    "reasoning": r["judge_reasoning"],
                    "label_source": "v1_eval_json",
                })
    print(f"[A] v1 eval-json rows: {len(rows)} from {len(files)} files")
    return rows


def load_source_b() -> list[dict]:
    """v2 Gemini relabel × parquet candidates (join by rollout_hash)."""
    cands: dict[str, dict] = {}
    with V2_CANDS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cands[c["rollout_hash"]] = c

    rows: list[dict] = []
    n_lines = n_join_miss = n_err = 0
    with V2_RELABEL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            r = json.loads(line)
            if r.get("gemini_v2_err"):
                n_err += 1
                continue
            v = r.get("gemini_v2_verdict")
            if v not in ("Correct", "Incorrect"):
                continue
            rh = r.get("rollout_hash")
            cand = cands.get(rh)
            if not cand:
                n_join_miss += 1
                continue
            reasoning = (r.get("gemini_v2_raw") or "").strip()
            if not reasoning:
                continue
            rows.append({
                "sid": cand["sid"],
                "cat": cand.get("cat", "Unknown"),
                "query": cand["query"],
                "final": cand.get("final", "") or "",
                "expected": cand.get("expected", "") or "",
                "before": cand.get("before", "") or "",
                "after": cand.get("after", "") or "",
                "verdict": v,
                "reasoning": reasoning,
                "label_source": "v2_relabel",
            })
    print(f"[B] v2 relabel rows: {len(rows)} "
          f"(scanned {n_lines}, join_miss {n_join_miss}, gemini_err {n_err})")
    return rows


# ── Main ───────────────────────────────────────────────────
def main() -> None:
    holdout_doc = json.load(HOLDOUT_JSON.open())
    holdout = set(holdout_doc["holdout_sids"])
    print(f"holdout sids: {len(holdout)}")

    rows = load_source_a() + load_source_b()
    print(f"combined raw: {len(rows)}")

    # Drop holdout sids
    pre = len(rows)
    rows = [r for r in rows if r["sid"] not in holdout]
    print(f"after holdout drop: {len(rows)} (dropped {pre - len(rows)})")

    # Dedupe by (query, final, before, after)
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        k = dedupe_key(r["query"], r["final"], r["before"], r["after"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    print(f"after dedupe: {len(deduped)}")

    verdicts = Counter(r["verdict"] for r in deduped)
    sources = Counter(r["label_source"] for r in deduped)
    cats = Counter(r["cat"] for r in deduped)
    print(f"  verdicts: {dict(verdicts)}")
    print(f"  sources:  {dict(sources)}")
    print(f"  cats:     {dict(cats)}")

    # Split 95/5 by sid
    train_rows, val_rows = [], []
    train_sids, val_sids = set(), set()
    for r in deduped:
        if is_val(r["sid"]):
            val_rows.append(r)
            val_sids.add(r["sid"])
        else:
            train_rows.append(r)
            train_sids.add(r["sid"])
    leak = train_sids & val_sids
    assert not leak, f"sid leak across train/val: {sorted(leak)[:5]}"
    print(f"split: train={len(train_rows)} ({len(train_sids)} sids), "
          f"val={len(val_rows)} ({len(val_sids)} sids)")
    print(f"  train verdicts: {Counter(r['verdict'] for r in train_rows)}")
    print(f"  val   verdicts: {Counter(r['verdict'] for r in val_rows)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRAIN_PATH.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(to_messages(r), ensure_ascii=False) + "\n")
    with VAL_PATH.open("w") as f:
        for r in val_rows:
            f.write(json.dumps(to_messages(r), ensure_ascii=False) + "\n")

    meta = {
        "built_at": "2026-05-07",
        "sources": {
            "v1_eval_json": sources.get("v1_eval_json", 0),
            "v2_relabel": sources.get("v2_relabel", 0),
        },
        "holdout_sids_excluded": len(holdout),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_sids": len(train_sids),
        "val_sids": len(val_sids),
        "verdict_counts_total": dict(verdicts),
        "category_counts": dict(cats),
        "user_prompt_template": "rl_train.py / evaluation.py byte-identical",
        "seed": SEED,
        "val_pct": VAL_PCT,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {TRAIN_PATH}")
    print(f"Wrote {VAL_PATH}")
    print(f"Wrote {META_PATH}")


if __name__ == "__main__":
    main()
