#!/usr/bin/env python3
"""Build judge SFT data from existing Gemini-labelled eval JSONs.

Source: runs/**/eval/checkpoint-*.json — each row has
  query, expected, final_output, before (text), after (text),
  verdict ('Correct'/'Incorrect'), judge_reasoning, cal, qi.

Pipeline (per local_judge.md §1.1):
  1. Glob and iterate any top-level section with 'results'.
  2. Dedupe by sha1(query + '|' + final_output).
  3. Split 95/5 by sha1(cal|qi) so all instances of a scenario stay on one side.
  4. Train side only: duplicate Incorrect rows x2 for class balance.
  5. Emit judge_data/{train,val}.jsonl with messages = [system, user, assistant].

The user prompt is byte-identical to rl_train.py:425-441 so the trained judge
sees the same shape RL inference will send.

Asserts no scenario leaks between train and val.
"""
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter

# Make calendar_agent importable (script may be run without PYTHONPATH set).
# Repo root = parent of scripts/training/judge/judge_data_prep.py = up 4 levels.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT  # noqa: E402
from calendar_agent.paths import JUDGE_DATA_DIR, JUDGE_TRAIN_JSONL, JUDGE_VAL_JSONL  # noqa: E402

EVAL_GLOB = os.path.join(_REPO_ROOT, "runs", "**", "eval", "checkpoint-*.json")
OUT_DIR = str(JUDGE_DATA_DIR)
TRAIN_PATH = str(JUDGE_TRAIN_JSONL)
VAL_PATH = str(JUDGE_VAL_JSONL)

VAL_PCT = 5  # 5% val, 95% train
SEED = 42


def build_user_prompt(query: str, final_output: str, expected: str,
                      before_text: str, after_text: str) -> str:
    """Byte-identical to rl_train.py:425-441 (which is also evaluation.py:62-75)."""
    return f"""\
Query: {query}

Response: {final_output if final_output else '(no response)'}

Expected: {expected if expected else '(not specified)'}

Before:
{before_text}

After:
{after_text}

Was the task completed correctly? End with one word: Correct or Incorrect."""


def scenario_key(row: dict) -> str:
    return f"cal_{row.get('cal')}_q_{row.get('qi')}"


def dedupe_key(row: dict) -> str:
    s = f"{row.get('query', '')}|{row.get('final_output', '')}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def is_val(scenario_id: str) -> bool:
    h = int(hashlib.sha1(scenario_id.encode("utf-8")).hexdigest(), 16)
    return (h % 100) < VAL_PCT


def to_jsonl_record(row: dict) -> dict:
    user_prompt = build_user_prompt(
        query=row["query"],
        final_output=row.get("final_output", "") or "",
        expected=row.get("expected", "") or "",
        before_text=row.get("before", "") or "",
        after_text=row.get("after", "") or "",
    )
    return {
        "messages": [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": row["judge_reasoning"]},
        ]
    }


def main() -> None:
    random.seed(SEED)
    files = sorted(glob.glob(EVAL_GLOB, recursive=True))
    print(f"Eval JSONs matched: {len(files)}")
    if not files:
        sys.exit("No eval JSONs found — check EVAL_GLOB")

    # 1. Collect rows from any top-level section that has 'results'.
    raw_rows = []
    for f in files:
        d = json.load(open(f))
        for section_key, section in d.items():
            if not isinstance(section, dict) or "results" not in section:
                continue
            for r in section["results"]:
                # Skip rows missing the fields we need.
                if not r.get("query") or r.get("verdict") not in ("Correct", "Incorrect"):
                    continue
                if not r.get("judge_reasoning"):
                    continue
                raw_rows.append(r)
    print(f"Total raw rows: {len(raw_rows)}")

    # 2. Dedupe by (query, final_output). Keep first occurrence.
    seen = set()
    deduped = []
    for r in raw_rows:
        k = dedupe_key(r)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    print(f"After dedupe: {len(deduped)}")

    verdict_counts = Counter(r["verdict"] for r in deduped)
    print(f"  verdicts: {dict(verdict_counts)}")

    # 3. Split by scenario hash.
    train_rows, val_rows = [], []
    train_scenarios, val_scenarios = set(), set()
    for r in deduped:
        sid = scenario_key(r)
        if is_val(sid):
            val_rows.append(r)
            val_scenarios.add(sid)
        else:
            train_rows.append(r)
            train_scenarios.add(sid)

    # Assert no scenario leaks across train/val.
    leak = train_scenarios & val_scenarios
    assert not leak, f"Scenario leak across train/val: {sorted(leak)[:5]}..."
    print(f"Split: train={len(train_rows)} ({len(train_scenarios)} scenarios), "
          f"val={len(val_rows)} ({len(val_scenarios)} scenarios)")

    # 4. Train side: duplicate Incorrect rows ×2 (each appears 2x total).
    balanced_train = []
    for r in train_rows:
        balanced_train.append(r)
        if r["verdict"] == "Incorrect":
            balanced_train.append(r)  # extra copy
    random.shuffle(balanced_train)
    print(f"Train after Incorrect ×2 oversample: {len(balanced_train)} "
          f"({Counter(r['verdict'] for r in balanced_train)})")
    print(f"Val (no oversample): {len(val_rows)} "
          f"({Counter(r['verdict'] for r in val_rows)})")

    # 5. Emit JSONL.
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(TRAIN_PATH, "w") as f:
        for r in balanced_train:
            f.write(json.dumps(to_jsonl_record(r)) + "\n")
    with open(VAL_PATH, "w") as f:
        for r in val_rows:
            f.write(json.dumps(to_jsonl_record(r)) + "\n")

    print(f"\nWrote {TRAIN_PATH}")
    print(f"Wrote {VAL_PATH}")


if __name__ == "__main__":
    main()
