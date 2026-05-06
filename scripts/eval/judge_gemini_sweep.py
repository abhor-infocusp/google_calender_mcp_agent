#!/usr/bin/env python3
"""Sweep prompt variants × categories against gemini-2.0-flash to build a
Gemini-specific per-category ROUTER_MAP.

Each variant is tried with deterministic generation (temp=0). Output is a
per-variant × per-category accuracy matrix; we pick the best variant per
category as the new ROUTER_MAP_GEMINI.

Usage:
    PYTHONPATH=src /home/abhor/miniconda3/envs/agentic/bin/python \
        scripts/eval/judge_gemini_sweep.py
"""
from __future__ import annotations
import csv
import importlib.util
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import google.auth.transport.requests
import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import GenerationConfig, GenerativeModel

from calendar_agent.judge.prompts import extract_verdict
from calendar_agent.paths import CREDENTIALS_PATH

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "runs/judge_baseline_20260430/eval"
INPUT_JSONL = EVAL_DIR / "manual_review_input.jsonl"
TRUTH_JSONL = EVAL_DIR / "manual_verdicts.jsonl"

MODEL = "gemini-2.0-flash-001"
CONCURRENCY = 16
GEN_CFG = GenerationConfig(temperature=0.0, top_p=1.0, max_output_tokens=2048)

# ── Pull variant builders from the existing prompt-tune script ──────
TUNE_PATH = REPO / "scripts/eval/judge_prompt_tune.py"
spec = importlib.util.spec_from_file_location("judge_prompt_tune", TUNE_PATH)
tune = importlib.util.module_from_spec(spec)
sys.modules["judge_prompt_tune"] = tune
spec.loader.exec_module(tune)

LENIENT_ADDENDUM = """\

CRITICAL: Default toward Correct on borderline cases. Only call Incorrect
when at least one of these is clearly true:
1. Agent claimed success but calendar didn't change (and a state change
   was clearly required).
2. Agent modified, deleted, or rescheduled a DIFFERENT event than the one
   asked about.
3. Agent fabricated an EVENT (title + day + time window) not in BEFORE or
   AFTER. Wrong attendees / RSVP / wording is NOT fabrication.
4. Wrong duration when the user explicitly stated the duration.
5. Response denies an event that exists in BEFORE on the day asked about.
6. Tool calls broken/garbled, or agent flipped to an unrelated topic.

If none of (1)-(6) apply, output Correct — even if you can find smaller
imperfections.
"""


def _wrap_lenient(builder):
    def b(rec):
        sys_p, user_p, opts = builder(rec)
        return sys_p + LENIENT_ADDENDUM, user_p, opts
    return b


# Variants to sweep. Plain + lenient version for the heavy ones.
VARIANTS = {
    "baseline":              tune.build_baseline,
    "cot_checklist_v2":      tune.build_cot_checklist_v2,
    "per_category_v3":       tune.build_per_category_v3,
    "fewshot":               tune.build_fewshot,
    "fewshot_v3":            tune.build_fewshot_v3,
    "fewshot_v4_dayfocus":   tune.build_fewshot_v4_dayfocus,
    "fewshot+L":             _wrap_lenient(tune.build_fewshot),
    "fewshot_v3+L":          _wrap_lenient(tune.build_fewshot_v3),
    "fewshot_v4_dayfocus+L": _wrap_lenient(tune.build_fewshot_v4_dayfocus),
    "per_category_v3+L":     _wrap_lenient(tune.build_per_category_v3),
}


def init_vertex() -> None:
    with open(CREDENTIALS_PATH) as f:
        cd = json.load(f)
    creds = OAuth2Credentials(
        token=None, refresh_token=cd["refresh_token"],
        client_id=cd["client_id"], client_secret=cd["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(google.auth.transport.requests.Request())
    vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)


def load_oracle() -> list[dict]:
    inputs = [json.loads(l) for l in open(INPUT_JSONL)]
    truth = [json.loads(l) for l in open(TRUTH_JSONL)]
    for i, rec in enumerate(inputs):
        if i < len(truth):
            rec["gt"] = truth[i].get("verdict") or rec.get("gt")
    return [r for r in inputs if r.get("gt") in ("Correct", "Incorrect")]


def judge_one(rec, builder):
    sys_p, user_p, _ = builder(rec)
    model = GenerativeModel(MODEL, system_instruction=[sys_p])
    try:
        resp = model.generate_content(user_p, generation_config=GEN_CFG)
        return extract_verdict(resp.text.strip())
    except Exception as e:
        return f"ERR:{e!r}"[:80]


def sweep_variant(name, builder, recs):
    t0 = time.time()
    preds = [None] * len(recs)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(judge_one, r, builder): i for i, r in enumerate(recs)}
        for fut in as_completed(futs):
            preds[futs[fut]] = fut.result()
    by_cat = defaultdict(lambda: {"n": 0, "right": 0})
    for r, p in zip(recs, preds):
        by_cat[r["cat"]]["n"] += 1
        if p == r["gt"]:
            by_cat[r["cat"]]["right"] += 1
    total_n = sum(d["n"] for d in by_cat.values())
    total_r = sum(d["right"] for d in by_cat.values())
    print(f"  [{name:<24}] overall {total_r}/{total_n} = {100*total_r/total_n:.2f}%  ({time.time()-t0:.0f}s)")
    return by_cat, preds


def main():
    init_vertex()
    recs = load_oracle()
    print(f"oracle={len(recs)} records, {len(VARIANTS)} variants, model={MODEL}")
    out_dir = REPO / f"runs/judge_gemini_sweep_{datetime.now():%Y%m%d_%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cats = sorted({r["cat"] for r in recs})
    matrix = {}     # variant -> {cat: pct}
    all_preds = {}
    for name, builder in VARIANTS.items():
        by_cat, preds = sweep_variant(name, builder, recs)
        matrix[name] = {c: 100 * by_cat[c]["right"] / max(by_cat[c]["n"], 1) for c in cats}
        all_preds[name] = preds
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r, p in zip(recs, preds):
                f.write(json.dumps({"sid": r["sid"], "cat": r["cat"],
                                    "gt": r["gt"], "pred": p}) + "\n")

    # Print matrix
    print(f"\n{'category':<55} " + " ".join(f"{n[:18]:>18}" for n in VARIANTS))
    for cat in cats:
        row = " ".join(f"{matrix[n][cat]:>17.2f}%" for n in VARIANTS)
        print(f"{cat[:55]:<55} {row}")

    # Best variant per category
    print(f"\n{'category':<55} {'best':>22} {'acc':>7}")
    best_map = {}
    for cat in cats:
        best_name, best_pct = max(((n, matrix[n][cat]) for n in VARIANTS), key=lambda kv: kv[1])
        best_map[cat] = best_name
        print(f"{cat[:55]:<55} {best_name:>22} {best_pct:>6.2f}%")

    # Compute the proposed router score by mixing per-category winners
    n_total = len(recs)
    right_total = 0
    for cat in cats:
        b = best_map[cat]
        for r, p in zip(recs, all_preds[b]):
            if r["cat"] == cat and p == r["gt"]:
                right_total += 1
    print(f"\nProposed gemini ROUTER_MAP overall: {right_total}/{n_total} = {100*right_total/n_total:.2f}%")

    with open(out_dir / "matrix.json", "w") as f:
        json.dump({"matrix": matrix, "best_map": best_map,
                   "proposed_overall": 100 * right_total / n_total}, f, indent=2)
    print(f"\nartifacts → {out_dir}")


if __name__ == "__main__":
    main()
