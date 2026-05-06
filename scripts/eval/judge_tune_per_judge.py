#!/usr/bin/env python3
"""Per-judge per-category prompt tuning with 5-fold stratified CV.

Sweeps a pool of variants × categories × judges and reports held-out mean
accuracy with stdev. Output identifies the best variant per (judge, cat) for
inclusion in ROUTER_MAP_QWEN_V2 / ROUTER_MAP_GEMINI_V2.

Usage:
    PYTHONPATH=src /home/abhor/miniconda3/envs/agentic/bin/python \
        scripts/eval/judge_tune_per_judge.py \
        --judges qwen,gemini \
        --cats vague,complex,chaos,reltime \
        --labels data/judge/v2_20260502/manual_v2_labels.jsonl

Requires:
- Local judge vLLM available at http://localhost:8000/v1 (the live judge service).
- Vertex creds at the path resolved by calendar_agent.paths.CREDENTIALS_PATH.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import google.auth.transport.requests
import httpx
import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import GenerationConfig, GenerativeModel

from calendar_agent.judge.prompts import extract_verdict
from calendar_agent.paths import CREDENTIALS_PATH

REPO = Path(__file__).resolve().parents[2]
ORACLE_INPUT = REPO / "runs/judge_baseline_20260430/eval/manual_review_input.jsonl"
ORACLE_TRUTH = REPO / "runs/judge_baseline_20260430/eval/manual_verdicts.jsonl"
HOLDOUT_JSON = REPO / "data/judge/v2_20260502/holdout_sids.json"

CAT_FULL = {
    "vague": "Vague & Contextual (Reasoning Required)",
    "complex": "Complex Logic & Conflict (Advanced)",
    "chaos": "Human Chaos (Edge Cases/Fragments)",
    "reltime": "Relative Time References (today, tomorrow, yesterday, this week)",
    "schedule": "Schedule a Single Event",
    "modifier": "Modifier & Correction (Rescheduling/Updates)",
    "ir": "Information Retrieval (Querying)",
}

# ---- Variant pool ----
TUNE_PATH = REPO / "scripts/eval/judge_prompt_tune.py"
spec = importlib.util.spec_from_file_location("judge_prompt_tune", TUNE_PATH)
tune = importlib.util.module_from_spec(spec)
sys.modules["judge_prompt_tune"] = tune
spec.loader.exec_module(tune)

LENIENT = """\

CRITICAL: Default toward Correct on borderline cases. Only call Incorrect when at least one of these is clearly true:
1. Agent claimed success but calendar didn't change (and a state change was clearly required).
2. Agent modified, deleted, or rescheduled a DIFFERENT event than the one asked about.
3. Agent fabricated an EVENT (title + day + time window) not in BEFORE or AFTER.
4. Wrong duration when the user explicitly stated the duration.
5. Response denies an event that exists in BEFORE on the day asked about.
6. Tool calls broken/garbled, or agent flipped to an unrelated topic.
If none of (1)-(6) apply, output Correct."""


def _lenient(b):
    def f(rec):
        s, u, o = b(rec)
        return s + LENIENT, u, o
    return f


# Two new variants targeting documented Complex / multi-step failure mode
MULTI_STEP_PREFIX = """\
You evaluate a calendar assistant. Apply this procedure verbatim.

STEP 1 — Enumerate the user's requested actions. List each as a bullet.

STEP 2 — For each bullet, verify the calendar diff (BEFORE → AFTER) reflects
that action on the right weekday at the right time. Cite the BEFORE/AFTER
lines you used.

STEP 3 — Apply hard-failure checks:
- claimed success but calendar unchanged when a state change was required;
- wrong event modified/deleted;
- fabricated EVENT (by title + day + time window) not in BEFORE or AFTER
  (attendee/RSVP/wording mismatch is NOT fabrication);
- wrong duration when the user stated one;
- denied an event that exists in BEFORE on the day asked about.

STEP 4 — Verdict line: print exactly Correct or Incorrect on the very last
line. Be lenient on cosmetic details if every step in STEP 1 verifies in
STEP 2 and no STEP 3 hard failure applies.
"""


def build_multi_step_checklist(rec: dict) -> tuple[str, str, dict]:
    diff = tune.diff_states(rec["before"], rec["after"])
    user = (
        f"User query: {rec['query']}\n\n"
        f"Assistant's user-facing response: {rec.get('final', '')}\n\n"
        f"Expected behavior (hint): {rec.get('expected', '')}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        f"End with Correct or Incorrect on the last line."
    )
    return MULTI_STEP_PREFIX, user, {}


STATE_GROUNDED_PREFIX = """\
You evaluate a calendar assistant. The calendar's BEFORE/AFTER state is the
sole ground truth. For every claim you make about the agent's behaviour, you
MUST cite the specific BEFORE or AFTER line that justifies it.

Procedure:
1. Restate the user's intent in one sentence.
2. List each event named or implied by the assistant's response, and for each
   write either: "BEFORE line: <quote>" / "AFTER line: <quote>" / "NOT IN
   STATE — possible fabrication".
3. If any event is NOT IN STATE and is presented as factual by the agent, mark
   it a hallucination → Incorrect.
4. If the user requested a state change, name the BEFORE→AFTER difference
   that satisfies it; if none, the agent failed unless asking-clarification
   was the right move.
5. Output exactly Correct or Incorrect on the very last line.
"""


def build_state_grounded_v1(rec: dict) -> tuple[str, str, dict]:
    diff = tune.diff_states(rec["before"], rec["after"])
    user = (
        f"User query: {rec['query']}\n\n"
        f"Assistant's user-facing response: {rec.get('final', '')}\n\n"
        f"Expected behavior (hint): {rec.get('expected', '')}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        f"End with Correct or Incorrect on the last line."
    )
    return STATE_GROUNDED_PREFIX, user, {}


VARIANTS: dict[str, callable] = {
    "baseline":              tune.build_baseline,
    "cot_checklist_v2":      tune.build_cot_checklist_v2,
    "per_category_v3":       tune.build_per_category_v3,
    "fewshot":               tune.build_fewshot,
    "fewshot_v3":            tune.build_fewshot_v3,
    "fewshot_v4_dayfocus":   tune.build_fewshot_v4_dayfocus,
    "fewshot_v3_dayfocus":   tune.build_fewshot_v3_dayfocus,
    "fewshot_v4":            tune.build_fewshot_v4,
    "fewshot+L":             _lenient(tune.build_fewshot),
    "fewshot_v3+L":          _lenient(tune.build_fewshot_v3),
    "fewshot_v4_dayfocus+L": _lenient(tune.build_fewshot_v4_dayfocus),
    "multi_step_checklist":  build_multi_step_checklist,
    "state_grounded_v1":     build_state_grounded_v1,
}


# ---- Backends ----
QWEN_BASE = "http://localhost:8000/v1"
QWEN_MODEL = "judge"


def call_qwen(client: httpx.Client, sys_p: str, user_p: str, no_think: bool = True) -> str:
    if no_think:
        sys_p = sys_p + "\n\n/no_think"
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    try:
        r = client.post(f"{QWEN_BASE}/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"<<<ERR:{e!r}>>>"[:120]


_GEMINI_INITED = False


def init_gemini() -> None:
    global _GEMINI_INITED
    if _GEMINI_INITED:
        return
    with open(CREDENTIALS_PATH) as f:
        cd = json.load(f)
    creds = OAuth2Credentials(
        token=None, refresh_token=cd["refresh_token"], client_id=cd["client_id"],
        client_secret=cd["client_secret"], token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(google.auth.transport.requests.Request())
    vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)
    _GEMINI_INITED = True


GEN_CFG = GenerationConfig(temperature=0.0, top_p=1.0, max_output_tokens=2048)


def call_gemini(sys_p: str, user_p: str) -> str:
    m = GenerativeModel("gemini-2.0-flash-001", system_instruction=[sys_p])
    try:
        r = m.generate_content(user_p, generation_config=GEN_CFG)
        return r.text.strip()
    except Exception as e:
        return f"<<<ERR:{e!r}>>>"[:120]


# ---- Data ----
def load_labeled_pool(extra_labels_path: Path | None) -> list[dict]:
    """Combine 285 oracle (already labeled) with optional new labels from agent.

    Returns full records (with cat/query/final/expected/before/after/gt) for
    each labeled item, EXCLUDING holdout sids.
    """
    holdout = set(json.load(open(HOLDOUT_JSON))["holdout_sids"])
    inputs = [json.loads(l) for l in open(ORACLE_INPUT)]
    truth = [json.loads(l) for l in open(ORACLE_TRUTH)]
    pool: list[dict] = []
    for i, rec in enumerate(inputs):
        gt = (truth[i].get("verdict") if i < len(truth) else None) or rec.get("gt")
        if gt not in ("Correct", "Incorrect"):
            continue
        if rec["sid"] in holdout:
            continue
        rec = dict(rec); rec["gt"] = gt; rec["label_source"] = "oracle"
        pool.append(rec)

    if extra_labels_path and extra_labels_path.exists():
        # The "to_label" pool has full content; the agent's _labels.jsonl has
        # verdict only. Join by label_id.
        to_label_path = REPO / "data/judge/v2_20260502/labels_to_collect.jsonl"
        if not to_label_path.exists():
            print(f"warning: {to_label_path} missing — skipping new labels")
        else:
            content = {json.loads(l)["label_id"]: json.loads(l) for l in open(to_label_path)}
            for line in open(extra_labels_path):
                lab = json.loads(line)
                lid = lab["label_id"]
                if lid not in content:
                    continue
                rec = dict(content[lid])
                rec["gt"] = lab["verdict"]
                rec["label_source"] = lab.get("labeler", "manual_v2_agent")
                if rec.get("sid") in holdout:
                    continue
                pool.append(rec)
    return pool


def stratified_kfold(pool: list[dict], cat: str, k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    import random
    idxs = [i for i, r in enumerate(pool) if r["cat"] == cat]
    rng = random.Random(seed)
    rng.shuffle(idxs)
    folds = [idxs[i::k] for i in range(k)]
    splits = []
    for i in range(k):
        test = folds[i]
        train = [j for f in folds for j in f if j not in test]
        splits.append((train, test))
    return splits


def sweep_variant(judge: str, variant_name: str, pool: list[dict], idx_set: list[int],
                  qwen_client: httpx.Client | None) -> dict[int, str]:
    """Score a single variant on a list of pool indices. Returns dict idx → pred."""
    builder = VARIANTS[variant_name]
    preds: dict[int, str] = {}

    def one(i: int) -> tuple[int, str]:
        rec = pool[i]
        sys_p, user_p, _ = builder(rec)
        if judge == "qwen":
            raw = call_qwen(qwen_client, sys_p, user_p, no_think=True)
        else:
            raw = call_gemini(sys_p, user_p)
        return i, extract_verdict(raw if not raw.startswith("<<<ERR") else "Incorrect")

    workers = 16 if judge == "gemini" else 8
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, i): i for i in idx_set}
        for fut in as_completed(futs):
            i, p = fut.result()
            preds[i] = p
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="qwen,gemini")
    ap.add_argument("--cats", default="vague,complex,chaos,reltime")
    ap.add_argument("--variants", default=None,
                    help="Comma-separated subset of VARIANTS; default = all")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260502)
    ap.add_argument("--labels", type=Path,
                    default=REPO / "data/judge/v2_20260502/manual_v2_labels.jsonl")
    args = ap.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    cats = [CAT_FULL[c.strip()] for c in args.cats.split(",")]
    variants = [v.strip() for v in (args.variants or ",".join(VARIANTS)).split(",")]

    pool = load_labeled_pool(args.labels)
    print(f"labeled pool: {len(pool)} records  ({sum(1 for r in pool if r['label_source']=='oracle')} oracle + "
          f"{sum(1 for r in pool if r['label_source']!='oracle')} new)")
    by_cat = defaultdict(int)
    for r in pool:
        by_cat[r["cat"]] += 1
    for c in cats:
        print(f"  {c[:55]:<57} {by_cat[c]} records")

    if "gemini" in judges:
        init_gemini()
    qwen_client = httpx.Client(timeout=httpx.Timeout(120.0)) if "qwen" in judges else None

    out_dir = REPO / f"runs/judge_tune_per_judge_{datetime.now():%Y%m%d_%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # results[judge][cat][variant] = list of fold accuracies
    results: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    t_start = time.time()
    for judge in judges:
        for cat in cats:
            splits = stratified_kfold(pool, cat, args.folds, args.seed)
            n_records = sum(1 for r in pool if r["cat"] == cat)
            for fold_i, (train_idx, test_idx) in enumerate(splits):
                # No training step here — variants are static prompts.
                # Score each variant on the test fold.
                for vname in variants:
                    t0 = time.time()
                    preds = sweep_variant(judge, vname, pool, test_idx, qwen_client)
                    right = sum(1 for i in test_idx if preds.get(i) == pool[i]["gt"])
                    acc = right / max(len(test_idx), 1)
                    results[judge][cat][vname].append(acc)
                    print(f"  [{judge:6} | {cat[:18]:<18} | fold {fold_i+1}/{args.folds} | {vname:<24}] "
                          f"acc={100*acc:.1f}%  ({time.time()-t0:.0f}s)")

    print(f"\nTotal sweep time: {(time.time()-t_start)/60:.1f} min\n")

    # Aggregate + pick best per (judge, cat)
    summary: dict = {}
    for judge in judges:
        summary[judge] = {}
        for cat in cats:
            stats = {}
            for v in variants:
                accs = results[judge][cat][v]
                if not accs: continue
                stats[v] = {
                    "mean": statistics.mean(accs),
                    "stdev": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                    "min": min(accs), "max": max(accs),
                }
            # Score = mean − 0.5 * stdev (penalize variance)
            ranked = sorted(stats.items(), key=lambda kv: -(kv[1]["mean"] - 0.5 * kv[1]["stdev"]))
            summary[judge][cat] = {"per_variant": stats, "ranked": [v for v, _ in ranked]}

    # Print headline
    print(f"{'judge':<8} {'cat':<10} {'best variant':<24} {'mean':>7} {'stdev':>7}  runner-up")
    for judge in judges:
        for cat in cats:
            r = summary[judge][cat]
            best = r["ranked"][0]
            runner = r["ranked"][1] if len(r["ranked"]) > 1 else "—"
            cm = r["per_variant"][best]
            cat_short = next((k for k, v in CAT_FULL.items() if v == cat), cat[:10])
            print(f"{judge:<8} {cat_short:<10} {best:<24} {100*cm['mean']:>6.2f}% "
                  f"{100*cm['stdev']:>6.2f}%   {runner}")

    # Divergence constraint
    print("\nDivergence check (Qwen vs Gemini per cat):")
    for cat in cats:
        if "qwen" in summary and "gemini" in summary:
            q_top = summary["qwen"][cat]["ranked"][0]
            g_top = summary["gemini"][cat]["ranked"][0]
            note = "DIFFERENT" if q_top != g_top else "SAME (consider divergence)"
            print(f"  {cat[:25]:<27} qwen={q_top:<22} gemini={g_top:<22}  {note}")

    # Persist
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"judges": judges, "cats": cats, "variants": variants,
                   "folds": args.folds, "seed": args.seed,
                   "labeled_pool_size": len(pool),
                   "results": {j: {c: {v: list(accs) for v, accs in vs.items()}
                                   for c, vs in cs.items()}
                               for j, cs in results.items()},
                   "summary": summary}, f, indent=2)
    with open(out_dir / "best_per_cat.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["judge", "cat", "best_variant", "mean_acc", "stdev"])
        for judge in judges:
            for cat in cats:
                best = summary[judge][cat]["ranked"][0]
                cm = summary[judge][cat]["per_variant"][best]
                w.writerow([judge, cat, best, f"{cm['mean']:.4f}", f"{cm['stdev']:.4f}"])
    print(f"\nartifacts → {out_dir}")


if __name__ == "__main__":
    main()
