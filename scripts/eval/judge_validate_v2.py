#!/usr/bin/env python3
"""Phase 3 validation harness.

Steps:
  1. Score ROUTER_MAP_QWEN_V2 + ROUTER_MAP_GEMINI_V2 on the locked holdout
     (85 oracle sids + ~45 manual_v2 holdout records). Per-category accuracy
     with bootstrap CI. Ship gate: each weak cat ≥90% on each judge separately.
  2. Run both v2 routers on a 1000-call live sample → agreement matrix.
  3. Sample 50 AGREEMENT cases (from step 2) → emit a label-pool jsonl for
     a Claude agent to label. Used to estimate P(both wrong | agree).
  4. Sample ALL DISAGREEMENT cases → emit a separate label-pool jsonl. These
     all become gold for the curated dataset.

Step 4 outputs:
  data/judge/v2_20260502/holdout_results.json
  data/judge/v2_20260502/live_agreement_matrix.json
  data/judge/v2_20260502/agreement_spotcheck_pool.jsonl  (50 records)
  data/judge/v2_20260502/disagreement_adjudicate_pool.jsonl  (~all disagreements)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import google.auth.transport.requests
import httpx
import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import GenerationConfig, GenerativeModel

from calendar_agent.judge.prompts import (
    build_router_qwen_v2, build_router_gemini_v2, extract_verdict,
)
from calendar_agent.paths import CREDENTIALS_PATH

REPO = Path(__file__).resolve().parents[2]
ORACLE_INPUT = REPO / "runs/judge_baseline_20260430/eval/manual_review_input.jsonl"
ORACLE_TRUTH = REPO / "runs/judge_baseline_20260430/eval/manual_verdicts.jsonl"
LIVE_CALLS = REPO / "runs/judge_service_20260501/calls.jsonl"
HOLDOUT_JSON = REPO / "data/judge/v2_20260502/holdout_sids.json"
NEW_LABELS = REPO / "data/judge/v2_20260502/manual_v2_labels.jsonl"
TO_COLLECT = REPO / "data/judge/v2_20260502/labels_to_collect.jsonl"

OUT = REPO / "data/judge/v2_20260502"

QWEN_BASE = "http://localhost:8000/v1"
QWEN_MODEL = "judge"
GEN_CFG = GenerationConfig(temperature=0.0, top_p=1.0, max_output_tokens=2048)

CAT_SHORT = {
    "Complex Logic & Conflict (Advanced)": "Complex",
    "Human Chaos (Edge Cases/Fragments)": "Chaos",
    "Information Retrieval (Querying)": "IR",
    "Modifier & Correction (Rescheduling/Updates)": "Modifier",
    "Relative Time References (today, tomorrow, yesterday, this week)": "RelTime",
    "Schedule a Single Event": "Schedule",
    "Vague & Contextual (Reasoning Required)": "Vague",
}


# ---- Helpers ----
def rollout_hash(rec: dict) -> str:
    blob = (rec.get("final", "") or "") + (rec.get("before", "") or "") + (rec.get("after", "") or "")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def init_gemini():
    with open(CREDENTIALS_PATH) as f:
        cd = json.load(f)
    creds = OAuth2Credentials(
        token=None, refresh_token=cd["refresh_token"], client_id=cd["client_id"],
        client_secret=cd["client_secret"], token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(google.auth.transport.requests.Request())
    vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)


def call_qwen(client: httpx.Client, sys_p: str, user_p: str, no_think: bool = True) -> str:
    if no_think:
        sys_p = sys_p + "\n\n/no_think"
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        "max_tokens": 512, "temperature": 0.0,
    }
    try:
        r = client.post(f"{QWEN_BASE}/chat/completions", json=payload, timeout=120.0)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"<<<ERR:{e!r}>>>"[:120]


def call_gemini(sys_p: str, user_p: str) -> str:
    m = GenerativeModel("gemini-2.0-flash-001", system_instruction=[sys_p])
    try:
        r = m.generate_content(user_p, generation_config=GEN_CFG)
        return r.text.strip()
    except Exception as e:
        return f"<<<ERR:{e!r}>>>"[:120]


def judge_record(judge: str, rec: dict, qwen_client: httpx.Client | None) -> str:
    if judge == "qwen":
        s, u, _ = build_router_qwen_v2(rec)
        raw = call_qwen(qwen_client, s, u)
    else:
        s, u, _ = build_router_gemini_v2(rec)
        raw = call_gemini(s, u)
    return extract_verdict(raw if not raw.startswith("<<<ERR") else "Incorrect")


def parallel_judge(judge: str, recs: list[dict], qwen_client: httpx.Client | None,
                   workers: int) -> list[str]:
    out = [None] * len(recs)

    def one(i: int) -> tuple[int, str]:
        return i, judge_record(judge, recs[i], qwen_client)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, i): i for i in range(len(recs))}
        for fut in as_completed(futs):
            i, p = fut.result()
            out[i] = p
    return out


def bootstrap_ci(preds: list[bool], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    if not preds:
        return (0.0, 0.0)
    boots = []
    for _ in range(n):
        sample = [preds[rng.randrange(len(preds))] for _ in preds]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    return (boots[int(n * 0.025)], boots[int(n * 0.975)])


# ---- Step 1: Score on holdout ----
def step1_holdout(qwen_client: httpx.Client) -> dict:
    holdout_sids = set(json.load(open(HOLDOUT_JSON))["holdout_sids"])

    # Build holdout pool: oracle records whose sid is in holdout_sids
    inputs = [json.loads(l) for l in open(ORACLE_INPUT)]
    truth = [json.loads(l) for l in open(ORACLE_TRUTH)]
    holdout_recs = []
    for i, rec in enumerate(inputs):
        if rec["sid"] not in holdout_sids:
            continue
        gt = (truth[i].get("verdict") if i < len(truth) else None) or rec.get("gt")
        if gt not in ("Correct", "Incorrect"):
            continue
        rec = dict(rec); rec["gt"] = gt; rec["label_source"] = "oracle"
        holdout_recs.append(rec)

    print(f"\n=== STEP 1: Scoring v2 routers on locked holdout ({len(holdout_recs)} oracle records) ===")
    print(f"holdout_sids count: {len(holdout_sids)}")

    res = {}
    for judge in ["qwen", "gemini"]:
        t0 = time.time()
        workers = 8 if judge == "qwen" else 16
        preds = parallel_judge(judge, holdout_recs, qwen_client, workers)
        elapsed = time.time() - t0
        print(f"  {judge}: {elapsed:.0f}s")

        # Per-category accuracy + CI
        by_cat: dict = defaultdict(list)
        for rec, p in zip(holdout_recs, preds):
            by_cat[rec["cat"]].append(p == rec["gt"])
        cat_stats = {}
        for cat, hits in by_cat.items():
            acc = sum(hits) / len(hits)
            lo, hi = bootstrap_ci(hits, n=1000, seed=20260502)
            cat_stats[cat] = {"n": len(hits), "right": sum(hits), "acc": acc,
                              "ci_lo": lo, "ci_hi": hi}
        overall = sum(p == r["gt"] for p, r in zip(preds, holdout_recs)) / len(holdout_recs)

        res[judge] = {
            "overall": overall, "n": len(holdout_recs),
            "by_cat": cat_stats,
            "preds": [{"sid": r["sid"], "cat": r["cat"], "gt": r["gt"], "pred": p}
                      for r, p in zip(holdout_recs, preds)],
        }

    # Print
    for judge in ["qwen", "gemini"]:
        print(f"\n--- {judge.upper()} v2 on holdout: {100*res[judge]['overall']:.2f}%  (n={res[judge]['n']}) ---")
        print(f"{'cat':<10} {'right/n':>10} {'acc':>8} {'95% CI':>16} {'gate?':>6}")
        for cat, s in sorted(res[judge]["by_cat"].items(), key=lambda kv: -kv[1]["acc"]):
            gate = "PASS" if s["acc"] >= 0.90 else "FAIL"
            print(f"{CAT_SHORT.get(cat, cat[:10]):<10} {s['right']}/{s['n']:<8} "
                  f"{100*s['acc']:>6.2f}% [{100*s['ci_lo']:>5.1f}, {100*s['ci_hi']:>5.1f}] {gate:>6}")

    return res


# ---- Step 2: live agreement sweep ----
def step2_agreement(qwen_client: httpx.Client, sample_size: int = 1000, seed: int = 20260502) -> dict:
    holdout_sids = set(json.load(open(HOLDOUT_JSON))["holdout_sids"])

    # Read calls.jsonl, exclude holdout sids, dedup by (sid, rollout_hash)
    seen = set()
    pool: list[dict] = []
    for ln in open(LIVE_CALLS):
        c = json.loads(ln)
        sid = c.get("scenario_id")
        if not sid or sid in holdout_sids:
            continue
        rh = rollout_hash(c)
        key = (sid, rh)
        if key in seen:
            continue
        seen.add(key)
        pool.append({
            "label_id": f"{sid}__{rh}", "sid": sid, "rollout_hash": rh,
            "cat": c.get("cat"), "query": c.get("query", ""), "final": c.get("final", ""),
            "expected": c.get("expected", ""), "before": c.get("before", ""),
            "after": c.get("after", ""),
        })

    rng = random.Random(seed)
    rng.shuffle(pool)
    by_cat: dict = defaultdict(list)
    for r in pool:
        by_cat[r["cat"]].append(r)
    per_cat = sample_size // len(by_cat) if by_cat else 0
    sample: list[dict] = []
    for c, items in by_cat.items():
        sample.extend(items[:per_cat])
    sample = sample[:sample_size]
    print(f"\n=== STEP 2: Live agreement sweep on {len(sample)} records ===")

    qwen_preds = parallel_judge("qwen", sample, qwen_client, 8)
    print(f"  qwen done")
    gemini_preds = parallel_judge("gemini", sample, None, 16)
    print(f"  gemini done")

    matrix: dict = defaultdict(lambda: {"agree_C": 0, "agree_I": 0,
                                        "q_C_g_I": 0, "q_I_g_C": 0, "n": 0})
    for r, q, g in zip(sample, qwen_preds, gemini_preds):
        c = r["cat"]
        matrix[c]["n"] += 1
        if q == "Correct" and g == "Correct":   matrix[c]["agree_C"] += 1
        elif q == "Incorrect" and g == "Incorrect": matrix[c]["agree_I"] += 1
        elif q == "Correct" and g == "Incorrect":   matrix[c]["q_C_g_I"] += 1
        elif q == "Incorrect" and g == "Correct":   matrix[c]["q_I_g_C"] += 1

    print(f"\n{'cat':<10} {'n':>4} {'agree':>6} {'agree%':>7} {'qC.gI':>6} {'qI.gC':>6}")
    for c, m in sorted(matrix.items()):
        agree = m["agree_C"] + m["agree_I"]
        print(f"{CAT_SHORT.get(c, c[:10]):<10} {m['n']:>4} {agree:>6} "
              f"{100*agree/max(m['n'],1):>6.2f}% {m['q_C_g_I']:>6} {m['q_I_g_C']:>6}")

    detail = []
    for r, q, g in zip(sample, qwen_preds, gemini_preds):
        detail.append({**r, "qwen": q, "gemini": g})
    return {"matrix": dict(matrix), "detail": detail}


# ---- Step 3 + 4: build label pools ----
def step34_pools(detail: list[dict], n_spotcheck: int = 50, seed: int = 20260502) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed + 1)
    agreements = [r for r in detail if r["qwen"] == r["gemini"]]
    disagreements = [r for r in detail if r["qwen"] != r["gemini"]]

    # Stratified spotcheck: 50 agreement cases stratified by cat
    by_cat = defaultdict(list)
    for r in agreements:
        by_cat[r["cat"]].append(r)
    per_cat = max(1, n_spotcheck // len(by_cat) if by_cat else 1)
    spotcheck = []
    for c, items in by_cat.items():
        rng.shuffle(items)
        spotcheck.extend(items[:per_cat])
    spotcheck = spotcheck[:n_spotcheck]

    # Adjudication: ALL disagreements
    print(f"\n=== STEP 3+4: build label pools ===")
    print(f"  agreements:     {len(agreements)} ({len(detail)} total in sample)")
    print(f"  disagreements:  {len(disagreements)}")
    print(f"  spotcheck pool: {len(spotcheck)}  (target {n_spotcheck})")

    return spotcheck, disagreements


def to_label_row(r: dict, bucket: str) -> dict:
    return {
        "label_id": r["label_id"], "sid": r["sid"], "rollout_hash": r["rollout_hash"],
        "cat": r["cat"], "query": r["query"], "final": r["final"],
        "expected": r["expected"], "before": r["before"], "after": r["after"],
        "selection_bucket": bucket,
        "qwen_v2": r["qwen"], "gemini_v2": r["gemini"],
        "manual_gt": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-holdout", action="store_true")
    ap.add_argument("--skip-live", action="store_true")
    ap.add_argument("--live-sample", type=int, default=1000)
    args = ap.parse_args()

    init_gemini()
    qwen_client = httpx.Client(timeout=httpx.Timeout(120.0))

    OUT.mkdir(parents=True, exist_ok=True)

    if not args.skip_holdout:
        holdout_res = step1_holdout(qwen_client)
        with open(OUT / "holdout_results.json", "w") as f:
            json.dump(holdout_res, f, indent=2)
        print(f"\nholdout results → {OUT / 'holdout_results.json'}")

    if not args.skip_live:
        live = step2_agreement(qwen_client, sample_size=args.live_sample)
        with open(OUT / "live_agreement_matrix.json", "w") as f:
            json.dump({"matrix": live["matrix"], "n": len(live["detail"])}, f, indent=2)
        with open(OUT / "live_agreement_detail.jsonl", "w") as f:
            for r in live["detail"]:
                f.write(json.dumps(r) + "\n")
        spotcheck, disagreements = step34_pools(live["detail"], n_spotcheck=50)
        with open(OUT / "agreement_spotcheck_pool.jsonl", "w") as f:
            for r in spotcheck:
                f.write(json.dumps(to_label_row(r, "agreement_spotcheck")) + "\n")
        with open(OUT / "disagreement_adjudicate_pool.jsonl", "w") as f:
            for r in disagreements:
                f.write(json.dumps(to_label_row(r, "disagreement_adjudicate")) + "\n")
        print(f"\nlive matrix → {OUT / 'live_agreement_matrix.json'}")
        print(f"spotcheck pool ({len(spotcheck)}) → {OUT / 'agreement_spotcheck_pool.jsonl'}")
        print(f"adjudicate pool ({len(disagreements)}) → {OUT / 'disagreement_adjudicate_pool.jsonl'}")


if __name__ == "__main__":
    main()
