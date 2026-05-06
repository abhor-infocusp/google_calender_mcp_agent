#!/usr/bin/env python3
"""Assemble student-judge SFT dataset from candidates + relabels + gold.

Inputs:
  data/judge/v2_20260502/student_candidates.jsonl    — 5808 candidate rollouts
  data/judge/v2_20260502/relabel_qwen.jsonl          — Qwen-v2 verdict + raw CoT (target)
  data/judge/v2_20260502/relabel_gemini.jsonl        — Gemini-v2 verdict (silver gate)
  data/judge/v2_20260502/train.jsonl                 — existing v2 gold (744)
  data/judge/v2_20260502/disagreements.jsonl         — 85 adjudicated gold
  data/judge/v2_20260502/eval.jsonl                  — 110 holdout (NEVER touched)
  data/judge/v2_20260502/manual_v2_labels.jsonl      — 139 agent labels (sid-level)

Pipeline:
  1. Match candidates against gold by (sid, hash8) → label, label_source=gold.
  2. Non-gold candidates: silver iff cat NOT in {Complex, Vague}
                                AND qwen_v2 == gemini_v2.
  3. Drop rows where Qwen-v2 raw CoT verdict disagrees with assigned label
     (gold or silver) — bad CoT poisons SFT.
  4. Per-cat class-balance to natural live ratio (73% Correct / 27% Incorrect ± 5%).
  5. Per-cat cap silver at SILVER_CAP × gold_count.
  6. Train-dev split: ~10% unique sids per cat to dev; rest to train.
     Held-out scenarios — no sid leakage between train and dev.

Outputs:
  data/judge/v2_20260502/student_sft_train.jsonl
  data/judge/v2_20260502/student_sft_dev.jsonl
  data/judge/v2_20260502/student_sft_metadata.json

Each output row:
  {sid, rollout_hash, cat, label, label_source,
   prompt_system, prompt_user, target_cot,
   judge_qwen_v2, judge_gemini_v2}
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from calendar_agent.judge.prompts import build_router_qwen_v2, extract_verdict  # noqa: E402

V2 = REPO / "data/judge/v2_20260502"

# Silver-block was a v2-dataset rule (P(both wrong | agree) > 5% on Complex/Vague).
# For SFT distillation we tolerate ~10% label noise to gain ~30× more rows; the
# locked tier-1 holdout (eval.jsonl) is unchanged and still gold-only.
SILVER_BLOCKED: set[str] = set()
SILVER_CAP = 200  # silver per cat ≤ SILVER_CAP × gold per cat (effectively unlimited)
MIN_COT_CHARS = 200  # drop rows where teacher CoT is too short (skipped reasoning)
# Per-query (sid) caps: Correct rollouts share the same underlying fact-claim
# and are mostly surface-variations, so they're more redundant than Incorrect
# rollouts (which span diverse failure modes). Cap accordingly.
MAX_CORRECT_PER_SID = 3
MAX_INCORRECT_PER_SID = 10
# Target ratio = 3 Correct : 10 Incorrect per sid → 0.23 Correct fraction.
# Per-sid caps already enforce this structurally; the cat-level balance pass
# below only kicks in if a cat is wildly off (e.g. RelTime is mostly Correct).
TARGET_POS_FRAC = 0.23
POS_TOL = 0.10
DEV_FRAC = 0.10
SEED = 20260504


def h8(final: str, before: str, after: str) -> str:
    s = f"{final or ''}{before or ''}{after or ''}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open()]


def load_relabel_index(p: Path, prefix: str) -> dict[str, dict]:
    """Index by rollout_hash (12-char from miner)."""
    if not p.exists():
        return {}
    out = {}
    for l in p.open():
        r = json.loads(l)
        if r.get(f"{prefix}_v2_err"):
            continue
        out[r["rollout_hash"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-train", default=str(V2 / "student_sft_train.jsonl"))
    ap.add_argument("--out-dev", default=str(V2 / "student_sft_dev.jsonl"))
    ap.add_argument("--out-meta", default=str(V2 / "student_sft_metadata.json"))
    ap.add_argument("--strict", action="store_true",
                    help="error if any expected gold source is missing")
    args = ap.parse_args()
    random.seed(SEED)

    # ---- Load
    cand_combined = V2 / "student_candidates_combined.jsonl"
    candidates = load_jsonl(cand_combined if cand_combined.exists() else V2 / "student_candidates.jsonl")
    cand_by_h12 = {r["rollout_hash"]: r for r in candidates}
    qwen_idx = load_relabel_index(V2 / "relabel_qwen.jsonl", "qwen")
    gemini_idx = load_relabel_index(V2 / "relabel_gemini.jsonl", "gemini")

    # Existing curated gold (rows on disk in v2 dataset)
    gold_train = load_jsonl(V2 / "train.jsonl")
    gold_disagree = load_jsonl(V2 / "disagreements.jsonl")
    gold_eval_holdout = load_jsonl(V2 / "eval.jsonl")  # never use, but record sids to exclude
    holdout_sid_set = {r["sid"] for r in gold_eval_holdout}

    # Build a lookup of gold rows by (sid, rollout_hash) — these were 8-char hashed in v2 work
    gold_by_key: dict[tuple[str, str], dict] = {}
    for r in gold_train + gold_disagree:
        gold_by_key[(r["sid"], r["rollout_hash"])] = r

    print(f"candidates: {len(candidates)}  qwen_relabeled: {len(qwen_idx)}  gemini_relabeled: {len(gemini_idx)}")
    print(f"existing v2 gold: train={len(gold_train)} disagreements={len(gold_disagree)}  holdout={len(gold_eval_holdout)}")

    # ---- For each candidate: assign label_source + label
    rows: list[dict] = []
    skip = Counter()
    for cand in candidates:
        if cand["sid"] in holdout_sid_set:
            skip["holdout_sid"] += 1
            continue
        h12 = cand["rollout_hash"]
        qrec = qwen_idx.get(h12)
        if not qrec:
            skip["no_qwen_relabel"] += 1
            continue
        qwen_verdict = qrec["qwen_v2_verdict"]
        qwen_raw = qrec["qwen_v2_raw"]
        # CoT must end in a parseable verdict that matches qwen_verdict
        # (extract_verdict already returns Correct/Incorrect; trust it)

        # Gold lookup: 8-char hash join on (sid, h8)
        h8_key = h8(cand["final"], cand["before"], cand["after"])
        gold = gold_by_key.get((cand["sid"], h8_key))

        if gold is not None:
            label = gold["label"]
            label_source = gold["label_source"]
            gemini_verdict = gold.get("judge_gemini_v2")  # may be None
        else:
            # Silver path
            if cand["cat"] in SILVER_BLOCKED:
                skip["silver_blocked_cat_no_gold"] += 1
                continue
            grec = gemini_idx.get(h12)
            if not grec:
                skip["no_gemini_relabel"] += 1
                continue
            gemini_verdict = grec["gemini_v2_verdict"]
            if qwen_verdict != gemini_verdict:
                skip["judge_disagreement"] += 1
                continue
            label = qwen_verdict
            label_source = "two_way_agree"

        # CoT-quality gate: Qwen's CoT must match the label
        if qwen_verdict != label:
            skip["qwen_cot_mismatch_label"] += 1
            continue
        # CoT-quality gate: minimum length (avoid teaching student to skip reasoning)
        if len(qwen_raw) < MIN_COT_CHARS:
            skip["cot_too_short"] += 1
            continue

        # Build prompt (router_qwen_v2)
        rec_for_prompt = {
            "cat": cand["cat"], "query": cand["query"], "final": cand["final"],
            "expected": cand["expected"], "before": cand["before"], "after": cand["after"],
        }
        sys_p, user_p, _ = build_router_qwen_v2(rec_for_prompt)
        sys_p = sys_p + "\n\n/no_think"  # match deployment

        rows.append({
            "sid": cand["sid"], "rollout_hash": h12, "cat": cand["cat"],
            "label": label, "label_source": label_source,
            "prompt_system": sys_p, "prompt_user": user_p,
            "target_cot": qwen_raw,
            "judge_qwen_v2": qwen_verdict,
            "judge_gemini_v2": gemini_verdict,
            # Convenience: SFT trainer reads `messages` directly.
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p},
                {"role": "assistant", "content": qwen_raw},
            ],
        })

    print(f"assembled rows pre-cap: {len(rows)}")
    print(f"skip reasons: {dict(skip)}")

    # ---- Per-(sid, label) cap: max 3 Correct, max 10 Incorrect per query.
    # Correct rollouts share the same underlying fact-claim → more redundant.
    # Incorrect rollouts span diverse failure modes → keep more.
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["sid"], r["label"])].append(r)
    capped_rows: list[dict] = []
    cap_drops = 0
    for (sid, label), rs in by_key.items():
        cap = MAX_CORRECT_PER_SID if label == "Correct" else MAX_INCORRECT_PER_SID
        if len(rs) > cap:
            random.shuffle(rs)
            cap_drops += len(rs) - cap
            rs = rs[:cap]
        capped_rows.extend(rs)
    print(f"per-(sid,label) cap: dropped {cap_drops}, kept {len(capped_rows)} (max {MAX_CORRECT_PER_SID}C / {MAX_INCORRECT_PER_SID}I per sid)")
    rows = capped_rows

    # ---- Per-cat class balance + silver cap
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r)

    balanced: list[dict] = []
    bal_log = {}
    for cat, rs in by_cat.items():
        gold = [r for r in rs if r["label_source"] != "two_way_agree"]
        silver = [r for r in rs if r["label_source"] == "two_way_agree"]
        random.shuffle(silver)
        # silver cap
        cap = SILVER_CAP * max(len(gold), 1)
        silver = silver[:cap]
        merged = gold + silver
        # class balance on merged: trim majority class
        n_pos = sum(1 for r in merged if r["label"] == "Correct")
        n_neg = sum(1 for r in merged if r["label"] == "Incorrect")
        n_total = n_pos + n_neg
        if n_total == 0:
            continue
        pos_frac = n_pos / n_total
        # Target frac per global TARGET_POS_FRAC ± POS_TOL
        # If pos_frac too high → drop some Correct silver. If too low → drop some Incorrect silver.
        target_lo = TARGET_POS_FRAC - POS_TOL
        target_hi = TARGET_POS_FRAC + POS_TOL
        if pos_frac > target_hi:
            # too many Correct: drop silver-Correct
            silver_pos = [r for r in merged if r["label_source"] == "two_way_agree" and r["label"] == "Correct"]
            others = [r for r in merged if not (r["label_source"] == "two_way_agree" and r["label"] == "Correct")]
            target_pos = int(target_hi * n_total)
            keep = max(target_pos - sum(1 for r in others if r["label"] == "Correct"), 0)
            random.shuffle(silver_pos)
            merged = others + silver_pos[:keep]
        elif pos_frac < target_lo:
            silver_neg = [r for r in merged if r["label_source"] == "two_way_agree" and r["label"] == "Incorrect"]
            others = [r for r in merged if not (r["label_source"] == "two_way_agree" and r["label"] == "Incorrect")]
            target_neg = int((1 - target_lo) * n_total)
            keep = max(target_neg - sum(1 for r in others if r["label"] == "Incorrect"), 0)
            random.shuffle(silver_neg)
            merged = others + silver_neg[:keep]
        bal_log[cat] = {
            "gold": len(gold), "silver_in": len([r for r in rs if r["label_source"] == "two_way_agree"]),
            "silver_capped": len(silver), "final_total": len(merged),
            "final_pos_frac": round(sum(1 for r in merged if r["label"] == "Correct") / max(len(merged), 1), 3),
        }
        balanced.extend(merged)

    print(f"balanced rows: {len(balanced)}")
    for cat, info in bal_log.items():
        print(f"  {cat[:48]:48s} {info}")

    # ---- Train-dev split: held-out scenarios, stratified per cat
    by_cat = defaultdict(list)
    for r in balanced:
        by_cat[r["cat"]].append(r)

    # Held-out calendars (not held-out sids). Stronger generalization test:
    # if cal_X is in dev, no rollout of any cal_X_q_Y appears in train, so the
    # model has never seen cal_X's calendar layout / persona / event corpus.
    # Pick dev calendars ONCE GLOBALLY (not per-cat) — otherwise a calendar
    # held out for one cat can leak into train via a different cat.
    cal_of = lambda sid: sid.split("_q_")[0]
    all_cals = sorted({cal_of(r["sid"]) for r in balanced})
    random.shuffle(all_cals)
    n_dev_cals = max(1, int(round(len(all_cals) * DEV_FRAC)))
    dev_cals = set(all_cals[:n_dev_cals])
    train, dev = [], []
    for r in balanced:
        (dev if cal_of(r["sid"]) in dev_cals else train).append(r)

    print(f"final: train={len(train)} dev={len(dev)}")
    print(f"  train cats:", Counter(r["cat"] for r in train))
    print(f"  dev cats:", Counter(r["cat"] for r in dev))

    # ---- Write
    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out_dev, "w") as f:
        for r in dev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "seed": SEED,
        "silver_cap_x_gold": SILVER_CAP,
        "target_pos_frac": TARGET_POS_FRAC,
        "pos_tol": POS_TOL,
        "dev_frac": DEV_FRAC,
        "silver_blocked_cats": sorted(SILVER_BLOCKED),
        "n_candidates": len(candidates),
        "n_qwen_relabel": len(qwen_idx),
        "n_gemini_relabel": len(gemini_idx),
        "skip_reasons": dict(skip),
        "per_cat_balance": bal_log,
        "n_train": len(train),
        "n_dev": len(dev),
        "train_pos_frac": round(sum(1 for r in train if r["label"] == "Correct") / max(len(train), 1), 3),
        "dev_pos_frac": round(sum(1 for r in dev if r["label"] == "Correct") / max(len(dev), 1), 3),
    }
    with open(args.out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    # Sanity assert: no sid leakage AND no calendar leakage
    train_sids = {r["sid"] for r in train}
    dev_sids = {r["sid"] for r in dev}
    leaked = train_sids & dev_sids
    assert not leaked, f"SID LEAK: {sorted(leaked)[:5]}"
    train_cals = {cal_of(s) for s in train_sids}
    dev_cals = {cal_of(s) for s in dev_sids}
    cal_leaked = train_cals & dev_cals
    assert not cal_leaked, f"CALENDAR LEAK: {sorted(cal_leaked)[:5]}"
    holdout_in_train = train_sids & holdout_sid_set
    assert not holdout_in_train, f"HOLDOUT LEAK: {sorted(holdout_in_train)[:5]}"
    print(f"✓ no sid leakage; no calendar leakage ({len(train_cals)} train cals, {len(dev_cals)} dev cals)")
    print(f"  no holdout leakage")


if __name__ == "__main__":
    main()
