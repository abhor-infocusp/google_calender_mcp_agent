#!/usr/bin/env python3
"""Evaluate the v3 judge LoRA against the 285-trajectory manual oracle.

Sends the byte-identical training prompt (EVAL_SYSTEM_PROMPT + rl_train.py
user template) to the model — bypassing the router-service prompt wrapping.
Reports overall accuracy, per-category accuracy, and a confusion matrix.

Usage:
    PYTHONPATH=src python scripts/eval/judge_v3_manual_eval.py \\
        --lora runs/judge_v3_qwen3_14b_20260507/checkpoints/final \\
        --base Qwen/Qwen3-14B \\
        --out runs/judge_v3_qwen3_14b_20260507/eval/manual_oracle.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT  # noqa

INPUT_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_review_input.jsonl"
TRUTH_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_verdicts.jsonl"

CAT_SHORT = {
    "Complex Logic & Conflict (Advanced)": "Complex",
    "Human Chaos (Edge Cases/Fragments)": "Chaos",
    "Information Retrieval (Querying)": "IR",
    "Modifier & Correction (Rescheduling/Updates)": "Modifier",
    "Relative Time References (today, tomorrow, yesterday, this week)": "RelTime",
    "Schedule a Single Event": "Schedule",
    "Vague & Contextual (Reasoning Required)": "Vague",
}


def build_user_prompt(query, final_output, expected, before_text, after_text):
    return f"""\
Query: {query}

Response: {final_output if final_output else '(no response)'}

Expected: {expected if expected else '(not specified)'}

Before:
{before_text}

After:
{after_text}

Was the task completed correctly? End with one word: Correct or Incorrect."""


def extract_verdict(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        low = line.lower()
        if low.endswith("correct") and "incorrect" not in low:
            return "Correct"
        if low.endswith("incorrect"):
            return "Incorrect"
        if low == "correct":
            return "Correct"
        if low == "incorrect":
            return "Incorrect"
    return "Unknown"


def load_dataset() -> list[dict]:
    inputs = [json.loads(l) for l in INPUT_JSONL.open()]
    truth = [json.loads(l) for l in TRUTH_JSONL.open()]
    truth_by_idx = {t["idx"]: t["verdict"] for t in truth}
    out: list[dict] = []
    for i, r in enumerate(inputs):
        if i not in truth_by_idx:
            continue
        out.append({
            **r,
            "manual_verdict": truth_by_idx[i],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-14B")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    rows = load_dataset()
    print(f"loaded {len(rows)} manual-oracle rows")
    print(f"manual verdict counts: {Counter(r['manual_verdict'] for r in rows)}")

    print(f"loading base: {args.base}  (4bit={args.load_in_4bit})")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                                                    torch_dtype=torch.bfloat16, device_map="auto")
    else:
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                    device_map="auto")
    print(f"loading LoRA: {args.lora}")
    model = PeftModel.from_pretrained(base, args.lora)
    model.eval()
    print("model ready")

    t0 = time.time()
    results = []
    for i, r in enumerate(rows):
        user_prompt = build_user_prompt(
            r["query"], r.get("final", ""), r.get("expected", ""),
            r.get("before", ""), r.get("after", ""),
        )
        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        verdict = extract_verdict(gen)
        results.append({
            "sid": r.get("sid"),
            "cat": r.get("cat"),
            "manual": r["manual_verdict"],
            "judge": verdict,
            "raw": gen,
        })
        if (i + 1) % 20 == 0 or (i + 1) == len(rows):
            n_done = i + 1
            elapsed = time.time() - t0
            n_correct = sum(1 for x in results if x["judge"] == x["manual"])
            print(f"  [{n_done}/{len(rows)}] acc={n_correct/n_done:.3f}  "
                  f"elapsed={elapsed:.1f}s  pace={elapsed/n_done:.2f}s/item")

    # Summary
    n_total = len(results)
    n_correct = sum(1 for r in results if r["judge"] == r["manual"])
    n_unknown = sum(1 for r in results if r["judge"] == "Unknown")
    print(f"\nOVERALL: {n_correct}/{n_total} = {n_correct/n_total:.3%} (unknown: {n_unknown})")

    # Per-cat
    by_cat = defaultdict(lambda: {"n": 0, "correct": 0, "unknown": 0})
    for r in results:
        cat = CAT_SHORT.get(r["cat"], r["cat"])
        by_cat[cat]["n"] += 1
        if r["judge"] == r["manual"]:
            by_cat[cat]["correct"] += 1
        if r["judge"] == "Unknown":
            by_cat[cat]["unknown"] += 1

    print("\nPer-cat accuracy:")
    for cat in sorted(by_cat):
        s = by_cat[cat]
        acc = s["correct"] / max(1, s["n"])
        print(f"  {cat:10s} {s['correct']:3d}/{s['n']:3d} = {acc:.3%}  unknown={s['unknown']}")

    # Confusion matrix
    cm = defaultdict(int)
    for r in results:
        cm[(r["manual"], r["judge"])] += 1
    print("\nConfusion (manual → judge):")
    for k, v in sorted(cm.items()):
        print(f"  {k[0]:10s} → {k[1]:10s}  {v}")

    # Persist
    summary = {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total,
        "n_unknown": n_unknown,
        "per_cat": {k: dict(v) for k, v in by_cat.items()},
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in cm.items()},
        "base": args.base,
        "lora": args.lora,
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
