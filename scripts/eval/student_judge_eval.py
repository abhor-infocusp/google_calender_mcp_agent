#!/usr/bin/env python3
"""Tier-0/1/2 evaluation for a student-judge checkpoint.

Tiers (ship-gate definitions in docs/judge/):
  0. Train-dev      data/judge/v2_20260502/student_sft_dev.jsonl
                    Used for ckpt selection during SFT (loose gate).
  1. Anchor         data/judge/v2_20260502/eval.jsonl                (110 sacred)
                    Per-cat ship gates: Modifier/IR ≥90, Schedule/Chaos/RelTime ≥87,
                    Vague ≥85, Complex ≥73.
  2. Hard cases     data/judge/v2_20260502/disagreements.jsonl       (85 adjudicated)
                    Ship gate: ≥70%.

Calls a vLLM server (default http://localhost:8000/v1) with the student
checkpoint loaded. Sends the same router-qwen-v2 + /no_think prompt the
deployed judge service uses, parses the final verdict from raw output,
compares to label.

Reports per-cat accuracy + 95% bootstrap CI, overall accuracy, confusion
matrix, and a JSON dump for downstream analysis.

Usage:
  PYTHONPATH=src python scripts/eval/student_judge_eval.py \
      --tier 1 --model student-judge --base http://localhost:8000/v1
"""
from __future__ import annotations
import argparse, json, random, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from calendar_agent.judge.prompts import build_router_qwen_v2, extract_verdict  # noqa: E402

V2 = REPO / "data/judge/v2_20260502"
TIERS = {
    "0": V2 / "student_sft_dev.jsonl",
    "1": V2 / "eval.jsonl",
    "2": V2 / "disagreements.jsonl",
}


def load_tier(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def call_vllm(client: httpx.Client, base: str, model: str, sys_p: str, user_p: str,
              max_tokens: int, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        "max_tokens": max_tokens, "temperature": temperature,
    }
    try:
        r = client.post(f"{base}/chat/completions", json=payload, timeout=120.0)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"<<<ERR:{type(e).__name__}>>>"


def bootstrap_ci(values: list[int], n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(values) for _ in range(n)) / n)
    means.sort()
    return (sum(values) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def predict_one(client, base, model, rec, max_tokens, temperature) -> dict:
    # Some tier files don't have all the keys build_router_qwen_v2 expects; fill defaults.
    rec_for_prompt = {
        "cat": rec.get("cat", ""), "query": rec.get("query", ""),
        "final": rec.get("final", "") or "", "expected": rec.get("expected", "") or "",
        "before": rec.get("before", "") or "", "after": rec.get("after", "") or "",
    }
    sys_p, user_p, opts = build_router_qwen_v2(rec_for_prompt)
    sys_p = sys_p + "\n\n/no_think"
    raw = call_vllm(client, base, model, sys_p, user_p, max_tokens, temperature)
    verdict = extract_verdict(raw if not raw.startswith("<<<ERR") else "Incorrect")
    return {"raw": raw, "pred": verdict, "label": rec.get("label", rec.get("verdict")), "cat": rec.get("cat", "")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["0", "1", "2", "all"], default="all")
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="served-model-name on the vLLM server")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", default=None,
                    help="if set, write per-tier predictions json to this dir")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    tiers_to_run = ["0", "1", "2"] if args.tier == "all" else [args.tier]

    client = httpx.Client(timeout=httpx.Timeout(180.0))

    overall_summary: dict = {"model": args.model, "base": args.base, "tiers": {}}
    for tier in tiers_to_run:
        path = TIERS[tier]
        if not path.exists():
            print(f"Tier {tier}: missing {path}; skipped")
            continue
        recs = load_tier(path)
        print(f"\n=== Tier {tier}: {path.name} (n={len(recs)}) ===")
        t0 = time.monotonic()
        preds: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(predict_one, client, args.base, args.model, r, args.max_tokens, args.temperature): r for r in recs}
            for i, fut in enumerate(as_completed(futs)):
                preds.append(fut.result())
                if (i + 1) % 25 == 0 or i + 1 == len(recs):
                    print(f"  {i+1}/{len(recs)}  ({time.monotonic()-t0:.0f}s)")

        # Aggregate
        per_cat = defaultdict(list)
        per_label = defaultdict(list)
        per_cat_label = defaultdict(list)
        confusion = Counter()
        for p in preds:
            ok = 1 if p["pred"] == p["label"] else 0
            per_cat[p["cat"]].append(ok)
            per_label[p["label"]].append(ok)
            per_cat_label[(p["cat"], p["label"])].append(ok)
            confusion[(p["label"], p["pred"])] += 1
        all_ok = [v for vs in per_cat.values() for v in vs]
        overall_acc, lo, hi = bootstrap_ci(all_ok)
        print(f"  Overall: {overall_acc*100:.2f}%  [95% CI {lo*100:.2f}, {hi*100:.2f}]  n={len(all_ok)}")
        # Pos-stratified accuracy (catches prior-domination)
        for label in ("Correct", "Incorrect"):
            vs = per_label.get(label, [])
            if vs:
                a, l, h = bootstrap_ci(vs)
                print(f"  on label={label:9s}  {a*100:5.1f}% [{l*100:5.1f}, {h*100:5.1f}]  n={len(vs)}")
        per_cat_summary = {}
        for cat, vs in sorted(per_cat.items()):
            acc, l, h = bootstrap_ci(vs)
            per_cat_summary[cat] = {"n": len(vs), "acc": round(acc, 4), "ci_lo": round(l, 4), "ci_hi": round(h, 4)}
            print(f"  {cat[:48]:48s} {acc*100:5.1f}% [{l*100:5.1f}, {h*100:5.1f}]  n={len(vs)}")
            for label in ("Correct", "Incorrect"):
                vsx = per_cat_label.get((cat, label), [])
                if vsx:
                    ax, lx, hx = bootstrap_ci(vsx)
                    print(f"    label={label:9s} {ax*100:5.1f}% [{lx*100:5.1f}, {hx*100:5.1f}]  n={len(vsx)}")
                    per_cat_summary[cat][f"acc_{label.lower()}"] = round(ax, 4)
        print(f"  confusion (label→pred): {dict(confusion)}")
        overall_summary["tiers"][tier] = {
            "path": str(path), "n": len(preds),
            "overall_acc": round(overall_acc, 4),
            "ci": (round(lo, 4), round(hi, 4)),
            "per_cat": per_cat_summary,
            "confusion": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
        }
        if out_dir:
            with (out_dir / f"tier{tier}_predictions.jsonl").open("w") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

    if out_dir:
        with (out_dir / "summary.json").open("w") as f:
            json.dump(overall_summary, f, indent=2)
        print(f"\nWrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
