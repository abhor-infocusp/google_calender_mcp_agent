#!/usr/bin/env python3
"""Phase-1 ship-gate eval: trained judge vs Gemini labels on ART trajectories.

For each ART rollout under
  runs/rl_adaptive_qwen3_14b_20260424/.art/.../trajectories/train/*.parquet
we have:
  - messages    : full convo (system, user, assistant, tool, ...)
  - metrics     : JSON with 'verdict' ∈ {0, 1, -1}  (Gemini label at train time)
  - metadata    : JSON with 'scenario_id'='cal_<N>_q_<I>' and 'category'

To re-judge a rollout we need (query, final_output, expected, before_days, after_days).
Reconstruction:
  query        = first user message
  final_output = last assistant message.content (strip <think>...</think>)
  expected     = rl_data/queries/<cal>.txt[<q>]['expected_behavior']
  before_days  = snapshot of fresh env at scenario.current_time, filtered to addressed_days
  after_days   = snapshot after replaying every dispatched tool call from messages

Then send the prompt (EVAL_SYSTEM_PROMPT + the byte-identical user template) to
the trained judge LoRA and compare its verdict to metrics['verdict'].

Output: <run-dir>/eval/art_holdout.json with overall + per-category agreement
and a 2×2 confusion matrix.

Usage:
    PYTHONPATH=src python scripts/eval/eval_judge_on_art.py \\
        --checkpoint runs/judge_v1_qwen3_7b_20260425/checkpoints/checkpoint-final \\
        --num-samples 2000
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

# Repo importability
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from calendar_agent.core import (
    compute_fallback_now,
    dispatch_tool_call,
    filter_by_days,
    snapshot_events,
)
from calendar_agent.environment import CalendarEnvironment
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT
from calendar_agent.paths import RL_JSON_CALENDAR_DIR, RL_QUERY_DIR

DEFAULT_PARQUET_GLOB = os.path.join(
    _REPO_ROOT,
    "runs/rl_adaptive_qwen3_14b_20260424/.art/calendar-agent/models/"
    "calendar-agent-001/trajectories/train/*.parquet",
)
DEFAULT_OUTPUT = os.path.join(
    _REPO_ROOT, "runs/judge_v1_qwen3_7b_20260425/eval/art_holdout.json"
)

BASE_MODEL = "Qwen/Qwen3-8B"


# ── Prompt (byte-identical to rl_train.py:425-441) ─────────

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


def format_day_state_text(by_day):
    """Same as evaluation.format_day_state_text — re-imported for clarity."""
    from calendar_agent.evaluation import format_day_state_text as _f
    return _f(by_day)


def extract_verdict(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        if line.lower() == "correct":
            return "Correct"
        if line.lower() == "incorrect":
            return "Incorrect"
    for line in reversed(lines):
        ll = line.lower()
        if "incorrect" in ll:
            return "Incorrect"
        if "correct" in ll:
            return "Correct"
    return "Incorrect"


# ── Scenario index ─────────────────────────────────────────

def load_scenario_index() -> dict[str, dict]:
    """Map scenario_id 'cal_<N>_q_<I>' → {expected, addressed_days, current_time, calendar_file_path}."""
    idx = {}
    for cal_index in range(50):
        cal_path = os.path.join(str(RL_JSON_CALENDAR_DIR), f"{cal_index}.txt")
        query_path = os.path.join(str(RL_QUERY_DIR), f"{cal_index}.txt")
        if not (os.path.exists(cal_path) and os.path.exists(query_path)):
            continue
        fallback_now = compute_fallback_now(cal_path)
        with open(query_path) as f:
            queries = json.load(f)
        for q_index, q in enumerate(queries):
            current_time = q.get("current_time", "")
            current_time = current_time.replace("T", " ") if current_time else fallback_now
            idx[f"cal_{cal_index}_q_{q_index}"] = {
                "expected": q.get("expected_behavior", ""),
                "addressed_days": q.get("addressed_days", []),
                "current_time": current_time,
                "calendar_file_path": os.path.abspath(cal_path),
            }
    return idx


# ── Trajectory replay ──────────────────────────────────────
#
# IDs are *not* stable across env loads: CalendarEnvironment.load_json_calendar
# generates fresh `evt_<uuid>` ids on every load, so the agent's original
# update_event(event_id=...) / delete_event(event_id=...) calls reference IDs
# that don't exist in our fresh env, dispatch fails silently, and Before==After.
#
# We fix this by harvesting (orig_id → signature) pairs from prior tool-result
# messages in the trajectory, matching each signature to a current-env event,
# and remapping the event_id argument before dispatch.

import re

# list_events / format_summary line:
#   "id: evt_xxx | <summary> — <Day> HH:MM-HH:MM"
# Use lazy match for summary because summaries can contain spaces and dashes.
_FMT_SUMMARY_RE = re.compile(
    r"id:\s*(evt_[0-9a-f]+)\s*\|\s*(.+?)\s+—\s+([A-Za-z]{3})\s+(\d{2}:\d{2})-(\d{2}:\d{2})"
)
# format_detail block:
#   "<summary>\n  ID: evt_xxx\n  Time: <Day> Mon DD, HH:MM - HH:MM"
_FMT_DETAIL_RE = re.compile(
    r"^(.+?)\n\s*ID:\s*(evt_[0-9a-f]+)\s*\n\s*Time:\s*([A-Za-z]{3})[^,]*,\s*(\d{2}:\d{2})\s*-\s*(\d{2}:\d{2})",
    re.M,
)


def _harvest_signatures(text: str):
    """Return [(orig_id, summary, day_short, start_hm, end_hm), ...] from a tool result."""
    if not text:
        return []
    out = []
    for m in _FMT_SUMMARY_RE.finditer(text):
        eid, summ, day, sh, eh = m.groups()
        out.append((eid, summ.strip(), day, sh, eh))
    for m in _FMT_DETAIL_RE.finditer(text):
        summ, eid, day, sh, eh = m.groups()
        out.append((eid, summ.strip(), day, sh, eh))
    return out


def _find_env_event_id(env, summary: str, day_short: str, sh: str, eh: str):
    """Return current id of env event matching the given signature, or None."""
    for e in env.calendar.events:
        if (e.summary == summary
                and e.start.strftime("%a") == day_short
                and e.start.strftime("%H:%M") == sh
                and e.end.strftime("%H:%M") == eh):
            return e.id
    return None


def reconstruct_state(scenario: dict, messages: list[dict]) -> tuple[str, str, str] | None:
    """Returns (final_output, before_text, after_text) or None if reconstruction fails."""
    env = CalendarEnvironment()
    events = CalendarEnvironment.load_json_calendar(scenario["calendar_file_path"])
    env.initialize(events=events, now=scenario["current_time"])

    before_snap = snapshot_events(env)
    before_days = filter_by_days(before_snap, scenario["addressed_days"])

    # Maps original (training-time) event_id → current env event_id.
    # Built incrementally from tool result messages as we walk the trajectory.
    id_remap: dict[str, str] = {}

    final_output = None
    for msg in messages:
        role = msg.get("role")

        # Tool results carry the original event ids alongside their human-readable
        # signature. Use them to populate id_remap so subsequent assistant calls
        # can be remapped to current-env ids.
        if role == "tool":
            content = msg.get("content") or ""
            for orig_id, summ, day, sh, eh in _harvest_signatures(content):
                if orig_id in id_remap:
                    continue
                cur_id = _find_env_event_id(env, summ, day, sh, eh)
                if cur_id:
                    id_remap[orig_id] = cur_id
            continue

        if role != "assistant":
            continue
        tcs_field = msg.get("tool_calls")
        # tool_calls comes back as a JSON string in parquet form.
        tool_calls = []
        if tcs_field:
            if isinstance(tcs_field, str):
                try:
                    tool_calls = json.loads(tcs_field)
                except Exception:
                    tool_calls = []
            elif isinstance(tcs_field, list):
                tool_calls = tcs_field
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_str = fn.get("arguments", "") or "{}"
                if name == "return_final_answer":
                    try:
                        final_output = json.loads(args_str).get("answer", "")
                    except Exception:
                        final_output = ""
                    continue
                try:
                    args = json.loads(args_str) if args_str else {}
                except Exception:
                    args = {}
                # Remap stale event_ids to current-env ids when we have a mapping.
                orig_eid = args.get("event_id")
                if orig_eid and orig_eid in id_remap:
                    args = {**args, "event_id": id_remap[orig_eid]}
                try:
                    dispatch_tool_call(env, name, args)
                except Exception:
                    # Tool dispatch occasionally fails; the state at this point
                    # still represents what the agent achieved before the error.
                    pass
        else:
            # Assistant text message — last one wins as final_output.
            content = msg.get("content") or ""
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            if content:
                final_output = content

    after_snap = snapshot_events(env)
    after_days = filter_by_days(after_snap, scenario["addressed_days"])
    return (
        final_output or "",
        format_day_state_text(before_days),
        format_day_state_text(after_days),
    )


def first_user_query(messages) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content") or ""
    return ""


# ── Load trajectories + stratified sample ──────────────────

def collect_judged_trajectories(parquet_glob: str) -> list[dict]:
    """Return list of {scenario_id, category, gt_verdict, messages} for judged rows."""
    files = sorted(glob.glob(parquet_glob))
    if not files:
        sys.exit(f"No parquet files at {parquet_glob}")
    print(f"Loaded {len(files)} parquet files")
    out = []
    for f in files:
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            try:
                m = json.loads(row["metrics"])
                meta = json.loads(row["metadata"])
            except Exception:
                continue
            v = m.get("verdict", -1)
            if v not in (0, 1):
                continue
            out.append({
                "scenario_id": meta.get("scenario_id", ""),
                "category": meta.get("category", "Unknown"),
                "gt_verdict": "Correct" if v == 1 else "Incorrect",
                "messages": list(row["messages"]),
            })
    return out


def stratified_sample(trajs: list[dict], n: int, seed: int = 42) -> list[dict]:
    if n >= len(trajs):
        return trajs
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for t in trajs:
        buckets[(t["category"], t["gt_verdict"])].append(t)
    # Allocate quota per bucket proportionally, with floor of 1 if bucket non-empty.
    total = sum(len(v) for v in buckets.values())
    sampled = []
    for key, items in buckets.items():
        rng.shuffle(items)
        take = max(1, round(len(items) / total * n))
        sampled.extend(items[:take])
    rng.shuffle(sampled)
    return sampled[:n]


# ── Judge inference (transformers in-process) ──────────────

class LocalJudge:
    def __init__(self, checkpoint: str | None, base_model: str = BASE_MODEL,
                 load_in_4bit: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if not getattr(self.tokenizer, "chat_template", None):
            sys.exit("Tokenizer has no chat_template — cannot proceed")
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            print(f"Loading base model {base_model} (4-bit nf4 / bf16 compute)…")
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model, quantization_config=bnb, device_map="auto",
            )
        else:
            print(f"Loading base model {base_model} (bf16)…")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, device_map="auto",
            )
        if checkpoint:
            print(f"Loading LoRA adapter from {checkpoint}…")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, checkpoint)
        self.model.eval()

    @torch.no_grad()
    def judge(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return extract_verdict(text), text


# ── Main ───────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="LoRA checkpoint dir, or '' to evaluate the base model with no adapter")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--parquet-glob", default=DEFAULT_PARQUET_GLOB)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load base model in 4-bit nf4 (needed for 14B on a 24 GiB MIG slice)")
    args = ap.parse_args()

    print("=" * 60)
    print("Phase-1 ART-trajectory judge eval")
    print(f"  checkpoint:   {args.checkpoint}")
    print(f"  base model:   {args.base_model}")
    print(f"  parquet glob: {args.parquet_glob}")
    print(f"  num-samples:  {args.num_samples}")
    print("=" * 60)

    # 1. Collect judged trajectories.
    trajs = collect_judged_trajectories(args.parquet_glob)
    print(f"Judged trajectories available: {len(trajs)}")
    cat_counts = Counter(t["category"] for t in trajs)
    print(f"  per-category: {dict(cat_counts)}")
    verdict_counts = Counter(t["gt_verdict"] for t in trajs)
    print(f"  verdicts:     {dict(verdict_counts)}")

    # 2. Stratified sample.
    sampled = stratified_sample(trajs, args.num_samples, seed=args.seed)
    print(f"Sampled: {len(sampled)}")

    # 3. Build scenario index.
    scen_idx = load_scenario_index()
    print(f"Scenario index entries: {len(scen_idx)}")

    # 4. Load judge.
    judge = LocalJudge(args.checkpoint, base_model=args.base_model,
                       load_in_4bit=args.load_in_4bit)

    # 5. Score.
    per_traj = []
    skipped_no_scen = 0
    skipped_replay = 0
    for i, t in enumerate(sampled):
        sid = t["scenario_id"]
        scen = scen_idx.get(sid)
        if scen is None:
            skipped_no_scen += 1
            continue

        try:
            recon = reconstruct_state(scen, t["messages"])
        except Exception as e:
            skipped_replay += 1
            print(f"  [{i}] {sid} replay failed: {e}")
            continue
        if recon is None:
            skipped_replay += 1
            continue

        final_output, before_text, after_text = recon
        query = first_user_query(t["messages"])

        user_prompt = build_user_prompt(
            query=query,
            final_output=final_output,
            expected=scen["expected"],
            before_text=before_text,
            after_text=after_text,
        )
        pred, raw = judge.judge(EVAL_SYSTEM_PROMPT, user_prompt)
        agree = pred == t["gt_verdict"]
        per_traj.append({
            "scenario_id": sid,
            "category": t["category"],
            "gt_verdict": t["gt_verdict"],
            "pred_verdict": pred,
            "agree": agree,
            "query": query,
            "final_output": final_output,
            "expected": scen["expected"],
            "before_text": before_text,
            "after_text": after_text,
            "user_prompt": user_prompt,
            "raw": raw,
        })
        # Per-trajectory debug print so we can inspect prompt/reasoning/verdict
        # live and after the fact (sbatch log captures stdout).
        marker = "✓" if agree else "✗"
        print(f"\n{'=' * 80}")
        print(f"[{i+1}/{len(sampled)}] {sid}  category={t['category']}")
        print(f"  gt={t['gt_verdict']}  pred={pred}  {marker}")
        print(f"--- USER PROMPT ---")
        print(user_prompt)
        print(f"--- MODEL RAW ---")
        print(raw)
        print(f"{'=' * 80}", flush=True)
        if (i + 1) % 50 == 0:
            agree_so_far = sum(1 for r in per_traj if r["agree"]) / len(per_traj) * 100
            print(f"  [{i+1}/{len(sampled)}] running agreement: {agree_so_far:.1f}%", flush=True)

    # 6. Aggregate.
    n = len(per_traj)
    if n == 0:
        sys.exit("No trajectories were scored — check scenario index and replay")
    agree_total = sum(1 for r in per_traj if r["agree"])
    overall_pct = agree_total / n * 100

    by_cat = defaultdict(lambda: {"n": 0, "agree": 0})
    for r in per_traj:
        by_cat[r["category"]]["n"] += 1
        by_cat[r["category"]]["agree"] += int(r["agree"])
    per_cat_pct = {
        c: {"n": v["n"], "agree": v["agree"],
            "pct": v["agree"] / v["n"] * 100 if v["n"] else 0.0}
        for c, v in by_cat.items()
    }

    # 2x2 confusion matrix (rows = gt, cols = pred).
    cm = {
        ("Correct", "Correct"): 0,
        ("Correct", "Incorrect"): 0,
        ("Incorrect", "Correct"): 0,
        ("Incorrect", "Incorrect"): 0,
    }
    for r in per_traj:
        cm[(r["gt_verdict"], r["pred_verdict"])] += 1
    cm_serial = {f"{gt}->{pr}": v for (gt, pr), v in cm.items()}

    summary = {
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "n_scored": n,
        "n_skipped_no_scenario": skipped_no_scen,
        "n_skipped_replay": skipped_replay,
        "overall_agreement_pct": round(overall_pct, 2),
        "per_category": per_cat_pct,
        "confusion_matrix": cm_serial,
        "ship_gate_A_overall_ge_95": overall_pct >= 95.0,
        "ship_gate_B_per_cat_ge_90": all(v["pct"] >= 90.0 for v in per_cat_pct.values()),
        "per_traj": per_traj,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"Overall agreement: {overall_pct:.2f}% ({agree_total}/{n})")
    print("Per category:")
    for c, v in sorted(per_cat_pct.items()):
        print(f"  {c}: {v['pct']:.2f}% ({v['agree']}/{v['n']})")
    print("Confusion matrix:")
    for k, v in cm_serial.items():
        print(f"  {k}: {v}")
    print(f"Gate A (overall ≥95%): {'PASS' if summary['ship_gate_A_overall_ge_95'] else 'FAIL'}")
    print(f"Gate B (per-cat ≥90%): {'PASS' if summary['ship_gate_B_per_cat_ge_90'] else 'FAIL'}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
