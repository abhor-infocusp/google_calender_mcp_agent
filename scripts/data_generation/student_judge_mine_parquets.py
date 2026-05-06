#!/usr/bin/env python3
"""Mine candidate rows from RL trajectory parquets via replay.

Each parquet row stores the agent's message sequence (system, user,
assistant, tool, ...) plus reward / metrics / metadata.scenario_id.
We:

  1. Load all RL scenarios (data/rl/json_calender/*.txt + data/rl/queries/*.txt).
  2. For each parquet rollout:
       - look up scenario by sid
       - init CalendarEnvironment from the scenario's calendar + current_time
       - snapshot before
       - replay each assistant tool_call (dispatch_tool_call on env)
       - snapshot after
       - format before/after via filter_by_days + format_day_state_text
       - extract final answer (last assistant content OR return_final_answer arg)
  3. Class-balance per (sid, cat): with multiple rollouts per group, prefer
     Incorrect ones for non-Complex/Vague cats (silver-pool class balance).
     Complex/Vague kept as-is (silver-blocked anyway).
  4. De-dup by sha256(final|before|after)[:12].

Output: data/judge/v2_20260502/student_candidates_parquet.jsonl
Schema matches student_judge_mine.py output.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, random, sys, traceback
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from calendar_agent.environment import CalendarEnvironment  # noqa: E402
from calendar_agent.core import (  # noqa: E402
    dispatch_tool_call, format_tool_result, snapshot_events,
    filter_by_days, compute_fallback_now,
)
from calendar_agent.evaluation import format_day_state_text  # noqa: E402

DEFAULT_OUT = REPO / "data/judge/v2_20260502/student_candidates_parquet.jsonl"
HOLDOUT = REPO / "data/judge/v2_20260502/holdout_sids.json"
RL_CAL = REPO / "data/rl/json_calender"
RL_QRY = REPO / "data/rl/queries"

PARQUET_GLOBS = [
    str(REPO / "runs/rl_adaptive_qwen3_14b_20260424/.art/calendar-agent/models/calendar-agent-001/trajectories/train/*.parquet"),
    str(REPO / "runs/rl_adaptive_qwen3_14b_base_20260426/.art/adaptive-base-20260426/models/qwen3-14b-adaptive/trajectories/train/*.parquet"),
]

SILVER_BLOCKED = {
    "Complex Logic & Conflict (Advanced)",
    "Vague & Contextual (Reasoning Required)",
}


def rhash(final: str, before: str, after: str) -> str:
    s = f"{final or ''}\x1f{before or ''}\x1f{after or ''}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def load_scenarios() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cal_path in sorted(RL_CAL.glob("*.txt")):
        cal_index = cal_path.stem
        if not cal_index.isdigit():
            continue
        cal_index = int(cal_index)
        qry_path = RL_QRY / f"{cal_index}.txt"
        if not qry_path.exists():
            continue
        fallback_now = compute_fallback_now(str(cal_path))
        with qry_path.open() as f:
            queries = json.load(f)
        for qi, q in enumerate(queries):
            sid = f"cal_{cal_index}_q_{qi}"
            current_time = q.get("current_time", "") or fallback_now
            current_time = current_time.replace("T", " ") if current_time else fallback_now
            out[sid] = {
                "sid": sid,
                "query": q["query"],
                "expected": q.get("expected_behavior", ""),
                "category": q.get("category", "Unknown"),
                "addressed_days": q.get("addressed_days", []),
                "current_time": current_time,
                "calendar_file_path": str(cal_path.resolve()),
            }
    return out


def replay_one(scenario: dict, messages) -> tuple[str, str, str] | None:
    """Returns (final_answer, before_text, after_text) or None on failure."""
    try:
        env = CalendarEnvironment()
        events = CalendarEnvironment.load_json_calendar(scenario["calendar_file_path"])
        env.initialize(events=events, now=scenario["current_time"])
        before_snap = snapshot_events(env)

        final_answer: str | None = None
        for m in messages:
            role = m.get("role")
            if role == "assistant":
                tcs_raw = m.get("tool_calls")
                if tcs_raw:
                    if isinstance(tcs_raw, str):
                        try:
                            tcs = json.loads(tcs_raw)
                        except Exception:
                            tcs = []
                    else:
                        tcs = list(tcs_raw)
                    for tc in tcs:
                        fn = tc.get("function") or {}
                        name = fn.get("name")
                        args_raw = fn.get("arguments") or "{}"
                        try:
                            args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
                        except Exception:
                            args = {}
                        if name == "return_final_answer":
                            final_answer = args.get("answer", "") or final_answer
                            continue
                        try:
                            dispatch_tool_call(env, name, args)
                        except Exception:
                            # Tool failure: continue replay; the recorded tool result
                            # in the next message represents the same mistake.
                            pass
                # Fallback: assistant content (no tool call) = final answer
                if not tcs_raw:
                    c = m.get("content") or ""
                    if "</think>" in c:
                        c = c.split("</think>")[-1].strip()
                    if c.strip():
                        final_answer = c.strip()

        after_snap = snapshot_events(env)
        before_days = filter_by_days(before_snap, scenario["addressed_days"])
        after_days = filter_by_days(after_snap, scenario["addressed_days"])
        before_text = format_day_state_text(before_days)
        after_text = format_day_state_text(after_days)
        if final_answer is None:
            return None
        return final_answer, before_text, after_text
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--per-sid-cap", type=int, default=15,
                    help="max rollouts per sid (after class-balance preference)")
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260504)
    args = ap.parse_args()
    random.seed(args.seed)

    scen = load_scenarios()
    print(f"loaded scenarios: {len(scen)}")

    holdout_doc = json.load(open(HOLDOUT))
    holdout = set(holdout_doc["holdout_sids"])
    print(f"holdout sids: {len(holdout)}")

    files: list[str] = []
    for g in PARQUET_GLOBS:
        files.extend(sorted(glob.glob(g)))
    if args.limit_files:
        files = files[:args.limit_files]
    print(f"parquet files: {len(files)}")

    # Stage 1: replay all rollouts, group by sid + correctness
    by_sid_verdict: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"Correct": [], "Incorrect": []})
    n_rows = n_replay_ok = n_replay_fail = n_holdout = n_no_scen = 0
    for fpath in files:
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            continue
        for _, row in df.iterrows():
            n_rows += 1
            try:
                meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else (row["metadata"] or {})
                metrics = json.loads(row["metrics"]) if isinstance(row["metrics"], str) else (row["metrics"] or {})
            except Exception:
                continue
            sid = meta.get("scenario_id")
            if not sid:
                continue
            if sid in holdout:
                n_holdout += 1
                continue
            scenario = scen.get(sid)
            if not scenario:
                n_no_scen += 1
                continue
            cat = meta.get("category") or scenario.get("category") or "Unknown"
            try:
                replayed = replay_one(scenario, row["messages"])
            except Exception:
                replayed = None
            if replayed is None:
                n_replay_fail += 1
                continue
            n_replay_ok += 1
            final_answer, before_text, after_text = replayed
            verdict_int = metrics.get("verdict")
            prior = "Correct" if verdict_int == 1 else "Incorrect"  # -1/0 → Incorrect, 1 → Correct
            cand = {
                "sid": sid,
                "cat": cat,
                "query": scenario["query"],
                "final": final_answer,
                "expected": scenario["expected"],
                "before": before_text,
                "after": after_text,
                "src": "parquet",
                "src_step": meta.get("step"),
                "prior_verdict": prior,
                "prior_judge": "rl-step-judge",
            }
            cand["rollout_hash"] = rhash(cand["final"], cand["before"], cand["after"])
            by_sid_verdict[sid][prior].append(cand)

    print(f"rows={n_rows} replay_ok={n_replay_ok} replay_fail={n_replay_fail} holdout={n_holdout} no_scen={n_no_scen}")

    # Stage 2: per-sid cap with class-balance preference for non-blocked cats
    # For non-blocked cats: prefer Incorrect first (push silver-pool toward 73/27 target);
    # for blocked cats: keep mix (we drop them later anyway unless gold-matched).
    seen_hash: set[str] = set()
    out_rows: list[dict] = []
    for sid, buckets in by_sid_verdict.items():
        cat = (buckets["Correct"] + buckets["Incorrect"])[0]["cat"]
        # de-dup within sid
        for v in ("Correct", "Incorrect"):
            uniq = []
            for r in buckets[v]:
                if r["rollout_hash"] in seen_hash:
                    continue
                seen_hash.add(r["rollout_hash"])
                uniq.append(r)
            buckets[v] = uniq

        cap = args.per_sid_cap
        if cat in SILVER_BLOCKED:
            # take whatever we have, capped uniformly
            random.shuffle(buckets["Correct"])
            random.shuffle(buckets["Incorrect"])
            picks = (buckets["Correct"] + buckets["Incorrect"])[:cap]
        else:
            # silver-class-balance preference: bias to Incorrect, fill with Correct
            random.shuffle(buckets["Correct"])
            random.shuffle(buckets["Incorrect"])
            target_inc = int(round(cap * 0.45))  # aim ~45% Incorrect in silver pool;
            # post Qwen∧Gemini agreement filter we lose ~30% of Incorrect, ending near 73/27.
            inc = buckets["Incorrect"][:target_inc]
            cor = buckets["Correct"][:cap - len(inc)]
            picks = inc + cor
        out_rows.extend(picks)

    print(f"after per-sid cap (max {args.per_sid_cap}): {len(out_rows)}")
    by_cat = Counter(r["cat"] for r in out_rows)
    by_prior = Counter(r["prior_verdict"] for r in out_rows)
    print("by cat:", dict(by_cat))
    print("prior verdicts (untrusted):", dict(by_prior))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
