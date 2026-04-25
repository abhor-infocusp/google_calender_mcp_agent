#!/usr/bin/env python3
"""Batch evaluation of SFT model on training data and RL data.

Usage:
    PYTHONPATH=src python scripts/eval/eval_batch.py --mode sft
    PYTHONPATH=src python scripts/eval/eval_batch.py --mode rl --num-calendars 20
    PYTHONPATH=src python scripts/eval/eval_batch.py --mode both --num-calendars 20
"""

import argparse
import json
import glob
import os
import signal
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

from openai import OpenAI

from calendar_agent.environment import CalendarEnvironment
from calendar_agent.core import (
    DAY_NAMES, compute_fallback_now,
    dispatch_tool_call, filter_by_days, get_query_now, snapshot_events,
)
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT, evaluate_trajectory, format_day_state_text
from calendar_agent.paths import SFT_DATA_DIR, RL_DATA_DIR, TEST_DATA_DIR, CREDENTIALS_PATH
from calendar_agent.core import format_tool_result
from calendar_agent.tools import get_openai_tools_minimal, get_openai_tools

OPENAI_TOOLS = get_openai_tools()  # Full descriptions for zero-shot eval


# ── Agent loop ───────────────────────────────────────────────

def run_query(client, model_name, tools, system_prompt, env, query, max_turns=8):
    """Run a single query, return trajectory."""
    trajectory = []
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    trajectory.append({"role": "user", "content": query})

    for turn in range(1, max_turns + 1):
        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages, tools=tools, temperature=0.7,
            )
        except Exception as e:
            trajectory.append({"role": "error", "content": str(e)})
            break

        msg = response.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if msg.content:
            trajectory.append({"role": "assistant", "content": msg.content})

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result = dispatch_tool_call(env, tool_name, args)
            result_str = format_tool_result(result)
            trajectory.append({"role": "tool_call", "name": tool_name, "args": args, "result": result_str})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

    return trajectory


# ── Eval helpers ─────────────────────────────────────────────

def get_sft_training_queries():
    """Get only the queries that have trajectories (were used for training).
    Returns list of (cal_idx, query_idx, query_dict)."""
    sft_dir = str(SFT_DATA_DIR)
    traj_dir = os.path.join(sft_dir, "trajectories")
    query_dir = os.path.join(sft_dir, "queries")

    tasks = []
    for traj_file in sorted(glob.glob(os.path.join(traj_dir, "*.json"))):
        cal_idx = os.path.basename(traj_file).replace(".json", "")
        # Load trajectories to get which queries were solved
        trajs = json.load(open(traj_file))
        # Load all queries for this calendar
        query_path = os.path.join(query_dir, f"{cal_idx}.txt")
        if not os.path.exists(query_path):
            continue
        all_queries = json.load(open(query_path))

        # Match trajectories to queries by query text
        solved_queries = {t["query"] for t in trajs}
        for qi, q in enumerate(all_queries):
            if q["query"] in solved_queries:
                tasks.append((cal_idx, qi, q))

    return tasks


def get_rl_queries(num_calendars=20):
    """Get RL data queries for first N calendars."""
    rl_dir = str(RL_DATA_DIR)
    query_dir = os.path.join(rl_dir, "queries")

    cal_files = sorted(glob.glob(os.path.join(query_dir, "*.txt")),
                       key=lambda f: int(os.path.basename(f).replace(".txt", "")))

    tasks = []
    for f in cal_files[:num_calendars]:
        cal_idx = os.path.basename(f).replace(".txt", "")
        queries = json.load(open(f))
        for qi, q in enumerate(queries):
            tasks.append((cal_idx, qi, q))

    return tasks


def get_test_queries(num_calendars=50):
    """Get held-out test data queries for first N calendars."""
    test_dir = str(TEST_DATA_DIR)
    query_dir = os.path.join(test_dir, "queries")

    cal_files = sorted(glob.glob(os.path.join(query_dir, "*.txt")),
                       key=lambda f: int(os.path.basename(f).replace(".txt", "")))

    tasks = []
    for f in cal_files[:num_calendars]:
        cal_idx = os.path.basename(f).replace(".txt", "")
        queries = json.load(open(f))
        for qi, q in enumerate(queries):
            tasks.append((cal_idx, qi, q))

    return tasks


def load_calendar(base_dir, cal_idx):
    """Load calendar events."""
    cal_path = os.path.join(base_dir, "json_calender", f"{cal_idx}.txt")
    events = CalendarEnvironment.load_json_calendar(cal_path)
    fallback_now = compute_fallback_now(cal_path)
    return events, fallback_now


def eval_tasks(client, model_name, eval_model, tasks, base_dir, label):
    """Evaluate a list of (cal_idx, query_idx, query_dict) tasks."""
    tools = list(OPENAI_TOOLS)
    system_prompt = "/no_think\nYou are a calendar assistant. Use the provided tools to manage events. Call get_current_time first to know the current date."

    results = []
    correct = 0
    incorrect = 0
    error = 0
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 10

    cat_results = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, (cal_idx, qi, q) in enumerate(tasks):
        events, fallback_now = load_calendar(base_dir, cal_idx)
        now = get_query_now(q, fallback_now)
        env = CalendarEnvironment()
        env.initialize(events=events, now=now)

        query_text = q["query"]
        expected = q.get("expected_behavior", "")
        category = q.get("category", "unknown")
        addressed_days = q.get("addressed_days", [])
        display_days = addressed_days if addressed_days else DAY_NAMES

        before = snapshot_events(env)
        before_days = filter_by_days(before, display_days)

        print(f"  [{i+1}/{len(tasks)}] cal={cal_idx} q={qi} [{category[:30]}] {query_text[:50]}...",
              end=" ", flush=True)

        def _alarm_handler(signum, frame):
            raise TimeoutError("Query timed out")

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(60)  # 60s hard timeout per query
        try:
            trajectory = run_query(client, model_name, tools, system_prompt, env, query_text)

            # If trajectory has errors (e.g. context overflow), skip judge — score as Incorrect
            has_error = any(s["role"] == "error" for s in trajectory)
            if has_error:
                err_msg = next((s["content"] for s in trajectory if s["role"] == "error"), "unknown")
                incorrect += 1
                consecutive_errors += 1
                cat_results[category]["total"] += 1
                print(f"Incorrect (agent error: {err_msg[:120]})", flush=True)
                results.append({"cal": cal_idx, "qi": qi, "category": category, "verdict": "Incorrect",
                                "query": query_text, "expected": expected, "trajectory": trajectory})
            else:
                consecutive_errors = 0
                after = snapshot_events(env)
                after_days = filter_by_days(after, display_days)

                final_output = next(
                    (s["content"] for s in reversed(trajectory) if s["role"] == "assistant"), ""
                )

                verdict, reasoning = evaluate_trajectory(eval_model, query_text, final_output, expected, before_days, after_days)

                before_text = format_day_state_text(before_days)
                after_text = format_day_state_text(after_days)

                if verdict == "Correct":
                    correct += 1
                    cat_results[category]["correct"] += 1
                    print(f"Correct", flush=True)
                else:
                    print(f"{verdict}", flush=True)
                    if verdict == "Incorrect":
                        incorrect += 1

                cat_results[category]["total"] += 1
                results.append({"cal": cal_idx, "qi": qi, "category": category, "verdict": verdict,
                                "query": query_text, "expected": expected, "final_output": final_output,
                                "trajectory": trajectory, "before": before_text, "after": after_text,
                                "judge_reasoning": reasoning})

        except TimeoutError:
            error += 1
            consecutive_errors += 1
            print(f"TIMEOUT", flush=True)
            results.append({"cal": cal_idx, "qi": qi, "category": category, "verdict": "Error",
                            "query": query_text, "expected": expected})
        except Exception as e:
            error += 1
            consecutive_errors += 1
            print(f"ERROR: {e}", flush=True)
            results.append({"cal": cal_idx, "qi": qi, "category": category, "verdict": "Error",
                            "query": query_text, "expected": expected, "error": str(e)})
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(f"\n  !!! {MAX_CONSECUTIVE_ERRORS} consecutive errors — vLLM likely hung, aborting eval !!!")
            break

        time.sleep(0.1)

    # Summary
    total = len(tasks)
    print()
    print(f"  === {label} RESULTS ===")
    print(f"  Total: {total}, Correct: {correct}, Incorrect: {incorrect}, Error: {error}")
    print(f"  Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print()
    print(f"  By category:")
    for cat in sorted(cat_results.keys()):
        cr = cat_results[cat]
        pct = cr["correct"] / cr["total"] * 100 if cr["total"] > 0 else 0
        print(f"    {cat}: {cr['correct']}/{cr['total']} ({pct:.0f}%)")

    return results, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sft", "rl", "test", "both"], default="both")
    parser.add_argument("--model", default="sft-v2")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--num-calendars", type=int, default=20, help="Number of RL calendars to eval")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--max-queries", type=int, default=0, help="Max queries per mode (0=all)")
    args = parser.parse_args()

    from openai import Timeout
    client = OpenAI(
        base_url=args.base_url, api_key="token-abc123",
        timeout=Timeout(connect=5, read=30, write=30, pool=5),
        max_retries=0,
    )

    # Init Gemini eval judge
    import vertexai
    from vertexai.generative_models import GenerativeModel
    from google.oauth2.credentials import Credentials as OAuth2Credentials

    with open(str(CREDENTIALS_PATH)) as f:
        cd = json.load(f)
    creds = OAuth2Credentials(
        token=None, refresh_token=cd["refresh_token"],
        client_id=cd["client_id"], client_secret=cd["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)
    eval_model = GenerativeModel("gemini-2.0-flash-001", system_instruction=[EVAL_SYSTEM_PROMPT])

    all_results = {}

    if args.mode in ("sft", "both"):
        print("=" * 60)
        print("SFT TRAINING DATA EVALUATION")
        print("  (only queries with existing trajectories)")
        print("=" * 60)
        tasks = get_sft_training_queries()
        if args.max_queries > 0:
            tasks = tasks[:args.max_queries]
        print(f"  {len(tasks)} queries across {len(set(t[0] for t in tasks))} calendars")
        print()
        results, correct, total = eval_tasks(
            client, args.model, eval_model, tasks, str(SFT_DATA_DIR), "SFT TRAINING DATA"
        )
        all_results["sft"] = {"results": results, "correct": correct, "total": total}

    if args.mode in ("rl", "both"):
        print()
        print("=" * 60)
        print(f"RL DATA EVALUATION ({args.num_calendars} calendars)")
        print("=" * 60)
        tasks = get_rl_queries(args.num_calendars)
        if args.max_queries > 0:
            tasks = tasks[:args.max_queries]
        print(f"  {len(tasks)} queries across {len(set(t[0] for t in tasks))} calendars")
        print()
        results, correct, total = eval_tasks(
            client, args.model, eval_model, tasks, str(RL_DATA_DIR), "RL DATA"
        )
        all_results["rl"] = {"results": results, "correct": correct, "total": total}

    if args.mode == "test":
        print()
        print("=" * 60)
        print(f"TEST DATA EVALUATION (held-out, {args.num_calendars} calendars)")
        print("=" * 60)
        tasks = get_test_queries(args.num_calendars)
        if args.max_queries > 0:
            tasks = tasks[:args.max_queries]
        print(f"  {len(tasks)} queries across {len(set(t[0] for t in tasks))} calendars")
        print()
        results, correct, total = eval_tasks(
            client, args.model, eval_model, tasks, str(TEST_DATA_DIR), "TEST DATA"
        )
        all_results["test"] = {"results": results, "correct": correct, "total": total}

    if args.save:
        with open(args.save, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
