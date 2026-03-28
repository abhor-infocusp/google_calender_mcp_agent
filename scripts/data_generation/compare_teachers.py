#!/usr/bin/env python3
"""Compare Gemini teacher models for trajectory generation quality.

Runs the same queries through multiple models and reports solve rates per category.

Usage:
    PYTHONPATH=src python scripts/data_generation/compare_teachers.py
"""

import json
import glob
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
import google.auth.transport.requests
from vertexai.generative_models import GenerativeModel
from collections import defaultdict

from calendar_agent.core import CALENDAR_TOOL, SYSTEM_PROMPT
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT
from calendar_agent.paths import SFT_DATA_DIR as _SFT_DATA_DIR, CREDENTIALS_PATH

sys.stdout.reconfigure(line_buffering=True)

SFT_DATA_DIR = str(_SFT_DATA_DIR)
JSON_CALENDAR_DIR = os.path.join(SFT_DATA_DIR, "json_calender")
QUERY_DIR = os.path.join(SFT_DATA_DIR, "queries")

# Test on calendars 0-4 (70 queries, 10 per category)
TEST_CALENDARS = [0, 2, 3, 4, 5]  # Cal 1 doesn't exist in queries

MODELS_TO_TEST = [
    "gemini-2.0-flash-001",
    "gemini-2.5-pro",
]


def run_single(model, eval_model, cal_path, query_dict, max_turns=10):
    """Run a single trajectory and return verdict."""
    from calendar_agent.environment import CalendarEnvironment
    from calendar_agent.core import (
        compute_fallback_now, dispatch_tool_call, snapshot_events,
        filter_by_days, get_query_now, DAY_NAMES,
    )
    from calendar_agent.evaluation import evaluate_trajectory
    from calendar_agent.tools import serialize_tool_result
    from vertexai.generative_models import Part

    events = CalendarEnvironment.load_json_calendar(cal_path)
    fallback_now = compute_fallback_now(cal_path)
    now = get_query_now(query_dict, fallback_now)
    env = CalendarEnvironment()
    env.initialize(events=events, now=now)

    addressed_days = query_dict.get("addressed_days", [])
    display_days = addressed_days if addressed_days else DAY_NAMES
    before = snapshot_events(env)
    before_days = filter_by_days(before, display_days)

    chat = model.start_chat()
    query_text = query_dict["query"]
    trajectory = [{"role": "user", "content": query_text}]

    try:
        response = chat.send_message(query_text)
    except Exception as e:
        return "Error"

    for turn in range(1, max_turns + 1):
        function_calls = []
        text_parts = []
        for part in response.candidates[0].content.parts:
            try:
                if part.function_call.name:
                    function_calls.append(part.function_call)
                    continue
            except AttributeError:
                pass
            try:
                if part.text:
                    text_parts.append(part.text)
            except AttributeError:
                pass

        if text_parts:
            trajectory.append({"role": "assistant", "content": "\n".join(text_parts)})

        if not function_calls:
            break

        response_parts = []
        for fc in function_calls:
            args = dict(fc.args)
            result = dispatch_tool_call(env, fc.name, args)
            if result is None:
                result = {"status": "ok"}
            result = serialize_tool_result(result)
            trajectory.append({"role": "tool_call", "name": fc.name, "args": args, "result": result})
            # Vertex AI requires dict for function responses; wrap lists
            resp_for_vertex = result if isinstance(result, dict) else {"result": result}
            response_parts.append(Part.from_function_response(name=fc.name, response=resp_for_vertex))

        try:
            response = chat.send_message(response_parts)
        except Exception as e:
            break

    after = snapshot_events(env)
    after_days = filter_by_days(after, display_days)
    final_output = next(
        (s["content"] for s in reversed(trajectory) if s["role"] == "assistant"), ""
    )
    expected = query_dict.get("expected_behavior", "")

    verdict = evaluate_trajectory(eval_model, query_text, final_output, expected, before_days, after_days)
    return verdict


def main():
    # Init credentials
    with open(str(CREDENTIALS_PATH)) as f:
        cd = json.load(f)
    creds = OAuth2Credentials(
        token=None, refresh_token=cd["refresh_token"],
        client_id=cd["client_id"], client_secret=cd["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(google.auth.transport.requests.Request())
    vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)
    refresh_req = google.auth.transport.requests.Request()

    # Always use 2.0-flash for eval judge
    eval_model = GenerativeModel("gemini-2.0-flash-001", system_instruction=[EVAL_SYSTEM_PROMPT])

    # Collect test queries
    tasks = []
    for cal_idx in TEST_CALENDARS:
        query_path = os.path.join(QUERY_DIR, f"{cal_idx}.txt")
        cal_path = os.path.join(JSON_CALENDAR_DIR, f"{cal_idx}.txt")
        if not os.path.exists(query_path) or not os.path.exists(cal_path):
            print(f"Skipping cal {cal_idx}: missing files")
            continue
        queries = json.load(open(query_path))
        for qi, q in enumerate(queries):
            tasks.append({"cal_idx": cal_idx, "qi": qi, "query": q, "cal_path": cal_path})

    print(f"Test set: {len(tasks)} queries from calendars {TEST_CALENDARS}")
    print(f"Models: {MODELS_TO_TEST}")
    print()

    # Run each model
    all_results = {}

    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"TESTING: {model_name}")
        print(f"{'='*60}")

        gen_model = GenerativeModel(
            model_name,
            tools=[CALENDAR_TOOL],
            system_instruction=[SYSTEM_PROMPT],
        )

        cat_results = defaultdict(lambda: {"correct": 0, "total": 0})
        total_correct = 0

        for i, task in enumerate(tasks):
            # Refresh token if needed
            if not creds.valid or creds.expired:
                creds.refresh(refresh_req)
                vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)
                gen_model = GenerativeModel(model_name, tools=[CALENDAR_TOOL], system_instruction=[SYSTEM_PROMPT])
                eval_model = GenerativeModel("gemini-2.0-flash-001", system_instruction=[EVAL_SYSTEM_PROMPT])
                print("  [TOKEN REFRESHED]")

            q = task["query"]
            cat = q.get("category", "unknown")
            cat_short = cat.split("(")[0].strip()

            print(f"  [{i+1}/{len(tasks)}] cal={task['cal_idx']} [{cat_short[:20]}] {q['query'][:40]}...", end=" ", flush=True)

            try:
                verdict = run_single(gen_model, eval_model, task["cal_path"], q)
                print(verdict, flush=True)
            except Exception as e:
                verdict = "Error"
                print(f"ERROR: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()

            cat_results[cat]["total"] += 1
            if verdict == "Correct":
                cat_results[cat]["correct"] += 1
                total_correct += 1

            time.sleep(0.3)

        # Summary for this model
        print(f"\n--- {model_name} Results ---")
        print(f"Overall: {total_correct}/{len(tasks)} ({total_correct/len(tasks)*100:.1f}%)")
        for cat in sorted(cat_results):
            r = cat_results[cat]
            pct = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
            short = cat.split("(")[0].strip()
            print(f"  {short:<40} {r['correct']}/{r['total']} ({pct:.0f}%)")

        all_results[model_name] = {
            "total_correct": total_correct,
            "total": len(tasks),
            "per_category": {k: dict(v) for k, v in cat_results.items()},
        }

    # Final comparison
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")

    # Get all categories
    all_cats = sorted(set(
        cat for r in all_results.values() for cat in r["per_category"]
    ))

    header = f"{'Category':<40}"
    for model_name in MODELS_TO_TEST:
        short_name = model_name.replace("gemini-", "").replace("-001", "")
        header += f" {short_name:>12}"
    print(header)
    print("-" * (40 + 13 * len(MODELS_TO_TEST)))

    for cat in all_cats:
        short = cat.split("(")[0].strip()
        row = f"{short:<40}"
        for model_name in MODELS_TO_TEST:
            r = all_results[model_name]["per_category"].get(cat, {"correct": 0, "total": 0})
            if r["total"] > 0:
                row += f" {r['correct']}/{r['total']} ({r['correct']/r['total']*100:.0f}%)"
            else:
                row += f" {'N/A':>12}"
        print(row)

    print("-" * (40 + 13 * len(MODELS_TO_TEST)))
    row = f"{'TOTAL':<40}"
    for model_name in MODELS_TO_TEST:
        r = all_results[model_name]
        row += f" {r['total_correct']}/{r['total']} ({r['total_correct']/r['total']*100:.1f}%)"
    print(row)

    # Save results
    out_path = os.path.join(SFT_DATA_DIR, "teacher_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
